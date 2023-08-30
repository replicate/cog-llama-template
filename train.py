import argparse
import os
import shutil
from subprocess import call
import logging
import subprocess
from typing import Optional
from zipfile import ZipFile
import psutil
import typing as tp
from dataclasses import fields, is_dataclass


import torch
from cog import BaseModel, Input, Path
from tensorizer import TensorSerializer
from transformers import LlamaForCausalLM
from llama_recipes.configs.training import train_config

from config import (
    download_file,
    LOCAL_TRAINING_WEIGHTS_PATH, 
    REMOTE_TRAINING_WEIGHTS_PATH, 
    LOCAL_TRAINING_WEIGHTS_CONFIG_PATH,
    REMOTE_TRAINING_WEIGHTS_CONFIG_PATH,
    REMOTE_TRAINING_FILES_TO_DOWNLOAD,
    log_memory_stuff
)

from src.utils import maybe_download_with_pget


MODEL_OUT = "/src/tuned_weights.tensors"
CHECKPOINT_DIR = "checkpoints"
SAVE_STRATEGY = "epoch"
OUTPUT_DIR = "training_output"


class TrainingOutput(BaseModel):
    weights: Path

def train(
    train_data: Path = Input(
        description="path to data file to use for fine-tuning your model"
    ),
    num_train_epochs: int = Input(
        description="number of training epochs", 
        ge=1, default=3,
    ),
    train_batch_size: int = Input(
        description="Global batch size. This specifies the batch size that will be used to calculate gradients.",
        default=4, ge=1,
    ),
    micro_batch_size: int = Input(
        description="Micro batch size. This specifies the on-device batch size, if this is less than `train_batch_size`, gradient accumulation will be activated.", 
        default=4, ge=1
    ),
    num_validation_samples: int = Input(
        description=("Number of samples to use for validation." \
                     "If `run_validation` is `True` and `validation_data` is not specified, this number of samples" \
                     "will be selected from the tail of the training data. If `validation_data` is specified, this" \
                     "number of samples will be selected from the head of the validation data, up to the size of the validation data."
        ),
        default=50, ge=1
    ),
    validation_data: Path = Input(
        description="path to optional evaluation data file to use for model eval",
        default=None,
    ),
    validation_batch_size: int = Input(
        description="Batch size for evaluation", 
        default=1, ge=1
    ),
    run_validation: bool = Input(
        description="Whether to run validation during training.", 
        default=True
    ),
    validation_prompt: str = Input(
        description="Prompt to use for generation during validation. If provided, a response to this prompt will be sampled and logged during validation.",
        default=None,
    ),
    learning_rate: float = Input(
        description="learning rate, for learning!", default=1e-4, ge=0
    ),
    pack_sequences: bool = Input(
        description="If 'True', sequences will be packed into a single sequences up to a given length. This improves computational efficiency.",
        default=False,
    ),
    wrap_packed_sequences: bool = Input(
        description="If 'pack_sequences' is 'True', this will wrap packed sequences across examples, ensuring a constant sequence length but breaking prompt formatting.",
        default=False
    ),
    chunk_size: int = Input(
        description="If 'pack_sequences' is 'True', this will chunk sequences into chunks of this size.",
        default=2048, ge=1
    ),
    seed: int = Input(
        description="random seed to use for training", 
        default=42
    ),
    # weights: Path = Input(
    #     description="location of weights that are going to be fine-tuned", default=None
    # ),
    #
    # warmup_ratio: float = Input(
    #     description="pct of steps for a linear learning rate warmup",
    #     ge=0,
    #     le=0.5,
    #     default=0.03,
    # ),

    # max_steps: int = Input(
    #     description="number of steps to run training for, supersedes num_train_epochs",
    #     default=-1,
    # ),
    # logging_steps: int = Input(
    #     description="number of steps between logging epoch & loss", default=1
    #),
    lora_rank: int = Input(
        description="Rank of the lora matrices", default=8, ge=1),
    lora_alpha: int = Input(description="Alpha parameter for scaling lora weights; weights are scaled by alpha/rank", default=16, ge=1),
    lora_dropout: float = Input(description="Dropout for lora training", default=0.05, ge=0.0, le=1.0),
    # lora_target_modules: str = Input(description="Comma-separated list of lora modules to target, i.e. 'q_proj,v_proj'. Leave blank for default.", default="q_proj,v_proj")
) -> TrainingOutput:

    weights = REMOTE_TRAINING_WEIGHTS_PATH

    if 'http' in weights:
       
        model_path = maybe_download_with_pget(
            LOCAL_TRAINING_WEIGHTS_PATH, 
            weights, 
            REMOTE_TRAINING_FILES_TO_DOWNLOAD,
        )

    root_path = os.getcwd()

    output_dir = OUTPUT_DIR
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    num_gpus = torch.cuda.device_count()

    print(f"Local Output Dir: {output_dir}")
    print(f"Number of GPUs: {num_gpus}")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_DATASETS_CACHE"] = "/src/.hf-cache"



    args = [
        # Hard coded for now
        "python3", "-m", "torch.distributed.run"
        f"--nnodes=1",
        f"--nproc_per_node={num_gpus}",
        f"llama_recipes/llama_finetuning.py",
        f"--enable_fsdp",
        f"--use_peft",
        f"--peft_method=lora",
        f"--model_name={model_path}",
        f"--pure_bf16",
        f"--output_dir={output_dir}",

        # User specified arguments -----

        # Preprocessing arguments
        f"--pack_sequences={pack_sequences}",
        f"--wrap_packed_sequences={wrap_packed_sequences}",
        f"--chunk_size={chunk_size}",

        # Train arguments
        f"--data_path={train_data}",
        f"--num_epochs={num_train_epochs}",
        f"--batch_size_training={train_batch_size}",
        f"--micro_batch_size={micro_batch_size}",
        f"--lr={learning_rate}",
        f"--lora_rank={lora_rank}",
        f"--lora_alpha={lora_alpha}",
        f"--lora_dropout={lora_dropout}",

        # Validation arguments
        f"--run_validation={'False' if not run_validation else 'True'}",
        f"--num_validation_samples={num_validation_samples}",
        f"--validation_data_path={validation_data}",
        f"--val_batch_size={validation_batch_size}",
        f"--validation_prompt={validation_prompt}",
        # Other arguments
        f"--seed={seed}",
    ]

    p = None
    try:
        p = subprocess.Popen(args, close_fds=False)
        p.wait()
        return_code = p.poll()
        if return_code != 0:
            raise Exception(f"Training failed with exit code {return_code}! Check logs for details")
        out_path = "training_output.zip"

        directory = Path(output_dir)
        with ZipFile(out_path, "w") as zip:
            for file_path in directory.rglob("*"):
                print(file_path)
                zip.write(file_path, arcname=file_path.relative_to(directory))

        return TrainingOutput(weights=Path(out_path))
    finally: 
        if p and p.poll() is None:
            top = psutil.Process(p.pid)
            children = top.children(recursive=True)
            for process in children + [top]:
                process.terminate()
            _, alive = psutil.wait_procs(children + [top], timeout = 5)
            if alive:
                for process in alive:
                    print(f"process {process.pid} survived termination")
            else:
                print("terminated all processes successfully")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune a language model on a text dataset"
    )
    parser.add_argument(
        "--train_data", type=Path, required=True, help="Path to the json dataset"
    )
    parser.add_argument(
        "--eval_data",
        type=Path,
        required=False,
        help="Path to the json dataset",
        default=None,
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="The model class to fine-tune on HF or as a local path (e.g. 'google/flan-t5-xxl'",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, required=True, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size for training"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="Number of warmup steps for the learning rate scheduler",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="Number of training steps to run, overrides num_train_epochs, useful for testing",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of training steps to run, overrides num_train_epochs, useful for testing",
    )
    parser.add_argument("--logging_steps", type=int, default=1)
    some_args = parser.parse_args()
    train(**vars(some_args))
