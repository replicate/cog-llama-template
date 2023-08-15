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


def _build_subprocess_command(
    prefix: str,
    train_args: tp.Dict[str, tp.Any],
    arg_mapping: tp.Dict[str, str],
    train_config: Optional[tp.Type] = None
) -> list:
    subprocess_command = prefix
    
    for train_arg, subprocess_arg in arg_mapping.items():
        if train_arg in train_args and subprocess_arg is not None: 
            # Extract actual value based on the type of input
            value = train_args[train_arg]
            
            if isinstance(value, Path):
                value = str(value)  # Convert Path object to string   

            if train_config is not None:
                # If config is provided, validate the argument
                f = next((field for field in fields(train_config) if field.name == subprocess_arg), None)
                
                if f is None:
                    raise ValueError(f"No field named {subprocess_arg} in config dataclass")
                
                if not isinstance(value, f.type):
                    raise ValueError(f"Invalid type for argument {f.name}: expected {f.type}, but got {type(value)}")
                
            subprocess_command.append(f"--{subprocess_arg}")
            subprocess_command.append(str(value))
            
    return subprocess_command


def train(
    train_data: Path = Input(
        description="path to data file to use for fine-tuning your model"
    ),
    num_train_epochs: int = Input(
        description="number of training epochs", 
        ge=1, default=1,
    ),
    train_batch_size: int = Input(
        description="Global batch size. This specifies the batch size that will be used to calculate gradients.",
        default=4, ge=1,
    ),
    micro_batch_size: int = Input(
        description="Micro batch size. This specifies the on-device batch size, if this is less than `train_batch_size`, gradient accumulation will be activated.", 
        default=4, ge=1
    ),
    eval_data: Path = Input(
        description="path to optional evaluation data file to use for model eval",
        default=None,
    ),
    eval_batch_size: int = Input(
        description="Batch size for evaluation", 
        default=4, ge=1
    ),
    run_validation: bool = Input(
        description="Whether to run validation during training.", 
        default=False
    ),
    learning_rate: float = Input(
        description="learning rate, for learning!", default=1e-4, ge=0
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
    # ),
    # lora_rank: int = Input(
    #     description="Rank of the lora matrices", default=8, ge=1),
    # lora_alpha: int = Input(description="Alpha parameter for scaling lora weights; weights are scaled by alpha/rank", default=16, ge=1),
    # lora_dropout: float = Input(description="Dropout for lora training", default=0.1, ge=0.0, le=1.0),
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

    arg_mapping = {
        'train_data': 'data_path',
        'train_batch_size': 'batch_size_training',
        'num_train_epochs': 'num_epochs',
        'micro_batch_size': 'micro_batch_size',
        'eval_data': None,  # No equivalent in train_config, so set as None
        'eval_batch_size': 'val_batch_size',
        'run_validation': 'run_validation',
        'seed': 'seed',
        'weights': None,  # No equivalent in train_config, so set as None
        'learning_rate': 'lr',
        'warmup_ratio': None,  # No equivalent in train_config, so set as None
        'max_steps': None,  # No equivalent in train_config, so set as None
        'logging_steps': None,  # No equivalent in train_config, so set as None
        'lora_rank': None,  # No equivalent in train_config, so set as None
        'lora_alpha': None,  # No equivalent in train_config, so set as None
        'lora_dropout': None,  # No equivalent in train_config, so set as None
        'lora_target_modules': None  # No equivalent in train_config, so set as None
    }

    command_prefix = f"torchrun --nnodes=1 --nproc_per_node={num_gpus} llama_recipes/llama_finetuning.py "
    command_prefix += f"--enable_fsdp --use_peft --pure_bf16 --model_name {model_path} --output_dir {output_dir} "
    command_prefix = command_prefix.split(" ")
    train_args = locals()
    args = _build_subprocess_command(command_prefix, train_args, arg_mapping, train_config)
    print("Training args: ",  ' '.join(args))

    def _arg_if_present(var, var_name):
        """Need to wrap any arguments whose default value in train() is `None`"""
        if var:
            return f" --{var_name} {var}"
        return " "
    

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