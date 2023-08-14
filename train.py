import argparse
import os
import shutil
from subprocess import call
import logging
import subprocess
from typing import Optional
from zipfile import ZipFile
import psutil

import torch
from cog import BaseModel, Input, Path
from tensorizer import TensorSerializer
from transformers import LlamaForCausalLM

from config import BASE_WEIGHTS_PATH, download_file, LOCAL_BASE_WEIGHTS, log_memory_stuff
from scripts.utils import maybe_download_with_pget

MODEL_OUT = "/src/tuned_weights.tensors"
CHECKPOINT_DIR = "checkpoints"
SAVE_STRATEGY = "epoch"
DIST_OUT_DIR = "tmp/model"


class TrainingOutput(BaseModel):
    weights: Path


def train(
    train_data: Path = Input(
        description="path to data file to use for fine-tuning your model"
    ),
    num_epochs: int = Input(
        description="number of training epochs", ge=1, default=1
    ),
) -> TrainingOutput:
    # input_weights = BASE_WEIGHTS_PATH
    # print('*' * 80)
    # print(f"Input Weights: {input_weights}")

    # if 'http' in input_weights or 'gs' in input_weights:
    #     # doing this once instead of 4x
    #     download_file(input_weights, LOCAL_BASE_WEIGHTS)
    #     input_weights = LOCAL_BASE_WEIGHTS


    print(f"LOCAL_BASE_WEIGHTS: {LOCAL_BASE_WEIGHTS}")
    print(f"BASE_WEIGHTS_PATH: {BASE_WEIGHTS_PATH}")

    N_SHARDS = 2
    REMOTE_FILES_TO_DOWNLOAD = [
        f"model-{str(i+1).zfill(5)}-of-{str(N_SHARDS).zfill(5)}.safetensors"
        for i in range(N_SHARDS)
    ]

    REMOTE_FILES_TO_DOWNLOAD += [
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "tokenizer.model",
    ]

    # DEFAULT_REMOTE_INFERENCE_WEIGHTS_PATH = "https://storage.googleapis.com/replicate-weights/llama-2-7b"

    weights_path = maybe_download_with_pget(
        LOCAL_BASE_WEIGHTS, BASE_WEIGHTS_PATH, REMOTE_FILES_TO_DOWNLOAD,
    )

    root_path = os.getcwd()

    output_dir = DIST_OUT_DIR
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    num_gpus = torch.cuda.device_count()
    num_gpus_flag = f"--num_gpus={num_gpus}"

    print(f"Local Output Dir: {output_dir}")
    print(f"Number of GPUs: {num_gpus}")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_DATASETS_CACHE"] = "/src/.hf-cache"

    def _arg_if_present(var, var_name):
        """Need to wrap any arguments whose default value in train() is `None`"""
        if var:
            return f" --{var_name} {var}"
        return " "
    
    
    print(f"weights_path: {weights_path}")
    output_dir = 'training_output'

    args = [
        "torchrun",
        "--nnodes", "1",
        "--nproc_per_node", "4",
        "llama_recipes/llama_finetuning.py",
        "--enable_fsdp",
        "--use_peft",
        "--model_name", weights_path,
        "--pure_bf16",
        "--output_dir", output_dir,
        "--run_validation", "False",
        "--data_path", train_data,
        "--num_epochs", "1",
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
