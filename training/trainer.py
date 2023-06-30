import argparse
import copy
import json
import os
import time
import logging
from collections import OrderedDict
from typing import List, Optional, Union

import torch
from cog import Input, Path
from peft import (LoraConfig, get_peft_model)
from torch.utils.data import Dataset
from transformers import LlamaForCausalLM, Trainer, TrainingArguments, AutoConfig
from tensorizer import TensorDeserializer
from tensorizer.utils import no_init_or_tensor
import sys
sys.path.append('/src/')

from config import DEFAULT_MODEL_NAME, load_tokenizer, load_tensorizer

MODEL_OUT = "/src/tuned_weights.tensors"
CHECKPOINT_DIR = "checkpoints"
SAVE_STRATEGY = "epoch"
DIST_OUT_DIR = "tmp/model"
IGNORE_INDEX = -100


class DatasetBuilder:
    """Dataset agnostic class to take in input_ids and labels and spit out tokens"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def batch_tokenize(self, texts):
        """Tokenizes text. Presently doesn't pad inputs, just returns input ids."""
        tokenized = [
            self.tokenizer(
                prompt, return_tensors="pt", padding="longest", truncation=True
            ).input_ids
            for prompt in texts
        ]
        return tokenized

    def construct_dataset(self, input_data):
        prompts = [val["prompt"] for val in input_data]
        tokenized_input_ids = self.batch_tokenize(prompts)
        labels = [val["completion"] for val in input_data]
        tokenized_labels = self.batch_tokenize(labels)
        return TuneDataset(tokenized_input_ids, tokenized_labels)


class CausalDatasetBuilder(DatasetBuilder):
    """Builds generative dataset for Causal LM."""

    def __init__(self, tokenizer, train_on_prompt=True):
        super().__init__(tokenizer)
        self.train_on_prompt = train_on_prompt

    def construct_dataset(self, input_data):
        labels = [
            val["prompt"] + "\n" + val["completion"] + self.tokenizer.eos_token
            for val in input_data
        ]
        input_ids = [val.squeeze() for val in self.batch_tokenize(labels)]
        labels = copy.deepcopy(input_ids)
        if self.train_on_prompt:
            return TuneDataset(input_ids, labels)
        # masking prompt
        prompts = [val["prompt"] for val in input_data]
        tokenized_prompts = self.batch_tokenize(prompts)
        prompt_lens = [val.shape[1] for val in tokenized_prompts]

        for label, source_len in zip(labels, prompt_lens):
            label[:source_len] = IGNORE_INDEX
        return TuneDataset(input_ids, labels)


class TuneDataset(Dataset):
    """Dead simple torch dataset wrapper. Attention masks are created in collator"""

    def __init__(self, input_ids, labels):
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


class SequenceDataCollator:
    """Collate examples for dynamic batch construction in supervised fine-tuning."""

    def __init__(self, tokenizer, multiple_of=None):
        self.tokenizer = tokenizer
        self.multiple_of = multiple_of
        self.cache_count = 0

    def pad_to_multiple(self, tensor, value):
        # taking advantage of tensor cores, perhaps
        multiple = self.multiple_of
        target_length = (tensor.size(0) + multiple - 1) // multiple * multiple
        return torch.nn.functional.pad(
            tensor, (0, target_length - tensor.size(0)), value=value
        )

    def __call__(self, instances):
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        if self.multiple_of:
            input_ids = [
                self.pad_to_multiple(val, self.tokenizer.pad_token_id)
                for val in input_ids
            ]
            labels = [self.pad_to_multiple(val, -100) for val in labels]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )  # -100 tells torch to ignore these tokens in loss computation.

        if self.cache_count < 1:
            torch.cuda.empty_cache()
            self.cache_count += 1

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def load_data(path):
    if path.suffix == ".json":
        return load_json(path)
    elif path.suffix == ".jsonl":
        return load_jsonl(path)
    else:
        raise Exception(
            f"file type {path} not supported. Currently supported types are json, jsonl"
        )


def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            json_object = json.loads(line)
            data.append(json_object)
    return data


def load_json(path):
    """Loads a single json blob"""
    with open(path, "r") as f:
        data = json.load(f)
    return data


def load_model(model_name_or_path):
    print(f"Rank : {os.environ['RANK']}, device: {torch.cuda.current_device()}")
    torch.cuda.set_device(int(os.environ['RANK']))
    if model_name_or_path is None:
        model_name_or_path = DEFAULT_MODEL_NAME
    model = load_tensorizer(model_name_or_path, plaid_mode=False, cls=LlamaForCausalLM)
    return model


def load_peft_model(
    model_name_or_path, lora_rank: int, lora_alpha: int, lora_dropout: float, lora_target_modules: Optional[Union[List[str], str]]
):
    if lora_target_modules:
        lora_target_modules = lora_target_modules.split(",")
    print("Using LoRA...")
    model = load_model(model_name_or_path)
        
    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=lora_target_modules,
        task_type="CAUSAL_LM",
    )
    print(f"LoRA config: {config}")
    model = get_peft_model(model, config)
    return model


def train(
    train_data: Path = Input(
        description="path to data file to use for fine-tuning your model"
    ),
    eval_data: Path = Input(
        description="path to optional evaluation data file to use for model eval",
        default=None,
    ),
    weights: Path = Input(
        description="location of weights that are going to be fine-tuned", default=None
    ),
    train_batch_size: int = Input(description="batch size per GPU", default=4, ge=1),
    gradient_accumulation_steps: int = Input(
        description="number of training steps to update gradient for before performing a backward pass",
        default=8,
    ),
    lr_scheduler_type: str = Input(
        description="learning rate scheduler",
        default="cosine",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "inverse_sqrt",
            "constant",
            "constant_with_warmup",
        ],
    ),
    learning_rate: float = Input(
        description="learning rate, for learning!", default=2e-4, ge=0
    ),
    warmup_ratio: float = Input(
        description="pct of steps for a linear learning rate warmup",
        ge=0,
        le=0.5,
        default=0.03,
    ),
    num_train_epochs: int = Input(
        description="number of training epochs", ge=1, default=1
    ),
    max_steps: int = Input(
        description="number of steps to run training for, supersedes num_train_epochs",
        default=-1,
        ge=0,
    ),
    logging_steps: int = Input(
        description="number of steps between logging epoch & loss", default=1
    ),
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    lora_target_modules: Optional[Union[List[str], str]] = None, 
    local_output_dir: str = None,
    local_rank: int = -1,
    deepspeed: str = None
) -> None:
    print("Loading model...")
    model = load_peft_model(weights, lora_rank, lora_alpha, lora_dropout, lora_target_modules)
    tokenizer = load_tokenizer()

    print(f"Loading dataset {train_data}...")
    print(train_data)
    train_data = load_data(train_data)
    p = CausalDatasetBuilder(tokenizer)
    train_dataset = p.construct_dataset(train_data)
    eval_dataset = None
    if eval_data:
        eval_data = load_json(eval_data)
        eval_dataset = p.construct_dataset(eval_data)

    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("high")


    print("Training...")
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=TrainingArguments(
            output_dir=CHECKPOINT_DIR,
            per_device_train_batch_size=train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            save_strategy="no",
            logging_steps=logging_steps,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            max_steps=max_steps,
            tf32=True,
            fp16=True,
            half_precision_backend="cuda_amp",
            deepspeed=deepspeed,
            local_rank=local_rank
        ),
        data_collator=SequenceDataCollator(tokenizer, 8),  # depends on bf16 value
    )
    trainer.train()
    print("model saving!")
    model.save_pretrained(local_output_dir)
    return


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
        "--deepspeed", type=str, default=None, help="Path to deepspeed config file."
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
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
    )
    parser.add_argument(
        "--local_output_dir",
        type=str,
        help="Write directly to this local path",
        required=True,
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Provided by deepspeed to identify which instance this process is when performing multi-GPU training.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help="Number of training steps to run, overrides num_train_epochs, useful for testing",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="Number of training steps to run, overrides num_train_epochs, useful for testing",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.4,
        help="Number of training steps to run, overrides num_train_epochs, useful for testing",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str, 
        default=None,
        help="Comma-separated list of lora modules to target, i.e. 'q_proj,v_proj'. Leave blank for default"
    )
    some_args = parser.parse_args()
    train(**vars(some_args))
