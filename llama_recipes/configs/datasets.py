# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class samsum_dataset:
    dataset: str = "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    input_length: int = 2048


@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "ft_datasets/grammar_dataset/gtrain_10k.csv"
    test_split: str = "ft_datasets/grammar_dataset/grammar_validation.csv"
    input_length: int = 2048


@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "ft_datasets/alpaca_data.json"


@dataclass
class completion:
    """
    A generic class for completion format datasets. Format is expected
    to be JSONL like:
        ```
        {"text": "..."}
        ```
    or
        ```
        {"text": "prompt ...", "completion": "..."}
        ```
    """

    dataset: str = "completion"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = None
    num_validation_samples: int = 100
    run_validation: bool = True
    validation_data_path: str = None
    pack_sequences: bool = True
    wrap_packed_sequences: bool = False
    chunk_size: int = 2048
    max_seq_length: int = 4096
