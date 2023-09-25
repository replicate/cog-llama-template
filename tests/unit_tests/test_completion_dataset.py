import pytest

import sys

sys.path.append(".")

from llama_recipes.ft_datasets.completion_dataset import (
    load_data,
    format_data,
    tokenize_data,
)

from dataclasses import dataclass


@pytest.fixture(scope="session")
def dataset_config():
    @dataclass
    class completion:
        dataset: str = "completion"
        train_split: str = "train"
        test_split: str = "val"
        data_path: str = "tests/data/200_samples.jsonl"
        num_validation_samples: int = 100
        run_validation: bool = True
        validation_data_path: str = None
        pack_sequences: bool = True
        wrap_packed_sequences: bool = True
        chunk_size: int = 100

    return completion


@pytest.fixture(scope="session")
def tokenizer():
    from transformers import LlamaTokenizer

    tokenizer = LlamaTokenizer.from_pretrained(
        "tests/assets/llama_tokenizer", legacy=False
    )
    tokenizer.add_special_tokens(
        {
            "pad_token": "<PAD>",
            "eos_token": "</s>",
            "bos_token": "<s>",
        }
    )
    return tokenizer


def test__load_data_train(dataset_config):
    dataset_config.run_validation = False
    dataset = load_data(dataset_config, split="train")
    assert len(dataset) == 200
    for example in dataset:
        assert example["text"].startswith("Write a response to the following message")


def test__load_data_train_with_val_split(dataset_config):
    dataset_config.run_validation = True
    train_dataset = load_data(dataset_config, split="train")

    train_texts = [example["text"] for example in train_dataset]

    val_dataset = load_data(dataset_config, split="val")
    assert len(val_dataset) == 100
    for example in val_dataset:
        assert example["text"].startswith("Write a response to the following message")
        assert example["text"] not in train_texts


@pytest.fixture(scope="session")
def dataset(dataset_config):
    dataset_config.run_validation = False
    dataset = load_data(dataset_config, split="train")
    return dataset


def test_format_data(dataset, tokenizer):
    formatted_data = format_data(dataset, tokenizer, dataset_config)
    for example in formatted_data:
        assert example["text"].startswith("Write a response to the following message")
        assert example["text"].endswith(tokenizer.eos_token)


@pytest.fixture(scope="session")
def formatted_dataset(dataset, tokenizer):
    return format_data(dataset, tokenizer, dataset_config)


def test_tokenize_data_with_wrapped_packing(
    formatted_dataset, tokenizer, dataset_config
):
    dataset_config.pack_sequences = True
    dataset_config.wrap_packed_sequences = True

    tokenized_data = tokenize_data(formatted_dataset, tokenizer, dataset_config)

    for tokenized_example in tokenized_data:
        assert "labels" in tokenized_example

    decoded_data = tokenizer.batch_decode(
        tokenized_data["input_ids"], skip_special_tokens=False
    )

    decoded_data = tokenizer.batch_decode(
        tokenized_data["input_ids"], skip_special_tokens=True
    )

    at_least_one_wrapped = False
    for example in decoded_data:
        if not example.startswith("Write a response to the following message"):
            at_least_one_wrapped = True

    assert at_least_one_wrapped

    for tokenized_example in tokenized_data["input_ids"]:
        assert len(tokenized_example) == dataset_config.chunk_size


def test_tokenize_data_without_wrapped_packing_small_chunk(
    formatted_dataset, tokenizer, dataset_config
):
    dataset_config.pack_sequences = True
    dataset_config.wrap_packed_sequences = False
    dataset_config.chunk_size: int = 100

    tokenized_data = tokenize_data(formatted_dataset, tokenizer, dataset_config)

    for tokenized_example in tokenized_data:
        assert tokenized_example["input_ids"][-1] == tokenizer.eos_token_id
        assert "labels" in tokenized_example

    decoded_data = tokenizer.batch_decode(
        tokenized_data["input_ids"], skip_special_tokens=False
    )

    for example in decoded_data:
        prefix = " ".join(
            [tokenizer.bos_token, "Write a response to the following message"]
        )
        assert example.startswith(prefix)

    recovered_data = []
    for decoded_sequence in decoded_data:
        for decoded_example in decoded_sequence.split(tokenizer.eos_token)[:-1]:
            decoded_example = decoded_example.removeprefix(tokenizer.bos_token + " ")
            decoded_example += tokenizer.eos_token
            recovered_data.append(decoded_example)

    for i in range(len(recovered_data)):
        assert recovered_data[i] == formatted_dataset[i]["text"]


def test_tokenize_data_without_wrapped_packing_large_chunk(
    formatted_dataset, tokenizer, dataset_config
):
    dataset_config.pack_sequences = True
    dataset_config.wrap_packed_sequences = False
    dataset_config.chunk_size: int = 2048

    tokenized_data = tokenize_data(formatted_dataset, tokenizer, dataset_config)

    for tokenized_example in tokenized_data:
        assert tokenized_example["input_ids"][-1] == tokenizer.eos_token_id
        assert "labels" in tokenized_example

    decoded_data = tokenizer.batch_decode(
        tokenized_data["input_ids"], skip_special_tokens=False
    )

    for example in decoded_data:
        prefix = " ".join(
            [tokenizer.bos_token, "Write a response to the following message"]
        )
        assert example.startswith(prefix)

    recovered_data = []
    for decoded_sequence in decoded_data:
        for decoded_example in decoded_sequence.split(tokenizer.eos_token)[:-1]:
            decoded_example = decoded_example.removeprefix(tokenizer.bos_token + " ")
            decoded_example += tokenizer.eos_token
            recovered_data.append(decoded_example)

    for i in range(len(recovered_data)):
        assert recovered_data[i] == formatted_dataset[i]["text"]


def test_tokenize_data_without_packing(formatted_dataset, tokenizer, dataset_config):
    dataset_config.pack_sequences = False
    tokenized_data = tokenize_data(formatted_dataset, tokenizer, dataset_config)

    for tokenized_example in tokenized_data["input_ids"]:
        assert tokenized_example[-1] == tokenizer.eos_token_id

    decoded_data = tokenizer.batch_decode(
        tokenized_data["input_ids"], skip_special_tokens=True
    )
    for i, example in enumerate(decoded_data):
        assert example.startswith("Write a response to the following message")
        assert example + tokenizer.eos_token == formatted_dataset[i]["text"]
