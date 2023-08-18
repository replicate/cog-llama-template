
import pytest

import sys
sys.path.append('.')

from llama_recipes.ft_datasets.completion_dataset import (
    get_completion_dataset, 
    load_data,
    format_data,
    tokenize_data
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
        pack: bool = True
    
    return completion

@pytest.fixture(scope="session")
def tokenizer():
    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained("tests/assets/llama_tokenizer", legacy=False)
    tokenizer.add_special_tokens(
            {
                "pad_token": "<PAD>",
            }
        )
    return tokenizer

def test__load_data_train(dataset_config):
    dataset_config.run_validation = False
    dataset = load_data(dataset_config, split='train')
    assert len(dataset) == 200
    for example in dataset:
        assert example['text'].startswith("Write a response to the following message")


def test__load_data_train_with_val_split(dataset_config):
    dataset_config.run_validation = True
    train_dataset = load_data(dataset_config, split='train')

    train_texts = [example['text'] for example in train_dataset]

    val_dataset = load_data(dataset_config, split='val')
    assert len(val_dataset) == 100
    for example in val_dataset:
        assert example['text'].startswith("Write a response to the following message")
        assert example['text'] not in train_texts

@pytest.fixture(scope="session")
def dataset(dataset_config):
    dataset_config.run_validation = False
    dataset = load_data(dataset_config, split='train')
    return dataset

def test_format_data(dataset, tokenizer):
    formatted_data = format_data(dataset, tokenizer, dataset_config)
    assert formatted_data[0]['text'].endswith(tokenizer.eos_token)
    for example in formatted_data:
        assert len(formatted_data == 1)

@pytest.fixture(scope="session")
def formatted_dataset(dataset, tokenizer):
    return format_data(dataset, tokenizer, dataset_config)

def test_tokenize_data(formatted_dataset, tokenizer, dataset_config):
    dataset_config.pack = True
    tokenized_data = tokenize_data(formatted_dataset, tokenizer, dataset_config)
    print(type(tokenized_data))
    print(len(tokenized_data))

    decoded_data = tokenizer.batch_decode(tokenized_data['input_ids'], skip_special_tokens=True)
    print(decoded_data[0])
    print('-----')
    print(decoded_data[1])
    for example in decoded_data:
        assert example.endswith(tokenizer.eos_token)
        assert example.startswith("Write a response to the following message")
    # assert tokenized_data['input_ids'].shape[0] == 1000
  






