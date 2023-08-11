# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import torch

from functools import partial

from ft_datasets.utils import Concatenator

from ft_datasets import (
    get_grammar_dataset,
    get_alpaca_dataset,
    get_samsum_dataset,
)
from typing import Optional


def get_completion_dataset(path: str, tokenizer, split: str = "train"):
    import json
    from datasets import Dataset

    data = []
    with open(path, 'r') as file:
        for line in file:
            data.append(json.loads(line))

    dataset = Dataset.from_dict({
        key: [item[key] for item in data] for key in data[0]},
    )

    def join_fields(example):
        joined_text = example['completion'] + '\n' + example['prompt']
        return {'text': joined_text}

    dataset = dataset.map(join_fields)

    dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        remove_columns=list(dataset.features),
    ).map(Concatenator(), batched=True)

    return dataset
 


    


DATASET_PREPROC = {
    "alpaca_dataset": partial(get_alpaca_dataset, max_words=224),
    "grammar_dataset": get_grammar_dataset,
    "samsum_dataset": get_samsum_dataset,
    "completion": get_completion_dataset,
    
}


def get_preprocessed_dataset(
    tokenizer, dataset_config, split: str = "train"
) -> torch.utils.data.Dataset:
    if not dataset_config.dataset in DATASET_PREPROC:
        return DATASET_PREPROC[dataset_config.dataset](
            dataset_config,
            tokenizer,
            get_split(),
        )


    def get_split():
        return (
            dataset_config.train_split
            if split == "train"
            else dataset_config.test_split
        )
    
    return DATASET_PREPROC[dataset_config.dataset](
        dataset_config,
        tokenizer,
        get_split(),
    )
