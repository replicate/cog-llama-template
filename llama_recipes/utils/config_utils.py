# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import inspect
from dataclasses import fields
from peft import (
    LoraConfig,
    AdaptionPromptConfig,
    PrefixTuningConfig,
)

from transformers import BitsAndBytesConfig

import configs.datasets as datasets
from configs import (
    lora_config,
    llama_adapter_config,
    prefix_config,
    train_config,
    qlora_config,
    bitsandbytes_config,
)
from .dataset_utils import DATASET_PREPROC


def update_config(config, **kwargs):
    if isinstance(config, (tuple, list)):
        for c in config:
            update_config(c, **kwargs)
    else:
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
            elif "." in k:
                # allow --some_config.some_param=True
                config_name, param_name = k.split(".")
                if type(config).__name__ == config_name:
                    if hasattr(config, param_name):
                        setattr(config, param_name, v)
                    else:
                        # In case of specialized config we can warm user
                        print(f"Warning: {config_name} does not accept parameter: {k}")
            elif isinstance(config, train_config):
                print(f"Warning: unknown parameter {k}")


def generate_peft_config(peft_method, kwargs):
    # Config mapping for train_config.peft_method to its corresponding config class
    config_mapping = {
        "lora": lora_config,
        "llama_adapter": llama_adapter_config,
        "prefix": prefix_config,
        "bitsandbytes_config": bitsandbytes_config,
        "qlora": qlora_config,
        # Add other mappings as needed
    }

    # Mapping from config class to its corresponding PEFT config
    peft_config_mapping = {
        lora_config: LoraConfig,
        llama_adapter_config: AdaptionPromptConfig,
        prefix_config: PrefixTuningConfig,
        bitsandbytes_config: BitsAndBytesConfig,
        qlora_config: LoraConfig,
        # Add other mappings as needed
    }

    # Step 2: Updated assertion
    assert peft_method in config_mapping.keys(), f"Peft config not found: {peft_method}"

    # Step 3: Fetch the correct configuration class based on train_config.peft_method
    config = config_mapping[peft_method]
    update_config(config, **kwargs)
    params = {k.name: getattr(config, k.name) for k in fields(config)}

    # Step 5: Fetch the correct PEFT config based on the configuration class
    peft_config_class = peft_config_mapping[config]
    peft_config = peft_config_class(**params)

    return peft_config


# def generate_peft_config(train_config, kwargs):
#     configs = (lora_config, llama_adapter_config, prefix_config, qlora_config)
#     peft_configs = (LoraConfig, AdaptionPromptConfig, PrefixTuningConfig)
#     names = tuple(c.__name__.rstrip("_config") for c in configs)

#     assert train_config.peft_method in names, f"Peft config not found: {train_config.peft_method}"

#     config = configs[names.index(train_config.peft_method)]
#     update_config(config, **kwargs)
#     params = {k.name: getattr(config, k.name) for k in fields(config)}
#     peft_config = peft_configs[names.index(train_config.peft_method)](**params)

#     return peft_config


def generate_dataset_config(train_config, kwargs):
    names = tuple(DATASET_PREPROC.keys())

    assert train_config.dataset in names, f"Unknown dataset: {train_config.dataset}"

    dataset_config = {k: v for k, v in inspect.getmembers(datasets)}[
        train_config.dataset
    ]
    update_config(dataset_config, **kwargs)

    return dataset_config
