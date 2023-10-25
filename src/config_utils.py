"""
An entirely self-contained config parsing util that should, if all goes well, dramatically simplify our configuration. 
"""


from typing import List, Optional

from pydantic import BaseModel

class Weights(BaseModel):
    local_path: str
    remote_path: str
    remote_files: List[str]

def get_fp16_file_list(n_shards: int):
    """
    Assumes safetensors
    """
    base_files = [f"model-0000{val}-of-0000{n_shards}.safetensors" for val in range (1, n_shards+1)]
    base_files += ["config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "model.safetensors.index.json"]
    return base_files


def get_gptq_file_list(base_model_name: str):
    """
    name of <model>.safetensors varies
    """
    base_files = [base_model_name]
    base_files += [
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "quantize_config.json"]
    return base_files

def exllama_kwargs(weights: Weights, config_overrides: Optional[dict] = None):
    exllama_default = {
        "weights": weights,
        "fused_attn": True
    }
    if config_overrides:
        exllama_default.update(config_overrides)
    return exllama_default

def vllm_kwargs(weights: Weights, config_overrides: Optional[dict] = None):
    vllm_default = {
        "weights": weights,
        "dtype": "auto",
    }
    if config_overrides:
        vllm_default.update(config_overrides)
    return vllm_default


