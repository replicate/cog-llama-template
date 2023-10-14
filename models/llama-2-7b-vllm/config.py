import os
import subprocess

import torch
from dotenv import load_dotenv
from transformers import LlamaTokenizer
from src.inference_engines.vllm_exllama_engine import ExllamaVllmEngine

from src.utils import get_env_var_or_default

load_dotenv()

MODEL_NAME = "llama-2-7b-vllm"
# INFERENCE CONFIGURATION
#######################################################################
# --------------------Notes--------------------------------------------
# We sometimes implement inference differently for models that have not
# been trained/fine-tuned vs. those that have been trained/fine-tuned. We refer to the
# former as "default" and the latter as "trained". Below, you can
# set your "default inference configuration" and your "trained
# inference configuration".
#
# GENERAL INFERENCE CONFIGURATION
# -------------------------------
# This section defines the general inference configuration,
# which is used for both trained and untrained models.
# -------------------------------

LOAD_IN_4BIT = False
TOKENIZER_PATH = f"models/{MODEL_NAME}/model_artifacts/default_inference_weights"
USE_SYSTEM_PROMPT = False
USE_EXLLAMA_FOR_UNTRAINED_WEIGHTS = False

# ENGINE CONFIGURATION
# -------------------------------
# Here we define the specific inference engine we intend to use for inference, and all appropriate kwargs.
# -------------------------------

from src.inference_engines.vllm_engine import vLLMEngine

ENGINE = ExllamaVllmEngine
LOCAL_DEFAULT_INFERENCE_WEIGHTS_PATH = f"models/{MODEL_NAME}/model_artifacts/default_inference_weights"

REMOTE_VLLM_WEIGHTS_PATH = get_env_var_or_default(
    "REMOTE_VLLM_INFERENCE_WEIGHTS_PATH",
    "remote/path/to/your/weights/here",
)

REMOTE_VLLM_INFERENCE_FILES_TO_DOWNLOAD = [
    "checklist.chk",
    "config.json",
    "model.safetensors.index.json",
    "params.json",
    "tokenizer.model",
    "tokenizer_checklist.chk",
    "consolidated.00.pth",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
]

EXLLAMA_ENGINE_KWARGS = {
    "fused_attn": True
}
VLLM_ENGINE_KWARGS = {
    "tokenizer_path": TOKENIZER_PATH, "dtype": "auto",     "vllm_model_info": 
        {
            "remote_path": REMOTE_VLLM_WEIGHTS_PATH,
            "remote_files": REMOTE_VLLM_INFERENCE_FILES_TO_DOWNLOAD,
            "local_path": f"models/{MODEL_NAME}/model_artifacts/lora_inference_weights"
        }
}
ENGINE_KWARGS = {
    "exllama_args": EXLLAMA_ENGINE_KWARGS,
    "vllm_args": VLLM_ENGINE_KWARGS,
}

# DEFAULT INFERENCE CONFIGURATION
# -------------------------------
# This section defines the default inference configuration, which may differ from
# how we implement inference for a trained model.
# -------------------------------


REMOTE_DEFAULT_INFERENCE_WEIGHTS_PATH = get_env_var_or_default(
    "REMOTE_DEFAULT_INFERENCE_WEIGHTS_PATH",
    "remote/path/to/your/weights/here",
)

N_SHARDS = 2
REMOTE_TRAINING_FILES_TO_DOWNLOAD = [
    f"model-{str(i+1).zfill(5)}-of-{str(N_SHARDS).zfill(5)}.safetensors"
    for i in range(N_SHARDS)
]

# REMOTE_DEFAULT_INFERENCE_FILES_TO_DOWNLOAD = ["model.safetensors"]
# N_SHARDS=3
# REMOTE_DEFAULT_INFERENCE_FILES_TO_DOWNLOAD = [
# f"pytorch_model-{str(i+1).zfill(5)}-of-{str(N_SHARDS).zfill(5)}.bin"
# for i in range(N_SHARDS)
# ]



REMOTE_DEFAULT_INFERENCE_FILES_TO_DOWNLOAD = ["gptq_model-4bit-128g.safetensors"]

REMOTE_DEFAULT_INFERENCE_FILES_TO_DOWNLOAD += [
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "quantize_config.json",
]

# TRAINED INFERENCE CONFIGURATION
# -------------------------------
# This section defines the inference configuration for fine-tuned models
# -------------------------------

LOCAL_TRAINING_WEIGHTS_PATH = f"models/{MODEL_NAME}/model_artifacts/training_weights"

REMOTE_TRAINING_WEIGHTS_PATH = get_env_var_or_default(
    var_name="REMOTE_TRAINING_WEIGHTS_PATH",
    default_value="remote/path/to/your/weights/here"
)

LOCAL_TRAINING_WEIGHTS_CONFIG_PATH = f"models/{MODEL_NAME}/model_artifacts/training_weights/config.json"

REMOTE_TRAINING_WEIGHTS_CONFIG_PATH = get_env_var_or_default(
    var_name="REMOTE_TRAINING_WEIGHTS_CONFIG_PATH",
    default_value="remote/path/to/your/weights/here"
)

REMOTE_TRAINING_FILES_TO_DOWNLOAD = [
    f"model-{str(i+1).zfill(5)}-of-{str(N_SHARDS).zfill(5)}.safetensors"
    for i in range(N_SHARDS)
]

REMOTE_TRAINING_FILES_TO_DOWNLOAD += [
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "model.safetensors.index.json"
]


# -------------------------------

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "</s>"
