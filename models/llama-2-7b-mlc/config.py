import os
import subprocess

import torch
from dotenv import load_dotenv
from transformers import LlamaTokenizer

from src.utils import get_env_var_or_default

load_dotenv()

MODEL_NAME = "llama-2-7b-mlc"
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
TOKENIZER_PATH = "huggyllama/llama-7b" 
USE_SYSTEM_PROMPT = False
USE_EXLLAMA_FOR_UNTRAINED_WEIGHTS = False

# ENGINE CONFIGURATION
# -------------------------------
# Here we define the specific inference engine we intend to use for inference, and all appropriate kwargs.
# -------------------------------

from src.inference_engines.mlc_engine import MLCEngine

ENGINE = MLCEngine
ENGINE_KWARGS = {
    "tokenizer_path": TOKENIZER_PATH,
}

# DEFAULT INFERENCE CONFIGURATION
# -------------------------------
# This section defines the default inference configuration, which may differ from
# how we implement inference for a trained model.
# -------------------------------

LOCAL_DEFAULT_INFERENCE_WEIGHTS_PATH = f"models/{MODEL_NAME}/model_artifacts/default_inference_weights"

REMOTE_DEFAULT_INFERENCE_WEIGHTS_PATH = get_env_var_or_default(
    "REMOTE_DEFAULT_INFERENCE_WEIGHTS_PATH",
    "remote/path/to/your/weights/here",
)

N_SHARDS = 2
REMOTE_TRAINING_FILES_TO_DOWNLOAD = [
    f"model-{str(i+1).zfill(5)}-of-{str(N_SHARDS).zfill(5)}.safetensors"
    for i in range(N_SHARDS)
]

N_SHARDS = 115
REMOTE_DEFAULT_INFERENCE_FILES_TO_DOWNLOAD = [
    f"params/params_shard_{shard_idx}.bin" for shard_idx in range(N_SHARDS)
]

REMOTE_DEFAULT_INFERENCE_FILES_TO_DOWNLOAD += [
    "llama-2-7b-hf-q4f16_1-cuda.so",
    "mod_cache_before_build.pkl",
    "params/mlc-chat-config.json",
    "params/ndarray-cache.json",
    "params/tokenizer.json",
    "params/tokenizer_config.json",
    "params/tokenizer.model",
]

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "</s>"
