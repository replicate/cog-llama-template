from collections import OrderedDict
import logging
import re
import time
from transformers import LlamaTokenizer, AutoConfig, LlamaForCausalLM
import torch
import subprocess
from subprocess import DEVNULL, STDOUT
from tensorizer import TensorDeserializer
from tensorizer.utils import no_init_or_tensor
import os
from dotenv import load_dotenv

from src.utils import get_env_var_or_default

from subclass import YieldingLlama

# add parent directory to path
import sys

load_dotenv()

MODEL_NAME = 'llama-2-70b-chat'
# INFERENCE CONFIGURATION
#######################################################################
# --------------------Notes--------------------------------------------
# We are trying our very best to no longer have different inference code paths
# for trained and untrained weights :)
#
# INFERENCE CONFIGURATION
# -------------------------------
# This section defines the general inference configuration,
# which is used for both trained and untrained models.
# -------------------------------

TOKENIZER_PATH = f"models/{MODEL_NAME}/model_artifacts/tokenizer"
USE_SYSTEM_PROMPT = True


# ENGINE CONFIGURATION
# -------------------------------
# Here we define the specific inference engine we intend to use for inference, and all appropriate kwargs. 
# -------------------------------

from src.exllama_predictor import ExllamaEngine

ENGINE = ExllamaEngine
ENGINE_KWARGS = {
    "fused_attn": True
}

# WEIGHTS CONFIGURATION
# -------------------------------
# Which base weights do we use for inference with this model?  
# -------------------------------


LOCAL_DEFAULT_INFERENCE_WEIGHTS_PATH = f"models/{MODEL_NAME}/model_artifacts/default_inference_weights"

REMOTE_DEFAULT_INFERENCE_WEIGHTS_PATH = get_env_var_or_default(
    "REMOTE_DEFAULT_INFERENCE_WEIGHTS_PATH", 
    "remote/path/to/your/weights/here",

)

# N_SHARDS = 2
# REMOTE_TRAINING_FILES_TO_DOWNLOAD = [
#     f"model-{str(i+1).zfill(5)}-of-{str(N_SHARDS).zfill(5)}.safetensors"
#     for i in range(N_SHARDS)
# ]

REMOTE_DEFAULT_INFERENCE_FILES_TO_DOWNLOAD = ["gptq_model-4bit--1g.safetensors"]

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

N_SHARDS = 15
REMOTE_TRAINING_FILES_TO_DOWNLOAD = [
    f"model-{str(i+1).zfill(5)}-of-{str(N_SHARDS).zfill(5)}.safetensors"
    for i in range(N_SHARDS)
]

REMOTE_TRAINING_FILES_TO_DOWNLOAD += [
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
]


# -------------------------------

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "</s>"


def log_memory_stuff(prompt=None):
    """One method to barf out everything we'd ever want to know about memory"""
    
    if prompt is not None:
        print(prompt)
    os.system("nvidia-smi")
    print(torch.cuda.memory_summary())


def load_tokenizer():
    """Same tokenizer, agnostic from tensorized weights/etc"""
    tok = LlamaTokenizer.from_pretrained(TOKENIZER_PATH, cache_dir="pretrained_weights", legacy=False)
    tok.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
            "pad_token": DEFAULT_PAD_TOKEN,
        }
    )
    return tok
 

def load_tensorizer(
    weights, plaid_mode: bool = True, cls: LlamaForCausalLM = YieldingLlama
):
    st = time.time()
    weights = str(weights)

    if 'http' in weights:
        if not (os.path.exists(LOCAL_TRAINING_WEIGHTS_PATH)):
            download_file(weights, LOCAL_TRAINING_WEIGHTS_PATH)
        weights = LOCAL_TRAINING_WEIGHTS_PATH
    
    if not os.path.exists(LOCAL_TRAINING_WEIGHTS_CONFIG_PATH):
        download_file(REMOTE_TRAINING_WEIGHTS_CONFIG_PATH, LOCAL_TRAINING_WEIGHTS_CONFIG_PATH)

    config = AutoConfig.from_pretrained(LOCAL_TRAINING_WEIGHTS_CONFIG_PATH)

    logging.disable(logging.WARN)
    model = no_init_or_tensor(
        lambda: cls.from_pretrained(
            None, config=config, state_dict=OrderedDict(), torch_dtype=torch.float16
        )
    )
    logging.disable(logging.NOTSET)

    des = TensorDeserializer(weights, plaid_mode=plaid_mode)
    des.load_into_module(model)
    print(f"weights loaded in {time.time() - st}")
    
    # We don't know what device model was tensorized in or what dtype was used.
    # If a GPU is available, we need to ensure that the model is on the GPU and cast to fp16.
    if next(model.parameters()).is_cuda:
        model = model.half()
    else:
        if torch.cuda.is_available():
            model.to("cuda")
            model = model.half()

    return model
