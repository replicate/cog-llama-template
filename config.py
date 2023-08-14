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

from subclass import YieldingLlama


# - Specify the files you want to download from the remote path.
# N_SHARDS = 1
# REMOTE_FILES_TO_DOWNLOAD = [
#     f"model-{str(i+1).zfill(5)}-of-{str(N_SHARDS).zfill(5)}.safetensors"
#     for i in range(N_SHARDS)
# ]

# UPDATE THESE VARIABLES FOR YOUR MODEL CONFIGURATION
#######################################################
# --------------------Notes---------------------------
# 1. We currently do not include weights in images, so they need to be downloaded.
# 2. We are currently serving GPTQ weights, but training with fp16 weights. 
# 3. We're not currently converting fine-tuned weights to GPTQ, so we need to support prediction with weights taht aren't gptq format. 
# 4. Accordingly, we need to support both GPTQ and non-GPTQ weights for prediction and non-GPTQ weights for training.
############################################
#
# DO YOU WANT TO USE A SYSTEM PROMPT (E.G. FOR A CHAT MODEL?)
USE_SYSTEM_PROMPT = False
# Path to directory where tokenizer is stored. 
TOKENIZER_NAME = "llama_weights/tokenizer"
# 
# DEFAULT INFERENCE CONFIGURATION
# -------------------------------
# This section defines the default inference configuration, which may differ from
# how we implement inference for a trained model.
# -------------------------------
# - Specify the local path where your weights are stored. If their not local, they'll be downloaded to this directory.
#
DEFAULT_INFERENCE_USE_EXLLAMA = True
# 
DEFAULT_LOCAL_INFERENCE_WEIGHTS_PATH = "default_base_weights"
# Specify the remote path where your GPTQ weights are stored. If they're not local, they'll be downloaded from this path.
DEFAULT_REMOTE_INFERENCE_WEIGHTS_PATH = "https://storage.googleapis.com/replicate-weights/Llama-2-7B-GPTQ"
DEFAULT_REMOTE_INFERENCE_WEIGHTS_PATH = DEFAULT_REMOTE_INFERENCE_WEIGHTS_PATH.rstrip("/") if DEFAULT_REMOTE_INFERENCE_WEIGHTS_PATH else None
# - Specify the files that should be downloaded from this remote path.
REMOTE_FILES_TO_DOWNLOAD = ["gptq_model-4bit-128g.safetensors"]
# N_SHARDS = 3
# REMOTE_FILES_TO_DOWNLOAD = [
#     f"pytorch_model-{str(i+1).zfill(5)}-of-{str(N_SHARDS).zfill(5)}.safetensors"
#     for i in range(N_SHARDS)
# ]

REMOTE_FILES_TO_DOWNLOAD += [
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "quantize_config.json",
]
# --------------------------------
#
# Base weights configuration
#
# Specify the path to your base weights. The format of this path has implications for model loading:
#   1. If it's a path to a local directory, we'll attempt to use `.from_pretrained` to load the weights from the provided directory.
#   2. If it's a local path to a file that ends with `.tensors`, we'll try to load the file with Tensorizer.
#   3. If it's a remote path to a file that ends with `.tensors`, we'll try to download the file and load it with Tensorizer.
#   4. If it's something else that won't work under those expectations, it probably won't work.
BASE_WEIGHTS_PATH = "https://weights.replicate.delivery/default/llama-2-7b"
# BASE_WEIGHTS_PATH = "llama_weights/llama_weights/llama-2-7b"
# BASE_WEIGHTS_PATH = "llama_weights/LLongMA-2-13b-16k/llongma-2-13b-16k.tensors"
# Specify the path to the model config --- this is necessary for loading tensorized weights.
CONFIG_LOCATION = "llama_weights/test/"
LOCAL_BASE_WEIGHTS = CONFIG_LOCATION

# LOCAL_BASE_WEIGHTS = os.path.join(CONFIG_LOCATION, BASE_WEIGHTS_PATH.split('/')[-1])


# - If the Hugging Face loader is used, should the model be loaded in 4bit?
LOAD_IN_4BIT = False
############################################

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
    tok = LlamaTokenizer.from_pretrained(TOKENIZER_NAME, cache_dir="pretrained_weights", legacy=False)
    tok.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
            "pad_token": DEFAULT_PAD_TOKEN,
        }
    )
    return tok
 
def download_file(file, local_filename):
    print(f"Downloading {file}")
    if os.path.exists(local_filename):
        os.remove(local_filename)
    command = ['pget', file, local_filename]
    subprocess.check_call(command)
    return


def load_tensorizer(
    weights, plaid_mode: bool = True, cls: LlamaForCausalLM = YieldingLlama
):
    st = time.time()
    weights = str(weights)

    if 'http' in weights:
        if not (os.path.exists(LOCAL_BASE_WEIGHTS)):
            download_file(weights, LOCAL_BASE_WEIGHTS)
        weights = LOCAL_BASE_WEIGHTS

    config = AutoConfig.from_pretrained(CONFIG_LOCATION)

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
