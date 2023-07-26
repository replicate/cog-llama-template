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

from subclass import YieldingLlama

############################################
# Update these variables for the model you want to use
#
# - Specify the local path where your weights are stored. If their not local, they'll be downloaded.
LOCAL_GPTQ_WEIGHTS_PATH = "llama_weights/Llama-2-70B-chat-GPTQ"
#
# - If the Hugging Face loader is used, should the model be loaded in 4bit? Should be True for 70b models
LOAD_IN_4BIT = False
#
# - Specify the remote path where your GPTQ weights are stored. If they're not local, they'll be downloaded.
REMOTE_GPTQ_WEIGHTS_PATH = None
REMOTE_GPTQ_WEIGHTS_PATH = REMOTE_GPTQ_WEIGHTS_PATH.rstrip("/") if REMOTE_GPTQ_WEIGHTS_PATH else None
# - Specify the remote path to your HF weights or tensorizer weights.
BASE_WEIGHTS_PATH = None
#
###### YOU SHOULD LOOK AT THE FILE NAMES! ######
# - Specify the files you want to download from the remote path.
# N_SHARDS = 1
# REMOTE_FILES_TO_DOWNLOAD = [
#     f"model-{str(i+1).zfill(5)}-of-{str(N_SHARDS).zfill(5)}.safetensors"
#     for i in range(N_SHARDS)
# ]

REMOTE_FILES_TO_DOWNLOAD = ["gptq_model-4bit-32g.safetensors"]

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
############################################

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "</s>"

# If you do not modify the directory structure, you should not need to modify these lines.
TOKENIZER_NAME = "llama_weights/tokenizer"
CONFIG_LOCATION = "llama_weights/llama-2-70b-chat"


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
    local_weights = "/src/llama_tensors"
    print("Deserializing weights...")
    if 'http' in weights:
        download_file(weights, local_weights)
    else:
        local_weights = weights

    config = AutoConfig.from_pretrained(CONFIG_LOCATION)

    logging.disable(logging.WARN)
    model = no_init_or_tensor(
        lambda: cls.from_pretrained(
            None, config=config, state_dict=OrderedDict(), torch_dtype=torch.float16
        )
    )
    logging.disable(logging.NOTSET)

    des = TensorDeserializer(local_weights, plaid_mode=plaid_mode)
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