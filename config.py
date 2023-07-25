from collections import OrderedDict
import logging
import os
import time
from transformers import LlamaTokenizer, AutoConfig, LlamaForCausalLM
import torch
import subprocess
from subprocess import DEVNULL, STDOUT
from tensorizer import TensorDeserializer
from tensorizer.utils import no_init_or_tensor

from subclass import YieldingLlama

# path from which we pull weights when there's no COG_WEIGHTS environment variable
# If you want to use tensorized weights, set `DEFAULT_MODEL_NAME` to the path of the tensorized weights.
DEFAULT_MODEL_NAME = "llama_weights/llama-7b/llama_7b_fp16.tensors"# "llama_7b_fp16.tensors" if you have a GPU avaiable or "llama_7b_fp32.tensors" if you don't. - This is where the convert_to_tensors.py will save the tensorized weights.
TOKENIZER_NAME = "llama_weights/tokenizer"
CONFIG_LOCATION = "llama_weights/llama-7b"

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "</s>"

def log_memory_stuff():
    """One method to barf out everything we'd ever want to know about memory"""
    os.system("nvidia-smi")
    print(f"cur memory: {torch.cuda.memory_allocated()}")
    print(f"max allocated: {torch.cuda.max_memory_allocated()}")
    print(f"peak memory: {torch.cuda.max_memory_reserved()}")
    print(f"memory summary: {torch.cuda.memory_summary()}")


def load_tokenizer():
    """Same tokenizer, agnostic from tensorized weights/etc"""
    tok = LlamaTokenizer.from_pretrained(TOKENIZER_NAME, cache_dir="pretrained_weights")
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
