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

DEFAULT_MODEL_NAME = "{{model_name}}" # path from which we pull weights when there's no COG_WEIGHTS environment variable
TOKENIZER_NAME = "llama_weights/tokenizer"
CONFIG_LOCATION = "{{config_location}}"

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"


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


def pull_gcp_file(weights, local_filename):
    """Pulls weights from GCP to local storage"""
    pattern = r'https://pbxt\.replicate\.delivery/([^/]+/[^/]+)'
    match = re.search(pattern, weights)
    if match:
        weights = f"gs://replicate-files/{match.group(1)}"

    command = (
        f"/gc/google-cloud-sdk/bin/gcloud storage cp {weights} {local_filename}".split()
    )
    res = subprocess.run(command)
    if res.returncode != 0:
        raise Exception(
            f"gcloud storage cp command failed with return code {res.returncode}: {res.stderr.decode('utf-8')}"
        )
    return


def load_tensorizer(
    weights, plaid_mode: bool = True, cls: LlamaForCausalLM = YieldingLlama
):
    st = time.time()
    weights = str(weights)
    local_weights = "/src/llama_tensors"
    print("Deserializing weights...")
    if 'http' in weights:
        pull_gcp_file(weights, local_weights)
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
    return model
