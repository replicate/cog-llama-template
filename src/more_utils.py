import logging
import os
import time
from collections import OrderedDict

import torch
from transformers import AutoConfig, LlamaForCausalLM, LlamaTokenizer

# circular imports. 
# from config import (
#     LOCAL_TRAINING_WEIGHTS_CONFIG_PATH,
#     LOCAL_TRAINING_WEIGHTS_PATH,
#     TOKENIZER_PATH,
# )
from tensorizer import TensorDeserializer
from tensorizer.utils import no_init_or_tensor


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


def load_tokenizer(tokenizer_path):
    """Same tokenizer, agnostic from tensorized weights/etc"""
    tok = LlamaTokenizer.from_pretrained(
        tokenizer_path, cache_dir="pretrained_weights", legacy=False
    )
    tok.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
            "pad_token": DEFAULT_PAD_TOKEN,
        }
    )
    return tok


# def load_tensorizer(
#     weights, plaid_mode: bool = True, cls: LlamaForCausalLM = YieldingLlama
# ):
#     st = time.time()
#     weights = str(weights)

#     if "http" in weights:
#         if not (os.path.exists(LOCAL_TRAINING_WEIGHTS_PATH)):
#             download_file(weights, LOCAL_TRAINING_WEIGHTS_PATH)
#         weights = LOCAL_TRAINING_WEIGHTS_PATH

#     if not os.path.exists(LOCAL_TRAINING_WEIGHTS_CONFIG_PATH):
#         download_file(
#             REMOTE_TRAINING_WEIGHTS_CONFIG_PATH, LOCAL_TRAINING_WEIGHTS_CONFIG_PATH
#         )

#     config = AutoConfig.from_pretrained(LOCAL_TRAINING_WEIGHTS_CONFIG_PATH)

#     logging.disable(logging.WARN)
#     model = no_init_or_tensor(
#         lambda: cls.from_pretrained(
#             None, config=config, state_dict=OrderedDict(), torch_dtype=torch.float16
#         )
#     )
#     logging.disable(logging.NOTSET)

#     des = TensorDeserializer(weights, plaid_mode=plaid_mode)
#     des.load_into_module(model)
#     print(f"weights loaded in {time.time() - st}")

#     # We don't know what device model was tensorized in or what dtype was used.
#     # If a GPU is available, we need to ensure that the model is on the GPU and cast to fp16.
#     if next(model.parameters()).is_cuda:
#         model = model.half()
#     else:
#         if torch.cuda.is_available():
#             model.to("cuda")
#             model = model.half()

#     return model
