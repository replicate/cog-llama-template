#!/usr/bin/env python


"""
This script uses CoreWeave's Tensorizer to convert model weights to tensorized format.
"""

import torch
import os
import sys

from tensorizer import TensorSerializer
from transformers import AutoModelForCausalLM
import torch

sys.path.append(".")


def main(weights_dir):
    # append parent directory to path

    fname = weights_dir.split("/")[-1]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = "fp16" if torch.cuda.is_available() else "fp32"

    if dtype == "fp16":
        dtype = torch.float16
        fname = fname + "_fp16.tensors"
        tensorizer_path = os.path.join(weights_dir, fname)
    elif dtype == "fp32":
        dtype = torch.float32
        fname = fname + "_fp32.tensors"
        tensorizer_path = os.path.join(weights_dir, fname)

    model = AutoModelForCausalLM.from_pretrained(
        weights_dir, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)

    serializer = TensorSerializer(tensorizer_path)
    serializer.write_module(model)
    serializer.close()

    print("Wrote tensorized model to: ", tensorizer_path)


# setup module execution

if __name__ == "__main__":
    import os
    import sys


    # fname = "llama_7b_chat_fp16.tensors"
    # weights_dir = "llama_weights/Llama-2-7b-chat" #This is the folder than contains the weights in a transformers-compatible format (multiple .bin files) which were previously converted from a PyTorch checkpoint format (consolidated.00.pth).
    weights_dir = sys.argv[1]
    main(weights_dir)
