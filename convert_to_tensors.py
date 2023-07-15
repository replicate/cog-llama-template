#!/usr/bin/env python


"""
This script uses CoreWeave's Tensorizer to convert model weights to tensorized format.
"""

import torch
import os
import argparse
import logging 
import sys

from tensorizer import TensorSerializer
from transformers import AutoModelForCausalLM, AutoConfig
import torch

model_path = 'weights/'#This is the folder than contains the weights in a transformers-compatible format (multiple .bin files) which were previously converted from a PyTorch checkpoint format (consolidated.00.pth).

# append parent directory to path
sys.path.append('.')

from config import DEFAULT_MODEL_NAME

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = "fp16" if torch.cuda.is_available() else "fp32"

if dtype == "fp16":
    dtype = torch.float16
    tensorizer_path = os.path.join(DEFAULT_MODEL_NAME, "llama_13b_fp16.tensors")
elif dtype == "fp32":
    dtype = torch.float32
    tensorizer_path = os.path.join(DEFAULT_MODEL_NAME, "llama_13b_fp32.tensors")

model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=dtype, low_cpu_mem_usage=True
).to(device)

serializer = TensorSerializer(tensorizer_path)
serializer.write_module(model)
serializer.close()

print('Wrote tensorized model to: ', tensorizer_path)
