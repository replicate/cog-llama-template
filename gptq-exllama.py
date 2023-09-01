import os
import sys
import glob 
import torch 
import time
import json
import typing as tp 
# from src.exllama_predictor import ExllamaGenerator

from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
from exllama.tokenizer import ExLlamaTokenizer
from exllama.generator import ExLlamaGenerator

model_directory = '../weights/codellama-13b-instruct/gptq'
# generator = ExllamaGenerator(weights)

tokenizer_path = os.path.join(model_directory, "tokenizer.model")
model_config_path = os.path.join(model_directory, "config.json")
st_pattern = os.path.join(model_directory, "*.safetensors")
model_path = glob.glob(st_pattern)[0]
with open(model_config_path, 'r') as f:
    base_model_config = json.load(f)
    if "max_position_embeddings" in base_model_config:
        max_seq_len = base_model_config["max_position_embeddings"]
    else:
        max_seq_len = 4096


config = ExLlamaConfig(model_config_path)               # create config from config.json
config.model_path = model_path                          # supply path to model weights file

# Override exllam's default settings to use full llama v2 context
# config.max_seq_len = max_seq_len
config.max_seq_len = max_seq_len
config.max_input_len = max_seq_len
config.max_attention_size = max_seq_len**2

config.repetition_penalty = 1.15
config.repetition_penalty_sustain = 256
config.token_repetition_penalty_decay = 128
config.temperature = 0.95
config.top_p = 0.65
config.top_k = 50
config.max_new_tokens = 128
config.min_new_tokens = 0
config.beams = 1
config.beam_length = 1

model = ExLlama(config)                                 # create ExLlama instance and load the weights
tokenizer = ExLlamaTokenizer(tokenizer_path)            # create tokenizer from tokenizer model file


#FIXME: make sure this cache isnt passing statefulness
cache = ExLlamaCache(model)                             # create cache for inference
generator = ExLlamaGenerator(model, tokenizer, cache)   # create generator


prompt = '''<s> [INST] <<SYS>>
All responses must be written in TypeScript.
<</SYS>>

Write a function that sums 2 integers together and returns the results.
[/INST]'''


def run():
    print(generator.generate_simple(prompt, 512))
if __name__ == "__main__":
    run()