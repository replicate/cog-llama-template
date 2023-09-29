import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os


def save_model(model):

    curdir = os.path.join('tmp', model)
    os.makedirs(curdir, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16)
    print('saving')
    model.save_pretrained(curdir, safe_serialization=True)

    tok = AutoTokenizer.from_pretrained(model)
    tok.save_pretrained(curdir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    args = parser.parse_args()
    save_model(args.model)

