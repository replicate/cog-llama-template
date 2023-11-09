import os


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "</s>"


def log_memory_stuff(prompt=None):
    """One method to barf out everything we'd ever want to know about memory"""
    import torch

    if prompt is not None:
        print(prompt)
    os.system("nvidia-smi")
    print(torch.cuda.memory_summary())


def load_tokenizer(tokenizer_path):
    """Same tokenizer, agnostic from tensorized weights/etc"""
    from transformers import LlamaTokenizer

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
