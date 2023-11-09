import re
from transformers import LlamaTokenizer
import subprocess

DEFAULT_MODEL_NAME = "{{model_name}}"  # path from which we pull weights when there's no COG_WEIGHTS environment variable
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
    pattern = r"https://pbxt\.replicate\.delivery/([^/]+/[^/]+)"
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
