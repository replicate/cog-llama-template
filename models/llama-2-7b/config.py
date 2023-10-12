from dotenv import load_dotenv
from src.inference_engines.vllm_transformers import vLLMTransformersEngine
from src.utils import get_env_var_or_default
from src.more_utils import load_tokenizer
from functools import partial

load_dotenv()

MODEL_NAME = "llama-2-7b"
# INFERENCE CONFIGURATION
#######################################################################
# --------------------Notes--------------------------------------------
# We are trying our very best to no longer have different inference code paths
# for trained and untrained weights :)
#
# INFERENCE CONFIGURATION

# -------------------------------
# This section defines the general inference configuration,
# which is used for both trained and untrained models.
# -------------------------------


TOKENIZER_PATH = f"models/{MODEL_NAME}/model_artifacts/default_inference_weights"
USE_SYSTEM_PROMPT = False

# ENGINE CONFIGURATION
# -------------------------------
# Here we define the specific inference engine we intend to use for inference, and all appropriate kwargs.
# -------------------------------

ENGINE = vLLMTransformersEngine
VLLM_ARGS = {
    "tokenizer_path": TOKENIZER_PATH, "dtype": "auto", "max_num_seqs": 4096,
}
TRANSFORMERS_ARGS = {
    "tokenizer_func" : partial(load_tokenizer, TOKENIZER_PATH)
}
ENGINE_KWARGS = {
    "vllm_args": VLLM_ARGS,
    "transformers_args": TRANSFORMERS_ARGS
}

LOCAL_DEFAULT_INFERENCE_WEIGHTS_PATH = f"models/{MODEL_NAME}/model_artifacts/default_inference_weights"

REMOTE_DEFAULT_INFERENCE_WEIGHTS_PATH = get_env_var_or_default(
    "REMOTE_DEFAULT_INFERENCE_WEIGHTS_PATH",
    "remote/path/to/your/weights/here",
)

N_SHARDS = 2
REMOTE_TRAINING_FILES_TO_DOWNLOAD = [
    f"model-{str(i+1).zfill(5)}-of-{str(N_SHARDS).zfill(5)}.safetensors"
    for i in range(N_SHARDS)
]

# REMOTE_DEFAULT_INFERENCE_FILES_TO_DOWNLOAD = ["model.safetensors"]
# N_SHARDS=3
# REMOTE_DEFAULT_INFERENCE_FILES_TO_DOWNLOAD = [
# f"pytorch_model-{str(i+1).zfill(5)}-of-{str(N_SHARDS).zfill(5)}.bin"
# for i in range(N_SHARDS)
# ]

REMOTE_DEFAULT_INFERENCE_FILES_TO_DOWNLOAD = [
    "checklist.chk",
    "config.json",
    "model.safetensors.index.json",
    "params.json",
    "tokenizer.model",
    "tokenizer_checklist.chk",
    "consolidated.00.pth",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
]

# TRAINED INFERENCE CONFIGURATION
# -------------------------------
# This section defines the inference configuration for fine-tuned models
# -------------------------------

LOCAL_TRAINING_WEIGHTS_PATH = f"models/{MODEL_NAME}/model_artifacts/training_weights"

REMOTE_TRAINING_WEIGHTS_PATH = get_env_var_or_default(
    var_name="REMOTE_TRAINING_WEIGHTS_PATH",
    default_value="remote/path/to/your/weights/here"
)

LOCAL_TRAINING_WEIGHTS_CONFIG_PATH = f"models/{MODEL_NAME}/model_artifacts/training_weights/config.json"

REMOTE_TRAINING_WEIGHTS_CONFIG_PATH = get_env_var_or_default(
    var_name="REMOTE_TRAINING_WEIGHTS_CONFIG_PATH",
    default_value="remote/path/to/your/weights/here"
)

REMOTE_TRAINING_FILES_TO_DOWNLOAD = [
    f"model-{str(i+1).zfill(5)}-of-{str(N_SHARDS).zfill(5)}.safetensors"
    for i in range(N_SHARDS)
]

REMOTE_TRAINING_FILES_TO_DOWNLOAD += [
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "model.safetensors.index.json"
]


# -------------------------------

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "</s>"
