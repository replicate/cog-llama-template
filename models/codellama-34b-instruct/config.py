from dotenv import load_dotenv
from src.utils import get_env_var_or_default

load_dotenv()

MODEL_NAME = "codellama-34b-instruct"
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

from src.inference_engines.vllm_engine import vLLMEngine

ENGINE = vLLMEngine
ENGINE_KWARGS = {
    "tokenizer_path": TOKENIZER_PATH,
    "dtype": "auto",
    "max_num_seqs": 16384,
}

# WEIGHTS CONFIGURATION
# -------------------------------
# Which base weights do we use for inference with this model?
# -------------------------------

LOCAL_DEFAULT_INFERENCE_WEIGHTS_PATH = (
    f"models/{MODEL_NAME}/model_artifacts/default_inference_weights"
)

REMOTE_DEFAULT_INFERENCE_WEIGHTS_PATH = get_env_var_or_default(
    "REMOTE_DEFAULT_INFERENCE_WEIGHTS_PATH",
    "remote/path/to/your/weights/here",
)

N_SHARDS=7
REMOTE_DEFAULT_INFERENCE_FILES_TO_DOWNLOAD = [
# This evaluates to `model-00001-of-00002.safetensors`...
f"pytorch_model-{str(i+1).zfill(5)}-of-{str(N_SHARDS).zfill(5)}.bin"
for i in range(N_SHARDS)
]

REMOTE_DEFAULT_INFERENCE_FILES_TO_DOWNLOAD += [
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "quantize_config.json",
]


# TRAINING CONFIGURATION
# -------------------------------
# This section defines the configuration for weights used to train/fine-tune models
# -------------------------------

LOCAL_TRAINING_WEIGHTS_PATH = f"models/{MODEL_NAME}/model_artifacts/training_weights"

REMOTE_TRAINING_WEIGHTS_PATH = get_env_var_or_default(
    var_name="REMOTE_TRAINING_WEIGHTS_PATH",
    default_value="remote/path/to/your/weights/here",
)

LOCAL_TRAINING_WEIGHTS_CONFIG_PATH = (
    f"models/{MODEL_NAME}/model_artifacts/training_weights/config.json"
)

REMOTE_TRAINING_WEIGHTS_CONFIG_PATH = get_env_var_or_default(
    var_name="REMOTE_TRAINING_WEIGHTS_CONFIG_PATH",
    default_value="remote/path/to/your/weights/here",
)

N_SHARDS = 7
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
    "pytorch_model.bin.index.json"
]
