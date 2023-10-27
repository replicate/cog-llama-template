import os

from dotenv import load_dotenv
from src.config_utils import Weights, get_mlc_file_list, mlc_kwargs
from src.inference_engines.mlc_engine import MLCEngine
from src.utils import get_env_var_or_default

load_dotenv()

MODEL_NAME = "llama-2-7b-chat-hf-mlc"

# Inference weights
mlc_file_list = get_mlc_file_list(
    model_name="Llama-2-7b-chat-hf-q4f16_1", n_shards=115)

LOCAL_PATH = f"models/{MODEL_NAME}/model_artifacts/default_inference_weights"

# ensure directories exist
for path in mlc_file_list:
    path_directory = os.path.dirname(path)
    if path_directory:
        path_directory = os.path.join(LOCAL_PATH, path_directory)
        os.makedirs(path_directory, exist_ok=True)

mlc_weights = Weights(
    local_path=LOCAL_PATH,
    remote_path=get_env_var_or_default(
        "REMOTE_DEFAULT_INFERENCE_WEIGHTS_PATH", None),
    remote_files=mlc_file_list,
)

# Inference config
TOKENIZER_PATH = "huggyllama/llama-7b"
USE_SYSTEM_PROMPT = True

ENGINE = MLCEngine
ENGINE_KWARGS = mlc_kwargs(
    mlc_weights, tokenizer_path=TOKENIZER_PATH, is_chat=False)

# Training config

LOAD_IN_4BIT = False

LOCAL_TRAINING_WEIGHTS_PATH = f"models/{MODEL_NAME}/model_artifacts/training_weights"
REMOTE_TRAINING_WEIGHTS_PATH = get_env_var_or_default(
    "REMOTE_TRAINING_WEIGHTS_PATH", None,)
LOCAL_TRAINING_WEIGHTS_CONFIG_PATH = f"models/{MODEL_NAME}/model_artifacts/training_weights/config.json"
REMOTE_TRAINING_WEIGHTS_CONFIG_PATH = get_env_var_or_default(
    var_name="REMOTE_TRAINING_WEIGHTS_CONFIG_PATH", default_value=None,)
REMOTE_TRAINING_FILES_TO_DOWNLOAD = mlc_file_list
