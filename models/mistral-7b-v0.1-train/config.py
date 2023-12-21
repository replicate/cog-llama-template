from dotenv import load_dotenv
from src.config_utils import (
    Weights,
    get_fp16_file_list,
    get_mlc_file_list,
    mlc_kwargs,
    transformers_kwargs,
    vllm_kwargs,
)
from src.inference_engines.transformers_engine import TransformersEngine
from src.utils import get_env_var_or_default

load_dotenv()

MODEL_NAME = "mistral-7b-v01-train"


LOCAL_PATH = f"models/{MODEL_NAME}/model_artifacts/default_inference_weights"

print("LOCAL_PATH", LOCAL_PATH)
transformers_weights = Weights(
    local_path=f"models/{MODEL_NAME}/model_artifacts/lora_inference_weights",
    remote_path=get_env_var_or_default("REMOTE_FINE_TUNE_INFERENCE_WEIGHTS_PATH", ""),
    remote_files=get_fp16_file_list(2),
)

# Inference config
USE_SYSTEM_PROMPT = False

ENGINE = TransformersEngine
ENGINE_KWARGS = transformers_kwargs(transformers_weights)


# Training config
LOAD_IN_4BIT = False

LOCAL_TRAINING_WEIGHTS_PATH = f"models/{MODEL_NAME}/model_artifacts/training_weights"
REMOTE_TRAINING_WEIGHTS_PATH = get_env_var_or_default(
    "REMOTE_TRAINING_WEIGHTS_PATH",
    None,
)
LOCAL_TRAINING_WEIGHTS_CONFIG_PATH = (
    f"models/{MODEL_NAME}/model_artifacts/training_weights/config.json"
)
REMOTE_TRAINING_WEIGHTS_CONFIG_PATH = get_env_var_or_default(
    var_name="REMOTE_TRAINING_WEIGHTS_CONFIG_PATH",
    default_value=None,
)
REMOTE_TRAINING_FILES_TO_DOWNLOAD = get_fp16_file_list(2)
