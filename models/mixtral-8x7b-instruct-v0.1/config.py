from dotenv import load_dotenv
from src.config_utils import (
    Weights,
    get_fp16_file_list,
    vllm_kwargs,
)
from src.inference_engines.vllm_engine import vLLMEngine
from src.utils import get_env_var_or_default

load_dotenv()

MODEL_NAME = "mixtral-8x7b-instruct-v0.1"


LOCAL_PATH = f"models/{MODEL_NAME}/model_artifacts/default_inference_weights"

print("LOCAL_PATH", LOCAL_PATH)

REMOTE_FILES = [
    "config.json",
    "generation_config.json", 
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
]

vllm_weights = Weights(
    local_path=f"models/{MODEL_NAME}/model_artifacts/default_inference_weights",
    remote_path=get_env_var_or_default("REMOTE_DEFAULT_INFERENCE_WEIGHTS_PATH", ""),
    remote_files=REMOTE_FILES,
)

# Inference config
USE_SYSTEM_PROMPT = True
PROMPT_TEMPLATE = "[INST] {system_prompt}{prompt} [/INST]"
DEFAULT_SYSTEM_PROMPT = "Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity. "


ENGINE = vLLMEngine
vLLM_ENGINE_OVERRIDES = {
    "tensor_parallel_size": 2,
}

ENGINE_KWARGS = vllm_kwargs(
    weights=vllm_weights,
    config_overrides=vLLM_ENGINE_OVERRIDES,
)

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