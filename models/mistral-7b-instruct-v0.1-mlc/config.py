from dotenv import load_dotenv
from src.config_utils import (
    Weights,
    get_fp16_file_list,
    get_mlc_file_list,
    mlc_kwargs,
    vllm_kwargs,
)
from src.inference_engines.mlc_vllm_engine import MLCvLLMEngine
from src.utils import get_env_var_or_default

load_dotenv()

MODEL_NAME = "mistral-7b-instruct-v0.1-mlc"

# Inference weights
mlc_file_list = get_mlc_file_list(model_name="Mistral-7B-Instruct-v0.1-q4f16_1", n_shards=107)

LOCAL_PATH = f"models/{MODEL_NAME}/model_artifacts/default_inference_weights"

mlc_weights = Weights(
    local_path=LOCAL_PATH,
    remote_path=get_env_var_or_default("REMOTE_DEFAULT_INFERENCE_WEIGHTS_PATH", None),
    remote_files=mlc_file_list,
)

vllm_weights = Weights(
    local_path=f"models/{MODEL_NAME}/model_artifacts/lora_inference_weights",
    remote_path=get_env_var_or_default("REMOTE_VLLM_INFERENCE_WEIGHTS_PATH", None),
    remote_files=get_fp16_file_list(2),
)

# Inference config
USE_SYSTEM_PROMPT = True

# from mistral: "<s>[INST] + Instruction [/INST] Model answer</s>[INST] Follow-up instruction [/INST]"
PROMPT_TEMPLATE = "[INST] {system_prompt}{prompt} [/INST]"
DEFAULT_SYSTEM_PROMPT = "Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity. "

ENGINE = MLCvLLMEngine
ENGINE_KWARGS = {
    "mlc_args": mlc_kwargs(mlc_weights, is_chat=False),
    "vllm_args": vllm_kwargs(vllm_weights),
}

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
