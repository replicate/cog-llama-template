from dotenv import load_dotenv
import sys
from src.config_utils import exllama_kwargs, get_fp16_file_list, get_gptq_file_list, vllm_kwargs
from src.utils import get_env_var_or_default

from src.inference_engines.vllm_exllama_engine import ExllamaVllmEngine

load_dotenv()

MODEL_NAME = "llama-2-7b-chat"

# Inference weights

LOCAL_DEFAULT_INFERENCE_WEIGHTS_PATH = f"models/{MODEL_NAME}/model_artifacts/default_inference_weights"
REMOTE_DEFAULT_INFERENCE_WEIGHTS_PATH = get_env_var_or_default("REMOTE_DEFAULT_INFERENCE_WEIGHTS_PATH", None)
REMOTE_DEFAULT_INFERENCE_FILES_TO_DOWNLOAD = get_gptq_file_list("gptq_model-4bit-32g.safetensors")

local_vllm_weights_path = f"models/{MODEL_NAME}/model_artifacts/lora_inference_weights"
remote_vllm_inference_weights_path = get_env_var_or_default("REMOTE_VLLM_INFERENCE_WEIGHTS_PATH", None)
remote_vllm_inference_files_to_download = get_fp16_file_list(2)


# Inference config

TOKENIZER_PATH = f"models/{MODEL_NAME}/model_artifacts/default_inference_weights"
USE_SYSTEM_PROMPT = True

ENGINE = ExllamaVllmEngine
exllama_kw = exllama_kwargs()
vllm_kwarg_overrides = {"vllm_model_info": 
                        {
                            "remote_path": remote_vllm_inference_weights_path,
                            "remote_files": remote_vllm_inference_files_to_download,
                            "local_path": local_vllm_weights_path
                        }
                        }
vllm_kw = vllm_kwargs(TOKENIZER_PATH, vllm_kwarg_overrides)

ENGINE_KWARGS = {
    "exllama_args": exllama_kw,
    "vllm_args": vllm_kw,
}

# Training weights

LOCAL_TRAINING_WEIGHTS_PATH = f"models/{MODEL_NAME}/model_artifacts/training_weights"
REMOTE_TRAINING_WEIGHTS_PATH = get_env_var_or_default("REMOTE_TRAINING_WEIGHTS_PATH", None,)
LOCAL_TRAINING_WEIGHTS_CONFIG_PATH = f"models/{MODEL_NAME}/model_artifacts/training_weights/config.json"
REMOTE_TRAINING_WEIGHTS_CONFIG_PATH = get_env_var_or_default(var_name="REMOTE_TRAINING_WEIGHTS_CONFIG_PATH",default_value=None,)
REMOTE_TRAINING_FILES_TO_DOWNLOAD = get_fp16_file_list(2)
