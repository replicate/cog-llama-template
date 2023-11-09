from dotenv import load_dotenv
from src.config_utils import Weights, get_fp16_file_list, vllm_kwargs


from src.utils import get_env_var_or_default

load_dotenv()

MODEL_NAME = "llama-2-7b-vllm"

# Inference config

weights = Weights(
    local_path=f"models/{MODEL_NAME}/model_artifacts/default_inference_weights",
    remote_path=get_env_var_or_default(
        "REMOTE_DEFAULT_INFERENCE_WEIGHTS_PATH",
        "remote/path/to/your/weights/here",
    ),
    remote_files=get_fp16_file_list(2),
)

LOAD_IN_4BIT = False
TOKENIZER_PATH = f"models/{MODEL_NAME}/model_artifacts/default_inference_weights"
USE_SYSTEM_PROMPT = False
USE_EXLLAMA_FOR_UNTRAINED_WEIGHTS = False

# Engine config

from src.inference_engines.vllm_engine import vLLMEngine


ENGINE = vLLMEngine
ENGINE_KWARGS = vllm_kwargs(weights)


# TRAINED INFERENCE CONFIGURATION
# -------------------------------
# This section defines the inference configuration for fine-tuned models
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

REMOTE_TRAINING_FILES_TO_DOWNLOAD = get_fp16_file_list(2)


# -------------------------------

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "</s>"
