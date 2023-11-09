from dotenv import load_dotenv
from src.config_utils import Weights, get_fp16_file_list
from src.utils import get_env_var_or_default

load_dotenv()

MODEL_NAME = "llama-2-7b-transformers"
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

TOKENIZER_PATH = f"models/{MODEL_NAME}/model_artifacts/tokenizer"
USE_SYSTEM_PROMPT = False


# ENGINE CONFIGURATION
# -------------------------------
# Here we define the specific inference engine we intend to use for inference, and all appropriate kwargs.
# -------------------------------

from src.inference_engines.transformers_engine import TransformersEngine

# todo - this is probably wrong - now that different engines have different tokenizers, should we eliminate load_tokenizer & handle it all within the engine? I ...think so
from functools import partial
from src.more_utils import load_tokenizer

weights = Weights(
    local_path=f"models/{MODEL_NAME}/model_artifacts/default_inference_weights",
    remote_path=get_env_var_or_default("REMOTE_DEFAULT_INFERENCE_WEIGHTS_PATH", None),
    remote_files=get_fp16_file_list(2),
)

ENGINE = TransformersEngine
ENGINE_KWARGS = {
    "weights": weights,
    "tokenizer_func": partial(load_tokenizer, TOKENIZER_PATH),
}


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
