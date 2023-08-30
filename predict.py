import shutil
import time
import zipfile
from typing import Optional

import torch
from cog import BasePredictor, ConcatenateIterator, Input, Path

from config import (
    LOCAL_DEFAULT_INFERENCE_WEIGHTS_PATH,
    REMOTE_DEFAULT_INFERENCE_WEIGHTS_PATH,
    REMOTE_TRAINING_FILES_TO_DOWNLOAD,
    USE_EXLLAMA_FOR_UNTRAINED_WEIGHTS,
    REMOTE_DEFAULT_INFERENCE_FILES_TO_DOWNLOAD,
    LOCAL_TRAINING_WEIGHTS_PATH,
    REMOTE_TRAINING_WEIGHTS_PATH,
    LOAD_IN_4BIT,
    load_tokenizer,
    load_tensorizer,
    download_file,
    USE_SYSTEM_PROMPT,
)

from subclass import YieldingLlama
from src.utils import maybe_download_with_pget, StreamingTextStopSequenceHandler

import os


# This prompt formatting was copied from the original Llama v2 repo:
# https://github.com/facebookresearch/llama/blob/6c7fe276574e78057f917549435a2554000a876d/llama/generation.py#L44

# These are components of the prompt that should not be changed by the users
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
PROMPT_TEMPLATE = f"{B_INST} {B_SYS}{{system_prompt}}{E_SYS}{{instruction}} {E_INST}"

# Users may want to change the system prompt, but we use the recommended system prompt by default
DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant."""


class Predictor(BasePredictor):
    def setup(self, weights: Optional[Path] = None):
        print("starting setup")
        print("!" * 100)
        print("Weights directory is:", weights)
        print("!" * 100)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        from src.exllama_predictor import ExllamaGenerator

        base_weights = maybe_download_with_pget(
            LOCAL_DEFAULT_INFERENCE_WEIGHTS_PATH,
            REMOTE_DEFAULT_INFERENCE_WEIGHTS_PATH,
            REMOTE_DEFAULT_INFERENCE_FILES_TO_DOWNLOAD,
        )
        self.generator = ExllamaGenerator(base_weights)

        if weights is not None and weights.name == "weights":
            # bugfix
            weights = None
        if weights:
            # If weights are passed in, they are LoRa weights
            # so we need to download the fp16 weights and load with peft
            self.initialize_peft(weights)
        else:
            raise Exception(f"Fine-tuned weights {weights} were improperly formatted.")

    def initialize_peft(self, replicate_weights):
        if "http" in replicate_weights:  # weights are in the cloud
            print("Downloading peft weights")
            st = time.time()
            local_peft_weights = "local_weights.zip"
            download_file(replicate_weights, local_peft_weights)
            print(f"downloaded peft weights in {time.time() - st}")
        else:
            local_peft_weights = replicate_weights

        print("Unziping peft weights")
        st = time.time()
        peft_path = "/src/peft_dir"
        if os.path.exists(peft_path):
            shutil.rmtree(peft_path)
        with zipfile.ZipFile(local_peft_weights, "r") as zip_ref:
            zip_ref.extractall(peft_path)
        print(f"Unzipped peft weights in {time.time() - st}")

        print("Initializing peft model")
        st = time.time()
        self.generator.load_lora(peft_path)
        print(f"Initialized peft model initialized in {time.time() - st}")
        # remove file
        os.remove(local_peft_weights)

    def predict(
        self,
        replicate_weights: str = Input(
            description="Path to fine-tuned weights produced by a Replicate fine-tune job.",
            default=None,
        ),
        prompt: str = Input(description="Prompt to send to the model."),
        max_new_tokens: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens",
            ge=1,
            default=128,
        ),
        min_new_tokens: int = Input(
            description="Minimum number of tokens to generate. To disable, set to -1. A word is generally 2-3 tokens.",
            ge=-1,
            default=-1,
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75 is a good starting value.",
            ge=0.01,
            le=5,
            default=0.75,
        ),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.0,
            le=1.0,
            default=0.9,
        ),
        top_k: int = Input(
            description="When decoding text, samples from the top k most likely tokens; lower to ignore less likely tokens",
            ge=0,
            default=50,
        ),
        stop_sequences: str = Input(
            description="A comma-separated list of sequences to stop generation at. For example, '<end>,<stop>' will stop generation at the first instance of 'end' or '<stop>'.",
            default=None,
        ),
        debug: bool = Input(
            description="provide debugging output in logs", default=False
        ),
    ) -> ConcatenateIterator:

        if stop_sequences:
            stop_sequences = stop_sequences.split(",")

        if USE_SYSTEM_PROMPT:
            prompt = prompt.strip("\n").lstrip(B_INST).rstrip(E_INST).strip()
            prompt = PROMPT_TEMPLATE.format(
                system_prompt=system_prompt.strip(), instruction=prompt.strip()
            )

        print(f"Your formatted prompt is: \n{prompt}")

        if replicate_weights:
            self.initialize_peft(replicate_weights)
        n_tokens = 0
        st = time.time()

        for decoded_token in self.generator(
            prompt,
            repetition_penalty=1.15,
            repetition_penalty_sustain=256,
            token_repetition_penalty_decay=128,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            stop_sequences=stop_sequences,
        ):
            n_tokens += 1
            yield decoded_token
        t = time.time() - st

        if debug:
            print(f"cur memory: {torch.cuda.memory_allocated()}")
            print(f"max allocated: {torch.cuda.max_memory_allocated()}")
            print(f"peak memory: {torch.cuda.max_memory_reserved()}")
