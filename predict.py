import functools
import inspect
import os
import random
import shutil
import socket
import time
import zipfile
from typing import Optional

import torch
from cog import BasePredictor, ConcatenateIterator, Input, Path

from config import (
    LOAD_IN_4BIT,
    LOCAL_DEFAULT_INFERENCE_WEIGHTS_PATH,
    LOCAL_TRAINING_WEIGHTS_PATH,
    REMOTE_DEFAULT_INFERENCE_FILES_TO_DOWNLOAD,
    REMOTE_DEFAULT_INFERENCE_WEIGHTS_PATH,
    REMOTE_TRAINING_FILES_TO_DOWNLOAD,
    REMOTE_TRAINING_WEIGHTS_PATH,
    USE_EXLLAMA_FOR_UNTRAINED_WEIGHTS,
    USE_FUSED_ATTN,
    USE_SYSTEM_PROMPT,
    download_file,
    load_tensorizer,
    load_tokenizer,
)
from src.utils import StreamingTextStopSequenceHandler, maybe_download_with_pget
from subclass import YieldingLlama

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
        print("Starting setup")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        from src.exllama_predictor import ExllamaGenerator

        base_weights = maybe_download_with_pget(
            LOCAL_DEFAULT_INFERENCE_WEIGHTS_PATH,
            REMOTE_DEFAULT_INFERENCE_WEIGHTS_PATH,
            REMOTE_DEFAULT_INFERENCE_FILES_TO_DOWNLOAD,
        )
        self.generator = ExllamaGenerator(base_weights, fused_attn=USE_FUSED_ATTN)

        if weights is not None and weights.name == "weights":
            # bugfix
            weights = None
        if weights:
            # If weights are passed in, they are LoRa weights
            # so we need to download the fp16 weights and load with peft
            self.initialize_peft(weights)
        else:
            print("Not using old-style COG_WEIGHTS LoRA weights")
            # raise Exception(f"Fine-tuned weights {weights} were improperly formatted.")

    # todo: adaptive cache like CLOCK
    @functools.lru_cache(maxsize=10)
    def get_lora(self, replicate_weights: str) -> "ExLlamaLora":
        if "http" in str(replicate_weights):  # weights are in the cloud
            print("Downloading peft weights")
            st = time.time()
            local_peft_weights = "local_weights.zip"
            download_file(str(replicate_weights), local_peft_weights)
            print(f"Downloaded peft weights in {time.time() - st}")

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
        lora = self.generator.load_lora(peft_path)
        print(f"Initialized peft model in {time.time() - st}")
        # remove file
        os.remove(local_peft_weights)
        return lora

    current_path: str = None

    def initialize_peft(self, replicate_weights: str) -> None:
        if self.current_path != replicate_weights:
            print(
                f"previous weights were {self.current_path}, switching to {replicate_weights}"
            )
            self.generator.lora = self.get_lora(replicate_weights)
            self.current_path = replicate_weights
        else:
            print("correct lora is already loaded")

    def predict(
        self,
        replicate_weights: str = Input(
            description="Path to fine-tuned weights produced by a Replicate fine-tune job.",
            default=None,
        ),
        prompt: str = Input(description="Prompt to send to the model."),
        system_prompt: str = Input(
            description="System prompt to send to the model. This is prepended to the prompt and helps guide system behavior.",
            default=DEFAULT_SYSTEM_PROMPT,
        ),
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
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed",
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
            start = time.time()
            self.initialize_peft(replicate_weights)
            print(f"Overall initialize_peft took {time.time() - start:.4f}")
        else:
            print("Not using LoRA")

        if seed is not None:
            print(f"Setting seed to {seed}")
            os.environ["PYTHONHASHSEED"] = str(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            random.seed(seed)

            import numpy

            numpy.random.seed(seed)

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
        print(f"hostname: {socket.gethostname()}")
        if debug:
            print(f"cur memory: {torch.cuda.memory_allocated()}")
            print(f"max allocated: {torch.cuda.max_memory_allocated()}")
            print(f"peak memory: {torch.cuda.max_memory_reserved()}")

    # def remove(f: "Callable", defaults: "dict[str, Any]") -> "Callable":
    #     # pylint: disable=no-self-argument
    #     # for the purposes of inspect.signature as used by predictor.get_input_type,
    #     # remove the argument (system_prompt)
    #     wrapped = functools.partialmethod(f, **defaults)
    #     sig = inspect.signature(wrapped)
    #     # TypeError: functools.partialmethod(<function Predictor.predict at 0x7fa5d2136340>, , system_prompt=None) is not a callabl object
    #
    #     params = [p for name, p in sig.parameters.items() if name not in defaults]
    #     wrapped.__signature__ = sig.replace(parameters=params)
    #     return wrapped

    # if not USE_SYSTEM_PROMPT:
    #     predict = remove(predict, {"system_prompt": None})

    _predict = predict

    def base_predict(self, *args, **kwargs) -> ConcatenateIterator:
        kwargs["system_prompt"] = None
        return self._predict(*args, **kwargs)

    # for the purposes of inspect.signature as used by predictor.get_input_type,
    # remove the argument (system_prompt)
    if not USE_SYSTEM_PROMPT:
        wrapper = base_predict
        # wrapper = functools.partialmethod(base_predict, system_prompt=None)
        sig = inspect.signature(_predict)
        params = [p for name, p in sig.parameters.items() if name != "system_prompt"]
        wrapper.__signature__ = sig.replace(parameters=params)
        predict = wrapper
