import functools
import inspect
import io
import os
import random
import shutil
import socket
import time
import zipfile
from typing import Any, Optional

import torch
from cog import BasePredictor, ConcatenateIterator, Input, Path

from config import ENGINE, ENGINE_KWARGS, USE_SYSTEM_PROMPT
from src.download import Downloader
from src.utils import seed_all

# This prompt formatting was copied from the original Llama v2 repo:
# https://github.com/facebookresearch/llama/blob/6c7fe276574e78057f917549435a2554000a876d/llama/generation.py#L44

# These are components of the prompt that should not be changed by the users
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
PROMPT_TEMPLATE = f"{B_INST} {B_SYS}{{system_prompt}}{E_SYS}{{instruction}} {E_INST}"

# Users may want to change the system prompt, but we use the recommended system prompt by default
DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant."""

# Temporary hack to disable Top K from the API. We should get rid of this once engines + configs are better standardized.
USE_TOP_K = ENGINE.__name__ != "MLCEngine"

class Predictor(BasePredictor):
    def setup(self, weights: Optional[Path] = None):
        print("Starting setup")
        self.downloader = Downloader()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.engine = ENGINE(**ENGINE_KWARGS)

        if weights is not None and weights.name == "weights":
            # bugfix
            weights = None
        if weights:
            # If weights are passed in, they are LoRa weights
            # so we need to download the fp16 weights and load with peft
            self.initialize_peft(weights)
        else:
            print("Not using old-style COG_WEIGHTS LoRA weights")

    # todo: adaptive cache like CLOCK
    @functools.lru_cache(maxsize=10)
    def get_lora(self, replicate_weights: str) -> Any:
        if "http" in str(replicate_weights):  # weights are in the cloud
            print("Downloading peft weights")
            st = time.time()
            buffer = self.downloader.sync_download_file(str(replicate_weights))
            print(f"Downloaded peft weights in {time.time() - st:.3f}")
        else:
            # zipfile accepts either a file-like or path-like object
            buffer = replicate_weights
        st = time.time()
        with zipfile.ZipFile(buffer, "r") as zip_ref:
            data = {name: zip_ref.read(name) for name in zip_ref.namelist()}
        print(f"Unzipped peft weights in {time.time() - st:.3f}")
        st = time.time()
        lora = self.engine.load_lora(data)
        del data, zip_ref
        print(f"Initialized peft model in {time.time() - st:.3f}")
        return lora

    current_path: str = None

    def initialize_peft(self, replicate_weights: str) -> None:
        if self.current_path != replicate_weights:
            print(
                f"previous weights were different, switching to {replicate_weights}"
            )
            self.engine.set_lora(self.get_lora(replicate_weights))

            self.current_path = replicate_weights
        else:
            print("correct lora is already loaded")

    def delete_lora(self):
        self.current_path = None
        self.engine.delete_lora()

    def predict(
        self,
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
            default=0.7,
        ),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.0,
            le=1.0,
            default=0.95,
        ),
        top_k: int = Input(
            description="When decoding text, samples from the top k most likely tokens; lower to ignore less likely tokens",
            ge=-1,
            default=-1,
        ),
        repetition_penalty: float = Input(
            description="A parameter that controls how repetitive text can be. Lower means more repetitive, while higher means less repetitive. Set to 1.0 to disable.",
            ge=0.0,
            default=1.15,
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
        return_logits: bool = Input(
            description="if set, only return logits for the first token. only useful for testing, etc.", default=False
        ),
        replicate_weights: str = Input(
            description="Path to fine-tuned weights produced by a Replicate fine-tune job.",
            default=None,
        ),
    ) -> ConcatenateIterator[str]:
        if stop_sequences:
            stop_sequences = stop_sequences.split(",")

        if USE_SYSTEM_PROMPT:
            prompt = prompt.strip("\n").removeprefix(
                B_INST).removesuffix(E_INST).strip()
            prompt = PROMPT_TEMPLATE.format(
                system_prompt=system_prompt.strip(), instruction=prompt.strip()
            )

        print(f"Your formatted prompt is: \n{prompt}")

        if replicate_weights:
            start = time.time()
            self.initialize_peft(replicate_weights)
            print(f"Overall initialize_peft took {time.time() - start:.3f}")
        else:
            if 'COG_WEIGHTS' not in os.environ:
                self.delete_lora()
                print("Not using LoRA")

        if seed is not None:
            print(f"Setting seed to {seed}")
            seed_all(seed)

        n_tokens = 0
        st = time.time()

        if return_logits:
            logits = self.engine.get_logits(prompt)
            # serializing so we aren't returning a massive json
            logits_path = 'logits.pt'
            torch.save(logits, logits_path)
            yield Path(logits_path)

        # todo: may need to do something clever with kwargs if/when we add more engines.
        else:
            generated_text = ""
            for decoded_token in self.engine(
                prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                stop_sequences=stop_sequences,
            ):
                n_tokens += 1
                yield decoded_token
                generated_text += decoded_token
                if n_tokens == 1 and debug:
                    second_start = time.time()
                if seed is not None:
                    torch.manual_seed(seed)
            et = time.time()
            t = et - st
            print(f"hostname: {socket.gethostname()}")
            if debug:
                print("generated text:", generated_text)
                print(
                    f"after initialization, first token took {second_start - st:.3f}")
                print(f"Tokens per second: {n_tokens / t:.2f}")
                print(
                    f"Tokens per second not including time to first token: {(n_tokens -1) / (et - second_start):.2f}")
                print(f"cur memory: {torch.cuda.memory_allocated()}")
                print(f"max allocated: {torch.cuda.max_memory_allocated()}")
                print(f"peak memory: {torch.cuda.max_memory_reserved()}")

    # # we'd like this to work eventually
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
        if not USE_TOP_K:
            kwargs["top_k"] = None
        return self._predict(*args, **kwargs)

    # for the purposes of inspect.signature as used by predictor.get_input_type,
    # remove the argument (system_prompt)
    # this removes system_prompt from the Replicate API for non-chat models.
    if not USE_SYSTEM_PROMPT or not USE_TOP_K:
        params_to_remove = ["None"]
        if not USE_SYSTEM_PROMPT:
            params_to_remove.append("system_prompt")
        if not USE_TOP_K:
            params_to_remove.append("top_k")

        wrapper = base_predict
        # wrapper = functools.partialmethod(base_predict, system_prompt=None)
        sig = inspect.signature(_predict)
        params = []
        for name, p in sig.parameters.items():
            if name not in params_to_remove:
                params.append(p)
        wrapper.__signature__ = sig.replace(parameters=params)
        predict = wrapper
