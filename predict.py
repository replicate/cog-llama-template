import asyncio
import functools
import io
import sys
import time
import zipfile
from typing import Optional

import torch
from cog import BasePredictor, ConcatenateIterator, Input, Path

from config import (
    LOCAL_DEFAULT_INFERENCE_WEIGHTS_PATH,
    REMOTE_DEFAULT_INFERENCE_FILES_TO_DOWNLOAD,
    REMOTE_DEFAULT_INFERENCE_WEIGHTS_PATH,
)
from src.download import Downloader
from src.inference_engines.vllm_engine import vLLMEngine
from src.utils import maybe_download_with_pget

# This prompt formatting was copied from the original CodeLlama repo:
# https://github.com/facebookresearch/llama/blob/6c7fe276574e78057f917549435a2554000a876d/llama/generation.py#L44


class Predictor(BasePredictor):
    def setup(self, weights: Optional[Path] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if weights is not None and weights.name == "weights":
            weights = None

        # If weights aren't passed in, we'll use the default weights configuration
        if not weights:
            weights = LOCAL_DEFAULT_INFERENCE_WEIGHTS_PATH
            local_weights_path = maybe_download_with_pget(
                weights,
                REMOTE_DEFAULT_INFERENCE_WEIGHTS_PATH,
                REMOTE_DEFAULT_INFERENCE_FILES_TO_DOWNLOAD,
            )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        self.engine = vLLMEngine(
            model_path=local_weights_path,
            tokenizer_path=local_weights_path,
            dtype="float16",
        )
        self.tokenizer = self.engine.tokenizer
        self.downloader = Downloader()

    @functools.lru_cache(maxsize=10)
    def get_lora(self, replicate_weights: str) -> "ExLlamaLora":
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
        return data["adapter_config.json"], io.BytesIO(data["adapter_model.bin"])

    async def generate_stream(
        self,
        prompt,
        temperature=0.6,
        top_p=0.95,
        top_k=50,
        max_new_tokens=128,
        stop_str=None,
        stop_token_ids=None,
        repetition_penalty=1.0,
    ):
        results_generator = self.engine(
            prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            stop_str=stop_str,
            stop_token_ids=stop_token_ids,
            repetition_penalty=repetition_penalty,
        )

        async for generated_text in results_generator:
            yield generated_text

    def predict(
        self,
        lora_path: str = Input(
            description="Path to .zip of LoRA weights.",
            default="https://pub-df34620a84bb4c0683fae07a260df1ea.r2.dev/sql.zip",
        ),
        prompt: str = Input(description=f"Prompt to send to CodeLlama."),
        max_new_tokens: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens",
            ge=1,
            default=128,
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
    ) -> ConcatenateIterator[str]:
        if lora_path:
            adapter_config, adapter_model = self.get_lora(replicate_weights=lora_path)
            lora = self.engine.load_lora(adapter_model, adapter_config)
            self.engine.set_lora(lora)
        if stop_sequences:
            stop_sequences = stop_sequences.split(",")

        loop = asyncio.get_event_loop()

        start_time = time.time()
        gen = self.generate_stream(
            prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            stop_str=stop_sequences,
            repetition_penalty=1.0,
        )

        generated_text = ""
        num_tokens = 0
        while True:
            try:
                generated_tokens = loop.run_until_complete(gen.__anext__())
                num_tokens += 1
                yield generated_tokens
                generated_text += generated_tokens
            except StopAsyncIteration:
                end_time = time.time()
                break

        generation_speed = num_tokens / (end_time - start_time)
        print(
            f"Generated {num_tokens} tokens in {end_time - start_time:.3f} seconds ({generation_speed:.3f} tokens per second)"
        )
        return generated_text
