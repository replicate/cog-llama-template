import asyncio
import json
import os
from io import IOBase
from typing import BinaryIO, List, Union, get_args

import torch
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams

from .engine import Engine

FILE_LIKE = str | os.PathLike
BYTES_LIKE = str | BinaryIO | IOBase | bytes


class LoRA:

    def __init__(self, adapter_config: Union[str, bytes, bytearray], adapter_model: FILE_LIKE) -> None:
        self.adapter_config = json.loads(adapter_config)
        self.adapter_model = torch.load(adapter_model, map_location="cpu")

    @classmethod
    def load_from_path(cls, adapter_config_path: os.PathLike, adapter_model_path: os.PathLike) -> "LoRA":
        with open(adapter_config_path, "r") as f:
            adapter_config = f.read()

        with open(adapter_model_path, "rb") as f:
            adapter_model = f.read()

        return cls(adapter_config=adapter_config, adapter_model=adapter_model)

    @classmethod
    def load_from_bytes(cls, adapter_config_bytes: BYTES_LIKE, adapter_model_bytes: BYTES_LIKE) -> "LoRA":
        return cls(adapter_config=adapter_config_bytes, adapter_model=adapter_model_bytes)


# TODO (Moin): this class should inherit from engine
class vLLMEngine(Engine):
    """
    An inference engine that runs inference w/ vLLM
    """

    def __init__(self, model_path: os.PathLike, tokenizer_path: os.PathLike, dtype: str, max_num_seqs: int = 16384) -> None:
        args = AsyncEngineArgs(
            model=model_path,
            tokenizer=tokenizer_path,
            dtype=dtype,
            max_num_seqs=max_num_seqs,
        )
        self.engine = AsyncLLMEngine.from_engine_args(args)
        self.tokenizer = self.engine.engine.tokenizer

    def load_lora(self, adapter_model: FILE_LIKE | BYTES_LIKE, adapter_config: FILE_LIKE | BYTES_LIKE) -> LoRA:
        """
        loads a lora from files into the format that this particular engine expects. DOES NOT prepare the engine for inference.
        lora_data is a dictionary of file names & references from the zip file
        """

        if isinstance(adapter_model, get_args(FILE_LIKE)) and isinstance(adapter_config, get_args(FILE_LIKE)):
            lora = LoRA.load_from_path(
                adapter_config_path=adapter_config, adapter_model_path=adapter_model)
        elif isinstance(adapter_model, get_args(BYTES_LIKE)) and isinstance(adapter_config, get_args(BYTES_LIKE)):
            lora = LoRA.load_from_bytes(
                adapter_config_bytes=adapter_config, adapter_model_bytes=adapter_model)
        else:
            raise TypeError(
                "Both the adapter model and the adapter config must be either both file-like or bytes-like objects/primitives.")

        return lora

    def set_lora(self, lora: LoRA) -> None:
        """
        Given a loaded lora (created w/ load_lora), configures the engine to use that lora in combination with the loaded base weights.
        """

        self.engine.engine.load_lora(
            lora_config=lora.adapter_config, lora_state_dict=lora.adapter_model)

    def delete_lora(self) -> None:
        self.engine.engine.delete_lora()

    async def __call__(self, prompt: str, max_new_tokens: int, temperature: float, top_p: float, top_k: int, stop_str: str = None, stop_token_ids: List[int] = None, repetition_penalty: float = 1.0, incremental_generation: bool = True) -> str:
        """
        Given a prompt, runs generation on the language model with vLLM.

        Args:
        - prompt (str): the prompt to give the model.
        - max_new_tokens (int): the maximum number of new tokens to generate.
        - temperature (float): the parameter to anneal the sampling distribution with.
        - top_p (float): the amount to truncate the sampling distribution by.
        - top_k (int): the number of tokens to truncate the sampling distribution by.
        - stop_str (str): the string to stop generation at.
        - stop_token_ids (List[str]): a list of token ids to stop generation at.
        - repetition_penalty (float): the amount to penalize tokens that have already been generated, higher values penalize more.
        - incremental_generation: whether to yield the entire generated sequence or the next generated token at each step.

        Yields:
        - generated_text (str): the generated text, or next token, depending on the value of `incremental_generation`.
        """
        stop_token_ids = stop_token_ids or []
        stop_token_ids.append(self.tokenizer.eos_token_id)

        if isinstance(stop_str, str) and stop_str != "":
            stop = [stop_str]
        elif isinstance(stop_str, list) and len(stop_str) > 0:
            stop = stop_str
        else:
            stop = []

        for tid in stop_token_ids:
            stop.append(self.tokenizer.decode(tid))

        sampling_params = SamplingParams(
            n=1,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            use_beam_search=False,
            stop=stop,
            max_tokens=max_new_tokens,
            frequency_penalty=repetition_penalty,
        )
        results_generator = self.engine.generate(prompt, sampling_params, 0)

        generation_length = 0
        async for request_output in results_generator:
            assert len(request_output.outputs) == 1
            generated_text = request_output.outputs[0].text
            if incremental_generation:
                yield generated_text[generation_length:]
            else:
                yield generated_text
            generation_length = len(generated_text)


async def run_generation():
    """
    Helper class to run the generation for tests.
    """
    model_path = "/home/moin/Llama-2-7b"
    tokenizer_path = "/home/moin/Llama-2-7b"
    dtype = "auto"
    engine = vLLMEngine(model_path=model_path,
                        tokenizer_path=tokenizer_path, dtype=dtype)
    prompt = "Hello,"
    generated_text = engine(prompt=prompt, max_new_tokens=128,
                            temperature=1.0, top_p=0.9, top_k=50)
    async for text in generated_text:
        print(text, end="")

if __name__ == "__main__":
    asyncio.run(run_generation())
