import asyncio
import json
import os
from io import BytesIO, IOBase
from typing import AsyncIterator, BinaryIO, List, Optional, Union, get_args

import torch
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams

from src.config_utils import Weights

from .engine import Engine

FILE_LIKE = str | os.PathLike
BYTES_LIKE = str | BinaryIO | IOBase | bytes


class LoRA:
    def __init__(
        self, adapter_config: Union[str, bytes, bytearray], adapter_model: FILE_LIKE
    ) -> None:
        self.adapter_config = json.loads(adapter_config)
        self.adapter_model = torch.load(adapter_model, map_location="cuda")

    @classmethod
    def load_from_path(
        cls, adapter_config_path: os.PathLike, adapter_model_path: os.PathLike
    ) -> "LoRA":
        with open(adapter_config_path, "r") as f:
            adapter_config = f.read()

        with open(adapter_model_path, "rb") as f:
            adapter_model = f.read()

        return cls(adapter_config=adapter_config, adapter_model=adapter_model)

    @classmethod
    def load_from_bytes(
        cls, adapter_config_bytes: BYTES_LIKE, adapter_model_bytes: BYTES_LIKE
    ) -> "LoRA":
        return cls(
            adapter_config=adapter_config_bytes, adapter_model=adapter_model_bytes
        )


class vLLMEngine(Engine):
    """
    An inference engine that runs inference w/ vLLM
    """

    def __init__(self, weights: Weights, dtype: str) -> None:
        model_path = self.load_weights(weights)
        args = AsyncEngineArgs(
            model=model_path,
            tokenizer=model_path,
            dtype=dtype,
        )
        self.engine = AsyncLLMEngine.from_engine_args(args)
        self.tokenizer = self.engine.engine.tokenizer

    def load_lora(
        self,
        lora_state_dict: Optional[dict] = None,
        adapter_model: Optional[FILE_LIKE | BYTES_LIKE] = None,
        adapter_config: Optional[FILE_LIKE | BYTES_LIKE] = None,
    ) -> LoRA:
        """
        loads a lora from files into the format that this particular engine expects. DOES NOT prepare the engine for inference.
        lora_data is a dictionary of file names & references from the zip file
        """

        # TODO (Moin): I don't like this "pass a dict or the explicit params" -- but going to add it in and ship ASAP.
        if lora_state_dict is None and adapter_model is None and adapter_config is None:
            raise ValueError(
                "At least one of lora_state_dict, adapter_model, or adapter_config must be provided."
            )

        if lora_state_dict is not None and (
            adapter_model is not None or adapter_config is not None
        ):
            raise ValueError(
                "lora_state_dict cannot be provided if adapter_model or adapter_config is provided."
            )

        if lora_state_dict is not None:
            ADAPTER_CONFIG_KEY_NAME = "adapter_config.json"
            ADAPTER_MODEL_KEY_NAME = "adapter_model.bin"
            if (
                ADAPTER_CONFIG_KEY_NAME not in lora_state_dict.keys()
                or ADAPTER_MODEL_KEY_NAME not in lora_state_dict.keys()
            ):
                raise ValueError(
                    f"lora_state_dict must include at least: '{ADAPTER_MODEL_KEY_NAME}' and '{ADAPTER_CONFIG_KEY_NAME}'."
                )

            adapter_config, adapter_model = (
                lora_state_dict[ADAPTER_CONFIG_KEY_NAME],
                BytesIO(lora_state_dict[ADAPTER_MODEL_KEY_NAME]),
            )

        if isinstance(adapter_model, get_args(FILE_LIKE)) and isinstance(
            adapter_config, get_args(FILE_LIKE)
        ):
            lora = LoRA.load_from_path(
                adapter_config_path=adapter_config, adapter_model_path=adapter_model
            )
        elif isinstance(adapter_model, get_args(BYTES_LIKE)) and isinstance(
            adapter_config, get_args(BYTES_LIKE)
        ):
            lora = LoRA.load_from_bytes(
                adapter_config_bytes=adapter_config, adapter_model_bytes=adapter_model
            )
        else:
            raise TypeError(
                "Both the adapter model and the adapter config must be either both file-like or bytes-like objects/primitives."
            )

        return lora

    def is_lora_active(self) -> bool:
        """
        Returns True if the engine is currently configured to use a lora, False otherwise.
        """
        return self.engine.engine.is_lora_active()

    def set_lora(self, lora: LoRA) -> None:
        """
        Given a loaded lora (created w/ load_lora), configures the engine to use that lora in combination with the loaded base weights.
        """
        self.delete_lora()  # defensive check -- can move this out of the engine if everything works appropriately
        self.delete_lora()  # defensive check -- can move this out of the engine if everything works appropriately
        self.engine.engine.load_lora(
            lora_config=lora.adapter_config, lora_state_dict=lora.adapter_model
        )

    def delete_lora(self) -> None:
        self.engine.engine.delete_lora()

    async def generate_stream(
        self, prompt: str, sampling_params: SamplingParams
    ) -> AsyncIterator[str]:
        results_generator = self.engine.generate(prompt, sampling_params, 0)
        async for generated_text in results_generator:
            yield generated_text

    def __call__(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        stop_sequences: str | List[str] = None,
        stop_token_ids: List[int] = None,
        frequency_penalty: float = 1.0,
        incremental_generation: bool = True,
        *args,
        **kwargs,
    ) -> str:
        """
        Given a prompt, runs generation on the language model with vLLM.

        Args:
        - prompt (str): the prompt to give the model.
        - max_new_tokens (int): the maximum number of new tokens to generate.
        - temperature (float): the parameter to anneal the sampling distribution with.
        - top_p (float): the amount to truncate the sampling distribution by.
        - top_k (int): the number of tokens to truncate the sampling distribution by.
        - stop_sequences (str | List[str]): the string to stop generation at.
        - stop_token_ids (List[str]): a list of token ids to stop generation at.
        - frequency_penalty (float): the amount to penalize tokens that have already been generated, higher values penalize more.
        - incremental_generation: whether to yield the entire generated sequence or the next generated token at each step.

        Yields:
        - generated_text (str): the generated text, or next token, depending on the value of `incremental_generation`.
        """
        if top_k is None or top_k == 0:
            top_k = -1

        min_new_tokens = kwargs.pop("min_new_tokens", None)
        if min_new_tokens is not None and min_new_tokens > -1:
            raise ValueError(
                "min_new_tokens is currently not supported by vLLM Engine."
            )

        stop_token_ids = stop_token_ids or []
        stop_token_ids.append(self.tokenizer.eos_token_id)

        if isinstance(stop_sequences, str) and stop_sequences != "":
            stop = [stop_sequences]
        elif isinstance(stop_sequences, list) and len(stop_sequences) > 0:
            stop = stop_sequences
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
            frequency_penalty=frequency_penalty,
        )

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        gen = self.generate_stream(
            prompt,
            sampling_params,
        )

        generation_length = 0
        while True:
            try:
                request_output = loop.run_until_complete(gen.__anext__())
                assert len(request_output.outputs) == 1
                generated_text = request_output.outputs[0].text
                if incremental_generation:
                    # it takes multiple calls to gen.__anext__ to render one emoji. 
                    # this check keeps us from needlesly yielding empty strings
                    if len(generated_text) > generation_length:
                        yield generated_text[generation_length:]
                else:
                    yield generated_text
                generation_length = len(generated_text)
            except StopAsyncIteration:
                break


def run_generation():
    """
    Helper class to run the generation for tests.
    """
    model_path = "/home/moin/Llama-2-7b"
    tokenizer_path = "/home/moin/Llama-2-7b"
    dtype = "auto"
    engine = vLLMEngine(
        model_path=model_path, tokenizer_path=tokenizer_path, dtype=dtype
    )
    prompt = "Hello,"
    generated_text = engine(
        prompt=prompt, max_new_tokens=128, temperature=1.0, top_p=0.9, top_k=50
    )
    for text in generated_text:
        print(text, end="")


if __name__ == "__main__":
    run_generation()
