import asyncio
import json
import os
import time
from io import BytesIO, IOBase
from os import path
from queue import Queue
from threading import Thread
from typing import BinaryIO, List, Optional, Union, get_args

import torch
from cog import ConcatenateIterator
from mlc_chat import ChatConfig, ChatModule, ConvConfig, GenerationConfig
from mlc_chat.callback import StreamIterator
from src.config_utils import Weights
from transformers import AutoTokenizer

from .engine import Engine

class AsyncCompletionStream:
    def __init__(self, cm: ChatModule, generation_config: GenerationConfig):
        self.generation_config = generation_config
        self.cm = cm

    def __aiter__(self):
        return self

    async def get_next_msg(self):
        if not self.cm._stopped():
            self.cm._decode(generation_config=self.generation_config)
            msg = self.cm._get_message()
            return msg
        else:
            raise StopAsyncIteration

    async def __anext__(self):
        if not self.cm._stopped():
            task = asyncio.create_task(self.get_next_msg())
            msg = await task
            return msg
        else:
            raise StopAsyncIteration


class MLCEngine(Engine):
    """
    An inference engine that runs inference w/ vLLM
    """

    def __init__(self, weights: Weights, tokenizer_path: os.PathLike, is_chat: bool) -> None:
        model_path = self.load_weights(weights)
        self.is_chat = is_chat
        self.stop_str = "[INST]"
        self.stop_tokens = [2,]
        self.add_bos = True

        if self.is_chat:
            self.conv_template="llama-2"
        else:
            self.conv_template = "LM"

        conv_config = ConvConfig(
            stop_tokens=self.stop_tokens, add_bos=self.add_bos, stop_str=self.stop_str)
        chat_config = ChatConfig(
            conv_config=conv_config, conv_template=self.conv_template)

        model_path = path.join(model_path, "params")
        self.cm = ChatModule(model=model_path, chat_config=chat_config)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def get_logits(self):
        """
        Given a prompt, returns the logits from the language model.
        """
        raise NotImplementedError("MLC currently does not support logits.")

    def load_lora(self):
        """
        loads a lora from files into the format that this particular engine expects. DOES NOT prepare the engine for inference.
        lora_data is a dictionary of file names & references from the zip file
        """
        raise NotImplementedError("MLC currently does not support LoRAs.")

    def is_lora_active(self):
        """
        Returns True if the engine is currently configured to use a lora, False otherwise.
        """
        raise NotImplementedError("MLC currently does not support LoRAs.")

    def set_lora(self):
        """
        Given a loaded lora (created w/ load_lora), configures the engine to use that lora in combination with the loaded base weights.
        """
        raise NotImplementedError("MLC currently does not support LoRAs.")

    def delete_lora(self):
        print("MLC is currently not using any LoRAs.")

    def __call__(self, prompt: str, max_new_tokens: int, temperature: float, top_p: float, top_k: int, stop_sequences: str | List[str] = None, stop_token_ids: List[int] = [], repetition_penalty: float = 1.0, incremental_generation: bool = True, *args, **kwargs) -> ConcatenateIterator[str]:
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

        if top_k > 0:
            raise ValueError(
                "top_k is currently not supported by our generation engine.")

        stop_token_ids += self.stop_tokens
        # stop_sequences = [self.stop_str] + stop_sequences

        # TODO (Moin): add support for the system prompt on chat models
        # conv_config = ConvConfig(
        #     stop_tokens=stop_token_ids, add_bos=self.add_bos, stop_str=stop_sequences)
        # chat_config = ChatConfig(temperature=temperature, repetition_penalty=repetition_penalty,
        #                          top_p=top_p, max_gen_len=max_new_tokens,conv_config=conv_config, conv_template=self.conv_template)

        # new thing
        # this also shits the bed
        # self.cm.reset_chat()
        # generation_config = GenerationConfig(
        #     temperature=temperature,
        #     repetition_penalty=repetition_penalty,
        #     top_p=top_p,
        #     max_gen_len=max_new_tokens
        # )

        # something might not be thread safe? Try their StreamingResponse? 

        generation_config = GenerationConfig(
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            max_gen_len=max_new_tokens
        )
        self.cm.reset_chat()
        self.cm._prefill(input=prompt, generation_config=generation_config)

        min_new_tokens = kwargs.pop("min_new_tokens", None)
        if min_new_tokens is not None and min_new_tokens > -1:
            raise ValueError(
                "min_new_tokens is currently not supported by vLLM Engine.")

        if len(kwargs) > 0:
            raise ValueError(
                f"Unknown keyword arguments: {', '.join(kwargs.keys())}")

        # token_iterator = StreamIterator(callback_interval=1, timeout=None)
        # run the generation in the background
        # Thread(target=self.cm.generate, kwargs={
        #        "prompt": prompt, "progress_callback": token_iterator, "generation_config":generation_config}).start()
        # for token in token_iterator:
        #    yield token

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        stream = AsyncCompletionStream(self.cm, generation_config)
        generation_length = 0
        while True:
            try:
                out = loop.run_until_complete(stream.get_next_msg())
                yield out[generation_length:]
                
                generation_length = len(out)
            except StopAsyncIteration:
                break


        

