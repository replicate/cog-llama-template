import torch
import asyncio

from typing import Optional
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams

from cog import BasePredictor, ConcatenateIterator, Input, Path

from config import (
    LOCAL_DEFAULT_INFERENCE_WEIGHTS_PATH, 
    REMOTE_DEFAULT_INFERENCE_FILES_TO_DOWNLOAD, 
    REMOTE_DEFAULT_INFERENCE_WEIGHTS_PATH,
)

from src.utils import maybe_download_with_pget


# This prompt formatting was copied from the original CodeLlama repo:
# https://github.com/facebookresearch/llama/blob/6c7fe276574e78057f917549435a2554000a876d/llama/generation.py#L44


class Predictor(BasePredictor):
    def setup(self, weights: Optional[Path] = None):
        print('starting setup')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if weights is not None and weights.name == "weights":
            weights = None

        # If weights aren't passed in, we'll use the default weights configuration
        if not weights:
            weights = LOCAL_DEFAULT_INFERENCE_WEIGHTS_PATH
            weights = maybe_download_with_pget(
                weights, REMOTE_DEFAULT_INFERENCE_WEIGHTS_PATH, REMOTE_DEFAULT_INFERENCE_FILES_TO_DOWNLOAD,
            )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        args = AsyncEngineArgs(
            model=LOCAL_DEFAULT_INFERENCE_WEIGHTS_PATH,
            tokenizer=LOCAL_DEFAULT_INFERENCE_WEIGHTS_PATH,
            dtype="float16",
            max_num_seqs=16384,
        )
        self.engine = AsyncLLMEngine.from_engine_args(args)
        self.tokenizer = self.engine.engine.tokenizer

    async def generate_stream(
        self,
        prompt,
        temperature=0.6,
        top_p=0.95,
        top_k=50,
        max_new_tokens=128,
        stop_str=None,
        stop_token_ids=None,
        repetition_penalty=1.0
    ):
        context = prompt
        stop_token_ids = stop_token_ids or []
        stop_token_ids.append(self.tokenizer.eos_token_id)

        if isinstance(stop_str, str) and stop_str != "":
            stop = [stop_str]
        elif isinstance(stop_str, list) and stop_str != []:
            stop = stop_str
        else:
            stop = []

        for tid in stop_token_ids:
            stop.append(self.tokenizer.decode(tid))

        # make sampling params in vllm
        top_p = max(top_p, 1e-5)
        if temperature <= 1e-5:
            top_p = 1.0
        sampling_params = SamplingParams(
            n=1,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            use_beam_search=False,
            stop=stop,
            max_tokens=max_new_tokens,
            frequency_penalty=repetition_penalty,
        )
        results_generator = self.engine.generate(context, sampling_params, 0)

        async for request_output in results_generator:
            prompt = request_output.prompt
            yield request_output.outputs[-1].text

    def predict(
        self,
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
        
        if stop_sequences:
            stop_sequences = stop_sequences.split(",")
        
        print(f"Prompt: \n{prompt}")

        loop = asyncio.get_event_loop()

        gen = self.generate_stream(
            prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            stop_str=stop_sequences,
            repetition_penalty=1.0
        )

        prv_value = ""
        value = ""
        while True:
            prv_value = value
            try:
                value = loop.run_until_complete(gen.__anext__())
                yield value[len(prv_value) :]
            except StopAsyncIteration:
                break
