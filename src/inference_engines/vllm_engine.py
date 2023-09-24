import os
from typing import (IO, Any, BinaryIO, Callable, Dict, Optional, Tuple, Type,
                    Union, cast)

from typing_extensions import TypeAlias  # Python 3.10+
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer import detokenize_incrementally

# from engine import Engine

FILE_LIKE = str | os.PathLike | BinaryIO | IO[bytes]
BYTES_LIKE = str | BinaryIO | IO[bytes]


class LoRA:

    def __init__(self, adapter_config_file: Union[str, bytes, bytearray], adapter_model_file: FILE_LIKE):
        self.adapter_config = json.loads(adapter_config_file)
        self.adapter_model = torch.load(adapter_model_file, map_location="cpu")

    @classmethod
    def load_from_path(cls, adapter_config_path: os.PathLike, adapter_model_path: os.PathLike) -> "LoRA":
        with open(adapter_config_path, "r") as f:
            adapter_config = f.read()

        with open(adapter_model_path, "rb") as f:
            adapter_model = f.read()

        return cls(adapter_config=adapter_config, adapter_model=adapter_model)

    @classmethod
    def load_from_bytes(self, adapter_config_bytes: BYTES_LIKE, adapter_model_bytes: BYTES_LIKE) -> "LoRA":
        return cls(adapter_config=adapter_config_bytes, adapter_model=adapter_model_bytes)


class vLLMEngine():
    """
    An inference engine that runs inference w/ vLLM
    """

    def __init__(self, model_path: os.PathLike, tokenizer_path: os.PathLike, dtype, max_num_seqs: int = 16384):
        args = AsyncEngineArgs(
            model=model_path,
            tokenizer=tokenizer_path,
            dtype=dtype,
            max_num_seqs=max_num_seqs,
        )
        # from remote_pdb import RemotePdb
        # RemotePdb('0.0.0.0', 4444).set_trace()
        self.engine = AsyncLLMEngine.from_engine_args(args)
        self.tokenizer = self.engine.engine.tokenizer

    def load_lora(self, adapter_model, adapter_config):
        """
        loads a lora from files into the format that this particular engine expects. DOES NOT prepare the engine for inference.
        lora_data is a dictionary of file names & references from the zip file
        """

        print("Adapter model:", adapter_model)
        if isinstance(adapter_model, FILE_LIKE) and isinstance(adapter_config, FILE_LIKE):
            lora = LoRA.load_from_path(
                adapter_config_path=adapter_config, adapter_model_path=adapter_model)
        elif isinstance(adapter_model, BYTES_LIKE) and isinstance(adpater_config, BYTES_LIKE):
            lora = LoRA.load_from_bytes(
                adapter_config_bytes=adapter_config, adapter_model_bytes=adapter_model)
        else:
            raise TypeError(
                "Both the adapter model and the adapter config must be either both file-like or bytes-like objects/primitives.")

        return lora

    def set_lora(self, lora: LoRA):
        """
        Given a loaded lora (created w/ load_lora), configures the engine to use that lora in combination with the loaded base weights.
        """

        self.engine.engine.load_lora(
            config=lora.adapter_config, model=lora.adapter_model)

    def delete_lora(self):
        self.engine.engine.delete_lora()

    async def __call__(self, prompt, max_new_tokens: int, temperature: float, top_p: float, top_k: int, stop_str=None, stop_token_ids=None, repetition_penalty=1.0, incremental_generation: bool = True) -> str:
        """
        generation!
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
