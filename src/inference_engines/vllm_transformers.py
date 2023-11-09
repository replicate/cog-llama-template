import gc
from typing import Any, Optional, List

import torch

from .engine import Engine
from .vllm_engine import vLLMEngine
from .transformers_engine import TransformersEngine


class vLLMTransformersEngine(Engine):
    """
    It's vLLM until fine-tuning hits, and then it's transformers.
    """

    def __init__(
        self, model_path: str, vllm_args: dict, transformers_args: dict
    ) -> None:
        self.engine = vLLMEngine(model_path, **vllm_args)
        self.model_path = model_path
        self.transformers_args = transformers_args

    def load_lora(self, lora_data: dict) -> Any:
        """
        loads a lora from files into the format that this particular engine expects. DOES NOT prepare the engine for inference.
        lora_data is a dictionary of file names & references from the zip file
        """
        if isinstance(self.engine, vLLMEngine):
            print("Transitioning from vLLM to Transformers")
            for worker in self.engine.engine.engine.workers:  # needs more engine
                del worker.cache_engine.gpu_cache
                del worker.cache_engine.cpu_cache
                del worker.gpu_cache
                del worker.model

            del self.engine
            gc.collect()
            torch.cuda.empty_cache()
            self.engine = TransformersEngine(self.model_path, **self.transformers_args)

        return self.engine.load_lora(lora_data)

    def is_lora_active(self) -> bool:
        """
        Returns True if the engine is currently configured to use a lora, False otherwise.
        """
        if isinstance(self.engine, TransformersEngine):
            return self.engine.is_lora_active()
        return False

    def set_lora(self, lora: Any) -> None:
        """
        Given a loaded lora (created w/ load_lora), configures the engine to use that lora in combination with the loaded base weights.
        """
        if isinstance(self.engine, vLLMEngine):
            raise Exception(
                "Loras not supported with vLLM Engine! Invalid state reached."
            )
        self.engine.set_lora(lora)

    def delete_lora(self) -> None:
        self.engine.delete_lora()

    def __call__(
        self,
        prompt,
        max_new_tokens: int = 128,
        min_new_tokens: int = -1,
        temperature: float = 0.75,
        top_p: float = 0.9,
        top_k: int = 50,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ):
        gen = self.engine(
            prompt,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop_sequences,
            **kwargs,
        )
        for val in gen:
            yield val
