from typing import Any, Optional, List
import os

from .engine import Engine
from .vllm_engine import vLLMEngine


class MLCvLLMEngine(Engine):
    """
    MLC for base models, vllm for fine-tunes.
    """

    def __init__(self, mlc_args: dict, vllm_args: dict) -> None:
        # checks for old style loras & if this is booted as a fine-tuneable hotswap
        if os.getenv("COG_WEIGHTS") or os.getenv("REPLICATE_HOTSWAP") == "1":
            self.engine = vLLMEngine(**vllm_args)
        else:
            # can't run vllm if MLC is imported
            from .mlc_engine import MLCEngine

            self.engine = MLCEngine(**mlc_args)
            self.vllm_args = vllm_args

    def load_lora(self, lora_data: dict) -> Any:
        """
        loads a lora from files into the format that this particular engine expects. DOES NOT prepare the engine for inference.
        lora_data is a dictionary of file names & references from the zip file
        """
        if not isinstance(self.engine, vLLMEngine):
            # Really we should never need to do this.
            # print("Transitioning from MLC to vLLM")
            # del self.engine.cm
            # del self.engine.tokenizer
            # del self.engine

            # gc.collect()
            # torch.cuda.empty_cache()
            # self.engine = vLLMEngine(**self.vllm_args)
            raise Exception("Loras not supported with MLCEngine")

        return self.engine.load_lora(lora_data)

    def is_lora_active(self) -> bool:
        """
        Returns True if the engine is currently configured to use a lora, False otherwise.
        """
        if isinstance(self.engine, vLLMEngine):
            return self.engine.is_lora_active()
        return False

    def set_lora(self, lora: Any) -> None:
        """
        Given a loaded lora (created w/ load_lora), configures the engine to use that lora in combination with the loaded base weights.
        """
        if not isinstance(self.engine, vLLMEngine):
            raise Exception(
                "Loras not supported with MLC Engine! Invalid state reached."
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
        print(f"MLC: {not isinstance(self.engine, vLLMEngine)}")
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
