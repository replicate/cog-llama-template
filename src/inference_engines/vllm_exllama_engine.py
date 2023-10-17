import gc
from typing import Any, Optional, List

import torch
import os

from .engine import Engine
from .vllm_engine import vLLMEngine
from .exllama import ExllamaEngine
from src.utils import maybe_download_with_pget


class ExllamaVllmEngine(Engine):
    """
    It's exllama until fine-tuning hits, and then it's vllm. 
    """

    def __init__(self, model_path: str, vllm_args: dict, exllama_args: dict) -> None:
        # for old-style loras, should they happen
        if 'COG_WEIGHTS' in os.environ:
            vllm_model_info = vllm_args.pop('vllm_model_info')
            maybe_download_with_pget(
                vllm_model_info['local_path'],
                vllm_model_info['remote_path'],
                vllm_model_info['remote_files']
            )
            self.engine = vLLMEngine(vllm_model_info['local_path'], **self.vllm_args)
        else:
            self.engine = ExllamaEngine(model_path, **exllama_args)
            self.vllm_args = vllm_args


    def load_lora(self, lora_data:dict) -> Any:
        """
        loads a lora from files into the format that this particular engine expects. DOES NOT prepare the engine for inference.
        lora_data is a dictionary of file names & references from the zip file
        """
        if isinstance(self.engine, ExllamaEngine):
            print("Transitioning from vLLM to Exllama")
            del self.engine.model
            del self.engine.generator
            del self.engine 

            gc.collect()
            torch.cuda.empty_cache()
            vllm_model_info = self.vllm_args.pop('vllm_model_info')
            maybe_download_with_pget(
                vllm_model_info['local_path'],
                vllm_model_info['remote_path'],
                vllm_model_info['remote_files']
            )
            self.engine = vLLMEngine(vllm_model_info['local_path'], **self.vllm_args)
        
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
        if isinstance(self.engine, ExllamaEngine):
            raise Exception("Loras not supported with Exllama Engine! Invalid state reached.")
        self.engine.set_lora(lora)

    def delete_lora(self) -> None:
        self.engine.delete_lora()


    def __call__(self, 
                 prompt,
                 max_new_tokens:int =128,
                 min_new_tokens:int =-1,
                 temperature:float =0.75,
                 top_p:float =0.9,
                 top_k:int =50,
                 stop_sequences: Optional[List[str]] = None,
                 **kwargs):
        print(f"Exllama: {isinstance(self.engine, ExllamaEngine)}")
        gen = self.engine(prompt, 
                          max_new_tokens=max_new_tokens, 
                          min_new_tokens=min_new_tokens,
                          temperature=temperature,
                          top_p=top_p,
                          top_k=top_k,
                          stop_sequences=stop_sequences,
                          **kwargs)
        for val in gen:
            yield val