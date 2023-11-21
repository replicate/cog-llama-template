import time
from abc import ABC, abstractmethod
from typing import Any

from src.config_utils import Weights
from src.utils import maybe_download_with_pget


class Engine(ABC):
    """
    WIP - this is what the engine looks like at the moment, outlining this just as an exercise to see what our ABC looks like. It will change.
    """

    def load_weights(self, weights: Weights):
        start = time.time()
        maybe_download_with_pget(
            weights.local_path, weights.remote_path, weights.remote_files
        )
        print(f"downloading weights took {time.time() - start:.3f}s")
        return weights.local_path

    @abstractmethod
    def load_lora(self, lora_data: dict):
        """
        loads a lora from files into the format that this particular engine expects. DOES NOT prepare the engine for inference.
        lora_data is a dictionary of file names & references from the zip file
        """
        pass

    @abstractmethod
    def set_lora(self, lora: Any):
        """
        given a loaded lora (created w/load_lora), configures the engine to use that lora in combination with the loaded base weights.
        """
        pass

    @abstractmethod
    def is_lora_active(self) -> bool:
        """
        Checks whether a LoRA has currently been loaded onto the engine.
        """
        pass

    @abstractmethod
    def delete_lora(self):
        """
        Deletes a LoRA.
        """
        pass

    @abstractmethod
    def __call__(self, prompt, **kwargs):
        """
        generation!
        """
        pass
