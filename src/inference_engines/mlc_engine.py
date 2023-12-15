import os

from cog import ConcatenateIterator
from mlc_chat import ChatConfig, ChatModule, ConvConfig, GenerationConfig
from transformers import AutoTokenizer

from src.config_utils import Weights

from .engine import Engine

class MLCEngine(Engine):
    """
    An inference engine that runs inference w/ vLLM
    """

    def __init__(
        self, weights: Weights, is_chat: bool, num_shards: int = 1, tokenizer_path: os.PathLike = None
    ) -> None:
        weights_path = self.load_weights(weights)
        self.is_chat = is_chat
        self.num_shards = num_shards

        if self.is_chat:
            self.conv_template = "llama-2"
            self.stop_str = ""
            self.stop_tokens = []
            self.add_bos = None
        else:
            self.conv_template = "LM"
            self.stop_str = "[INST]"
            self.stop_tokens = [
                2,
            ]
            self.add_bos = True

        conv_config = ConvConfig(
            stop_tokens=self.stop_tokens, add_bos=self.add_bos, stop_str=self.stop_str
        )
        chat_config = ChatConfig(
            conv_config=conv_config, conv_template=self.conv_template, num_shards=self.num_shards
        )

        model_path = os.path.join(weights_path, "params")
        self.cm = ChatModule(model=model_path, chat_config=chat_config)

        # this isn't used!
        tokenizer_path = os.path.join(weights_path, "params")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def load_weights(self, weights: Weights) -> str:
        """
        Downloads the weights from the given Weights object and returns the path to the downloaded weights.

        Args:
        - weights (Weights): the weights to download.

        Returns:
        - weights_path (str): the path to the downloaded weights.
        """
        # ensure directories exist
        for path in weights.remote_files:
            path_directory = os.path.dirname(path)
            if path_directory:
                path_directory = os.path.join(weights.local_path, path_directory)
                os.makedirs(path_directory, exist_ok=True)

        return super().load_weights(weights)

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

    def __call__(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        stop_sequences: str | list[str] = None,
        stop_token_ids: list[int] = [],
        repetition_penalty: float = 1.0,
        incremental_generation: bool = True,
        *args,
        **kwargs,
    ) -> ConcatenateIterator[str]:
        """
        Given a prompt, runs generation on the language model with vLLM.

        Args:
        - prompt (str): the prompt to give the model.
        - max_new_tokens (int): the maximum number of new tokens to generate.
        - temperature (float): the parameter to anneal the sampling distribution with.
        - top_p (float): the amount to truncate the sampling distribution by.
        - top_k (int): the number of tokens to truncate the sampling distribution by.
        - stop_sequences (str | list[str]): the string to stop generation at.
        - stop_token_ids (list[str]): a list of token ids to stop generation at.
        - frequency_penalty (float): the amount to penalize tokens that have already been generated, higher values penalize more.
        - incremental_generation: whether to yield the entire generated sequence or the next generated token at each step.

        Yields:
        - generated_text (str): the generated text, or next token, depending on the value of `incremental_generation`.
        """

        if top_k is not None and top_k > 0:
            raise ValueError(
                "top_k is currently not supported by our generation engine."
            )

        stop_token_ids += self.stop_tokens
        # stop_sequences = [self.stop_str] + stop_sequences

        # TODO (Moin): add support for the system prompt on chat models
        conv_config = ConvConfig(
            stop_tokens=stop_token_ids, add_bos=self.add_bos, stop_str=stop_sequences
        )
        chat_config = ChatConfig(
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            max_gen_len=max_new_tokens,
            mean_gen_len=max_new_tokens,
            conv_config=conv_config,
            conv_template=self.conv_template,
            num_shards=self.num_shards
        )
        self.cm.reset_chat(chat_config)

        generation_config = GenerationConfig(
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            max_gen_len=max_new_tokens,
        )
        self.cm._prefill(input=prompt, generation_config=generation_config)

        min_new_tokens = kwargs.pop("min_new_tokens", None)
        if min_new_tokens is not None and min_new_tokens > -1:
            raise ValueError(
                "min_new_tokens is currently not supported by MLC's engine."
            )

        if len(kwargs) > 0:
            raise ValueError(f"Unknown keyword arguments: {', '.join(kwargs.keys())}")

        generation_length = 0
        while True:
            if self.cm._stopped():
                break
            self.cm._decode(generation_config=generation_config)
            out = self.cm._get_message()
            # stops us from yielding half an emoji, which breaks
            out = out.replace("\N{Replacement Character}", "") 
            if len(out) == generation_length:
                # don't yield an empty string
                continue
            yield out[generation_length:]
            generation_length = len(out)
