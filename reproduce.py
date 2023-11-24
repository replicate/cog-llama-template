import os
from mlc_chat import ChatConfig, ChatModule, ConvConfig, GenerationConfig
from transformers import AutoTokenizer
from repro_utils import Weights, get_mlc_file_list, maybe_download_with_pget, mlc_kwargs

if os.getenv("7B"):
    MODEL_NAME = "llama-2-7b-mlc"
    mlc_file_list = get_mlc_file_list(model_name="llama-2-7b-hf-q4f16_1", n_shards=115)
else:
    MODEL_NAME = "llama-2-70b-chat-hf-mlc-sm90"
    mlc_file_list = get_mlc_file_list(
        model_name="Llama-2-70b-chat-hf-q4f16_1", n_shards=483
    )

mlc_weights = Weights(
    local_path=f"models/{MODEL_NAME}/model_artifacts/default_inference_weights",
    remote_path=f"https://weights.replicate.delivery/default/{MODEL_NAME}",
    remote_files=mlc_file_list,
)

ENGINE_KWARGS = mlc_kwargs(mlc_weights, is_chat=False)


class MLCEngine:
    def __init__(
        self,
        weights: Weights,
        num_shards: int = 1,
        is_chat: bool = False,
        tokenizer_path: os.PathLike = None,
    ) -> None:
        weights_path = mlc_weights.local_path
        self.num_shards = num_shards

        self.conv_template = "LM"
        self.stop_str = "[INST]"
        self.stop_tokens = [
            2,
        ]
        self.add_bos = True
        maybe_download_with_pget(
            weights.local_path, weights.remote_path, weights.remote_files
        )

        conv_config = ConvConfig(
            stop_tokens=self.stop_tokens, add_bos=self.add_bos, stop_str=self.stop_str
        )
        chat_config = ChatConfig(
            conv_config=conv_config,
            conv_template=self.conv_template,
            num_shards=self.num_shards,
        )

        model_path = os.path.join(weights_path, "params")
        self.cm = ChatModule(model=model_path, chat_config=chat_config, device="cuda")

        tokenizer_path = os.path.join(weights_path, "params")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def __call__(
        self,
        prompt: str,
        *args,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = -1,
        stop_sequences: str | list[str] = None,
        stop_token_ids: list[int] = [],
        repetition_penalty: float = 1.0,
        incremental_generation: bool = True,
        **kwargs,
    ):
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
            num_shards=self.num_shards,
        )
        self.cm.reset_chat(chat_config)

        generation_config = GenerationConfig(
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            max_gen_len=max_new_tokens,
        )
        self.cm._prefill(input=prompt, generation_config=generation_config)

        generation_length = 0
        while True:
            if self.cm._stopped():
                break
            self.cm._decode(generation_config=generation_config)
            out = self.cm._get_message()
            yield out[generation_length:]
            generation_length = len(out)


if __name__ == "__main__":
    engine = MLCEngine(**ENGINE_KWARGS)
    print(next(engine("hi")))
