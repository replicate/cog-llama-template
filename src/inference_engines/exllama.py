import io
import os
import sys
import glob

import torch
import time
import typing as tp

from src.config_utils import Weights

exllama_path = os.path.abspath("exllama")
sys.path.insert(0, exllama_path)

from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
from exllama.lora import ExLlamaLora
from exllama.tokenizer import ExLlamaTokenizer
from exllama.generator import ExLlamaGenerator

from src.inference_engines.engine import Engine
from ..utils import StreamingTextStopSequenceHandler

torch.cuda._lazy_init()
torch.set_printoptions(precision=10)


def next_logits(
    generator, input_ids, apply_lora=None, last_id_only=True, input_mask=None
):
    n_logits = generator.model.forward(
        input_ids, generator.cache, last_id_only, lora=apply_lora, input_mask=input_mask
    )
    return n_logits


def begin(generator):
    if generator.cache is None:
        generator.cache = ExLlamaCache(generator.model)
    else:
        generator.cache.current_seq_len = 0
    return generator


def timer(name, func):
    t = time.time()
    ret = func()
    t = time.time() - t
    print(f" ** Time, {name}: {t:.2f} seconds")
    return ret


class ExllamaEngine(Engine):
    def __init__(self, weights: Weights, fused_attn=True):
        model_directory = self.load_weights(weights)
        tokenizer_path = os.path.join(model_directory, "tokenizer.model")
        model_config_path = os.path.join(model_directory, "config.json")
        st_pattern = os.path.join(model_directory, "*.safetensors")
        model_path = glob.glob(st_pattern)[0]

        config = ExLlamaConfig(model_config_path)  # create config from config.json
        config.model_path = model_path  # supply path to model weights file

        # Override exllam's default settings to use full llama v2 context
        config.max_seq_len = 2 * 2048
        config.max_input_len = 2 * 2048
        config.max_attention_size = 2 * 2048**2
        config.fused_attn = fused_attn

        self.model = model = ExLlama(
            config
        )  # create ExLlama instance and load the weights
        tokenizer = ExLlamaTokenizer(
            tokenizer_path
        )  # create tokenizer from tokenizer model file

        cache = ExLlamaCache(model)  # create cache for inference
        generator = ExLlamaGenerator(model, tokenizer, cache)  # create generator

        # warmup kernels

        warmup_ids = torch.randint(0, 31999, (1, 50)).cuda()
        print("warming up exllama kernels...")
        for i in range(1, 3):
            print(f" -- Warmup pass {i}...")
            begin(generator)
            logits = timer("Warmup", lambda: next_logits(generator, warmup_ids, None))

        self.generator = begin(generator)

    def delete_lora(self):
        self.generator.lora = None
        return

    def is_lora_active(self) -> bool:
        return self.generator.lora is None

    def load_lora(self, data_ref: dict) -> ExLlamaLora:
        return ExLlamaLora(
            self.model,
            data_ref["adapter_config.json"],
            io.BytesIO(data_ref["adapter_model.bin"]),
        )

    def set_lora(self, lora: ExLlamaLora | None) -> None:
        self.generator.lora = lora

    def __call__(
        self,
        prompt: str,
        repetition_penalty: float = 1.15,
        repetition_penalty_sustain: int = 256,
        token_repetition_penalty_decay: float = 128,
        temperature: float = 0.95,
        top_p: float = 0.65,
        top_k: int = 20,
        max_new_tokens: int = 128,
        min_new_tokens: int = 0,
        beams: int = 1,
        beam_length: int = 1,
        stop_sequences: tp.List[str] = None,
    ):
        if top_k <= 0:
            top_k = 20
        generator = begin(self.generator)
        generator.settings.token_repetition_penalty_max = repetition_penalty
        generator.settings.token_repetition_penalty_sustain = repetition_penalty_sustain
        generator.settings.token_repetition_penalty_decay = (
            token_repetition_penalty_decay
        )
        generator.settings.temperature = temperature
        generator.settings.top_p = top_p
        generator.settings.top_k = top_k
        generator.settings.beams = beams
        generator.settings.beam_length = beam_length

        in_tokens = generator.tokenizer.encode(prompt)
        n_in_tokens = in_tokens.shape[-1]
        if n_in_tokens >= generator.model.config.max_input_len:
            raise ValueError(
                f"Your input is too long. Max input length is {generator.model.config.max_input_len} tokens, but you supplied {n_in_tokens} tokens."
            )

        max_new_tokens = min(
            max_new_tokens, generator.model.config.max_seq_len - n_in_tokens
        )

        num_res_tokens = in_tokens.shape[-1]  # Decode from here

        generator.gen_begin(in_tokens)
        generator.begin_beam_search()

        stop_sequence_handler = StreamingTextStopSequenceHandler(
            stop_sequences=stop_sequences,
            eos_token=generator.tokenizer.eos_token,
        )

        for i in range(max_new_tokens):
            if i < min_new_tokens:
                generator.disallow_tokens(
                    [
                        generator.tokenizer.newline_token_id,
                        generator.tokenizer.eos_token_id,
                    ]
                )
            else:
                generator.disallow_tokens(None)

            gen_token = generator.beam_search()
            if gen_token.item() == generator.tokenizer.eos_token_id:
                break

            if gen_token.item() == generator.tokenizer.eos_token_id:
                generator.replace_last_token(generator.tokenizer.newline_token_id)

            num_res_tokens += 1
            text = generator.tokenizer.decode(
                generator.sequence_actual[:, -num_res_tokens:][0]
            )
            new_text = text[len(prompt):]

            if len(new_text.replace("�", "")) == 0:
                # if we're getting �, then we're halfway through an emoji; ignore it til it's fully generated. 
                continue
            skip_space = prompt.endswith(("\n", "[/INST]")) and new_text.startswith(
                " "
            )  # Bit prettier console output
            prompt += new_text
            if skip_space:
                new_text = new_text[1:]

            yielded_text = None
            for yielded_text in stop_sequence_handler(new_text):
                if yielded_text == stop_sequence_handler.eos_token:
                    break
                yield yielded_text

            if yielded_text == stop_sequence_handler.eos_token:
                break

        for yielded_text in stop_sequence_handler.finalize():
            yield yielded_text
