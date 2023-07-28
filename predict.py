import shutil
import time
from typing import Optional
import zipfile
import glob
import time 

import torch
from cog import BasePredictor, ConcatenateIterator, Input, Path

from config import (
    LOCAL_GPTQ_WEIGHTS_PATH, 
    REMOTE_GPTQ_WEIGHTS_PATH,
    REMOTE_FILES_TO_DOWNLOAD, 
    BASE_WEIGHTS_PATH, 
    LOAD_IN_4BIT,
    load_tokenizer, 
    load_tensorizer, 
    download_file,
    USE_SYSTEM_PROMPT
)

from subclass import YieldingLlama
from scripts.utils import maybe_download_with_pget

from peft import PeftModel
import os

# This prompt formatting was copied from the original Llama v2 repo:
# https://github.com/facebookresearch/llama/blob/6c7fe276574e78057f917549435a2554000a876d/llama/generation.py#L44

# These are components of the prompt that should not be changed by the users
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
PROMPT_TEMPLATE = f"{B_INST} {B_SYS}{{system_prompt}}{E_SYS}{{instruction}} {E_INST}"

# Users may want to change the system prompt, but we use the recommended system prompt by default
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


class Predictor(BasePredictor):
    def setup(self, weights: Optional[Path] = None):
        print('starting setup')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if weights is not None and weights.name == "weights":
            # bugfix
            weights = None
        
        # If weights aren't passed in, pull GPTQ weights from remote
        if not weights:
            weights = LOCAL_GPTQ_WEIGHTS_PATH
            from src.exllama_predictor import ExllamaGenerator
            weights = maybe_download_with_pget(
                weights, REMOTE_GPTQ_WEIGHTS_PATH, REMOTE_FILES_TO_DOWNLOAD,
            )
            self.generator = ExllamaGenerator(weights)
            self.use_exllama = True
        
        # If weights are passed in, they are LoRa weights
        # so we need to download the fp16 weights and load with peft
        elif '.zip' in weights:
            self.model = self.load_peft(weights)
            self.tokenizer = load_tokenizer()
            self.use_exllama = False


    def load_peft(self, weights):
        st = time.time()
        if 'tensors' in BASE_WEIGHTS_PATH:
            model = load_tensorizer(BASE_WEIGHTS_PATH, plaid_mode=False, cls=YieldingLlama)
        else:
            model = self.load_huggingface_model(BASE_WEIGHTS_PATH, load_in_4bit=LOAD_IN_4BIT)
        if 'https' in weights: # weights are in the cloud
            local_weights = 'local_weights.zip'
            download_file(weights, local_weights)
            weights = local_weights
        out = '/src/peft_dir'
        if os.path.exists(out):
            shutil.rmtree(out)
        with zipfile.ZipFile(weights, 'r') as zip_ref:
            zip_ref.extractall(out)
        model = PeftModel.from_pretrained(model, out)
        print(f"peft model loaded in {time.time() - st}")
        return model.to('cuda')

    def load_huggingface_model(self, weights=None):
        st = time.time()
        print(f"loading weights from {weights} w/o tensorizer")
        if LOAD_IN_4BIT:
            model = YieldingLlama.from_pretrained(
                weights, 
                cache_dir="pretrained_weights", 
                device_map='auto',
                load_in_4bit=LOAD_IN_4BIT,
            )
        else:
            model = YieldingLlama.from_pretrained(
                weights, cache_dir="pretrained_weights", torch_dtype=torch.float16
            )
            model.to(self.device)

        print(f"weights loaded in {time.time() - st}")
        return model

    def predict(
        self,
        prompt: str = Input(description=f"Prompt to send to Llama v2."),
        # system_prompt: str = Input(
        #     description="System prompt to send to Llama v2. This is prepended to the prompt and helps guide system behavior.", 
        #     default=DEFAULT_SYSTEM_PROMPT,
        # ),
        max_new_tokens: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens",
            ge=1,
            default=500,
        ),
        min_new_tokens: int = Input(
            description="Minimum number of tokens to generate. To disable, set to -1. A word is generally 2-3 tokens.",
            ge=-1,
            default=-1,
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75 is a good starting value.",
            ge=0.01,
            le=5,
            default=0.95,
        ),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.0,
            le=1.0,
            default=0.95,
        ),
        top_k: int = Input(
            description="When decoding text, samples from the top k most likely tokens; lower to ignore less likely tokens",
            ge=0,
            default=250,
        ),
        repetition_penalty: float = Input(
            description="Penalty for repeated words in generated text; 1 is no penalty, values greater than 1 discourage repetition, less than 1 encourage it.",
            ge=0.01,
            le=5,
            default=1.15,
        ),
        repetition_penalty_sustain: int = Input(
            description="Number of most recent tokens to apply repetition penalty to, -1 to apply to whole context",
            ge=-1,
            default=256,
        ),
        token_repetition_penalty_decay: int = Input(
            description="Gradually decrease penalty over this many tokens",
            ge=1,
            default=128,
        ),
        debug: bool = Input(
            description="provide debugging output in logs", default=False
        ),
    ) -> ConcatenateIterator: 
        
        if USE_SYSTEM_PROMPT:
            prompt = prompt.strip('\n').lstrip(B_INST).rstrip(E_INST).strip()
            prompt_templated = PROMPT_TEMPLATE.format(system_prompt=system_prompt.strip(), instruction=prompt.strip())
        else:
            prompt_templated = prompt
        
        if self.use_exllama:
            n_tokens = 0
            st = time.time()
            for decoded_token in self.generator(
                prompt_templated,
                repetition_penalty=repetition_penalty,
                repetition_penalty_sustain=repetition_penalty_sustain,
                token_repetition_penalty_decay=token_repetition_penalty_decay,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
            ):
                n_tokens += 1
                yield decoded_token
            t = time.time() - st
            print(f" ** Speed: {n_tokens / t:.2f} tokens/second")
        
        # This is our original generation code
        else:

            input = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

            with torch.inference_mode() and torch.autocast("cuda"):
                first_token_yielded = False
                prev_ids = []

                for output in self.model.generate(
                    input_ids=input,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                ):
                    cur_id = output.item()


                    # in order to properly handle spaces, we need to do our own tokenizing. Fun!
                    # we're building up a buffer of sub-word / punctuation tokens until we hit a space, and then yielding whole words + punctuation.
                    cur_token = self.tokenizer.convert_ids_to_tokens(cur_id)

                    # skip initial newline, which this almost always yields. hack - newline id = 13.
                    if not first_token_yielded and not prev_ids and cur_id in [13, 259]:
                        continue

                    # underscore means a space, means we yield previous tokens
                    if cur_token.startswith("▁"):  # this is not a standard underscore.
                        # first token
                        if not prev_ids:
                            prev_ids = [cur_id]
                            continue

                        # there are tokens to yield
                        else:
                            token = ' ' + self.tokenizer.decode(prev_ids)
                            prev_ids = [cur_id]

                            if not first_token_yielded:
                                # no leading space for first token
                                token = token.strip()
                                first_token_yielded = True
                            yield token
                    else:
                        prev_ids.append(cur_id)
                        continue

                # remove any special tokens such as </s>
                token = ' ' + self.tokenizer.decode(prev_ids, skip_special_tokens=True).rstrip('\n')
                if not first_token_yielded:
                    # no leading space for first token
                    token = token.strip()
                    first_token_yielded = True
                yield token 

        if debug:
            print(f"cur memory: {torch.cuda.memory_allocated()}")
            print(f"max allocated: {torch.cuda.max_memory_allocated()}")
            print(f"peak memory: {torch.cuda.max_memory_reserved()}")

