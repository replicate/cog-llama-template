import shutil
import time
from typing import Optional
import zipfile

import torch
from cog import BasePredictor, ConcatenateIterator, Input, Path

from config import DEFAULT_MODEL_NAME, load_tokenizer, load_tensorizer, pull_gcp_file
from subclass import YieldingLlama
from peft import PeftModel
import os

PROMPT_TEMPLATE = '''<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{instruction} [/INST]'''

class Predictor(BasePredictor):
    def setup(self, weights: Optional[Path] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if weights is not None and weights.name == "weights":
            # bugfix
            weights = None
        
        weights = DEFAULT_MODEL_NAME if weights is None else str(weights)

        if '.zip' in weights:
            self.model = self.load_peft(weights)
        elif "tensors" in weights:
            self.model = load_tensorizer(weights, plaid_mode=True, cls=YieldingLlama)
        else:
            self.model = self.load_huggingface_model(weights=weights)

        self.tokenizer = load_tokenizer()

    def load_peft(self, weights):
        st = time.time()
        if 'tensors' in DEFAULT_MODEL_NAME:
            model = load_tensorizer(DEFAULT_MODEL_NAME, plaid_mode=False, cls=YieldingLlama)
        else:
            model = self.load_huggingface_model(DEFAULT_MODEL_NAME)
        if 'https' in weights: # weights are in the cloud
            local_weights = 'local_weights.zip'
            pull_gcp_file(weights, local_weights)
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
        model = YieldingLlama.from_pretrained(
            weights, cache_dir="pretrained_weights", torch_dtype=torch.float16
        )
        model.to(self.device)
        print(f"weights loaded in {time.time() - st}")
        return model

    def predict(
        self,
        prompt: str = Input(description=f"Prompt to send to Llama v2."),
        max_length: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens",
            ge=1,
            default=500,
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75 is a good starting value.",
            ge=0.01,
            le=5,
            default=0.5,
        ),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.01,
            le=1.0,
            default=1.0,
        ),
        repetition_penalty: float = Input(
            description="Penalty for repeated words in generated text; 1 is no penalty, values greater than 1 discourage repetition, less than 1 encourage it.",
            ge=0.01,
            le=5,
            default=1,
        ),
        debug: bool = Input(
            description="provide debugging output in logs", default=False
        ),
    ) -> ConcatenateIterator[str]:
        prompt = PROMPT_TEMPLATE.format(instruction=prompt.strip())
        input = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        with torch.inference_mode() and torch.autocast("cuda"):
            first_token_yielded = False
            prev_ids = []
            previous_token_id = None  # This stores the previous token id so we can look for `\nUser:`

            for output in self.model.generate(
                input_ids=input,
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            ):
                cur_id = output.item()

                # Break if previous token id was 13 (newline) and current id is 2659 (user)
                if previous_token_id == 13 and cur_id == 2659:
                    break

                previous_token_id = cur_id  # Store the current token id to check in the next iteration


                # in order to properly handle spaces, we need to do our own tokenizing. Fun!
                # we're building up a buffer of sub-word / punctuation tokens until we hit a space, and then yielding whole words + punctuation.
                cur_token = self.tokenizer.convert_ids_to_tokens(cur_id)

                # skip initial newline, which this almost always yields. hack - newline id = 13.
                if not first_token_yielded and not prev_ids and cur_id == 13:
                    continue

                # underscore means a space, means we yield previous tokens
                if cur_token.startswith("▁"):  # this is not a standard underscore.
                    # first token
                    if not prev_ids:
                        prev_ids = [cur_id]
                        continue

                    # there are tokens to yield
                    else:
                        token = self.tokenizer.decode(prev_ids)
                        prev_ids = [cur_id]

                        if not first_token_yielded:
                            # no leading space for first token
                            token = token.strip()
                            first_token_yielded = True
                            if not token:
                                # first token is empty space sometimes 
                                continue
                        yield token
                else:
                    prev_ids.append(cur_id)
                    continue

            # remove any special tokens such as </s>
            token = self.tokenizer.decode(prev_ids, skip_special_tokens=True).rstrip('\n')
            if not first_token_yielded:
                # no leading space for first token
                token = token.strip()
                first_token_yielded = True
            yield token 

        if debug:
            print(f"cur memory: {torch.cuda.memory_allocated()}")
            print(f"max allocated: {torch.cuda.max_memory_allocated()}")
            print(f"peak memory: {torch.cuda.max_memory_reserved()}")


class EightBitPredictor(Predictor):
    """subclass s.t. we can configure whether a model is loaded in 8bit mode from cog.yaml"""

    def setup(self, weights: Optional[Path] = None):
        if weights is not None and weights.name == "weights":
            # bugfix
            weights = None
        # TODO: fine-tuned 8bit weights.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YieldingLlama.from_pretrained(
            DEFAULT_MODEL_NAME, load_in_8bit=True, device_map="auto"
        )
        self.tokenizer = load_tokenizer()
