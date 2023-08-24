import shutil
import time
from typing import Optional
import zipfile
import glob
import time 

import torch
from cog import BasePredictor, ConcatenateIterator, Input, Path

from config import (
    LOCAL_DEFAULT_INFERENCE_WEIGHTS_PATH, 
    REMOTE_DEFAULT_INFERENCE_WEIGHTS_PATH,
    REMOTE_TRAINING_FILES_TO_DOWNLOAD,
    USE_EXLLAMA_FOR_UNTRAINED_WEIGHTS,
    REMOTE_DEFAULT_INFERENCE_FILES_TO_DOWNLOAD, 
    LOCAL_TRAINING_WEIGHTS_PATH, 
    REMOTE_TRAINING_WEIGHTS_PATH,
    LOAD_IN_4BIT,
    load_tokenizer, 
    load_tensorizer, 
    download_file,
    USE_SYSTEM_PROMPT
)

from subclass import YieldingLlama
from src.utils import maybe_download_with_pget, StreamingTextStopSequenceHandler

from peft import PeftModel
import os

# This prompt formatting was copied from the original Llama v2 repo:
# https://github.com/facebookresearch/llama/blob/6c7fe276574e78057f917549435a2554000a876d/llama/generation.py#L44

# These are components of the prompt that should not be changed by the users
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
PROMPT_TEMPLATE = f"{B_INST} {B_SYS}{{system_prompt}}{E_SYS}{{instruction}} {E_INST}"

# Users may want to change the system prompt, but we use the recommended system prompt by default
DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant."""


class Predictor(BasePredictor):
    def setup(self, weights: Optional[Path] = None):
        print('starting setup')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        
        if weights is not None and weights.name == "weights":
            # bugfix
            weights = None
        # If weights aren't passed in, we'll use the default weights configuration
        if not weights:
            weights = LOCAL_DEFAULT_INFERENCE_WEIGHTS_PATH
            weights = maybe_download_with_pget(
                weights, REMOTE_DEFAULT_INFERENCE_WEIGHTS_PATH, REMOTE_DEFAULT_INFERENCE_FILES_TO_DOWNLOAD,
            )

            if USE_EXLLAMA_FOR_UNTRAINED_WEIGHTS:
                from src.exllama_predictor import ExllamaGenerator
                self.generator = ExllamaGenerator(weights)
                self.use_exllama = True
            
            else:
                if os.path.isdir(weights):
                    self.model = self.load_huggingface_model(weights, load_in_4bit=LOAD_IN_4BIT)
                    self.tokenizer = load_tokenizer()
                    self.use_exllama = False

        
        # If weights are passed in, they are LoRa weights
        # so we need to download the fp16 weights and load with peft
        elif '.zip' in str(weights):
            weights = str(weights)
            self.model = self.load_peft(weights)
            self.tokenizer = load_tokenizer()
            self.use_exllama = False
        else:
            raise Exception(f"Fine-tuned weights {weights} were improperly formatted.")


    def load_peft(self, weights):
        st = time.time()

        model_path = maybe_download_with_pget(
            LOCAL_TRAINING_WEIGHTS_PATH, 
            REMOTE_TRAINING_WEIGHTS_PATH, 
            REMOTE_TRAINING_FILES_TO_DOWNLOAD,
        )

        model = self.load_huggingface_model(model_path, load_in_4bit=LOAD_IN_4BIT)
        if 'http' in weights: # weights are in the cloud
            local_weights = 'local_weights.zip'
            if not os.path.exists(local_weights):
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

    def load_huggingface_model(self, weights=None, load_in_4bit=False):
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
                weights, cache_dir="pretrained_weights", torch_dtype=torch.float16, low_cpu_mem_usage=True
            ).to(self.device)

        print(f"weights loaded in {time.time() - st}")
        return model

    def predict(
        self,
        prompt: str = Input(description=f"Prompt to send to the model."),
        max_new_tokens: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens",
            ge=1,
            default=128,
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
            default=0.75,
        ),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.0,
            le=1.0,
            default=0.9,
        ),
        top_k: int = Input(
            description="When decoding text, samples from the top k most likely tokens; lower to ignore less likely tokens",
            ge=0,
            default=50,
        ),
        stop_sequences: str = Input(
            description="A comma-separated list of sequences to stop generation at. For example, '<end>,<stop>' will stop generation at the first instance of 'end' or '<stop>'.",
            default=None,
        ),
        debug: bool = Input(
            description="provide debugging output in logs", default=False
        ),
    ) -> ConcatenateIterator: 
        
        if stop_sequences:
            stop_sequences = stop_sequences.split(",")
        
        if USE_SYSTEM_PROMPT:
            prompt = prompt.strip('\n').lstrip(B_INST).rstrip(E_INST).strip()
            prompt = PROMPT_TEMPLATE.format(system_prompt=system_prompt.strip(), instruction=prompt.strip())

        print(f"Your formatted prompt is: \n{prompt}")
        if self.use_exllama:
            n_tokens = 0
            st = time.time()

            for decoded_token in self.generator(
                prompt,
                repetition_penalty=1.15,
                repetition_penalty_sustain=256,
                token_repetition_penalty_decay=128,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                stop_sequences=stop_sequences,
            ):
                n_tokens += 1
                yield decoded_token
            t = time.time() - st
        
        else:
            
            if stop_sequences:
                stop_sequences_token_ids = [self.tokenizer.encode(seq, add_special_tokens=False) for seq in stop_sequences]
            else:
                stop_sequences_token_ids = []

            stop_sequence_handler = StreamingTextStopSequenceHandler(
                stop_sequences=stop_sequences,
                eos_token=self.tokenizer.eos_token,
            )

            prompt_tokens = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            n_in_tokens = prompt_tokens.shape[-1]
            if n_in_tokens >= self.model.config.max_position_embeddings:
                raise ValueError(f"Your input is too long. Max input length is {self.model.config.max_position_embeddings} tokens, but you supplied {n_in_tokens} tokens.")

            max_new_tokens = min(max_new_tokens, self.model.config.max_position_embeddings - n_in_tokens)
            old_tokens = prompt_tokens.tolist()[0]
            old_text = self.tokenizer.bos_token + prompt
            with torch.inference_mode() and torch.autocast("cuda"):

                for token in self.model.generate(
                    input_ids=prompt_tokens,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=1,
                ):
                    
                    old_tokens.append(token.item())
                    text = self.tokenizer.decode(old_tokens)
                    new_text = text[len(old_text):]
                    old_text = text

                    for yielded_text in stop_sequence_handler(new_text):
                        if yielded_text == stop_sequence_handler.eos_token:
                            break
                        yield yielded_text
                    
                    if yielded_text == stop_sequence_handler.eos_token:
                        break

                for yielded_text in stop_sequence_handler.finalize():
                    yield yielded_text    


        if debug:
            print(f"cur memory: {torch.cuda.memory_allocated()}")
            print(f"max allocated: {torch.cuda.max_memory_allocated()}")
            print(f"peak memory: {torch.cuda.max_memory_reserved()}")



