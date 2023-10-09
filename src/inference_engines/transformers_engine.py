import os
import shutil
from transformers import AutoModelForCausalLM, TextIteratorStreamer, StoppingCriteria
from typing import Optional, List
from threading import Thread
from peft import PeftModel

import torch

from .engine import Engine

class ExtraStopSequence(StoppingCriteria):
    """
    Adds in an extra stop sequence. Assuming 1-D generation, not batch. 
    """
    # TODO: there's something silly to debug here. 
    def __init__(self, stop_sequence: torch.Tensor, device: str):
        self.stop_sequence = stop_sequence.to(device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        return torch.equal(self.stop_sequence, input_ids[:, self.stop_sequence.shape[-1]])



class TransformersEngine(Engine):
    """
    An inference engine that runs in vanilla transformers. 
    Vanilla is, at times, fantastic.
    """
    def __init__(self, weights, tokenizer_func=None, device="cuda"):
        self.model = AutoModelForCausalLM.from_pretrained(weights).to(device)
        self.tokenizer = tokenizer_func()
        self.device = device 
        print("Transformers engine initialized.")


    def load_lora(self, lora_weights: dict):
        # reset to non-lora model if previous lora exists, can't just swap loras out w/transformers
        if hasattr(self.model, 'unload') and callable(self.model.unload):
            self.model = self.model.unload()

        # serializing the dictionary of files and such - hf doesn't have quick and easy ways to load loras from file references, 
        # and this implementation isn't built for speed anyway
        model_dir = 'tmp/model'
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.makedirs(model_dir)
        for handle in lora_weights:
            fpath = os.path.join(model_dir, handle)
            with open(fpath, 'wb') as f:
                f.write(lora_weights[handle])

        model = PeftModel.from_pretrained(self.model, model_dir)
        shutil.rmtree(model_dir)
        return model
    
    def is_lora_active(self) -> bool:
        return hasattr(self.model, 'unload')
    
    def delete_lora(self):
        if hasattr(self.model, 'unload') and callable(self.model.unload):
            self.model = self.model.unload()

    def set_lora(self, lora):
        self.model = lora

    def __call__(self, 
                 prompt,
                 max_new_tokens:int =128,
                 min_new_tokens:int =-1,
                 temperature:float =0.75,
                 top_p:float =0.9,
                 top_k:int =50,
                 stop_sequences: Optional[List[str]] = None,
                 **kwargs):
        tokens_in = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)

        stopping_criteria_list = None
        if stop_sequences is not None:
            # stop sequences!
            stopping_criteria_list = []
            for seq in stop_sequences:
                stop_ids = self.tokenizer(seq, return_tensors='pt', add_special_tokens=False).input_ids[0]
                stopping_criteria_list.append(ExtraStopSequence(stop_ids, self.device))
   

        generate_kwargs = dict(
            input_ids=tokens_in,
            streamer=streamer,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stopping_criteria=stopping_criteria_list
        )

        t = Thread(target=self.model.generate, kwargs=generate_kwargs)
        t.start()

        for out in streamer:
            yield out
