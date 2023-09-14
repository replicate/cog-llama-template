from transformers import AutoModelForCausalLM, TextIteratorStreamer, StoppingCriteria
from typing import Optional, List
from threading import Thread

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

    def load_lora(sel, lora_weights):
        # todo - this is probably where the contract for loading needs to change a bit. 
        pass

    def set_lora(self, lora):
        # todo - this is probably more straightforward
        pass

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
