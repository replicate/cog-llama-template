# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from tqdm import tqdm
from itertools import chain
from torch.utils.data import Dataset

class Concatenator(object):
    def __init__(self, chunk_size=2096, wrap_samples=False):
        self.chunk_size=chunk_size
        self.residual = {"input_ids": [], "attention_mask": []}
        self.wrap_samples = False

    def _wrap_concat(self, batch):
        """
        When we pack samples into a single sequence, it's possible that the final
        sample's sequence will exceed `chunk_size`. In this case, the `_wrap_concat` 
        method will wrap the final sample around to the beginning of the next sequence.
        This breaks the sample into two parts and may introduce samples that violate prompt formats.
        However, it allows us to strictly enforce chunk size.  
        """
        concatenated_samples = {
            k: v + list(chain(*batch[k])) for k, v in self.residual.items()
        }
        
        total_length = len(concatenated_samples[list(concatenated_samples.keys())[0]])

        if total_length >= self.chunk_size:
            chunk_num = total_length // self.chunk_size
            result = {
                k: [
                    v[i : i + self.chunk_size]
                    for i in range(0, chunk_num * self.chunk_size, self.chunk_size)
                ]
                for k, v in concatenated_samples.items()
            }
            self.residual = {
                k: v[(chunk_num * self.chunk_size) :]
                for k, v in concatenated_samples.items()
            }
        else:
            result = concatenated_samples
            self.residual = {k: [] for k in concatenated_samples.keys()}

        result["labels"] = result["input_ids"].copy()

        return result
    
    def _concat(self, batch):
        """
        When we pack samples into a single sequence, it's possible that the final
        sample's sequence will exceed `chunk_size`. In this case, the `_concat` method
        will simply promote the final sample to the next sequence. This may introduce 
        sequences with variable lengths, e.g. some that are below `chunk_size`,
        but it allows us to pack sequences while strictly respecting formatting. 
        """
        
        # Initialize current sequences from residual or empty if none exists
        keys = batch.keys()
        current_sequences = {key: self.residual.get(key, []) for key in keys}

        # # We'll store packed sequences in results
        results = {key: [] for key in keys}

        # len_of_new_seq = len(batch[list(batch.keys())[0]])
        # len_of_current_seq = len(current_sequences[list(current_sequences.keys())[0]])

        num_samples = len(batch[next(iter(keys))])

        for idx in range(num_samples):
            # Check if adding next sample will exceed the chunk size for any key
            will_exceed = len(current_sequences[list(keys)[0]]) + len(batch[list(keys)[0]][idx]) > self.chunk_size

            if will_exceed:
                for key in keys:
                    results[key].append(current_sequences[key])
                    current_sequences[key] = []
                # After appending to results, extend current_sequences with the sample for all keys
                for key in keys:
                    current_sequences[key].extend(batch[key][idx])
            else:
                for key in keys:
                    current_sequences[key].extend(batch[key][idx])

        # Store unappended sequences as residual
        self.residual = current_sequences

        return results

        # if len_of_current_seq + len_of_new_seq < self.chunk_size:
        #     # Add new sequences to current sequences
        #     for key in keys:
        #         current_sequences[key] += batch[key]
        #         len_of_current_seq = len(current_sequences[list(current_sequences.keys())[0]])

        # elif len_of_current_seq + len_of_new_seq == self.chunk_size:
        #     for key in keys:
        #         current_sequences[key] += batch[key]
        #         len_of_current_seq = len(current_sequences[list(current_sequences.keys())[0]])
        #         return current_sequences
        
        # else:

        #     pass

  

        

    def __call__(self, batch):
        
        if self.wrap_samples:
            return self._wrap_concat(batch)
        else:
            return self._concat(batch)


class ConcatDataset(Dataset):
    def __init__(self, dataset, chunk_size=4096):
        self.dataset = dataset
        self.chunk_size = chunk_size
        
        self.samples = []
        
        buffer = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            }
        
        for sample in tqdm(self.dataset, desc="Preprocessing dataset"):
            buffer = {k: v + sample[k] for k,v in buffer.items()}
            
            while len(next(iter(buffer.values()))) > self.chunk_size:
                self.samples.append({k: v[:self.chunk_size] for k,v in buffer.items()})
                buffer = {k: v[self.chunk_size:] for k,v in buffer.items()}
                
    def __getitem__(self, idx):
        return self.samples[idx]
    
    def __len__(self):
        return len(self.samples)