import pytest

import sys
sys.path.append('.')

from src.src.utils import StreamingStopSequenceHandler

@pytest.fixture(scope="session")
def tokenizer():
    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained("tests/assets/llama_tokenizer", legacy=False)
    tokenizer.add_special_tokens(
            {
                "pad_token": "<PAD>",
                "eos_token": "</s>",
                "bos_token": "<s>",
            }
        )
    return tokenizer


def test_no_stop_sequences(tokenizer):
    stop_sequence_handler = StreamingStopSequenceHandler(stop_sequences_token_ids=None)
    prompt = "Hello world <end>"
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    output = []
    for token_id in token_ids:
        for yielded_token_id in stop_sequence_handler(token_id):
            output.append(yielded_token_id)

    assert output == token_ids # All tokens are yielded since no stop sequence was provided



def test_single_stop_sequence(tokenizer):
    stop_sequences = ["<end>"]
    stop_sequences_token_ids = [tokenizer.encode(seq, add_special_tokens=False) for seq in stop_sequences]
    stop_sequence_handler = StreamingStopSequenceHandler(stop_sequences_token_ids, eos_token_id=tokenizer.eos_token_id)
    
    prompt = "Hello world <end> how are you?"
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    expected_output = tokenizer.encode("Hello world", add_special_tokens=False)
    output = []
    for token_id in token_ids:
        for yielded_token_id in stop_sequence_handler(token_id):
            if yielded_token_id == stop_sequence_handler.eos_token_id:
                break
            
            output.append(yielded_token_id)

        if yielded_token_id == stop_sequence_handler.eos_token_id:
            break
    
    for yielded_token_id in stop_sequence_handler.finalize():
        output.append(yielded_token_id) 
    
    

    assert output == expected_output # All tokens are yielded since no stop sequence was provided


def test_multiple_stop_sequence(tokenizer):
    stop_sequences = ["<end>", "how are"]
    stop_sequences_token_ids = [tokenizer.encode(seq, add_special_tokens=False) for seq in stop_sequences]
    stop_sequence_handler = StreamingStopSequenceHandler(stop_sequences_token_ids, eos_token_id=tokenizer.eos_token_id)
    
    prompt = "Hello world <end! how are you?"
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    expected_output = tokenizer.encode("Hello world <end!", add_special_tokens=False)
    output = []
    import time
    st = time.time()
    for token_id in token_ids:
        for yielded_token_id in stop_sequence_handler(token_id):
            if yielded_token_id == stop_sequence_handler.eos_token_id:
                break
            
            output.append(yielded_token_id)

        if yielded_token_id == stop_sequence_handler.eos_token_id:
            break
    
    for yielded_token_id in stop_sequence_handler.finalize():
        output.append(yielded_token_id) 
    
    
    assert output == expected_output # All tokens are yielded since no stop sequence was provided

def test_multiple_overlapping_stop_sequence_1(tokenizer):
    stop_sequences = ["<end>", "how are"]
    stop_sequences_token_ids = [tokenizer.encode(seq, add_special_tokens=False) for seq in stop_sequences]
    stop_sequence_handler = StreamingStopSequenceHandler(stop_sequences_token_ids, eos_token_id=tokenizer.eos_token_id)
    
    prompt = "Hello world <end how are you?"
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    expected_output = tokenizer.encode("Hello world <end", add_special_tokens=False)
    output = []
    for token_id in token_ids:
        for yielded_token_id in stop_sequence_handler(token_id):
            if yielded_token_id == stop_sequence_handler.eos_token_id:
                break
            
            output.append(yielded_token_id)

        if yielded_token_id == stop_sequence_handler.eos_token_id:
            break
    
    for yielded_token_id in stop_sequence_handler.finalize():
        output.append(yielded_token_id) 
    
    

    assert output == expected_output 


def test_multiple_overlapping_stop_sequence_2(tokenizer):
    stop_sequences = ["<end>", "<end how"]
    stop_sequences_token_ids = [tokenizer.encode(seq, add_special_tokens=False) for seq in stop_sequences]
    stop_sequence_handler = StreamingStopSequenceHandler(stop_sequences_token_ids, eos_token_id=tokenizer.eos_token_id)
    
    prompt = "Hello world <end how are you?"
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    expected_output = tokenizer.encode("Hello world", add_special_tokens=False)
    output = []
    for token_id in token_ids:
        for yielded_token_id in stop_sequence_handler(token_id):
            if yielded_token_id == stop_sequence_handler.eos_token_id:
                break
            
            output.append(yielded_token_id)

        if yielded_token_id == stop_sequence_handler.eos_token_id:
            break
    
    for yielded_token_id in stop_sequence_handler.finalize():
        output.append(yielded_token_id) 
    
    

    assert output == expected_output 


def test_uncompleted_stop_tokens(tokenizer):
    stop_sequences = ["<end>"]
    stop_sequences_token_ids = [tokenizer.encode(seq, add_special_tokens=False) for seq in stop_sequences]
    stop_sequence_handler = StreamingStopSequenceHandler(stop_sequences_token_ids, eos_token_id=tokenizer.eos_token_id)
    
    prompt = "Hello world how are you <end"
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    expected_output = tokenizer.encode("Hello world how are you <end", add_special_tokens=False)
    output = []
    for token_id in token_ids:
        for yielded_token_id in stop_sequence_handler(token_id):
            if yielded_token_id == stop_sequence_handler.eos_token_id:
                break
            test_uncompleted_stop_tokens
            output.append(yielded_token_id)

        if yielded_token_id == stop_sequence_handler.eos_token_id:
            break
    
    for yielded_token_id in stop_sequence_handler.finalize():
        output.append(yielded_token_id) 
    

    assert output == expected_output 