import pytest

import sys
sys.path.append('.')

from src.src.utils import StreamingTokenStopSequenceHandler, StreamingTextStopSequenceHandler

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

def get_decoded_prompt_tokens(tokenizer, prompt):
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    tokens = [tokenizer.decode(token_id) for token_id in token_ids]
    return tokens


def test_no_stop_sequences(tokenizer):
    stop_sequences = None
    stop_sequence_handler = StreamingTextStopSequenceHandler(
        stop_sequences, 
        eos_token=tokenizer.eos_token
    )
    
    prompt = "Hello world"
    prompt_tokens = tokenizer.encode(prompt)

    response = "how are <end> you?"
    response_tokens = tokenizer.encode(response, add_special_tokens=False)

    old_text = tokenizer.decode(prompt_tokens)
    output = []
    for token in response_tokens:
        prompt_tokens.append(token)
        text = tokenizer.decode(prompt_tokens)
        new_text = text[len(old_text):]
        old_text = text

        for yielded_text in stop_sequence_handler(new_text):
            if yielded_text == stop_sequence_handler.eos_token:
                break
    
            output.append(yielded_text)

        if yielded_text == stop_sequence_handler.eos_token:
            break
    
    for yielded_text in stop_sequence_handler.finalize():
        output.append(yielded_text) 
    
    assert ''.join(output) == " how are <end> you?" # All tokens are yielded since no stop sequence was provided


def test_single_stop_sequence_1(tokenizer):
    stop_sequences = ["<end>"]
    stop_sequence_handler = StreamingTextStopSequenceHandler(
        stop_sequences, 
        eos_token=tokenizer.eos_token
    )
    
    prompt = "Hello world"
    prompt_tokens = tokenizer.encode(prompt)

    response = "how are <end> you?"
    response_tokens = tokenizer.encode(response, add_special_tokens=False)

    old_text = tokenizer.decode(prompt_tokens)
    output = []
    for token in response_tokens:
        prompt_tokens.append(token)
        text = tokenizer.decode(prompt_tokens)
        new_text = text[len(old_text):]
        old_text = text

        for yielded_text in stop_sequence_handler(new_text):
            if yielded_text == stop_sequence_handler.eos_token:
                break
    
            output.append(yielded_text)

        if yielded_text == stop_sequence_handler.eos_token:
            break
    
    for yielded_text in stop_sequence_handler.finalize():
        output.append(yielded_text) 
    
    assert ''.join(output) == " how are" # All tokens are yielded since no stop sequence was provided


def test_single_stop_sequence_2(tokenizer):
    stop_sequences = ["###"]
    stop_sequence_handler = StreamingTextStopSequenceHandler(
        stop_sequences, 
        eos_token=tokenizer.eos_token
    )
    
    prompt = "Hello world"
    prompt_tokens = tokenizer.encode(prompt)

    response = "how are ### you?"
    response_tokens = tokenizer.encode(response, add_special_tokens=False)

    old_text = tokenizer.decode(prompt_tokens)
    output = []
    for token in response_tokens:
        prompt_tokens.append(token)
        text = tokenizer.decode(prompt_tokens)
        new_text = text[len(old_text):]
        old_text = text

        for yielded_text in stop_sequence_handler(new_text):
            if yielded_text == stop_sequence_handler.eos_token:
                break
    
            output.append(yielded_text)

        if yielded_text == stop_sequence_handler.eos_token:
            break
    
    for yielded_text in stop_sequence_handler.finalize():
        output.append(yielded_text) 
    
    assert ''.join(output) == " how are" # All tokens are yielded since no stop sequence was provided


def test_multiple_stop_sequence(tokenizer):
    stop_sequences = ["<end>", "|STOP|"]
    stop_sequence_handler = StreamingTextStopSequenceHandler(
        stop_sequences, 
        eos_token=tokenizer.eos_token
    )
    
    prompt = "Hello world"
    prompt_tokens = tokenizer.encode(prompt)

    response = "how are <end you |STOP| today?"
    response_tokens = tokenizer.encode(response, add_special_tokens=False)

    old_text = tokenizer.decode(prompt_tokens)
    output = []
    for token in response_tokens:
        prompt_tokens.append(token)
        text = tokenizer.decode(prompt_tokens)
        new_text = text[len(old_text):]
        old_text = text

        for yielded_text in stop_sequence_handler(new_text):
            if yielded_text == stop_sequence_handler.eos_token:
                break
    
            output.append(yielded_text)

        if yielded_text == stop_sequence_handler.eos_token:
            break
    
    for yielded_text in stop_sequence_handler.finalize():
        output.append(yielded_text) 
    
    assert ''.join(output) == " how are <end you" # All tokens are yielded since no stop sequence was provided

def test_adjacent_stop_sequences(tokenizer):
    stop_sequences = ["<end>", "|STOP|"]
    stop_sequence_handler = StreamingTextStopSequenceHandler(
        stop_sequences, 
        eos_token=tokenizer.eos_token
    )
    
    prompt = "Hello world"
    prompt_tokens = tokenizer.encode(prompt)

    response = "how are <end |STOP| today?"
    response_tokens = tokenizer.encode(response, add_special_tokens=False)

    old_text = tokenizer.decode(prompt_tokens)
    output = []
    for token in response_tokens:
        prompt_tokens.append(token)
        text = tokenizer.decode(prompt_tokens)
        new_text = text[len(old_text):]
        old_text = text

        for yielded_text in stop_sequence_handler(new_text):
            if yielded_text == stop_sequence_handler.eos_token:
                break
    
            output.append(yielded_text)

        if yielded_text == stop_sequence_handler.eos_token:
            break
    
    for yielded_text in stop_sequence_handler.finalize():
        output.append(yielded_text) 
    
    assert ''.join(output) == " how are <end" # All tokens are yielded since no stop sequence was provided


def test_wtf(tokenizer):
    stop_sequences = ["John"]
    stop_sequence_handler = StreamingTextStopSequenceHandler(
        stop_sequences, 
        eos_token=tokenizer.eos_token
    )
    
    prompt = "My name is John Michael. My name is John Michael. My"
    prompt_tokens = tokenizer(prompt, return_tensors="pt").input_ids

    response = "name is John Michael."
    response_tokens = tokenizer.encode(response, add_special_tokens=False)

    old_tokens = prompt_tokens.tolist()[0]
    old_text = tokenizer.bos_token + prompt
    print('old text', old_text)
    output = []
    for token in response_tokens:
        old_tokens.append(token)
        text = tokenizer.decode(old_tokens)
        new_text = text[len(old_text):]
        old_text = text

        for yielded_text in stop_sequence_handler(new_text):
            if yielded_text == stop_sequence_handler.eos_token:
                break
    
            output.append(yielded_text)

        if yielded_text == stop_sequence_handler.eos_token:
            break
    
    for yielded_text in stop_sequence_handler.finalize():
        output.append(yielded_text) 
    
    assert ''.join(output) == " name is" # All tokens are yielded since no stop sequence was provided

