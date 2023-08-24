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



def test_overlapping_stop_sequence(tokenizer):
    stop_sequences = ["<end>", "<end |STOP|"]
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
    
    assert ''.join(output) == " how are" # All tokens are yielded since no stop sequence was provided
   
    

#     assert output == expected_output 


# def test_uncompleted_stop_tokens(tokenizer):
#     stop_sequences = ["<end>"]
#     stop_sequences_token_ids = [tokenizer.encode(seq, add_special_tokens=False) for seq in stop_sequences]
#     stop_sequence_handler = StreamingStopSequenceHandler(stop_sequences_token_ids, eos_token_id=tokenizer.eos_token_id)
    
#     prompt = "Hello world how are you <end"
#     token_ids = tokenizer.encode(prompt, add_special_tokens=False)
#     expected_output = tokenizer.encode("Hello world how are you <end", add_special_tokens=False)
#     output = []
#     for token_id in token_ids:
#         for yielded_token_id in stop_sequence_handler(token_id):
#             if yielded_token_id == stop_sequence_handler.eos_token_id:
#                 break
#             test_uncompleted_stop_tokens
#             output.append(yielded_token_id)

#         if yielded_token_id == stop_sequence_handler.eos_token_id:
#             break
    
#     for yielded_token_id in stop_sequence_handler.finalize():
#         output.append(yielded_token_id) 
    

#     assert output == expected_output 

##############

# def test_no_stop_sequences(tokenizer):
#     stop_sequence_handler = StreamingStopSequenceHandler(stop_sequences_token_ids=None)
#     prompt = "Hello world <end>"
#     token_ids = tokenizer.encode(prompt, add_special_tokens=False)
#     output = []
#     for token_id in token_ids:
#         for yielded_token_id in stop_sequence_handler(token_id):
#             output.append(yielded_token_id)

#     assert output == token_ids # All tokens are yielded since no stop sequence was provided



# def test_single_stop_sequence(tokenizer):
#     stop_sequences = ["<end>"]
#     stop_sequences_token_ids = [tokenizer.encode(seq, add_special_tokens=False) for seq in stop_sequences]
#     stop_sequence_handler = StreamingStopSequenceHandler(stop_sequences_token_ids, eos_token_id=tokenizer.eos_token_id)
    
#     prompt = "Hello world <end> how are you?"
#     token_ids = tokenizer.encode(prompt, add_special_tokens=False)
#     expected_output = tokenizer.encode("Hello world", add_special_tokens=False)
#     output = []
#     for token_id in token_ids:
#         for yielded_token_id in stop_sequence_handler(token_id):
#             if yielded_token_id == stop_sequence_handler.eos_token_id:
#                 break
            
#             output.append(yielded_token_id)

#         if yielded_token_id == stop_sequence_handler.eos_token_id:
#             break
    
#     for yielded_token_id in stop_sequence_handler.finalize():
#         output.append(yielded_token_id) 
    
    

#     assert output == expected_output # All tokens are yielded since no stop sequence was provided


# def test_multiple_stop_sequence(tokenizer):
#     stop_sequences = ["<end>", "how are"]
#     stop_sequences_token_ids = [tokenizer.encode(seq, add_special_tokens=False) for seq in stop_sequences]
#     stop_sequence_handler = StreamingStopSequenceHandler(stop_sequences_token_ids, eos_token_id=tokenizer.eos_token_id)
    
#     prompt = "Hello world <end! how are you?"
#     token_ids = tokenizer.encode(prompt, add_special_tokens=False)
#     expected_output = tokenizer.encode("Hello world <end!", add_special_tokens=False)
#     output = []
#     import time
#     st = time.time()
#     for token_id in token_ids:
#         for yielded_token_id in stop_sequence_handler(token_id):
#             if yielded_token_id == stop_sequence_handler.eos_token_id:
#                 break
            
#             output.append(yielded_token_id)

#         if yielded_token_id == stop_sequence_handler.eos_token_id:
#             break
    
#     for yielded_token_id in stop_sequence_handler.finalize():
#         output.append(yielded_token_id) 
    
    
#     assert output == expected_output # All tokens are yielded since no stop sequence was provided

# def test_multiple_overlapping_stop_sequence_1(tokenizer):
#     stop_sequences = ["<end>", "how are"]
#     stop_sequences_token_ids = [tokenizer.encode(seq, add_special_tokens=False) for seq in stop_sequences]
#     stop_sequence_handler = StreamingStopSequenceHandler(stop_sequences_token_ids, eos_token_id=tokenizer.eos_token_id)
    
#     prompt = "Hello world <end how are you?"
#     token_ids = tokenizer.encode(prompt, add_special_tokens=False)
#     expected_output = tokenizer.encode("Hello world <end", add_special_tokens=False)
#     output = []
#     for token_id in token_ids:
#         for yielded_token_id in stop_sequence_handler(token_id):
#             if yielded_token_id == stop_sequence_handler.eos_token_id:
#                 break
            
#             output.append(yielded_token_id)

#         if yielded_token_id == stop_sequence_handler.eos_token_id:
#             break
    
#     for yielded_token_id in stop_sequence_handler.finalize():
#         output.append(yielded_token_id) 
    
    

#     assert output == expected_output 


# def test_multiple_overlapping_stop_sequence_2(tokenizer):
#     stop_sequences = ["<end>", "<end how"]
#     stop_sequences_token_ids = [tokenizer.encode(seq, add_special_tokens=False) for seq in stop_sequences]
#     stop_sequence_handler = StreamingStopSequenceHandler(stop_sequences_token_ids, eos_token_id=tokenizer.eos_token_id)
    
#     prompt = "Hello world <end how are you?"
#     token_ids = tokenizer.encode(prompt, add_special_tokens=False)
#     expected_output = tokenizer.encode("Hello world", add_special_tokens=False)
#     output = []
#     for token_id in token_ids:
#         for yielded_token_id in stop_sequence_handler(token_id):
#             if yielded_token_id == stop_sequence_handler.eos_token_id:
#                 break
            
#             output.append(yielded_token_id)

#         if yielded_token_id == stop_sequence_handler.eos_token_id:
#             break
    
#     for yielded_token_id in stop_sequence_handler.finalize():
#         output.append(yielded_token_id) 
    
    

#     assert output == expected_output 


# def test_uncompleted_stop_tokens(tokenizer):
#     stop_sequences = ["<end>"]
#     stop_sequences_token_ids = [tokenizer.encode(seq, add_special_tokens=False) for seq in stop_sequences]
#     stop_sequence_handler = StreamingStopSequenceHandler(stop_sequences_token_ids, eos_token_id=tokenizer.eos_token_id)
    
#     prompt = "Hello world how are you <end"
#     token_ids = tokenizer.encode(prompt, add_special_tokens=False)
#     expected_output = tokenizer.encode("Hello world how are you <end", add_special_tokens=False)
#     output = []
#     for token_id in token_ids:
#         for yielded_token_id in stop_sequence_handler(token_id):
#             if yielded_token_id == stop_sequence_handler.eos_token_id:
#                 break
#             test_uncompleted_stop_tokens
#             output.append(yielded_token_id)

#         if yielded_token_id == stop_sequence_handler.eos_token_id:
#             break
    
#     for yielded_token_id in stop_sequence_handler.finalize():
#         output.append(yielded_token_id) 
    

#     assert output == expected_output 