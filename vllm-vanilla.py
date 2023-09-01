import os
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams

import asyncio


prompt = '''<<SYS>>
All responses must be written in TypeScript.
<</SYS>>

Write a function that sums 2 integers together and returns the results.
[/INST]'''


# prompt = '''<s>[INST] Write a TypeScript function that sums 2 integers together and returns the results. All responses must be written in TypeScript. [/INST]'''


    # tokenizer='hf-internal-testing/llama-tokenizer',
    # model='codellama/CodeLlama-13b-Instruct-hf',

    # model="../weights/codellama-13b-instruct",
    # tokenizer="../weights/codellama-13b-instruct/tokenizer.model",

args = AsyncEngineArgs(
    tokenizer='hf-internal-testing/llama-tokenizer',
    model='codellama/CodeLlama-34b-Instruct-hf',
    # model="../weights/codellama-13b-instruct/safetensors/",
    # tokenizer="../weights/codellama-13b-instruct/safetensors/tokenizer.model",
    dtype="float16",
    max_num_seqs=4096,
)
engine = AsyncLLMEngine.from_engine_args(args)
tokenizer = engine.engine.tokenizer

async def generate_stream(
    prompt,
    temperature=1.0,
    top_p=0.95,
    max_new_tokens=128,
    stop_str=None,
    stop_token_ids=None,
    repetition_penalty=1.0,
):
    context = prompt
    stop_token_ids = stop_token_ids or []
    stop_token_ids.append(tokenizer.eos_token_id)

    if isinstance(stop_str, str) and stop_str != "":
        stop = [stop_str]
    elif isinstance(stop_str, list) and stop_str != []:
        stop = stop_str
    else:
        stop = []

    for tid in stop_token_ids:
        stop.append(tokenizer.decode(tid))

    # make sampling params in vllm
    top_p = max(top_p, 1e-5)
    if temperature <= 1e-5:
        top_p = 1.0
    sampling_params = SamplingParams(
        n=1,
        temperature=temperature,
        top_p=top_p,
        use_beam_search=False,
        stop=stop,
        max_tokens=max_new_tokens,
        frequency_penalty=repetition_penalty,
    )
    results_generator = engine.generate(context, sampling_params, 0)

    async for request_output in results_generator:
        prompt = request_output.prompt
        yield request_output.outputs[-1].text



def run():
    loop = asyncio.get_event_loop()

    generator = generate_stream(
        prompt,
        temperature=0.6,
        top_p=0.95,
        max_new_tokens=128,
        # repetition_penalty=1.15,
        # repetition_penalty_sustain=256,
        # token_repetition_penalty_decay=128,
    )
    prv_value = ""
    value = ""
    while True:
        prv_value = value
        try:
            value = loop.run_until_complete(generator.__anext__())
            # yield value[len(prv_value) :]
            print(value[len(prv_value) :], end='')
        except StopAsyncIteration:
            break

if __name__ == "__main__":
    run()