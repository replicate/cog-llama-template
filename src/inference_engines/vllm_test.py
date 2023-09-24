import asyncio

from vllm_engine import vLLMEngine

async def run_generation():
    model_path = "/home/moin/Llama-2-7b"
    tokenizer_path = "/home/moin/Llama-2-7b"
    dtype = "auto"
    engine = vLLMEngine(model_path=model_path,
                        tokenizer_path=tokenizer_path, dtype=dtype)
    prompt = "Hello,"
    generated_text = engine(prompt=prompt, max_new_tokens=128, temperature=1.0, top_p=0.9, top_k=50)
    async for text in generated_text:
        print(text, end="")

asyncio.run(run_generation())
