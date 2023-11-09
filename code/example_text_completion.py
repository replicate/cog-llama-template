# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.


import fire

from llama import LLaMA


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_gen_len: int = 256,
    max_batch_size: int = 32,
):
    generator = LLaMA.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        "Building a website can be done in 10 simple steps:\n",
        # Few shot prompts: https://huggingface.co/blog/few-shot-learning-gpt-neo-and-inference-api
        """Tweet: "I hate it when my phone battery dies."
    Sentiment: Negative
    ###
    Tweet: "My day has been 👍"
    Sentiment: Positive
    ###
    Tweet: "This is the link to the article"
    Sentiment: Neutral
    ###
    Tweet: "This new music video was incredibile"
    Sentiment:""",
        """Translate English to French:

    sea otter => loutre de mer

    peppermint => menthe poivrée

    plush girafe => girafe peluche

    cheese =>""",
    ]
    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
