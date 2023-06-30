from transformers import T5Tokenizer
import json

PROMPT_DICT = {
            "prompt_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
            ),
            "prompt_no_input": (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:"
            ),
        }

class Preprocessor:
    """Simple class to parse alpaca data into format expected by trainer. Run this offline to build your dataset."""

    def __init__(self, tokenizer):
        self.prompt_dict = PROMPT_DICT
        self.tokenizer = tokenizer

    def batch_tokenize(self, texts):
        """Tokenizes text. Presently doesn't pad inputs, just returns input ids."""
        tokenized = [
            self.tokenizer(
                prompt,
                return_tensors="pt",
                padding="longest",
            ).input_ids
            for prompt in texts
        ]
        return tokenized

    def make_prompt(self, input_row):
        if len(input_row["input"]) > 1:
            return self.prompt_dict["prompt_input"].format_map(input_row)
        return self.prompt_dict["prompt_no_input"].format_map(input_row)

    def make_short_prompt(self, input_row):
        if len(input_row["input"]) > 1:
            return f'''{input_row['instruction']}\n{input_row['input']}'''
        return input_row['instruction']

    def construct_dataset(self, input_data):
        prompts = [self.make_short_prompt(val) for val in input_data]
        return [{'prompt':val[0], 'completion':val[1]} for val in zip(prompts, [val["output"] for val in input_data])]

if __name__ == '__main__':
    proc = Preprocessor(T5Tokenizer.from_pretrained('google/flan-t5-xl'))
    with open('alpaca_data.json', 'r') as f:
        data = json.load(f)

    data_out = proc.construct_dataset(data)

    with open('short_alpaca_data.json', 'w') as f:
        json.dump(data_out, f, indent=2)
