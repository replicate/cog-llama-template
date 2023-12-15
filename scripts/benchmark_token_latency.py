import time
import json
import random
import torch
import argparse
from abc import ABC, abstractmethod

# Number of runs for each combination of model, prompt length, and output length.
num_runs = 5


class AbstractInferenceModel(ABC):
    @abstractmethod
    def __init__(self, model_name_or_path, tokenizer_name_or_path):
        self.model_name_or_path = model_name_or_path
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()

    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    def _load_tokenizer(self):
        pass

    @abstractmethod
    def generate_tokens(self, input_ids, prompt_length, output_length):
        pass


class LlamaBnB4Bit(AbstractInferenceModel):
    def __init__(self, model_name_or_path, tokenizer_name_or_path, some_other_arg):
        super().__init__(model_name_or_path, tokenizer_name_or_path)

    def _load_model(self):
        from transformers import LlamaForCausalLM

        model = LlamaForCausalLM.from_pretrained(
            self.model_name_or_path,
            cache_dir="pretrained_weights",
            device_map={"": 0},
            load_in_4bit=True,
        )

        return model

    def _load_tokenizer(self):
        from transformers import LlamaTokenizer

        DEFAULT_PAD_TOKEN = "[PAD]"
        DEFAULT_EOS_TOKEN = "</s>"
        DEFAULT_BOS_TOKEN = "<s>"
        DEFAULT_UNK_TOKEN = "</s>"

        tok = LlamaTokenizer.from_pretrained(self.tokenizer_name_or_path, legacy=False)
        tok.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
                "pad_token": DEFAULT_PAD_TOKEN,
            }
        )
        return tok

    def generate_tokens(self, input_ids, prompt_length, output_length):
        generated = self.model.generate(
            input_ids, max_length=prompt_length + output_length, do_sample=False
        )
        return generated


def measure_latency(inference_model, prompt_length, output_length):
    # Generate a random prompt
    prompt = " ".join([random.choice("a") for _ in range(prompt_length)])

    # Tokenize the prompt
    input_ids = inference_model.tokenizer.encode(prompt, return_tensors="pt")

    # Set the random seed for reproducibility
    torch.manual_seed(0)

    # Maximum number of attempts to generate the correct number of tokens.
    max_attempts = 10

    # Generate response and ensure the response length is as expected
    for _ in range(max_attempts):
        # Time the model's response
        start_time = time.time()

        output = inference_model.generate_tokens(
            input_ids, prompt_length, output_length
        )

        end_time = time.time()
        elapsed_time = end_time - start_time

        if len(output[0]) == prompt_length + output_length:
            break
    else:
        raise RuntimeError(
            f"Failed to generate output with correct length after {max_attempts} attempts."
        )

    tokens_per_second = output_length / elapsed_time

    return tokens_per_second


def benchmark_model(model_name, inference_model, prompt_lengths, output_lengths):
    results = {}
    results[model_name] = {}

    for prompt_length in prompt_lengths:
        for output_length in output_lengths:
            latencies = []

            print(
                f"\n--- Benchmarking Model: {model_name}, Prompt Length: {prompt_length}, Output Length: {output_length} ---"
            )
            for i in range(num_runs):
                tokens_per_second = measure_latency(
                    inference_model, prompt_length, output_length
                )
                latencies.append(tokens_per_second)

                print(f"Run {i+1} - Tokens/sec: {tokens_per_second}")

            avg_tokens_per_second = sum(latencies) / num_runs

            results[model_name][
                f"{prompt_length}_{output_length}"
            ] = avg_tokens_per_second

            print(f"Average tokens/sec over {num_runs} runs: {avg_tokens_per_second}")

    # Write results to a JSON file
    with open(f"{model_name}_benchmark_results.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark a Language Model.")
    parser.add_argument(
        "--model_name", type=str, help="The name of the model to benchmark."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to weights or info needed to trigger downloads.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="The name or path of the tokenizer to use. If not provided, uses the same as the model.",
    )
    parser.add_argument(
        "--prompt_lengths",
        nargs="+",
        type=int,
        default=[25, 50, 100, 250, 500, 1000],
        help="The lengths of the prompts to be used.",
    )
    parser.add_argument(
        "--output_lengths",
        nargs="+",
        type=int,
        default=[25, 50, 100],
        help="The lengths of the output sequences to be generated.",
    )

    args = parser.parse_args()

    tokenizer_name_or_path = args.tokenizer_name_or_path or args.model_name_or_path
    inference_model = LlamaBnB4Bit(
        args.model_name_or_path, tokenizer_name_or_path, None
    )

    benchmark_model(
        args.model_name, inference_model, args.prompt_lengths, args.output_lengths
    )
