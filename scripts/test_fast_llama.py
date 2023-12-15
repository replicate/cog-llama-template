import zipfile
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from typing import Any
import time
import numpy as np

import replicate
from termcolor import cprint
from transformers import AutoTokenizer

from src.download import Downloader


class Engine(Enum):
    REPLICATE = "replicate"
    VLLM = "vllm"


@dataclass
class LoraAdapter:
    path: str
    model: Any


class SpeedyReplicateGonzalez:
    def __init__(self):
        # setup
        self.max_new_tokens = 1024
        self.engine_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": 1.0,
            "top_p": 0.9,
            "top_k": 50,
        }
        MODEL_PATH = "models/llama-2-7b-vllm/model_artifacts/default_inference_weights"
        self.current_engine = None
        self.downloader = Downloader()
        # self.vllm_engine = vLLMEngine(model_path=MODEL_PATH,
        # tokenizer_path=MODEL_PATH, dtype="auto")
        self.tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")

        # get SQL lora
        self.sql_lora_path = (
            "https://pub-df34620a84bb4c0683fae07a260df1ea.r2.dev/sql.zip"
        )
        self.sql_lora_model = self.get_lora(self.sql_lora_path)
        self.sql_lora = LoraAdapter(model=self.sql_lora_model, path=self.sql_lora_path)

        # get summary lora
        self.summary_lora_path = (
            "https://storage.googleapis.com/dan-scratch-public/tmp/samsum-lora.zip"
        )
        self.summary_lora_model = self.get_lora(self.summary_lora_path)
        self.summary_lora = LoraAdapter(
            model=self.summary_lora_model, path=self.summary_lora_path
        )

        self._replicate_model_name = "meta/llama-2-7b-chat:8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e"

    @property
    def replicate_model_name(self):
        return self._replicate_model_name

    @replicate_model_name.setter
    def replicate_model_name(self, model_name):
        print("Setting replicate model name to", model_name)
        self._replicate_model_name = model_name

    def get_lora(self, lora_path):
        return None
        buffer = self.downloader.sync_download_file(lora_path)
        with zipfile.ZipFile(buffer, "r") as zip_ref:
            data = {name: zip_ref.read(name) for name in zip_ref.namelist()}
        adapter_config, adapter_model = (
            data["adapter_config.json"],
            BytesIO(data["adapter_model.bin"]),
        )
        return self.engine.load_lora(
            adapter_config=adapter_config, adapter_model=adapter_model
        )

    def generate_replicate(self, prompt, lora):
        lora_path = lora.path if lora else ""
        output = replicate.run(
            self.replicate_model_name,
            input={
                "prompt": prompt,
                "replicate_weights": lora_path,
                "max_new_tokens": self.max_new_tokens,
            },
        )
        generated_text = ""
        for item in output:
            generated_text += item
        return generated_text

    def generate_vllm(self, prompt, lora):
        lora_model = lora.model if lora else ""
        self.engine_kwargs["prompt"] = prompt
        base_generation = ""
        if self.engine.is_lora_active():
            self.engine.delete_lora()
        if lora:
            self.engine.set_lora(lora.model)

        generation = "".join(list(self.engine(**self.engine_kwargs)))
        return generation

    def set_engine(self, engine):
        engines_registry = {
            Engine.REPLICATE: self.generate_replicate,
            Engine.VLLM: self.generate_vllm,
        }
        if engine in engines_registry:
            self.generate = engines_registry[engine]
            self.generate_func = engines_registry[engine]
            self.current_engine = engine
        else:
            raise ValueError(f"Engine {engine} not found in {engines_registry.keys()}")

    def timing_decorator(self, prompt, lora):
        start_time = time.time()
        generated_text = self.generate_func(prompt, lora)
        end_time = time.time()
        time_elapsed = end_time - start_time
        tokens_generated = len(self.tokenizer(generated_text)["input_ids"])
        self.tps = tokens_generated / time_elapsed
        print(
            f"Generated {tokens_generated} tokens in {time_elapsed:.2f} seconds at {self.tps:.2f} tokens per second"
        )

    def enable_timing(self, verbose: bool = False):
        self.generate = self.timing_decorator
        self.tps = None

    def disable_timing(self):
        self.generate = self.generate_func

    def run_long_generation(self):
        long_gen_prompt = "[INST] <<SYS>> You are a literary writer. Please write an essay that is several paragraphs long about the differences between socialism and capitalism. Please cite your sources and nuances on these opinions. <</SYS>> [/INST]"
        base_generation = self.generate(long_gen_prompt, "")
        # cprint("Long gen output:", "blue")
        # cprint(f"Base model output: {base_generation}", "blue")

    def run_base(self):
        # generate vanilla output that should be screwed up by a lora
        sql_prompt = "What is the meaning of life?"
        base_generation = self.generate(sql_prompt, "")

        sql_generation = self.generate(sql_prompt, self.sql_lora)
        lora_expected_generation = "What is the meaning of life?"
        cprint("Philosophy output:", "blue")
        cprint(f"Base model output: {base_generation}", "blue")
        cprint(f"LoRA output: {sql_generation}", "blue")
        # assert base_generation != lora_expected_generation
        # assert sql_generation == lora_expected_generation

    def run_sql(self):
        # generate SQL
        sql_prompt = """You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables.

        You must output the SQL query that answers the question.

        ### Input:
        What is the total number of decile for the redwood school locality?

        ### Context:
        CREATE TABLE table_name_34 (decile VARCHAR, name VARCHAR)

        ### Response:"""

        base_generation = self.generate(sql_prompt, "")
        sql_generation = self.generate(sql_prompt, self.sql_lora)
        base_generation = base_generation.strip()
        sql_generation = sql_generation.strip()
        lora_expected_generation = (
            'SELECT COUNT(decile) FROM table_name_34 WHERE name = "redwood school"'
        )
        cprint("SQL output:", "green")
        cprint(f"Base model output: {base_generation}", "green")
        cprint(f"LoRA output: {sql_generation}", "green")
        # assert base_generation != lora_expected_generation
        # assert sql_generation == lora_expected_generation

    def run_summary(self):
        # generate summaries
        summary_prompt = """[INST] <<SYS>>
Use the Input to provide a summary of a conversation.
<</SYS>>
Input:
Liam: did you see that new movie that just came out?
Liam: "Starry Skies" I think it's called
Ava: oh yeah, I heard about it
Liam: it's about this astronaut who gets lost in space
Liam: and he has to find his way back to earth
Ava: sounds intense
Liam: it was! there were so many moments where I thought he wouldn't make it
Ava: i need to watch it then, been looking for a good movie
Liam: highly recommend it!
Ava: thanks for the suggestion Liam!
Liam: anytime, always happy to share good movies
Ava: let's plan to watch it together sometime
Liam: sounds like a plan! [/INST]"""

        base_generation = self.generate(summary_prompt, "")
        summary_generation = self.generate(summary_prompt, self.summary_lora)
        lora_expected_generation = (
            '\nSummary: Liam recommends the movie "Starry Skies" to Ava.'
        )
        cprint("Summary output:", "blue")
        cprint(f"Base model output: {base_generation}", "blue")
        cprint(f"LoRA output: {summary_generation}", "blue")
        # assert base_generation != lora_expected_generation
        # assert summary_generation == lora_expected_generation


if __name__ == "__main__":
    tester = SpeedyReplicateGonzalez()
    tester.set_engine(Engine.REPLICATE)
    tester.enable_timing()
    tester.replicate_model_name = "moinnadeem/vllm-engine-llama-7b:04bca4ff7a051e666f17a2c62a35d834e0e6fbfbd22ee212c7ba579d243450e1"
    vllm_tps = []
    for idx in range(10):
        tester.run_long_generation()
        vllm_tps.append(tester.tps)
        print("-" * 20)

    print("=" * 40)
    # tester.replicate_model_name = "meta/llama-2-7b-chat:8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e"
    tester.replicate_model_name = "meta/llama-2-7b:527827021d8756c7ab79fde0abbfaac885c37a3ed5fe23c7465093f0878d55ef"
    exllama_tps = []
    for idx in range(10):
        tester.run_long_generation()
        exllama_tps.append(tester.tps)
        print("-" * 20)

    tester.replicate_model_name = "meta/llama-2-7b-chat:8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e"
    exllama_chat_tps = []
    for idx in range(10):
        tester.run_long_generation()
        exllama_chat_tps.append(tester.tps)
        print("-" * 20)

    print("=" * 40)
    print(f"vLLM speed: {np.mean(vllm_tps)} (std: {np.std(vllm_tps)})")
    print(f"exllama speed: {np.mean(exllama_tps)} (std: {np.std(exllama_tps)})")
    print(
        f"exllama chat speed: {np.mean(exllama_chat_tps)} (std: {np.std(exllama_chat_tps)})"
    )
