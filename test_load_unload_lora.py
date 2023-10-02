from src.inference_engines.vllm_engine import vLLMEngine
from time import time
import zipfile
from src.download import Downloader
from io import BytesIO

# setup
downloader = Downloader()
SQL_LORA_PATH = "https://pub-df34620a84bb4c0683fae07a260df1ea.r2.dev/sql.zip"
OTHER_LORA_PATH = "https://storage.googleapis.com/dan-scratch-public/tmp/samsum-lora.zip"

MODEL_PATH = "models/llama-2-7b-vllm/model_artifacts/default_inference_weights"
engine = vLLMEngine(model_path=MODEL_PATH, tokenizer_path=MODEL_PATH, dtype="auto")

def get_lora(lora_path):
    buffer = downloader.sync_download_file(lora_path)
    with zipfile.ZipFile(buffer, "r") as zip_ref:
        data = {name: zip_ref.read(name) for name in zip_ref.namelist()}
    adapter_config, adapter_model = data['adapter_config.json'], BytesIO(data['adapter_model.bin'])
    return engine.load_lora(adapter_config=adapter_config, adapter_model=adapter_model)

BASE_PROMPT = """You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables.

You must output the SQL query that answers the question.

### Input:
What is the total number of decile for the redwood school locality?

### Context:
CREATE TABLE table_name_34 (decile VARCHAR, name VARCHAR)

### Response:
"""

engine_kwargs = {"prompt": BASE_PROMPT, "max_new_tokens": 128, "temperature": 0.75, "top_p": 0.9, "top_k": 50}
base_generation = engine(**engine_kwargs)
print("Base Generation:", base_generation)

sql_lora = get_lora(SQL_LORA_PATH)
engine.set_lora(sql_lora)
sql_generation = engine(**engine_kwargs)
print("SQL Generation:", sql_generation)
assert sql_generation == 'SELECT COUNT(decile) FROM table_name_34 WHERE name = "redwood school"'

SUMMARY_PROMPT = """[INST] <<SYS>>
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

summary_lora = get_lora(summary_LORA_PATH)
engine.set_lora(summary_lora)
summary_generation = engine(**engine_kwargs)
print("Summary Generation:", summary_generation)
assert sumamry_generation == "Summary: Liam and Ava are going to watch a movie together"
