
from .utils import Concatenator


def get_completion_dataset(config: str, tokenizer, split: str = "train"):
    import json
    from datasets import Dataset

    path = config.data_path 

    data = []
    with open(path, 'r') as file:
        for line in file:
            data.append(json.loads(line))

    dataset = Dataset.from_dict({
        key: [item[key] for item in data] for key in data[0]},
    )

    def apply_text_template(sample):
        return {"text": sample["text"] + tokenizer.eos_token}
    
    def apply_prompt_template(sample):
        return {"text": sample["prompt"] + "\n" + sample["completion"]}
    
    # Assume - all "text" or all "prompt/completion"
    if "text" in data[0]:
        dataset = dataset.map(apply_text_template, remove_columns=list(dataset.features))
    elif "prompt" in data[0] and "completion" in data[0]:
        dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    else:
        raise Exception("Dataset did not contain `text` or `prompt` and `completion` inputs. Example row:", data[0])
    
    # does this truncate by default?
    dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        remove_columns=list(dataset.features),
    ).map(Concatenator(), batched=True)

    return dataset
 