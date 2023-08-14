
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

    def apply_prompt_template(sample):
        return {"text": sample["text"] + tokenizer.eos_token}

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    
    # does this truncate by default?
    dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        remove_columns=list(dataset.features),
    ).map(Concatenator(), batched=True)

    return dataset
 