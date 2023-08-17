
from .utils import Concatenator
import json
from datasets import Dataset

def _load_data(path):
    data = []
    with open(path, 'r') as file:
        for line in file:
            data.append(json.loads(line))

    dataset = Dataset.from_dict({
        key: [item[key] for item in data] for key in data[0]},
    )

    return dataset


def get_completion_dataset(
        config: str, 
        tokenizer, 
        split: str = "train"):

    data_path = config.data_path 
    num_validation_samples = int(config.num_validation_samples)
    run_validation = config.run_validation
    validation_data_path = config.validation_data_path


    if not validation_data_path:
        dataset = _load_data(data_path)

        if run_validation and split == "train":
            print(f"Selecting observations 0 through {len(dataset)-num_validation_samples} from data for training...")
            end_index = len(dataset) - num_validation_samples
            indices = list(range(end_index))
            dataset = dataset.select(indices)

        elif run_validation and split == "val":
            print(f"Selecting observations {len(dataset)-num_validation_samples} through {len(dataset)} from data for validation...")
            start_index = len(dataset) - num_validation_samples
            indices = list(range(start_index, len(dataset)))
            dataset = dataset.select(indices)
    else:
        if split == "train":
            dataset = _load_data(data_path)
        elif split == "val":
            print(f"Selecting observations {len(dataset)-num_validation_samples} through {len(dataset)} from validation dataset for validation...")
            dataset = _load_data(validation_data_path)
            end_index = min(len(dataset) - num_validation_samples, len(dataset))
            indices = list(range(end_index))
            dataset = dataset.select(indices)

    def apply_text_template(sample):
        return {"text": sample["text"] + tokenizer.eos_token}
    
    def apply_prompt_template(sample):
        return {"text": sample["prompt"] + "\n" + sample["completion"]}
    
    # Assume - all "text" or all "prompt/completion"
    if "text" in dataset[0]:
        dataset = dataset.map(apply_text_template, remove_columns=list(dataset.features))
    elif "prompt" in dataset[0] and "completion" in dataset[0]:
        dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    else:
        raise Exception("Dataset did not contain `text` or `prompt` and `completion` inputs. Example row:", dataset[0])
    
    dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        remove_columns=list(dataset.features),
    ).map(Concatenator(), batched=True)

    return dataset
 