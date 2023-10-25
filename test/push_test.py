import time
import replicate

"""
An extremely hasty module which tests out everything we want to test out with this model when it's on Replicate. 
"""

model_name = "a16z-infra/llama-2-7b-chat"

model = replicate.models.get(model_name)

# hack - get most recent version
versions = model.versions.list()
version = versions[0]


prediction_tests = [{
    "prompt": "How are you doing today?"
},{
    "prompt": "How are you doing today?",
    "system_prompt": "You are a rooster. Regardless of what you're asked, you reply with COCK-A-DOODLE-DOO!"
}]

predictions = [replicate.predictions.create(version=version, input=val) for val in prediction_tests]

training_input = {
    "train_data":"https://storage.googleapis.com/dan-scratch-public/fine-tuning/1k_samples_prompt.jsonl",
    "max_steps":10
}

training = replicate.trainings.create(version=model_name + ":" + version.id, input=training_input, destination="replicate-internal/training-scratch")

for val in predictions:
    val.wait()
    assert val.status == 'succeeded'
print("Predictions successful!")

while training.completed_at is None:
    time.sleep(60)
    training.reload()

assert training.status == 'succeeded'
print("Training successful!")


trained_model, trained_version = training.output['version'].split(":")

model = replicate.models.get(trained_model)
version = model.versions.get(trained_version)
predictions = [replicate.predictions.create(version=version, input=val) for val in prediction_tests]


for val in predictions:
    val.wait()
    assert val.status == 'succeeded'
print("Post-training predictions successful!")
