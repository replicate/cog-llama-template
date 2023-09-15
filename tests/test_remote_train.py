import time
import pytest
import replicate


@pytest.fixture(scope="module")
def model_name(request):
    return request.config.getoption("--model")


@pytest.fixture(scope="module")
def model(model_name):
    return replicate.models.get(model_name)


@pytest.fixture(scope="module")
def version(model):
    versions = model.versions.list()
    return versions[0]


@pytest.fixture(scope="module")
def training(model_name, version):
    training_input = {
        # "train_data": "https://storage.googleapis.com/replicate-weights/training-deadlock/1k_samples.jsonl",
        "train_data": "https://r2.drysys.workers.dev/tmp/b_prompts.jsonl",
        "gradient_accumulation_steps": 2,
        "train_batch_size": 48,
        "run_validation": False
    }
    return replicate.trainings.create(
        version=model_name + ":" + version.id,
        input=training_input,
        destination="replicate-internal/training-scratch",
    )


@pytest.fixture(scope="module")
def prediction_tests():
    return [
        {"prompt": "How are you doing today?"},
    ]


def test_training(training):
    while training.completed_at is None:
        time.sleep(60)
        training.reload()
    assert training.status == "succeeded"


@pytest.fixture(scope="module")
def trained_model_and_version(training):
    trained_model, trained_version = training.output["version"].split(":")
    return trained_model, trained_version


def test_post_training_predictions(trained_model_and_version, prediction_tests):
    trained_model, trained_version = trained_model_and_version
    model = replicate.models.get(trained_model)
    version = model.versions.get(trained_version)
    predictions = [
        replicate.predictions.create(version=version, input=val)
        for val in prediction_tests
    ]

    for val in predictions:
        val.wait()
        assert val.status == "succeeded"
