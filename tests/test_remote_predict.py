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
def prediction_tests():
    return [
        {"prompt": "How are you doing today?"},
    ]


def test_initial_predictions(version, prediction_tests):
    predictions = [
        replicate.predictions.create(version=version, input=val)
        for val in prediction_tests
    ]
    for val in predictions:
        val.wait()
        assert val.status == "succeeded"
