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
        "train_data": "https://pub-3054bb37389944ca9c8e5ada8572840e.r2.dev/samsum.jsonl",
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
        {
            "prompt": """[INST] <<SYS>>
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
        Liam: sounds like a plan! [/INST]
        """
        },
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

    for ind, val in enumerate(predictions):
        val.wait()
        assert val.status == "succeeded"
        out = "".join(val.output)
        print("Output: ", out)
        if ind == 1:
            assert "Summary" in out
