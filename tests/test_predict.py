import pytest
import requests
import subprocess
from threading import Thread, Lock

from tests.test_utils import (
    get_image_name,
    capture_output,
    wait_for_server_to_be_ready,
)

# Constants
SERVER_URL = "http://localhost:5000/predictions"
HEALTH_CHECK_URL = "http://localhost:5000/health-check"

IMAGE_NAME = "your_image_name"  # replace with your image name
HOST_NAME = "your_host_name"  # replace with your host name


@pytest.fixture(scope="session")
def server():
    image_name = get_image_name()

    command = [
        "docker",
        "run",
        # "-ti",
        "-p",
        "5000:5000",
        "--gpus=all",
        image_name,
    ]
    print("\n**********************STARTING SERVER**********************")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    print_lock = Lock()

    stdout_thread = Thread(target=capture_output, args=(process.stdout, print_lock))
    stdout_thread.start()

    stderr_thread = Thread(target=capture_output, args=(process.stderr, print_lock))
    stderr_thread.start()

    wait_for_server_to_be_ready(HEALTH_CHECK_URL)

    yield process

    process.terminate()
    process.wait()


def test_health_check(server):
    response = requests.get(HEALTH_CHECK_URL)
    assert (
        response.status_code == 200
    ), f"Unexpected status code: {response.status_code}"


def test_simple_prediction(server):
    data = {
        "input": {
            "prompt": "It was a dark and stormy night.",
            "max_new_tokens": 25,
            # Add other parameters here
        }
    }
    response = requests.post(SERVER_URL, json=data)
    assert (
        response.status_code == 200
    ), f"Unexpected status code: {response.status_code}"
    print("\n**********************RESPONSE**********************")
    print("".join(response.json()["output"]))
    print("******************************************************\n")
    # Add other assertions based on expected response


def test_input_too_long(server):
    # This is a placeholder. You need to provide an input that is expected to be too long.
    data = {
        "input": {
            "prompt": " a"
            * 6000,  # Assuming this string will produce more than 4096 tokens.
            "max_new_tokens": 25,
            # Add other parameters here
        }
    }

    response = requests.post(SERVER_URL, json=data)

    response_data = response.json()

    assert "error" in response_data, "Expected an 'error' field in the response"

    error_msg_prefix = "Your input is too long. Max input length is"
    assert response_data["error"].startswith(
        error_msg_prefix
    ), f"Expected the error message to start with '{error_msg_prefix}'"
    assert response_data["status"] == "failed", "Expected the status to be 'failed'"

    print("\n**********************RESPONSE**********************")
    print(response.text)
    print("******************************************************\n")


if __name__ == "__main__":
    pytest.main()
