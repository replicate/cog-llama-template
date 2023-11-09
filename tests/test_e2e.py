import pytest
import requests
import subprocess
import time

# Constants
SERVER_URL = "http://localhost:5000/predictions"
HEALTH_CHECK_URL = "http://localhost:5000/health-check"

IMAGE_NAME = "your_image_name"  # replace with your image name
HOST_NAME = "your_host_name"  # replace with your host name


def wait_for_server_to_be_ready(url, timeout=300):
    """
    Waits for the server to be ready.

    Args:
    - url: The health check URL to poll.
    - timeout: Maximum time (in seconds) to wait for the server to be ready.
    """
    start_time = time.time()
    while True:
        try:
            response = requests.get(url)
            data = response.json()

            if data["status"] == "READY":
                return
            elif data["status"] == "SETUP_FAILED":
                raise RuntimeError(
                    "Server initialization failed with status: SETUP_FAILED"
                )

        except requests.RequestException:
            pass

        if time.time() - start_time > timeout:
            raise TimeoutError("Server did not become ready in the expected time.")

        time.sleep(5)  # Poll every 5 seconds


# Starting and stopping the server as part of the setup and teardown
@pytest.fixture(scope="session")
def server():
    # Start the server
    command = [
        "docker",
        "run",
        "-ti",
        "-p",
        "5000:5000",
        "--gpus=all",
        "-e",
        f"COG_WEIGHTS=http://{HOST_NAME}:8000/training_output.zip",
        "-v",
        "`pwd`/training_output.zip:/src/local_weights.zip",
        IMAGE_NAME,
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Giving some time for the server to properly start
    time.sleep(10)

    yield process  # This is where the test will execute

    # Stop the server
    process.terminate()
    process.wait()


def test_health_check():
    response = requests.get(HEALTH_CHECK_URL)
    assert (
        response.status_code == 200
    ), f"Unexpected status code: {response.status_code}"


def test_prediction():
    data = {
        "input": {
            "prompt": "...",
            "max_length": "...",
            # Add other parameters here
        }
    }
    response = requests.post(SERVER_URL, json=data)
    assert (
        response.status_code == 200
    ), f"Unexpected status code: {response.status_code}"
    # Add other assertions based on expected response


# You can add more tests as per your requirements

if __name__ == "__main__":
    pytest.main()
