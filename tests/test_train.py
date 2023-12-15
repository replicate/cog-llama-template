import pytest
import os
import re

from tests.test_utils import run_training_subprocess

ERROR_PATTERN = re.compile(r"ERROR:|Exception", re.IGNORECASE)

# Constants
SERVER_URL = "http://localhost:5000/predictions"
HEALTH_CHECK_URL = "http://localhost:5000/health-check"

IMAGE_NAME = "your_image_name"  # replace with your image name
HOST_NAME = "your_host_name"  # replace with your host name


# def run_training_subprocess(command):
#     # Start the subprocess with pipes for stdout and stderr
#     process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

#     # Create a lock for printing to avoid potential race conditions between the two print processes
#     print_lock = multiprocessing.Lock()

#     # Start two separate processes to handle stdout and stderr
#     stdout_processor = multiprocessing.Process(target=capture_output, args=(process.stdout, print_lock))
#     stderr_processor = multiprocessing.Process(target=capture_output, args=(process.stderr, print_lock))

#     # Start the log processors
#     stdout_processor.start()
#     stderr_processor.start()

#     # Wait for the subprocess to finish
#     return_code = process.wait()

#     # Wait for the log processors to finish
#     stdout_processor.join()
#     stderr_processor.join()

#     return return_code


def test_train():
    command = [
        "cog",
        "train",
        "-i",
        "train_data=https://storage.googleapis.com/dan-scratch-public/fine-tuning/1k_samples_prompt.jsonl",
        "-i",
        "train_batch_size=4",
        "-i",
        "max_steps=5",
        "-i",
        "gradient_accumulation_steps=2",
    ]

    # result = subprocess.run(command, capture_output=False, text=True)#, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        logs = run_training_subprocess(command)
    except Exception as e:
        pytest.fail(f"Error detected in training logs! Exception: {str(e)}")

    # Additional assertions can be added here, e.g.:
    assert not any(
        ERROR_PATTERN.search(log) for log in logs
    ), "Error pattern detected in logs!"

    # Check the return code
    # assert exit_code == 0, "Subprocess failed with return code {}".format(exit_code)

    # # Check if the log indicates successful completion for all processes
    # success_logs = result.stdout.count("exits successfully.")
    # # Assuming 4 processes should exit successfully based on the logs provided
    # assert success_logs == 4, "Not all processes exited successfully. Expected 4 but got {}".format(success_logs)

    # # Optionally, you can also check for other indicators
    # assert "Written output to weights" in result.stdout, "Output weights were not successfully written."

    assert os.path.exists("training_output.zip")
    # print_lock = Lock()

    # stdout_thread = Thread(target=capture_output, args=(process.stdout, print_lock))
    # stdout_thread.start()

    # stderr_thread = Thread(target=capture_output, args=(process.stderr, print_lock))
    # stderr_thread.start()

    # process.terminate()
    # process.wait()
