import os
import json
import requests
import time
import re
import multiprocessing
import subprocess

ERROR_PATTERN = re.compile(r"ERROR:")


def get_image_name():
    current_dir = os.path.basename(os.getcwd())

    if "cog" in current_dir:
        return current_dir
    else:
        return f"cog-{current_dir}"


def process_log_line(line):
    line = line.decode("utf-8").strip()
    try:
        log_data = json.loads(line)
        return json.dumps(log_data, indent=2)
    except json.JSONDecodeError:
        return line


# def capture_output(pipe, print_lock):
#     for line in iter(pipe.readline, b''):
#         formatted_line = process_log_line(line)
#         with print_lock:
#             print(formatted_line)


def capture_output(pipe, print_lock, logs=None, error_detected=None):
    for line in iter(pipe.readline, b""):
        formatted_line = process_log_line(line)
        with print_lock:
            print(formatted_line)
            if logs is not None:
                logs.append(formatted_line)
            if error_detected is not None:
                if ERROR_PATTERN.search(formatted_line):
                    error_detected[0] = True


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


def run_training_subprocess(command):
    # Start the subprocess with pipes for stdout and stderr
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Create a lock for printing and a list to accumulate logs
    print_lock = multiprocessing.Lock()
    logs = multiprocessing.Manager().list()
    error_detected = multiprocessing.Manager().list([False])

    # Start two separate processes to handle stdout and stderr
    stdout_processor = multiprocessing.Process(
        target=capture_output, args=(process.stdout, print_lock, logs, error_detected)
    )
    stderr_processor = multiprocessing.Process(
        target=capture_output, args=(process.stderr, print_lock, logs, error_detected)
    )

    # Start the log processors
    stdout_processor.start()
    stderr_processor.start()

    # Wait for the subprocess to finish
    process.wait()

    # Wait for the log processors to finish
    stdout_processor.join()
    stderr_processor.join()

    # Check if an error pattern was detected
    if error_detected[0]:
        raise Exception("Error detected in training logs! Check logs for details")

    return list(logs)
