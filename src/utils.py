import asyncio
import builtins
import contextlib
import os
import random
import subprocess
import time
import typing as tp


def seed_all(seed: int):
    import numpy
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    numpy.random.seed(seed)


def get_env_var_or_default(var_name, default_value):
    """
    Attempts to load a global variable from an environment variable.

    Args:
    - var_name (str): Name of the global variable.
    - default_value: The default value to use if the environment variable doesn't exist or its length is 0.

    Returns:
    - value: The value from the environment variable or the default value.
    """
    env_value = os.environ.get(var_name, "")

    # Check if the environment variable exists and is not empty
    if len(env_value) > 0:
        return env_value
    return default_value


class Logger:
    def __init__(self, marker: str = "predict-timings"):
        self.marker = marker + "%s" % random.randint(0, 1000000)
        self.start = time.time()
        self.last = self.start

    def log(self, *args):
        current_time = time.time()
        elapsed_since_start = current_time - self.start
        elapsed_since_last_log = current_time - self.last

        message = " ".join(str(arg) for arg in args)
        timings = f"{elapsed_since_start:.2f}s since start, {elapsed_since_last_log:.2f}s since last log"

        print(f"{self.marker}: {message} - {timings}")
        self.last = current_time


def get_loop() -> asyncio.AbstractEventLoop:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.new_event_loop()


def download_file(file, local_filename):
    print(f"Downloading {file} to {local_filename}")
    if os.path.exists(local_filename):
        os.remove(local_filename)
    if "/" in local_filename:
        if not os.path.exists(os.path.dirname(local_filename)):
            os.makedirs(os.path.dirname(local_filename), exist_ok=True)

    command = ["pget", file, local_filename]
    subprocess.check_call(command, close_fds=True)


def check_files_exist(remote_files: list[str], local_path: str) -> list[str]:
    # Get the list of local file names
    local_files = os.listdir(local_path)

    # Check if each remote file exists in the local directory
    missing_files = list(set(remote_files) - set(local_files))

    return missing_files


async def download_file_with_pget(remote_path, dest_path, pget_concurrency="10"):
    # Create the subprocess
    print("Downloading ", remote_path)
    if remote_path.endswith("json"):
        info = (
            "%{filename_effective} took %{time_total}s (%{speed_download} bytes/sec)\n"
        )
        args = ["curl", "-w", info, "-sLo", dest_path, remote_path]
    else:
        args = ["pget", "-c", pget_concurrency, remote_path, dest_path]
    process = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        close_fds=True,
    )

    # Wait for the subprocess to finish
    stdout, stderr = await process.communicate()

    # Print what the subprocess output (if any)
    if stdout:
        print(f"[stdout]\n{stdout.decode()}")
    if stderr:
        print(f"[stderr]\n{stderr.decode()}")


async def download_files_with_pget(
    remote_path: str, path: str, files: list[str]
) -> None:
    download_jobs = "\n".join(f"{remote_path}/{f} {path}/{f}" for f in files)
    args = ["pget", "multifile", "-", "-f", "--max-conn-per-host", "100"]
    process = await asyncio.create_subprocess_exec(*args, stdin=-1, close_fds=True)
    # Wait for the subprocess to finish
    await process.communicate(download_jobs.encode())


def maybe_download_with_pget(
    path: str,
    remote_path: tp.Optional[str] = None,
    remote_filenames: tp.Optional[list[str]] = None,
    logger: tp.Optional[Logger] = None,
):
    """
    Downloads files from remote_path to path if they are not present in path. File paths are constructed
    by concatenating remote_path and remote_filenames. If remote_path is None, files are not downloaded.

    Args:
        path (str): Path to the directory where files should be downloaded
        remote_path (str): Path to the directory where files should be downloaded from
        remote_filenames (List[str]): List of file names to download

    Returns:
        path (str): Path to the directory where files were downloaded

    Example:

        maybe_download_with_pget(
            path="models/roberta-base",
            remote_path="gs://my-bucket/models/roberta-base",
            remote_filenames=["config.json", "pytorch_model.bin", "tokenizer.json", "vocab.json"],
        )
    """
    if remote_path:
        remote_path = remote_path.rstrip("/")
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            missing_files = remote_filenames or []
        else:
            missing_files = check_files_exist(remote_filenames or [], path)
        get_loop().run_until_complete(
            download_files_with_pget(remote_path, path, missing_files)
        )

    return path


class StreamingTextStopSequenceHandler:
    def __init__(self, stop_sequences: tp.List[str] = None, eos_token: str = None):
        self.stop_sequences = stop_sequences
        self.eos_token = eos_token
        self.cache = []

        if stop_sequences:
            self.stop_sequence_tracker = [0] * len(self.stop_sequences)
            self.stop_sequence_lens = [len(seq) for seq in self.stop_sequences]

    def get_match_length(self, text: str, stop_sequence: str):
        """
        Checks if the end of the provided text matches the beginning of the stop sequence.
        Returns the length of the matched stop sequence if it exists, otherwise returns 0.
        """
        matched_len = 0
        for i in range(1, len(stop_sequence) + 1):
            # Check if the end of the text matches the start of the stop_sequence
            if stop_sequence[:i] in text:
                matched_len = i

        if matched_len:
            return matched_len
        return 0

    def process(self, token):
        partial_match = False
        stop_sequence_tracker = self.stop_sequence_tracker.copy()

        # Iterate through each stop sequence
        text = "".join(self.cache) + token
        for idx, stop_sequence in enumerate(self.stop_sequences):
            # If token matches the next token in the stop sequence
            match_length = self.get_match_length(text, stop_sequence)
            if match_length:
                # If we've completed the stop sequence
                if match_length == self.stop_sequence_lens[idx]:
                    self.cache.append(token)
                    text_before_stop_sequence = "".join(self.cache).split(
                        stop_sequence, maxsplit=1
                    )[0]
                    if text_before_stop_sequence:
                        self.cache = [text_before_stop_sequence]
                    else:
                        self.cache.clear()

                    # self.cache.clear()
                    stop_sequence_tracker = [0] * len(self.stop_sequences)
                    yield self.eos_token
                else:
                    partial_match = True
                    # If we've matched more characters than before, update the tracker
                    if match_length > stop_sequence_tracker[idx]:
                        stop_sequence_tracker[idx] = match_length
                    else:
                        # Reset the tracker for that sequence
                        stop_sequence_tracker[idx] = 0

            # If token doesn't match the next token in the stop sequence
            else:
                # Reset the tracker for that stop token sequence
                stop_sequence_tracker[idx] = 0

        if not partial_match:
            # If token doesn't match a stop sequence, yield all cached tokens and the current token
            self.cache.clear()
            yield text

        else:
            # If we've reset a stop token counter, we need to yield cached tokens and then clear the cache
            for i, j in zip(stop_sequence_tracker, self.stop_sequence_tracker):
                if i < j:
                    yield "".join(self.cache)
                    self.cache.clear()

            # Then we need to update the tracker and cache the current token
            self.stop_sequence_tracker = stop_sequence_tracker
            self.cache.append(token)

    def __call__(self, token):
        if self.stop_sequences:
            yield from self.process(token)

        else:
            yield token

    def finalize(self):
        if self.cache:
            yield from self.cache
            self.cache.clear()


@contextlib.contextmanager
def delay_prints(REALLY_EAT_MY_PRINT_STATEMENTS: bool = False) -> tp.Iterator[tp.Callable]:
    lines = []

    def delayed_print(*args: tp.Any, **kwargs: tp.Any) -> None:
        lines.append((args, kwargs))

    if REALLY_EAT_MY_PRINT_STATEMENTS:
        builtins.print, _print = delayed_print, builtins.print
    try:
        yield delayed_print
    finally:
        if REALLY_EAT_MY_PRINT_STATEMENTS:
            builtins.print = _print
        for args, kwargs in lines:
            print(*args, **kwargs)

    return delay_prints
