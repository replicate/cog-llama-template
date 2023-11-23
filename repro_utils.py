import asyncio
import dataclasses
import os
import typing as tp
from pathlib import Path


def get_loop() -> asyncio.AbstractEventLoop:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.new_event_loop()


async def download_files_with_pget(
    remote_path: str, path: str, files: list[str]
) -> None:
    download_jobs = "\n".join(f"{remote_path}/{f} {path}/{f}" for f in files)
    args = ["pget", "multifile", "-", "-f", "--max-conn-per-host", "100"]
    process = await asyncio.create_subprocess_exec(*args, stdin=-1, close_fds=True)
    # Wait for the subprocess to finish
    await process.communicate(download_jobs.encode())


def check_files_exist(remote_files: list[str], local_path: str) -> list[str]:
    local_path_obj = Path(local_path)

    # Get the list of all local file paths relative to local_path
    local_files_relative = set(
        str(f.relative_to(local_path_obj)) for f in local_path_obj.rglob("*")
    )

    # Check if each remote file exists in the local directory
    return list(set(remote_files) - local_files_relative)


def maybe_download_with_pget(
    path: str,
    remote_path: tp.Optional[str] = None,
    remote_filenames: tp.Optional[list[str]] = None,
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
    for _path in remote_filenames:
        path_directory = os.path.dirname(_path)
        if path_directory:
            path_directory = os.path.join(path, path_directory)
            os.makedirs(path_directory, exist_ok=True)

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


def get_mlc_file_list(model_name: str, n_shards: int):
    files_to_download = [
        f"params/params_shard_{shard_idx}.bin" for shard_idx in range(n_shards)
    ]

    files_to_download += [
        f"{model_name}-cuda.so",
        "mod_cache_before_build.pkl",
        "params/mlc-chat-config.json",
        "params/ndarray-cache.json",
        "params/tokenizer.json",
        "params/tokenizer_config.json",
        "params/tokenizer.model",
        "params/config.json",
    ]
    return files_to_download


@dataclasses.dataclass
class Weights:
    local_path: str
    remote_path: str
    remote_files: list[str]


def mlc_kwargs(
    weights: Weights,
    is_chat: bool,
    num_shards: int = 1,
    tokenizer_path: str = None,
    config_overrides: None | dict = None,
):
    mlc_default = {
        "weights": weights,
        "tokenizer_path": tokenizer_path,
        "is_chat": is_chat,
        "num_shards": num_shards,
    }
    if config_overrides:
        mlc_default.update(config_overrides)
    return mlc_default
