import os
import subprocess
import random
import time
import typing as tp
import asyncio

class Logger:
    def __init__(self, marker: str = 'predict-timings'):
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


def check_files_exist(remote_files, local_path):
    # Get the list of local file names
    local_files = os.listdir(local_path)
    
    # Check if each remote file exists in the local directory
    missing_files = [file for file in remote_files if file not in local_files]
    
    return missing_files

async def download_file_with_pget(remote_path, dest_path):
    # Create the subprocess
    process = await asyncio.create_subprocess_exec(
        'pget', remote_path, dest_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    # Wait for the subprocess to finish
    stdout, stderr = await process.communicate()

    # Print what the subprocess output (if any)
    if stdout:
        print(f'[stdout]\n{stdout.decode()}')
    if stderr:
        print(f'[stderr]\n{stderr.decode()}')

async def download_files_with_pget(remote_path, path, files):
    await asyncio.gather(*(download_file_with_pget(f"{remote_path}/{file}", f"{path}/{file}") for file in files))

    # # Run the bash script for each missing file 
    # process = subprocess.Popen(["./src/download-with-pget.sh", remote_path, path, *files])
    # process.wait()

def maybe_download_with_pget(
    path, 
    remote_path: tp.Optional[str] = None, 
    remote_filenames: tp.Optional[tp.List[str]] = [],
    logger: tp.Optional[Logger] = None):
    """
    Downloads files from remote_path to path if they are not present in path. File paths are constructed 
    by concatenating remote_path and remote_filenames. If remote_path is None, files are not downloaded.

    Args:
        path (str): Path to the directory where files should be downloaded
        remote_path (str): Path to the directory where files should be downloaded from
        remote_filenames (List[str]): List of file names to download
        logger (Logger): Logger object to log progress
    
    Returns:
        path (str): Path to the directory where files were downloaded
    
    Example:

        maybe_download_with_pget(
            path="models/roberta-base",
            remote_path="gs://my-bucket/models/roberta-base",
            remote_filenames=["config.json", "pytorch_model.bin", "tokenizer.json", "vocab.json"],
            logger=logger
        )
    """
    if remote_path:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            missing_files = remote_filenames
        else:
            local_files = os.listdir(path)
            missing_files = check_files_exist(remote_filenames, path)

        if len(missing_files) > 0:
            print('Downloading weights...')
            st = time.time()
            if logger:
                logger.info(f"Downloading {missing_files} from {remote_path} to {path}")
            asyncio.run(download_files_with_pget(remote_path, path, missing_files))
            if logger:
                logger.info(f"Finished download")
            print(f"Finished download in {time.time() - st:.2f}s")


    return path



