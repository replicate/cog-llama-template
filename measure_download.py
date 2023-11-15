import os
import random
import statistics as stats
import time
from typing import Optional, Union

from cog import BasePredictor, Input

from config import mlc_weights as weights
from src import download
from src.download import Downloader, Method
from src import utils

download_args = (weights.local_path, weights.remote_path, weights.remote_files)


def time_down(method: Method, args):
    download.write_method = method
    downloader = Downloader(**args)
    start = time.perf_counter()
    print("downloading with method", method)
    if method == Method.PGET:
        utils.maybe_download_with_pget(*download_args)
    else:
        downloader.sync_maybe_download_files(*download_args)
    elapsed = time.perf_counter() - start
    for p in weights.remote_files:
        os.remove(f"{weights.local_path}/{p}")
    del downloader
    return elapsed


choices = [
    "DEST_MMAP",
    "MEMFD_SENDFILE",
    "MEMFD_SENDFILE_REUSE",
    "ANON_MMAP_COPYFILE",
    "PGET",
    "ALL",
]


class Predictor(BasePredictor):
    def predict(
        self,
        method: str = Input(default="ALL", choices=choices),
        repetitions: int = 2,
        concurrency: int = Input(default=-1),
        semaphore_multiplier: int = 2,
        output_format: str = Input(default="text", choices=["text", "json"]),
    ) -> Union[str, dict]:
        methods = list(Method) if method == "ALL" else [Method[method]]
        jobs = [m for m in methods for _ in range(repetitions)]
        # minimize order effects
        random.shuffle(jobs)
        method_times = {m: [] for m in methods}
        args = {
            "concurrency": concurrency if concurrency > -1 else None,
            "sem": semaphore_multiplier,
        }
        for m in jobs:
            method_times[m].append(time_down(m, args))
        if output_format == "json":
            return {str(k): v for k, v in method_times.items()}
        results = []
        for m, times in sorted(method_times.items(), key=str):
            info = {
                "mean": stats.mean(times),
                "stdev": stats.stdev(times) if len(times) > 1 else 0,
                "min": min(times),
            }
            msg = f"{m}: " + " ".join(f"{k}: {v:.3f}" for k, v in info.items())
            results.append(msg)
        return "\n".join(results)
