import os
import random
import statistics as stats
import time
from typing import Optional

from cog import BasePredictor, Input

from config import mlc_weights as weights
from src import download
from src.download import Downloader, Method
from src import utils

download_args = (weights.local_path, weights.remote_path, weights.remote_files)


def time_down(method: Method):
    download.write_method = method
    downloader = Downloader()
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


class Predictor(BasePredictor):
    def predict(
        self,
        method: str = Input(
            default="ALL", choices=["DEST_MMAP", "MEMFD_SENDFILE", "ANON_MMAP_COPYFILE", "PGET", "ALL"]
        ),
        repetitions: int = 2,
    ) -> str:
        methods = list(Method) if method == "ALL" else [Method[method]]
        jobs = [m for m in methods for _ in range(repetitions)]
        # minimize order effects
        random.shuffle(jobs)
        method_times = {m: [] for m in methods}
        for m in jobs:
            method_times[m].append(time_down(m))
        results = []
        for m, times in sorted(method_times.items()):
            info = {
                "mean": stats.mean(times),
                "stdev": stats.stdev(times) if len(times) > 1 else 0,
                "min": min(times),
            }
            msg = f"{m}: " + " ".join(f"{k}: {v:.3f}" for k, v in info.items())
            results.append(msg)
        return "\n".join(results)
