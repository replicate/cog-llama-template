import asyncio
import functools
import os
import random
import sys
import time
import typing as t
from io import IOBase
from concurrent.futures import ThreadPoolExecutor
import aiohttp
from yarl import URL
from .utils import check_files_exist, get_loop
from .aio_file import AIOFile

# some important tricks:
# 1. os.sched_getaffinity to get an accurate cpu count in containers
# 2. memoryview for less copies
# 3. keep redirects from the first head
# 4. mmap
# 5. thread for file writes

MIN_CHUNK_SIZE = 1024 * 1024 * 8  # 8mb

global_downloader = None


class Downloader:
    def __init__(self, concurrency: int | None = None) -> None:
        if not concurrency:
            concurrency = len(os.sched_getaffinity(0))
        self.concurrency = concurrency
        self.sem = asyncio.Semaphore(concurrency * 2)
        self.retries = 0
        self.loop = get_loop()
        global global_downloader
        global_downloader = self

    _session: aiohttp.ClientSession | None = None

    @property
    def session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit_per_host=self.concurrency),
                loop=self.loop,
            )
        return self._session

    _threadpool: ThreadPoolExecutor | None = None

    @property
    def threadpool(self) -> ThreadPoolExecutor:
        if not self._threadpool:
            self._threadpool = ThreadPoolExecutor(4)
        return self._threadpool

    async def get_remote_file_size(self, url: str | URL) -> "tuple[URL, int]":
        # try:
        #     direct_url = str(url).replace(
        #         "pbxt.replicate.delivery", "replicate-files.object.lga1.coreweave.com"
        #     )
        #     resp = await self.session.head(direct_url, timeout=5)
        #     if resp.status == 200:
        #         if resp.url != url:
        #             print(f"using {resp.url} instead of {url}")
        #         return resp.url, int(resp.headers["Content-Length"])
        #     print(f"direct link not available {resp}")
        # except (KeyError, asyncio.TimeoutError, aiohttp.ClientError) as e:
        #     print(f"direct link not available: {direct_url} with error {repr(e)}")
        for i in range(3):
            start = time.time()
            headers = {"Retry-Count": str(i)} if i else {}
            try:
                response = await self.session.head(
                    url, allow_redirects=True, headers=headers
                )
                if response.status >= 400:
                    print("HEAD failed:", response, response.headers.items())
                # https://docs.aiohttp.org/en/stable/client_reference.html#aiohttp.ClientResponse.url
                # .url is the url of the final request, as opposed to .real_url
                return response.url, int(response.headers["Content-Length"])
            except KeyError as e:
                print("HEAD failed", repr(e))
                print(response.headers, response)
            except asyncio.TimeoutError:
                print(f"HEAD {url} timed out after {time.time() - start:.4f}")
            except aiohttp.ClientError as e:
                print(f"HEAD {url} {repr(e)}")
            await asyncio.sleep(random.random() / 10)
        raise ValueError(f"Failed to HEAD {url} after multiple retries")

    async def download_chunk(
        self, url: str | URL, start: int, end: int, dest: AIOFile
    ) -> None:
        async with self.sem:
            for i in range(5):
                headers = {"Retry-Count": str(i)} if i else {}
                try:
                    headers |= {"Range": f"bytes={start}-{end}"}
                    async with self.session.get(url, headers=headers) as response:
                        await dest.write(await response.read(), start)
                        return
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    print(f"Error: {e}")
                    await asyncio.sleep(random.random() / 10)  # sleep 0-100ms
        raise ValueError(f"Failed to download {url} after multiple retries")

    files_processed = 0
    total_size = 0

    async def download_file(self, url: str | URL, dest_fd: int = -1) -> IOBase:
        """
        download file into a mmap

        if dest_fd is not passed, an anonymous map is used for a buffer
        """
        self.retries = 0
        url, file_size = await self.get_remote_file_size(url)
        self.total_size += file_size
        # lower this in proportion to how many files are in flight
        # when files > concurrency, splitting is bad
        # # to track requests in flight, except it's either full or 0 when we check:
        # allowed_concurrency = min(self.sem._value + 1, self.concurrency)
        # this way is kind of random but the assumption is the more data has gone over
        # the connection so far, the bigger the TCP window sizes, and the less benefit
        # from using additional connections
        allowed_concurrency = max(1, self.concurrency - self.files_processed // 2)
        self.files_processed += 1
        max_chunks = file_size // (MIN_CHUNK_SIZE * 1) or 1
        concurrency = min(allowed_concurrency, max_chunks)
        chunk_size = file_size // concurrency
        tasks = []
        dest = AIOFile(dest_fd, file_size, self.loop, self.threadpool)
        start_time = time.time()
        for i in range(concurrency):
            start = i * chunk_size
            end = start + chunk_size - 1 if i != concurrency - 1 else file_size - 1
            tasks.append(self.download_chunk(url, start, end, dest))

        await asyncio.gather(*tasks)
        print(
            f"Downloaded {os.path.basename(str(url))} as {concurrency} {chunk_size // 1024}"
            f" kB chunks in {time.time() - start_time:.3f}s with {self.retries} retries"
        )
        self.retries = 0
        return await dest.get_buf()

    async def download_file_to_disk(self, url: str, path: str) -> None:
        fd = os.open(path, os.O_RDWR | os.O_CREAT)
        await self.download_file(url, fd)

    async def maybe_download_files_to_disk(
        self, path: str, remote_path: str, filenames: list[str]
    ) -> None:
        remote_path = remote_path.rstrip("/")
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            missing_files = filenames
        else:
            missing_files = check_files_exist(filenames, path)
        start = time.time()
        coros = [
            self.download_file_to_disk(f"{remote_path}/{f}", f"{path}/{f}")
            for f in missing_files
        ]
        await asyncio.gather(*coros)
        elapsed = time.time() - start
        throughput = self.total_size / elapsed / 1024 / 1024
        print(
            f"downloaded {self.total_size / 1024 / 1024:.2f} MB in {elapsed:.3f}s ({throughput:.2f} MB/s)"
        )
        self.total_size = 0
        self.files_processed = 0  # loras can use a bunch of connections

    def sync(f: t.Callable) -> t.Callable:
        # pylint: disable=no-self-argument
        @functools.wraps(f)
        def wrapper(self: "Downloader", *args: t.Any, **kwargs: t.Any) -> t.Any:
            try:
                return self.loop.run_until_complete(f(self, *args, **kwargs))
            except RuntimeError as e:
                if e.args[0] == "Event loop is closed":
                    print("has to start a new event loop")
                    self.loop = get_loop()
                    self._session = None
                    return self.loop.run_until_complete(f(self, *args, **kwargs))
                if "another loop is running" in e.args[0]:
                    self.loop = get_loop()
                    self._session = None
                    return self.loop.run_until_complete(f(self, *args, **kwargs))
                raise e

        return wrapper

    sync_download_file = sync(download_file)
    sync_maybe_download_files = sync(maybe_download_files_to_disk)


if __name__ == "__main__":
    Downloader().sync_download_file(sys.argv[1])
