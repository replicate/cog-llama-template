import asyncio
import io
import functools
import os
import random
import shutil
import sys
import time
import typing as t
from concurrent.futures import ThreadPoolExecutor
import aiohttp
from yarl import URL
from .utils import check_files_exist

# some important tricks:
# 1. os.sched_getaffinity to work right in docker
# 2. memoryview for less copies
# 3. keep redirects from the first head
MIN_CHUNK_SIZE = 1024 * 1024  # 1MB

global_downloader = None


class Downloader:
    def __init__(self, concurrency: int | None = None) -> None:
        if not concurrency:
            concurrency = len(os.sched_getaffinity(0))
        self.concurrency = concurrency
        self.sem = asyncio.Semaphore(concurrency * 2)
        self.retries = 0
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
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
            self._threadpool = ThreadPoolExecutor(2)
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
        self, url: str | URL, start: int, end: int, buffer_view: memoryview
    ) -> None:
        async with self.sem:
            for i in range(5):
                headers = {"Retry-Count": str(i)} if i else {}
                try:
                    headers |= {"Range": f"bytes={start}-{end}"}
                    async with self.session.get(url, headers=headers) as response:
                        buffer_view[start : end + 1] = await response.read()
                        return
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    print(f"Error: {e}")
                    await asyncio.sleep(random.random() / 10)  # sleep 0-100ms
        raise ValueError(f"Failed to download {url} after multiple retries")

    files_processed = 0

    async def download_file(self, url: str | URL) -> io.BytesIO:
        self.retries = 0
        url, file_size = await self.get_remote_file_size(url)
        # lower this in proportion to how many files are in flight
        # when files > concurrency, splitting is bad
        # # to track requests in flight, except it's either full or 0 when we check:
        # allowed_concurrency = min(self.sem._value + 1, self.concurrency)
        # this way is kind of random but the assumption is the more data has gone over
        # the connection so far, the bigger the TCP window sizes, and the less benefit
        # from using additional connections
        allowed_concurrency = max(1, self.concurrency - self.files_processed)
        self.files_processed += 1
        max_chunks = file_size // (MIN_CHUNK_SIZE * 1) or 1
        concurrency = min(allowed_concurrency, max_chunks)
        chunk_size = file_size // concurrency
        tasks = []
        buf = io.BytesIO()
        buf.write(b"\0" * file_size)
        buf.seek(0)
        buffer_view = memoryview(buf.getbuffer())
        start_time = time.time()
        for i in range(concurrency):
            start = i * chunk_size
            end = start + chunk_size - 1 if i != concurrency - 1 else file_size - 1
            tasks.append(self.download_chunk(url, start, end, buffer_view))

        await asyncio.gather(*tasks)
        buf.seek(0)
        print(
            f"Downloaded {os.path.basename(str(url))} as {concurrency} {chunk_size // 1024}"
            f" kB chunks in {time.time() - start_time:.4f} with {self.retries} retries"
        )
        self.retries = 0
        return buf

    async def download_file_to_disk(self, url: str, path: str) -> None:
        buf = await self.download_file(url)
        # don't block the event loop for disk io
        await self.loop.run_in_executor(
            self.threadpool,
            lambda: shutil.copyfileobj(buf, open(path, "wb"), length=2 << 18),
        )

    async def maybe_download_files_to_disk(
        self, path: str, remote_path: str, filenames: list[str]
    ) -> None:
        remote_path = remote_path.rstrip("/")
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            missing_files = filenames
        else:
            missing_files = check_files_exist(filenames, path)
        coros = [
            self.download_file_to_disk(f"{remote_path}/{f}", f"{path}/{f}")
            for f in missing_files
        ]
        await asyncio.gather(*coros)
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
                    self.loop = asyncio.new_event_loop()
                    self._session = None
                    return self.loop.run_until_complete(f(self, *args, **kwargs))
                raise e

        return wrapper

    sync_download_file = sync(download_file)
    sync_maybe_download_files = sync(maybe_download_files_to_disk)


if __name__ == "__main__":
    Downloader().sync_download_file(sys.argv[1])
