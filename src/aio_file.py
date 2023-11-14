import os
import mmap
import io

# zipfile requires seekable
class SeekableMmap(mmap.mmap):
    def seekable(self) -> bool:
        return True

def write_mmap(view, start, end, buf):
    view[start : end] = buf

class AIOFile:
    def __init__(self, dest_fd, file_size, loop, threadpool):
        self.threadpool = threadpool
        self.loop = loop
        self.memory_backed = dest_fd == -1
        if not self.memory_backed:
            os.ftruncate(dest_fd, file_size)
            os.posix_fallocate(dest_fd, 0, file_size)
        self.buf = SeekableMmap(dest_fd, file_size)
        self.buffer_view = memoryview(self.buf)

    async def write(self, buf, start):
        end = start + len(buf)
        if self.memory_backed:
            self.buffer_view[start : end] = buf
        else:
            await self.loop.run_in_executor(
                self.threadpool,
                write_mmap, self.buffer_view, start, end, buf
            )

    async def get_buf(self) -> io.IOBase:
        self.buf.seek(0)
        return self.buf
