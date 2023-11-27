import asyncio
import dataclasses
import json
import time
from typing import Callable, Iterator

from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
    RTCDataChannel,
)


class ShutdownTimer(asyncio.Event):
    def __init__(self, timeout: int = 5) -> None:
        self.deadline = time.monotonic() + timeout
        self.timeout = timeout
        self.task = asyncio.create_task(self.exit())
        super().__init__()

    def reset(self) -> None:
        self.deadline = time.monotonic() + self.timeout

    async def exit(self) -> None:
        while not self.is_set():
            await asyncio.sleep(self.deadline - time.monotonic())
            if self.deadline < time.monotonic():
                print("ping deadline exceeded")
                self.set()


@dataclasses.dataclass
class RTC:
    offer: str

    def on_message(self, f: Callable[[dict], Iterator[dict]]) -> None:
        self.wrapped_message_handler = f

    def message_handler(self, message: bytes | str) -> Iterator[bytes | str]:
        if message[0] != "{":
            print("received invalid message", message)
            return
        args = json.loads(message)  # works for bytes or str
        id = args.pop("id", 0)
        for result in self.wrapped_message_handler(args):
            result["id"] = id
            yield json.dumps(result)

    def serve_with_loop(self, loop: asyncio.AbstractEventLoop) -> Iterator[str]:
        """
        this is so that you can do `yield from rtc.serve_with_loop(loop)`

        you can't `yield from` in an async function, so in that case the caller
        would need to do `yield await rtc.answer(); yield await rtc.wait_disconnect()`
        """
        yield loop.run_until_complete(self.answer())
        yield loop.run_until_complete(self.wait_disconnect())

    async def answer(self) -> str:
        print("handling offer")
        params = json.loads(self.offer)
        ice_servers = params.get("ice_servers", "[]")

        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        print("creating for", offer)
        config = RTCConfiguration([RTCIceServer(**a) for a in ice_servers])
        print("configured for", ice_servers)
        pc = RTCPeerConnection(configuration=config)
        print("made peerconnection", pc)

        # five seconds to establish a connection and ping!
        self.done = ShutdownTimer()

        @pc.on("datachannel")
        def on_datachannel(channel: RTCDataChannel) -> None:
            print(type(channel))

            @channel.on("message")
            async def on_message(message: str | bytes) -> None:
                print(message)
                if isinstance(message, str) and message.startswith("ping"):
                    # recepient can use our time + rt ping latency to estimate clock drift
                    # if they send time as the ping message and record received time,
                    # drift = (their time) - ((time we sent) + (roundtrip latency) / 2) 
                    channel.send(f"pong{message[4:]} {round(time.time() * 1000)}")
                    self.done.reset()
                else:
                    for result in self.message_handler(message):
                        channel.send(result)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            print("Connection state is %s", pc.connectionState)
            if pc.connectionState == "failed":
                await pc.close()
                self.done.set()

        # handle offer
        await pc.setRemoteDescription(offer)
        print("set remote description")

        # send answer
        answer = await pc.createAnswer()
        print("created answer", answer)
        await pc.setLocalDescription(answer)
        print("set local description")
        data = {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        return json.dumps(data)

    async def wait_disconnect(self) -> str:
        await self.done.wait()
        return "disconnected"
