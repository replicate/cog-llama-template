import asyncio
import time
from typing import Callable, Iterator, List, Optional

from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
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


async def accept_offer(
    offer: str,
    handler: Callable[[str | bytes], Iterator[str | bytes]],
    ice_servers: str,
) -> tuple[str, asyncio.Event]:

@dataclasses.dataclass
class RTC:
    offer: str
    ice_servers: str

    def on_message(self, f: Callable[[str | bytes], Iterator[str | bytes]) -> None:
        self.handler = f

    async def answer(self) -> str:
        print("handling offer")
        params = json.loads(self.offer)

        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        print("creating for", offer)
        config = RTCConfiguration([RTCIceServer(**a) for a in json.loads(self.ice_servers)])
        print("configured for", ice_servers)
        pc = RTCPeerConnection(configuration=config)
        print("made peerconnection", pc)

        # five seconds to establish a connection and ping!
        self.done = ShutdownTimer()

        @pc.on("datachannel")
        def on_datachannel(channel: aiortc.rtcdatachannel.RTCDataChannel) -> None:
            print(type(channel))

            @channel.on("message")
            async def on_message(message) -> None:
                print(message)
                if isinstance(message, str) and message.startswith("ping"):
                    channel.send(f"pong{message[4:]} {round(time.time() * 1000)}")
                    done.reset()
                else:
                    for result in self.handler(message):
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
        return json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        )

    async def wait_disconnect(self) -> str:
        await self.done()
        return "disconnected"
