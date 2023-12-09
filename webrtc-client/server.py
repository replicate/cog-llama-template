# Copyright (c) 2022 Dryad Systems
# pylint: disable=wrong-import-position,unspecified-encoding,unused-argument
import os
import time

server_start = time.time()
if os.getenv("BREAK"):
    time.sleep(60 * 60 * 24)
# import nyacomp

import asyncio
import contextlib
import json
import logging
import sys
import typing as t
import uuid

# from pathlib import Path

import aiortc

import replicate
# import torch
import aiohttp
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription

pc_logger = logging.getLogger("pc")
pc_logger.setLevel("DEBUG")
pcs = set()

logging.getLogger().setLevel("INFO")


class Counter:
    def __init__(self):
        self.count = 0

    @contextlib.contextmanager
    def start(self):
        self.count += 1
        try:
            yield self
        finally:
            self.count -= 1


class Live:
    html = open("index.html").read()

    async def index(self, req: web.Request) -> web.Response:
        return web.Response(body=self.html, content_type="text/html")

    async def js(self, req: web.Request) -> web.Response:
        return web.Response(
            body=open("client.js").read(),
            content_type="application/javascript",
            headers={"Cache-Control": "No-Cache"},
        )

    async def offer(self, request: web.Request) -> web.Response:
        print("!!!")
        print("handling offer")
        offer_data = await request.json()

        st = time.time()
        #model_name = "technillogue/llama-2-7b-chat-hf-mlc"
        model_name = "replicate-internal/llama-2-70b-chat-nix-webrtc-h100"
        model = replicate.models.get(model_name)
        print(model.latest_version.id)
        output = await replicate.async_run(
            #"technillogue/lcm-webrtc:d488a31186c9e6e2e92c89a7d21a7d0553e7e637bc8af4ea6747c7a644aa94ae",
            # "technillogue/llama-2-7b-chat-hf-mlc:87441c3b57069829eb86a9ec74bfa4731d244e36822b347c00e1fa169604792c",
            f"{model_name}:{model.latest_version.id}",
            input={"webrtc_offer": json.dumps(offer_data), "prompt": ""},
        )
        print(f"running prediction took {time.time()-st:.3f}")
        st = time.time()
        answer = next(output)
        print(f"got answer from iterator after {time.time()-st:.3f}")
        return web.Response(content_type="application/json", text=answer)

    async def on_startup(self, app: web.Application) -> None:
        launched = os.getenv("START")
        # await self.llama.async_setup()
        self.cs = cs = aiohttp.ClientSession()
        if launched:
            msg = f"wordmirror started {int(server_start - int(launched))}s after launch, ready {time.time() - server_start:.3f}s after start"
            await cs.post("https://imogen.fly.dev/admin", data=msg)
        req = await cs.get("https://ipinfo.io", headers={"User-Agent": "curl"})
        self.ipinfo = await req.json()
        if "city" in self.ipinfo and "region" in self.ipinfo:
            loc = ", ".join((self.ipinfo["city"], self.ipinfo["region"]))
            self.html = self.html.replace("<!--$LOC-->", f"location: {loc}")
            logging.info(f"got location: {self.ipinfo}")
        else:
            logging.info(f"couldn't get location: {self.ipinfo}")
        # idle exit needs to be in a task because all on_startups have to exit
        # asyncio.create_task(self.idle_exit())

    last_gen = time.time()

    # async def idle_exit(self) -> None:
    #     pod_id = os.getenv("RUNPOD_POD_ID")
    #     while pod_id:
    #         await asyncio.sleep(20 * 60)
    #         if time.time() - self.last_gen > 3600:
    #             await self.cs.post(
    #                 "https://imogen.fly.dev/admin",
    #                 data="mirror shutting down after 20m inactivity",
    #             )
    #             # TODO: if we don't have a volume, exit instead of suspend
    #             # query = 'mutation {podTerminate(input: {podId: "%s"})}' % pod_id
    #             query = 'mutation {podStop(input: {podId: "%s"})}' % pod_id
    #             await self.cs.post(
    #                 "https://api.runpod.io/graphql",
    #                 params={"api_key": os.getenv("RUNPOD_API_KEY")},
    #                 json={"query": query},
    #                 headers={"Content-Type": "application/json"},
    #             )
    #             sys.exit()


    # async def ws_only(self, req: web.Request) -> web.Response:
    #     return web.FileResponse("./ws-only.html")

    # async def next_index(self, req: web.Request) -> web.Response:
    #     return web.FileResponse("/app/next/index.html")

    async def conn_count(self, req: web.Request) -> web.Response:
        return web.Response(body=str(len(pcs) + len(self.connections)))


app = web.Application()
live = Live()
app.on_startup.append(live.on_startup)
app.add_routes(
    [
        # web.route("*", "/plain", live.index),
        web.route("*", "/client.js", live.js),
        web.post("/offer", live.offer),
        # web.get("/ws", live.handle_ws),
        # web.get("/ws-only", live.ws_only),
        web.route("*", "/", live.index),
        # web.route("*", "/", live.next_index),
        # web.static("/", "/app/next"),
    ]
)

if __name__ == "__main__":
    web.run_app(app, port=8080, host="0.0.0.0")
