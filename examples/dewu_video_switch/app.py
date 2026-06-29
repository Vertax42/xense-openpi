#!/usr/bin/env python
"""Video-playback laptop app for the dewu seamless video-switch demo (Plan B).

Topology
--------
    robot laptop (examples.bi_flexiv_rizon4_rt.main --forward)
        │  ws push  {state, head image}  (msgpack)
        ▼
    THIS PROCESS  (the video-playback laptop)
        ├─ obs ws server   :9100   ← laptop connects here, pushes frames
        ├─ detector (thread pool)  → SceneController (debounce)
        ├─ switch ws server :9101  → browsers connect, receive {"scene": id}
        └─ static http      :8080  → serves web/ (the seamless player)

Why three listeners: the laptop is a ws *client* (so it needs no inbound port),
the browser is a ws *client*, and the player HTML is plain static files. Detection
runs in a thread-pool executor on the *latest* frame only (older frames are
dropped while busy) so a heavy keypoint model never stalls the switch broadcast.

Run:
    python -m examples.dewu_video_switch.app          # from the openpi repo root
    # or, copied standalone onto the video-playback laptop:
    python app.py
Then open  http://<this-machine-ip>:8080  on the display.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import functools
import http.server
import json
import pathlib
import threading

# Support both `python -m examples.dewu_video_switch.app` and standalone `python app.py`.
try:
    from examples.dewu_video_switch import detector as _detector
    from examples.dewu_video_switch import msgpack_numpy
    from examples.dewu_video_switch.controller import SceneController
except ImportError:  # standalone copy on the video-playback laptop
    from controller import SceneController  # type: ignore
    import detector as _detector  # type: ignore
    import msgpack_numpy  # type: ignore

import websockets

WEB_DIR = pathlib.Path(__file__).parent / "web"


class LatestFrame:
    """Single-slot frame holder: producer overwrites, consumer drops staleness."""

    def __init__(self) -> None:
        self._frame: dict | None = None
        self._event = asyncio.Event()

    def put(self, frame: dict) -> None:
        self._frame = frame
        self._event.set()

    async def get(self) -> dict:
        await self._event.wait()
        self._event.clear()
        assert self._frame is not None
        return self._frame


class App:
    def __init__(self, detector: _detector.Detector, controller: SceneController) -> None:
        self._detector = detector
        self._controller = controller
        self._latest = LatestFrame()
        self._frontends: set = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    # ----- obs ingress (from robot laptop) -----

    async def _obs_handler(self, websocket) -> None:
        peer = getattr(websocket, "remote_address", "?")
        print(f"[obs] laptop connected: {peer}")
        try:
            async for message in websocket:
                if isinstance(message, str):
                    continue
                try:
                    frame = msgpack_numpy.unpackb(message)
                except Exception as e:
                    print(f"[obs] decode error: {e}")
                    continue
                self._latest.put(frame)
        except websockets.ConnectionClosed:
            pass
        finally:
            print(f"[obs] laptop disconnected: {peer}")

    # ----- switch egress (to browsers) -----

    async def _switch_handler(self, websocket) -> None:
        self._frontends.add(websocket)
        print(f"[switch] browser connected ({len(self._frontends)} total)")
        # Send the current scene immediately so a late-joining display syncs up.
        if self._controller.current is not None:
            await self._safe_send(websocket, self._controller.current)
        try:
            await websocket.wait_closed()
        finally:
            self._frontends.discard(websocket)
            print(f"[switch] browser disconnected ({len(self._frontends)} total)")

    async def _broadcast(self, scene: str) -> None:
        if not self._frontends:
            return
        await asyncio.gather(*(self._safe_send(ws, scene) for ws in list(self._frontends)))

    @staticmethod
    async def _safe_send(ws, scene: str) -> None:
        with contextlib.suppress(Exception):
            await ws.send(json.dumps({"scene": scene}))

    # ----- detection loop -----

    async def _detect_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            frame = await self._latest.get()
            # Run (possibly heavy) detection off the event loop.
            proposed = await loop.run_in_executor(None, self._detector.detect, frame)
            committed = self._controller.update(proposed)
            if committed is not None:
                print(f"[scene] → {committed}")
                await self._broadcast(committed)

    # ----- orchestration -----

    async def run(self, obs_port: int, switch_port: int) -> None:
        self._loop = asyncio.get_running_loop()
        async with (
            websockets.serve(self._obs_handler, "0.0.0.0", obs_port, max_size=None, compression=None),
            websockets.serve(self._switch_handler, "0.0.0.0", switch_port, max_size=None),
        ):
            print(f"[ws] obs server   on :{obs_port}  (robot laptop connects here)")
            print(f"[ws] switch server on :{switch_port} (browsers connect here)")
            await self._detect_loop()


def _serve_static(http_port: int) -> None:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(WEB_DIR))
    httpd = http.server.ThreadingHTTPServer(("0.0.0.0", http_port), handler)
    print(f"[http] player at http://0.0.0.0:{http_port}  (serving {WEB_DIR})")
    httpd.serve_forever()


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Detection + seamless video-switch app for the video-playback laptop (machine ③). "
            "Receives forwarded robot obs (head image + state) from the robot laptop, runs the "
            "detector + scene debounce, and pushes scene switches to the in-browser player."
        ),
        epilog=(
            "Three listeners start together: obs ws (robot laptop connects in), switch ws "
            "(browsers connect in), and static http (the player). Open http://<this-host>:<http-port> "
            "on the display. Feed it with examples.dewu_video_switch.replay_lerobot (real dataset) "
            "or .sim_laptop (synthetic link test)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--obs-port", type=int, default=9100,
        help="TCP port of the obs WebSocket server the robot laptop's ForwardSubscriber pushes frames into.",
    )
    p.add_argument(
        "--switch-port", type=int, default=9101,
        help="TCP port of the switch WebSocket server browsers connect to; it broadcasts {\"scene\": id} on each committed switch.",
    )
    p.add_argument(
        "--http-port", type=int, default=8080,
        help="TCP port of the static HTTP server for web/ (the seamless player HTML and the video clips).",
    )
    p.add_argument(
        "--detector", default="gripper",
        help="Which detector to run: 'gripper' = real grasp-state detector from the robot state; "
        "'stub' = brightness buckets, for the synthetic link test only. See detector.make_detector.",
    )
    p.add_argument(
        "--confirm-frames", type=int, default=5,
        help="Debounce: a proposed scene must repeat for this many consecutive frames before it may commit. "
        "Higher = steadier but slower to react.",
    )
    p.add_argument(
        "--min-dwell-s", type=float, default=1.0,
        help="Debounce: minimum seconds to hold the current scene before another switch is allowed. "
        "Raise to ~2.5 to suppress brief flicker on a marketing display.",
    )
    args = p.parse_args()

    detector = _detector.make_detector(args.detector)
    controller = SceneController(confirm_frames=args.confirm_frames, min_dwell_s=args.min_dwell_s)
    app = App(detector, controller)

    threading.Thread(target=_serve_static, args=(args.http_port,), daemon=True).start()
    try:
        asyncio.run(app.run(args.obs_port, args.switch_port))
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
