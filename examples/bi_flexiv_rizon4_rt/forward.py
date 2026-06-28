"""Forward selected observation channels to a downstream detection machine.

Plan B: the heavy keypoint-detection + seamless video-switching runs on a
SEPARATE machine (Windows / macOS). This subscriber lives in the robot host
laptop's inference process and only forwards the data that machine needs
(robot state + head camera raw image) over a websocket. All detection/render
cost stays off the control laptop.

Design constraints (see xense_client.runtime.runtime.Runtime._step, which calls
subscriber.on_step INSIDE the control loop):
- on_step must NEVER block. It only drops the latest frame into a single-slot
  queue (discarding any stale frame) and returns immediately.
- A daemon thread owns the websocket: it connects, sends, and reconnects on
  failure. Every exception is swallowed so a dead detection machine, a slow
  link, or a network hiccup can never disturb the robot's 30 Hz control loop or
  its home-on-exit motion.

The wire format is msgpack (xense_client.msgpack_numpy), the same codec the
inference channel uses. The detection machine can decode it with a vendored
copy of msgpack_numpy.py (no xense_client dependency required).
"""

import contextlib
import queue
import threading

from typing_extensions import override
import websockets.sync.client
from xense_client import msgpack_numpy
from xense_client.logger import get_logger
from xense_client.runtime import subscriber as _subscriber

logger = get_logger("ForwardSubscriber")


class ForwardSubscriber(_subscriber.Subscriber):
    """Push a slim per-step payload to a downstream websocket receiver.

    Connects OUT to the detection machine (which runs the websocket server), so
    the robot laptop needs no inbound ports. One-way: obs out, nothing read back.
    """

    def __init__(
        self,
        uri: str,
        *,
        cameras: tuple[str, ...] = ("head",),
        send_state: bool = True,
        send_stride: int = 1,
        connect_timeout_s: float = 2.0,
    ) -> None:
        """
        Args:
            uri: ws URI of the detection machine, e.g. "ws://192.168.1.50:9100".
            cameras: which raw cameras to forward from observation["images_raw"].
                Defaults to the head camera only (state + head image is what the
                detector needs; wrists are dropped to save bandwidth).
            send_state: also forward the 20-D robot state (observation["state"]).
            send_stride: forward every Nth step (1 = every step). Use >1 to
                throttle bandwidth if the detector runs slower than 30 Hz.
            connect_timeout_s: per-attempt websocket connect timeout.
        """
        self._uri = uri
        self._cameras = cameras
        self._send_state = send_state
        self._send_stride = max(1, int(send_stride))
        self._connect_timeout_s = connect_timeout_s

        self._packer = msgpack_numpy.Packer()
        # Single-slot: on_step always overwrites, so the sender thread only ever
        # ships the freshest frame and never builds a backlog.
        self._q: queue.Queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._ws = None
        self._step_idx = 0
        self._dropped = 0
        self._sent = 0

        self._worker = threading.Thread(target=self._run, name="forward-sender", daemon=True)
        self._worker.start()
        logger.info(f"ForwardSubscriber → {uri} (cameras={cameras}, state={send_state}, stride={self._send_stride})")

    # ----- Subscriber API (runs in the control loop; keep it cheap) -----

    @override
    def on_episode_start(self) -> None:
        self._step_idx = 0

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        self._step_idx += 1
        if self._step_idx % self._send_stride != 0:
            return

        payload: dict = {"step": self._step_idx}
        if self._send_state and "state" in observation:
            payload["state"] = observation["state"]

        raw = observation.get("images_raw", {})
        imgs = {cam: raw[cam] for cam in self._cameras if cam in raw}
        if imgs:
            payload["images"] = imgs

        # Optionally forward the model's action for overlay/debug downstream.
        act = action.get("actions") if isinstance(action, dict) else None
        if act is not None:
            payload["action"] = act

        # Non-blocking enqueue, drop-oldest. Never blocks the control loop.
        try:
            self._q.put_nowait(payload)
        except queue.Full:
            try:
                self._q.get_nowait()
                self._dropped += 1
            except queue.Empty:
                pass
            with contextlib.suppress(queue.Full):
                self._q.put_nowait(payload)

    @override
    def on_episode_end(self) -> None:
        # Don't tear down the socket between episodes; the detector keeps running.
        logger.info(f"ForwardSubscriber episode end: sent={self._sent}, dropped(stale)={self._dropped}")

    def close(self) -> None:
        self._stop.set()
        self._worker.join(timeout=2.0)
        self._safe_close_ws()

    # ----- Background sender thread (owns the socket; swallows all errors) -----

    def _run(self) -> None:
        self._connect()
        while not self._stop.is_set():
            try:
                payload = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if self._ws is None and not self._connect():
                continue
            try:
                self._ws.send(self._packer.pack(payload))
                self._sent += 1
            except Exception as e:
                logger.warning(f"forward send failed ({e}); reconnecting")
                self._reconnect()
        self._safe_close_ws()

    def _connect(self) -> bool:
        try:
            self._ws = websockets.sync.client.connect(
                self._uri,
                compression=None,
                max_size=None,
                open_timeout=self._connect_timeout_s,
            )
            logger.info(f"ForwardSubscriber connected to {self._uri}")
            return True
        except Exception as e:
            self._ws = None
            logger.debug(f"forward connect failed ({e}); will retry")
            return False

    def _reconnect(self) -> None:
        self._safe_close_ws()
        # Brief backoff without blocking shutdown responsiveness.
        self._stop.wait(0.5)
        if not self._stop.is_set():
            self._connect()

    def _safe_close_ws(self) -> None:
        try:
            if self._ws is not None:
                self._ws.close()
        except Exception:
            pass
        finally:
            self._ws = None


def make_forward_subscriber(
    uri: str,
    *,
    cameras: tuple[str, ...] = ("head",),
    send_state: bool = True,
    send_stride: int = 1,
) -> ForwardSubscriber:
    """Convenience factory mirroring make_recorder_subscriber's style."""
    return ForwardSubscriber(
        uri,
        cameras=cameras,
        send_state=send_state,
        send_stride=send_stride,
    )
