"""Pluggable keypoint detector for the dewu video-switch demo.

Contract
--------
A detector receives one forwarded frame and returns a *scene id* (a short
string) or None if it cannot decide this frame. The scene id is what drives the
front-end video switch (see web/index.html SCENES).

frame = {
    "step":   int,
    "state":  np.ndarray shape (20,)            # robot TCP state, optional
    "images": {"head": np.ndarray (H, W, 3) uint8}  # raw head camera, optional
    "action": np.ndarray (20,)                  # model action, optional
}

The real product detector (keypoint model over the head image) plugs in by
subclassing Detector and implementing detect(). Swap it in app.py. The
StubDetector below is a dependency-free placeholder so the whole pipeline runs
end-to-end before the real model is wired in — replace it, don't ship it.
"""

from __future__ import annotations

import abc

import numpy as np


class Detector(abc.ABC):
    """Maps a forwarded frame to a scene id (or None = undecided)."""

    @abc.abstractmethod
    def detect(self, frame: dict) -> str | None:
        ...


class StubDetector(Detector):
    """Placeholder: derive a scene from a coarse signal, no ML deps.

    Heuristic (PLACEHOLDER — replace with the real keypoint model):
    - If a head image is present, threshold its mean brightness into buckets.
    - Else fall back to the gripper state (last 2 dims of `state`).
    This exists only to prove the transport + switching loop; the scene ids it
    emits ("scene_a"/"scene_b"/"scene_c") match web/index.html's SCENES keys.
    """

    def __init__(self, scene_ids: tuple[str, ...] = ("scene_a", "scene_b", "scene_c", "scene_d")) -> None:
        if not scene_ids:
            raise ValueError("scene_ids must be non-empty")
        self._scenes = scene_ids

    def detect(self, frame: dict) -> str | None:
        imgs = frame.get("images") or {}
        head = imgs.get("head")
        if isinstance(head, np.ndarray) and head.size > 0:
            brightness = float(head.mean()) / 255.0  # 0..1
            idx = min(int(brightness * len(self._scenes)), len(self._scenes) - 1)
            return self._scenes[idx]

        state = frame.get("state")
        if isinstance(state, np.ndarray) and state.shape[-1] >= 2:
            grip = float(state[..., -1])  # right gripper as a crude proxy
            idx = min(int(abs(grip) * len(self._scenes)), len(self._scenes) - 1)
            return self._scenes[idx]

        return None


def make_detector(name: str = "stub", **kwargs) -> Detector:
    """Factory so app.py can select a detector by name / config.

    Register real detectors here, e.g.:
        if name == "keypoint":
            from dewu_video_switch.keypoint_detector import KeypointDetector
            return KeypointDetector(**kwargs)
    """
    if name == "stub":
        return StubDetector(**kwargs)
    raise ValueError(f"unknown detector: {name!r}")
