"""Per-shoe state-machine detector — the production detection logic.

States 0..4: 0 = standby / reset (arm at init), 1..4 = currently working shoe N.

  - Pick event (advances state N-1 -> N): the RIGHT TCP enters a 3D bounding box
    (the shoebox), grasps inside, then exits the box while still grasping
    (pose_events.BoxGraspEdgeEvent). Each pick -> scene "detecting".
  - Blue event (within states 1..4): an OpenCV check on the head image runs about
    once per second (vision_stride frames); when the blue insole bottom appears
    -> scene "next". Re-armed when a new shoe starts.
  - Reset (4 -> 0): in the final state, when BOTH TCPs return to the init/home
    pose (the 4-shoe episode is done and the arm homes) -> scene "standby".

Emission model (Option A): detect() returns the LEVEL scene id that should be
showing *now*, every frame. It is internally idempotent and only changes the
level when an event fires, so app.py's SceneController can debounce/min-dwell on
top unchanged. The three scene ids must match web/index.html SCENES.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

# Support `python -m examples.dewu_video_switch.app` and standalone `python app.py`.
try:
    from examples.dewu_video_switch.detector import Detector
    from examples.dewu_video_switch.pose_events import (
        LEFT_TCP_XYZ,
        RIGHT_TCP_XYZ,
        BoundingBox3D,
        BoxGraspEdgeEvent,
        HomePose,
    )
    from examples.dewu_video_switch.vision import BlueInsoleConfig
except ImportError:  # standalone copy on the video-playback laptop
    from detector import Detector  # type: ignore
    from pose_events import (  # type: ignore
        LEFT_TCP_XYZ,
        RIGHT_TCP_XYZ,
        BoundingBox3D,
        BoxGraspEdgeEvent,
        HomePose,
    )
    from vision import BlueInsoleConfig  # type: ignore

SCENE_STANDBY = "standby"
SCENE_DETECTING = "detecting"
SCENE_NEXT = "next"


@dataclass
class TransitionConfig:
    """One pick transition: which box/arm/gripper defines it."""

    box: BoundingBox3D
    tcp_idx: tuple = RIGHT_TCP_XYZ  # right TCP by default (data-driven per shoe later)
    grip_idx: int = 19
    grasp_is_low: bool = True
    grasp_threshold: float | None = None  # None => self-calibrate


@dataclass
class ShoeSMConfig:
    n_shoes: int = 4
    transitions: list = field(default_factory=list)  # len == n_shoes (one box per pick)
    blue: BlueInsoleConfig = field(default_factory=BlueInsoleConfig)
    vision_stride: int = 30   # run the blue check every N frames (~1 Hz @ 30 fps)
    vision_confirm: int = 2   # consecutive positive blue checks before firing
    home_tol: float = 0.05    # meters; both TCPs within tol of init => home
    home_pose: HomePose | None = None  # None => auto-capture from initial state-0 frames

    @classmethod
    def from_json(cls, path: str) -> "ShoeSMConfig":
        """Overlay a JSON config onto the defaults. See shoe_sm.example.json.

        Schema (all optional except a box source):
            n_shoes, vision_stride, vision_confirm, home_tol
            pick_box:   {x_min,x_max,y_min,y_max,z_min,z_max}   # reused for all picks
            pick_boxes: [ {..6..}, ... ]                        # one per pick (overrides pick_box)
            grip_idx, grasp_is_low, grasp_threshold            # applied to every transition
            blue: {h_lo,h_hi,s_lo,s_hi,v_lo,v_hi,min_area_frac,roi,open_ksize,close_ksize}
            home_pose: {left_xyz:[x,y,z], right_xyz:[x,y,z]}
        """
        with open(path) as f:
            d = json.load(f)
        cfg = cls()
        cfg.n_shoes = int(d.get("n_shoes", cfg.n_shoes))
        cfg.vision_stride = int(d.get("vision_stride", cfg.vision_stride))
        cfg.vision_confirm = int(d.get("vision_confirm", cfg.vision_confirm))
        cfg.home_tol = float(d.get("home_tol", cfg.home_tol))

        def _box(b):
            return BoundingBox3D(b["x_min"], b["x_max"], b["y_min"], b["y_max"], b["z_min"], b["z_max"])

        grip_idx = int(d.get("grip_idx", 19))
        grasp_is_low = bool(d.get("grasp_is_low", True))
        grasp_threshold = d.get("grasp_threshold")
        tcp_idx = tuple(d.get("tcp_idx", RIGHT_TCP_XYZ))
        if "pick_boxes" in d:
            boxes = [_box(b) for b in d["pick_boxes"]]
        elif "pick_box" in d:
            boxes = [_box(d["pick_box"])] * cfg.n_shoes
        else:
            boxes = []  # falls back to placeholder default below
        cfg.transitions = [
            TransitionConfig(box=b, tcp_idx=tcp_idx, grip_idx=grip_idx,
                             grasp_is_low=grasp_is_low, grasp_threshold=grasp_threshold)
            for b in boxes
        ]

        if "blue" in d:
            cfg.blue = BlueInsoleConfig(**{k: (tuple(v) if k == "roi" and v is not None else v)
                                           for k, v in d["blue"].items()})
        if "home_pose" in d:
            hp = d["home_pose"]
            cfg.home_pose = HomePose(tuple(hp["left_xyz"]), tuple(hp["right_xyz"]), cfg.home_tol)
        return cfg


def _placeholder_config() -> ShoeSMConfig:
    """Default with a PLACEHOLDER pick box — runs, but must be calibrated.

    The box below is a loose guess in robot-base meters; replace via a tuned
    shoe_sm.json (see replay_offline.py --calibrate-tcp).
    """
    cfg = ShoeSMConfig()
    box = BoundingBox3D(x_min=0.45, x_max=0.95, y_min=-0.35, y_max=0.35, z_min=-0.40, z_max=0.05)
    cfg.transitions = [TransitionConfig(box=box) for _ in range(cfg.n_shoes)]
    return cfg


class ShoeStateMachineDetector(Detector):
    def __init__(self, cfg: ShoeSMConfig | None = None, config_path: str | None = None) -> None:
        if config_path:
            cfg = ShoeSMConfig.from_json(config_path)
        cfg = cfg or _placeholder_config()
        if not cfg.transitions:
            cfg.transitions = _placeholder_config().transitions
        self.cfg = cfg

        self._picks = [
            BoxGraspEdgeEvent(box=t.box, tcp_idx=t.tcp_idx, grip_idx=t.grip_idx,
                              grasp_is_low=t.grasp_is_low, grasp_threshold=t.grasp_threshold)
            for t in cfg.transitions
        ]
        self._blue = None  # lazy (needs cv2)
        self._home = cfg.home_pose
        self._home_capture: list = []

        self._state = 0
        self._blue_fired = False
        self._blue_pos_count = 0
        self._frame_i = 0

    # ----- introspection (used by replay_offline timeline) -----
    @property
    def state(self) -> int:
        return self._state

    def _ensure_blue(self):
        if self._blue is None:
            try:
                from examples.dewu_video_switch.vision import BlueInsoleDetector
            except ImportError:
                from vision import BlueInsoleDetector  # type: ignore
            self._blue = BlueInsoleDetector(self.cfg.blue)
        return self._blue

    def _level(self) -> str:
        if self._state == 0:
            return SCENE_STANDBY
        return SCENE_NEXT if self._blue_fired else SCENE_DETECTING

    def _reset_cycle(self) -> None:
        self._state = 0
        self._blue_fired = False
        self._blue_pos_count = 0
        for p in self._picks:
            p.reset()

    def detect(self, frame: dict) -> str | None:
        self._frame_i += 1
        state = frame.get("state")
        if not isinstance(state, np.ndarray) or state.shape[-1] < 20:
            return self._level()

        # Auto-capture the home pose from the first state-0 frames (arm at init).
        if self._home is None and self._state == 0 and len(self._home_capture) < 5:
            self._home_capture.append(np.asarray(state, dtype=float))
            if len(self._home_capture) == 5:
                med = np.median(np.stack(self._home_capture), axis=0)
                self._home = HomePose(
                    left_xyz=tuple(float(med[i]) for i in LEFT_TCP_XYZ),
                    right_xyz=tuple(float(med[i]) for i in RIGHT_TCP_XYZ),
                    tol=self.cfg.home_tol,
                )

        # Pick event: advance to the next shoe state.
        if self._state < self.cfg.n_shoes:
            if self._picks[self._state].update(state):
                self._state += 1
                self._blue_fired = False
                self._blue_pos_count = 0
                return self._level()

        # Blue event (~vision_stride): only within an active shoe state, once each.
        if 1 <= self._state <= self.cfg.n_shoes and not self._blue_fired:
            if self._frame_i % max(1, self.cfg.vision_stride) == 0:
                head = (frame.get("images") or {}).get("head")
                if isinstance(head, np.ndarray) and head.ndim == 3:
                    present, _dbg = self._ensure_blue().detect(head)
                    self._blue_pos_count = self._blue_pos_count + 1 if present else 0
                    if self._blue_pos_count >= self.cfg.vision_confirm:
                        self._blue_fired = True
                        return self._level()

        # Reset 4 -> 0 when the arm returns to the init/home pose (episode done).
        if self._state == self.cfg.n_shoes and self._home is not None and self._home.at_home(state):
            self._reset_cycle()

        return self._level()
