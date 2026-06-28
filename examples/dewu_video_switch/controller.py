"""Scene debounce / hysteresis.

Raw per-frame detector output is noisy at 30 Hz — a bare value would make the
front-end video flap. SceneController only commits a new scene once the detector
has proposed it for `confirm_frames` consecutive frames AND the current scene
has been held for at least `min_dwell_s`. It returns the new scene id exactly
once on a committed change, else None.
"""

from __future__ import annotations

import time


class SceneController:
    def __init__(self, confirm_frames: int = 5, min_dwell_s: float = 1.0) -> None:
        self._confirm_frames = max(1, int(confirm_frames))
        self._min_dwell_s = float(min_dwell_s)

        self._current: str | None = None
        self._candidate: str | None = None
        self._candidate_count = 0
        self._last_switch_t = 0.0

    @property
    def current(self) -> str | None:
        return self._current

    def update(self, proposed: str | None) -> str | None:
        """Feed one detector output. Return a scene id only on a committed switch."""
        if proposed is None:
            self._candidate = None
            self._candidate_count = 0
            return None

        if proposed == self._current:
            # Already showing it; reset any pending candidate.
            self._candidate = None
            self._candidate_count = 0
            return None

        if proposed == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = proposed
            self._candidate_count = 1

        if self._candidate_count < self._confirm_frames:
            return None

        now = time.monotonic()
        if self._current is not None and (now - self._last_switch_t) < self._min_dwell_s:
            return None  # too soon since last switch; hold

        # Commit.
        self._current = self._candidate
        self._candidate = None
        self._candidate_count = 0
        self._last_switch_t = now
        return self._current
