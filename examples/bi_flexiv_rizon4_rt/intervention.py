"""Pico4 human intervention for BiFlexivRizon4RT inference.

Dual-grip (both side buttons held simultaneously) swaps the arm target from
policy output to the Pico4 controller pose; releasing either grip hands control
back to the policy and clears the ActionChunkBroker so the next step re-infers
from the current observation.

The module exposes three collaborating pieces:

* ``Pico4InterventionController`` — owns the BiPico4 teleop, decides whether
  intervention is active on each tick, and keeps the teleop's internal target
  pose synced to the real robot pose while the policy drives (without this the
  first post-release frame would snap the arm to the teleop's stale target).
* ``InterventionEnvironmentWrapper`` — Environment wrapper that polls the
  controller in ``get_observation`` and swaps the action in ``apply_action``.
* ``InterventionPolicyAgent`` — Agent wrapper that skips ``policy.infer`` while
  intervention is active (saves a WebSocket round-trip) and calls
  ``ActionChunkBroker.reset`` on the release edge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lerobot.utils.robot_utils import get_logger
import numpy as np
from typing_extensions import override
from xense_client import action_chunk_broker as _action_chunk_broker
from xense_client.runtime import agent as _agent
from xense_client.runtime import environment as _environment
from xense_client.runtime.agents import policy_agent as _policy_agent

if TYPE_CHECKING:
    from lerobot.teleoperators.bi_pico4 import BiPico4

    import examples.bi_flexiv_rizon4_rt.env as _env

logger = get_logger("Pico4Intervention")


# Packing order for the 20D action vector consumed by real_env.step.
# Mirrors _ACTION_LABELS in examples/bi_flexiv_rizon4_rt/main.py and the
# dict packing in examples/bi_flexiv_rizon4_rt/real_env.py.
_ACTION_KEYS_IN_ORDER: tuple[str, ...] = (
    "left_tcp.x",
    "left_tcp.y",
    "left_tcp.z",
    "left_tcp.r1",
    "left_tcp.r2",
    "left_tcp.r3",
    "left_tcp.r4",
    "left_tcp.r5",
    "left_tcp.r6",
    "right_tcp.x",
    "right_tcp.y",
    "right_tcp.z",
    "right_tcp.r1",
    "right_tcp.r2",
    "right_tcp.r3",
    "right_tcp.r4",
    "right_tcp.r5",
    "right_tcp.r6",
    "left_gripper.pos",
    "right_gripper.pos",
)


class Pico4InterventionController:
    """Owns the BiPico4 teleop and tracks intervention mode each tick."""

    def __init__(self, teleop: BiPico4, base_env: _env.BiFlexivRizon4RTEnvironment) -> None:
        self._teleop = teleop
        self._base_env = base_env
        self._active = False
        self._was_active = False
        self._pending_release = False

    @property
    def active(self) -> bool:
        return self._active

    def start(self) -> None:
        """Connect the BiPico4 to the already-running robot.

        Must be called after the underlying robot is connected (so the initial
        TCP pose is readable).  pre_init overlaps XenseVR SDK setup with robot
        init; here the robot is already up, so we just call connect directly.
        """
        left_pose, right_pose = self._base_env._env.robot.get_current_tcp_pose_quat()
        logger.info("Connecting BiPico4 teleop for intervention...")
        self._teleop.connect(
            calibrate=False,
            left_tcp_pose_quat=left_pose,
            right_tcp_pose_quat=right_pose,
        )
        logger.info("BiPico4 intervention armed (hold both grips to take over).")

    def poll_and_decide(self) -> bool:
        """Refresh intervention state; return True if we should override policy.

        Called once per control step from the environment wrapper's
        ``get_observation``.  When not intervening, resyncs the teleop's target
        pose to the real robot pose so a later handoff starts from the same
        point and does not jump.
        """
        left_pico = self._teleop._left_pico4
        right_pico = self._teleop._right_pico4
        xrt = self._teleop._xrt
        if xrt is None:
            # Teleop not connected: treat as never intervening.
            self._was_active = self._active
            self._active = False
            return False

        left_grip = float(xrt.get_left_grip())
        right_grip = float(xrt.get_right_grip())
        cfg = self._teleop.config
        # Apply BiPico4's hysteresis thresholds per side, but combine with AND.
        # Using raw grips (rather than each Pico4._enabled) keeps the decision
        # independent of whether get_action has already been called this frame.
        enter_hi = cfg.grip_enable_threshold
        exit_lo = cfg.grip_disable_threshold
        if self._active:
            still_active = left_grip > exit_lo and right_grip > exit_lo
            new_active = still_active
        else:
            new_active = left_grip > enter_hi and right_grip > enter_hi

        self._was_active = self._active
        self._active = new_active

        if not self._active:
            # Keep the teleop's internal target aligned with the real robot so
            # the first override frame does not snap the arm. BiPico4 seeds its
            # "reference" from _target_pos on the grip rising-edge, so we need
            # _target_pos to track the current TCP here.
            try:
                left_pose, right_pose = self._base_env._env.robot.get_current_tcp_pose_quat()
                self._teleop.reset_to_pose(
                    left_pose[:7],
                    right_pose[:7],
                    float(left_pose[7]),
                    float(right_pose[7]),
                )
            except Exception as e:
                # Non-fatal: if the read fails once, we try again next frame.
                logger.debug(f"teleop pose resync skipped: {e}")

        if self._was_active and not self._active:
            self._pending_release = True
            logger.info("Intervention released (grip < disable threshold).")
        elif not self._was_active and self._active:
            logger.info(f"Intervention engaged — policy paused (grips L={left_grip:.2f} R={right_grip:.2f}).")

        return self._active

    def consume_release_event(self) -> bool:
        """Return True exactly once after each intervention→policy transition."""
        if self._pending_release:
            self._pending_release = False
            return True
        return False

    def get_override_action(self) -> np.ndarray:
        """Sample the BiPico4 and pack into a 20D float32 vector.

        Key ordering follows _ACTION_KEYS_IN_ORDER, matching the 20D action
        vector consumed by BiFlexivRizon4RTRealEnv.step.
        """
        raw = self._teleop.get_action()
        return np.asarray(
            [float(raw[k]) for k in _ACTION_KEYS_IN_ORDER],
            dtype=np.float32,
        )

    def disconnect(self) -> None:
        try:
            self._teleop.disconnect()
        except Exception as e:
            logger.warning(f"Error disconnecting BiPico4: {e}")


class InterventionEnvironmentWrapper(_environment.Environment):
    """Wraps an Environment, swapping policy actions for teleop actions while
    intervention is active.  Stamps ``is_intervention`` onto the action dict
    for downstream subscribers (e.g. recorders) regardless of source.
    """

    def __init__(
        self,
        wrapped_env: _environment.Environment,
        controller: Pico4InterventionController,
    ) -> None:
        self._wrapped_env = wrapped_env
        self._controller = controller

    @override
    def reset(self) -> None:
        self._wrapped_env.reset()

    @override
    def is_episode_complete(self) -> bool:
        return self._wrapped_env.is_episode_complete()

    @override
    def get_observation(self) -> dict:
        self._controller.poll_and_decide()
        return self._wrapped_env.get_observation()

    @override
    def apply_action(self, action: dict) -> None:
        if self._controller.active:
            override_actions = self._controller.get_override_action()
            merged: dict[str, Any] = {**action, "actions": override_actions, "is_intervention": True}
        else:
            merged = {**action, "is_intervention": False}
        self._wrapped_env.apply_action(merged)

    def disconnect(self) -> None:
        self._controller.disconnect()
        inner_disconnect = getattr(self._wrapped_env, "disconnect", None)
        if callable(inner_disconnect):
            inner_disconnect()


class InterventionPolicyAgent(_agent.Agent):
    """Wraps a PolicyAgent: skips inference while intervening, clears the
    chunk broker on release so the next step triggers a fresh inference.
    """

    def __init__(
        self,
        inner_agent: _policy_agent.PolicyAgent,
        controller: Pico4InterventionController,
        broker: _action_chunk_broker.ActionChunkBroker,
    ) -> None:
        self._inner = inner_agent
        self._controller = controller
        self._broker = broker

    @override
    def get_action(self, observation: dict) -> dict:
        if self._controller.active:
            # Env wrapper will replace this with the teleop action; the
            # zero vector is purely a placeholder that satisfies the action
            # dict contract without hitting the inference server.
            return {"actions": np.zeros(20, dtype=np.float32)}

        if self._controller.consume_release_event():
            logger.info("Clearing ActionChunkBroker cache (intervention released).")
            self._broker.reset()

        return self._inner.get_action(observation)

    @override
    def reset(self) -> None:
        self._inner.reset()

    @override
    def warmup(self, observation: dict) -> None:
        self._inner.warmup(observation)
