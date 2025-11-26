import logging
import math
import threading
import time
from typing import Dict, Optional

import numpy as np  # noqa: F401
from typing_extensions import override

from openpi_client import base_policy as _base_policy
from openpi_client.action_queue import ActionQueue
from openpi_client.latency_tracker import LatencyTracker

logger = logging.getLogger(__name__)


class RTCActionChunkBroker(_base_policy.BasePolicy):
    """Wraps a policy to return actions using an RTC-style ActionQueue.

    This broker runs a background thread to fetch action chunks from the policy
    and maintains a thread-safe queue of actions. It handles:
    - Asynchronous action fetching
    - Latency tracking (basic)
    - Action queue management (merging/replacing based on delay)

    Args:
        policy: The underlying policy (e.g., WebsocketClientPolicy).
        frequency_hz: The control frequency in Hz.
        action_queue_size_to_get_new_actions: Threshold to request new actions.
        rtc_enabled: Whether to enable RTC mode (replace queue) or append mode.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        frequency_hz: float = 50.0,
        action_queue_size_to_get_new_actions: int = 20,
        rtc_enabled: bool = True,
        execution_horizon: int = 20,
    ):
        self._policy = policy
        self._frequency_hz = frequency_hz
        self._time_per_chunk = 1.0 / frequency_hz
        self._action_queue_size_to_get_new_actions = (
            action_queue_size_to_get_new_actions
        )
        self._execution_horizon = execution_horizon

        self._action_queue = ActionQueue(rtc_enabled=rtc_enabled)
        self._latency_tracker = LatencyTracker()
        self._latest_obs: Optional[Dict] = None
        self._latest_obs_lock = threading.Lock()

        self._stop_event = threading.Event()
        self._first_inference_done = (
            threading.Event()
        )  # Signal when first actions are ready
        self._thread = threading.Thread(target=self._get_actions_loop, daemon=True)
        self._thread_started = False

    def _start_thread_if_needed(self):
        if not self._thread_started:
            self._thread.start()
            self._thread_started = True
            logger.info("RTCActionChunkBroker background thread started")

    def _get_actions_loop(self):
        while not self._stop_event.is_set():
            try:
                # Check if we need more actions
                if (
                    self._action_queue.qsize()
                    <= self._action_queue_size_to_get_new_actions
                ):
                    # Get latest observation
                    with self._latest_obs_lock:
                        obs = self._latest_obs

                    if obs is None:
                        # No observation yet, wait a bit
                        time.sleep(0.01)
                        continue

                    # Prepare for inference
                    current_time = time.perf_counter()
                    action_index_before_inference = (
                        self._action_queue.get_action_index()
                    )

                    # Get leftover actions for RTC guidance
                    prev_chunk_left_over = self._action_queue.get_left_over()

                    # Estimate inference delay
                    inference_latency = self._latency_tracker.max()
                    estimated_delay_steps = math.ceil(
                        inference_latency / self._time_per_chunk
                    )

                    # Perform inference
                    results = self._policy.infer(
                        obs,
                        prev_chunk_left_over=prev_chunk_left_over,
                        inference_delay=estimated_delay_steps,
                        execution_horizon=self._execution_horizon,
                    )

                    # Calculate actual latency for next time
                    latency = time.perf_counter() - current_time
                    self._latency_tracker.add(latency)
                    inference_delay_steps = math.ceil(latency / self._time_per_chunk)

                    # Get actions
                    # Prefer original actions if available (for RTC correctness), else processed actions
                    # Note: ActionQueue expects both, but if we only have one, we use it for both.
                    # In the client-server setup, "actions" is usually the processed one.
                    # "actions_original" might be available if we modified the server.
                    processed_actions = results.get("actions")
                    original_actions = results.get(
                        "actions_original", processed_actions
                    )

                    if processed_actions is None:
                        logger.error("Policy returned no 'actions' key")
                        continue

                    # Merge into queue
                    self._action_queue.merge(
                        original_actions=original_actions,
                        processed_actions=processed_actions,
                        real_delay=inference_delay_steps,
                        action_index_before_inference=action_index_before_inference,
                    )

                    # Signal that first inference is done (queue now has actions)
                    if not self._first_inference_done.is_set():
                        logger.info(
                            "First inference completed, action queue initialized"
                        )
                        self._first_inference_done.set()
                else:
                    # Sleep to prevent busy waiting
                    time.sleep(0.005)

            except Exception as e:
                logger.error(f"Error in RTC background thread: {e}")
                time.sleep(0.1)

    @override
    def infer(self, obs: Dict) -> Dict:
        self._start_thread_if_needed()

        # Update latest observation for the background thread
        with self._latest_obs_lock:
            self._latest_obs = obs

        # Wait for first inference to complete (handles JIT compilation delay)
        if not self._first_inference_done.is_set():
            logger.info("Waiting for first inference to complete (JIT compilation)...")
            # Wait up to 120 seconds for first inference (JIT can be slow)
            if not self._first_inference_done.wait(timeout=120.0):
                raise RuntimeError(
                    "RTCActionChunkBroker: Timeout waiting for first inference. "
                    "Check if the policy server is running and responsive."
                )
            logger.info("First inference done, proceeding with action queue")

        # Get action from queue
        action = self._action_queue.get()

        if action is None:
            # Queue empty after first inference - this shouldn't happen often
            # Wait briefly and retry
            logger.warning("Action queue empty! Waiting...")
            start_wait = time.time()
            while action is None and (time.time() - start_wait) < 2.0:
                time.sleep(0.005)
                action = self._action_queue.get()

            if action is None:
                logger.error("Action queue empty after waiting!")
                raise RuntimeError("RTCActionChunkBroker: Action queue is empty.")

        # Return in the format expected by the agent (dict)
        # The agent expects a dict that might contain other keys, but usually just "actions"
        # Since we only queue the action array, we reconstruct the dict.
        return {"actions": action}

    @override
    def reset(self) -> None:
        self._policy.reset()
        # Clear the action queue for next episode
        self._action_queue.clear()
        self._first_inference_done.clear()  # Reset for next episode
        with self._latest_obs_lock:
            self._latest_obs = None

    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join()
