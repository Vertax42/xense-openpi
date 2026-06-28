#!/usr/bin/env python
"""Link test: drive the REAL ForwardSubscriber against a running detection app,
WITHOUT the inference server or the robot.

This exercises the exact production class (examples.bi_flexiv_rizon4_rt.forward
.ForwardSubscriber) by feeding it synthetic observations shaped like
env.get_observation() output: {"state": (20,), "images_raw": {"head": HWC}}.
It only needs xense_client importable (already present on the robot laptop) —
it does NOT import the robot SDK or connect to the policy server.

Usage
-----
1) On the detection machine, start the app:
       python -m examples.dewu_video_switch.app
   and open  http://<detection-machine-ip>:8080  in a browser.

2) On the robot laptop (or any machine with xense_client installed):
       python -m examples.dewu_video_switch.sim_laptop --uri ws://<detection-machine-ip>:9100

You should see the browser cycle scene_a → scene_b → scene_c → scene_d (~2 s
each) and the app console print "[scene] → ...". That confirms ports, firewall,
msgpack codec, and the whole switch path end-to-end.
"""

import argparse
import time

import numpy as np

from examples.bi_flexiv_rizon4_rt.forward import make_forward_subscriber


def main() -> None:
    ap = argparse.ArgumentParser(description="ForwardSubscriber link test (no inference server, no robot)")
    ap.add_argument("--uri", default="ws://127.0.0.1:9100", help="detection machine obs ws URI")
    ap.add_argument("--hz", type=float, default=30.0, help="send rate (mimics runtime_hz)")
    ap.add_argument("--hold-s", type=float, default=2.0, help="seconds to hold each scene")
    ap.add_argument("--seconds", type=float, default=0.0, help="total run time; 0 = until Ctrl+C")
    args = ap.parse_args()

    sub = make_forward_subscriber(args.uri)
    sub.on_episode_start()

    period = 1.0 / args.hz
    frames_per_scene = max(1, int(args.hold_s * args.hz))
    # Brightness buckets that StubDetector maps to scene_a..scene_d.
    levels = [0.1, 0.4, 0.6, 0.9]

    print(
        f"Sending synthetic frames to {args.uri} at {args.hz:.0f} Hz "
        f"(scene every {args.hold_s:.0f}s). Ctrl+C to stop."
    )
    t0 = time.time()
    i = 0
    try:
        while True:
            level = levels[(i // frames_per_scene) % len(levels)]
            head = np.full((480, 640, 3), int(level * 255), dtype=np.uint8)
            obs = {"state": np.zeros(20, dtype=np.float32), "images_raw": {"head": head}}
            action = {"actions": np.zeros(20, dtype=np.float32)}
            sub.on_step(obs, action)
            i += 1
            if args.seconds and (time.time() - t0) > args.seconds:
                break
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        sub.on_episode_end()
        sub.close()
        print("done")


if __name__ == "__main__":
    main()
