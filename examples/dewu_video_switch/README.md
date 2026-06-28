# dewu video switch (Plan B: off-laptop detection + seamless video switching)

Runs the heavy keypoint-detection and the seamless video player on a **separate
machine** (Windows / macOS), so the robot host laptop only forwards a slim obs
payload and spends zero extra cycles on detection or rendering.

```
robot laptop  examples.bi_flexiv_rizon4_rt.main --forward
    │  ws push {state, head image}  (msgpack, one-way, non-blocking)
    ▼
detection machine  examples.dewu_video_switch.app
    ├─ :9100  obs ws server     ← laptop connects here
    ├─ detector (thread pool)   → SceneController (debounce/hysteresis)
    ├─ :9101  switch ws server  → browsers connect, receive {"scene": id}
    └─ :8080  http (web/)        → seamless dual-/multi-video player
```

The forwarder lives in the laptop's inference process as a `Subscriber`
(`runtime.Runtime` calls it after each step). It only drops the latest frame
into a single-slot queue; a daemon thread owns the socket and swallows every
error, so a dead detection machine or a slow link can **never** disturb the
robot's 30 Hz control loop or its home-on-exit motion.

## Run

**Detection machine** (needs only `pip install -r requirements.txt` — no openpi):

```bash
python -m examples.dewu_video_switch.app        # from the openpi repo root
# or copy this folder onto the machine and run:  python app.py
```

Open `http://<detection-machine-ip>:8080` on the display. All clips play muted &
looping underneath; a switch is a pure opacity crossfade — no load, no seek, so
it's imperceptible. Cross-platform because it's just a browser (Chrome/Safari).

**Robot laptop** — add the forward flags to the usual launch:

```bash
python -m examples.bi_flexiv_rizon4_rt.main \
    --args.host 192.168.142.158 --args.port 8000 \
    --args.bi-mount-type side --args.inner-control-hz 1000 \
    --args.interpolate-cmds --args.runtime-hz 30 --args.rtc-enabled \
    --args.forward \
    --args.forward-uri ws://<detection-machine-ip>:9100
```

Useful forward flags: `--args.forward-cameras head` (which raw cameras),
`--args.forward-state` / `--no-args.forward-state`, `--args.forward-stride N`
(forward every Nth step to throttle bandwidth; raw head 640×480 @30 Hz ≈ 27 MB/s).

## Plugging in the real detector

`detector.py` ships a dependency-free `StubDetector` (brightness/gripper buckets)
only to prove the pipeline. Replace it with the real keypoint model:

1. Subclass `Detector` and implement `detect(frame) -> scene_id | None`.
   `frame` = `{"step", "state": (20,), "images": {"head": HWC uint8}, "action": (20,)}`.
2. Register it in `make_detector()` and select with `--detector <name>`.
3. Make `detect()` return scene ids that match `SCENES` in `web/index.html`.

`SceneController` (`--confirm-frames`, `--min-dwell-s`) debounces noisy per-frame
output so the video doesn't flicker.

## Videos

`web/videos/scene_{a,b,c,d}.mp4` — for local testing these are symlinked to the
repo's `videos/得物_0627_{1..4}.m4v`. On the real machine, drop product clips
under the same names. Binaries and symlinks here are git-ignored.
