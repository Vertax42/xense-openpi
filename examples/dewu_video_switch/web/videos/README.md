# Video assets

Put the loop clips here, named to match the `SCENES` map in `../index.html`
(and the scene ids your detector emits):

```
scene_a.mp4  scene_b.mp4  scene_c.mp4  scene_d.mp4
```

For local testing these are symlinked to the repo's `videos/得物_0627_{1..4}.m4v`
(ascii names so the browser needs no URL-encoding and the `.mp4` extension maps
to `video/mp4`). On the real detection machine, drop in the product clips under
these same names.

Tips for seamless switching:
- Encode each clip to loop cleanly (first/last frame match) — the player sets
  `loop` and never seeks, so a clean loop point keeps it imperceptible.
- Same resolution / aspect ratio across clips so the crossfade has no jump.
- H.264 MP4/M4V plays on both Chrome (Windows) and Safari (macOS).

These files are git-ignored (large, product-specific binaries).
