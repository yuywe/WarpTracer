# Warp playback

All visualization uses `warp.render.OpenGLRenderer` from our pinned Warp 1.17.
There is no browser scene server or HTML replay exporter. Pyglet provides the
OpenGL window/context; ImageIO/FFmpeg only encode the rendered frames as MP4.

## On your GPU computer

```bash
uv run warptracer demo oval-grid --view
```

The demo simulates, saves `outputs/oval-grid.npz`, then opens a native window.
For a trained policy, use `uv run warptracer evaluate --view`.
To reopen any recording:

```bash
uv run warptracer view outputs/oval-grid.npz
```

Space pauses/resumes, Left/Right seek by one display frame, and R restarts.
Drag the mouse to look, WASD moves the camera, mouse wheel zooms, and Esc closes.
The final frame stays visible until you close the window or restart. Camera
movement never reruns physics. This is recorded playback, not a live training view.

## On Colab

The driving and PPO notebooks request video automatically. The equivalent commands are:

```bash
uv run warptracer demo oval-grid --video
uv run warptracer evaluate --video
```

These use Warp to render offscreen, then encode `outputs/oval-grid.mp4` or
`outputs/ppo/evaluation.mp4`. The notebook displays the video with ordinary
play/pause/seeking controls. Its camera is fixed. Download the NPZ and use the
native viewer locally to move the camera interactively. To render an existing
recording, use `uv run warptracer view outputs/oval-grid.npz --video`.

## Recordings and requirements

Without `--view` or `--video`, demos/evaluation only save arrays and metadata;
no graphics module or display context is initialized. Training and benchmarks
also remain headless. Demo `--headless` additionally skips intermediate recording
copies and cannot be combined with `--view` or `--video`.

The current viewer requires a working NVIDIA CUDA driver/GPU and OpenGL context.
Warp 1.17's buffer fallback still allocates CUDA pinned memory, so CPU-only
machines without an NVIDIA driver can simulate and record but cannot use this
viewer. Colab needs a GPU runtime and EGL/OpenGL support; a missing context
produces an error after preserving the recording. Native CUDA/OpenGL rendering
cannot be verified on the CPU-only development host. Headless OpenGL may use
software rendering if the graphics driver is not configured; CUDA availability
alone does not establish hardware-accelerated drawing.

NPZ files now embed versioned JSON scene/metadata strings alongside numeric
arrays, including LiDAR directions/timestamps. They require no pickle or separate
PNG to replay. Older demo NPZs work with their matching JSON sidecars. Older PPO
evaluation recordings that lack geometry/directions must be regenerated from
their checkpoint. Checkpoint/training formats are unchanged.

The viewer samples body and scan timestamps independently at 30 display FPS,
retains the terminal frame, and draws a subset of ray lines while preserving all
beams in the recording. Rendering/video encoding happens after simulation and
is excluded from the reported simulator/PPO throughput. We have not measured a
rendering speedup from this replacement.
