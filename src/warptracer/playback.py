"""Portable recordings and Warp OpenGL playback; graphics imports stay lazy."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .maps import MapTrack, restore_track, track_state
from .scene import Track, Vehicle
from .terrain import OvalTrack

# WarpTracer's world is Z-up (X forward, Y left); the installed Warp 1.17 OpenGL
# renderer's up_axis="Z" mode disagrees with itself (its camera and its shape
# instances resolve the axis permutation differently, landing arbitrary rotated
# objects in the wrong place). Side-step that entirely: render in the renderer's
# native up_axis="Y" and convert every position/rotation/extent we hand it
# through this one fixed, verified change of basis.
_WORLD_TO_RENDERER = Rotation.from_euler("x", -90, degrees=True)


def _to_gl(points):
    return _WORLD_TO_RENDERER.apply(points)


def _extent_to_gl(extent):
    ex, ey, ez = extent
    return (ex, ez, ey)


def _quat_to_gl(quat):
    return tuple((_WORLD_TO_RENDERER * Rotation.from_quat(quat) * _WORLD_TO_RENDERER.inv()).as_quat())


def save_recording(trajectory, track, vehicle, path):
    """Save arrays and embedded scene JSON without pickle or rendering."""
    path = Path(path).with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    scene = {"version": 1, "track_kind": "png" if isinstance(track, MapTrack) else
             "oval" if isinstance(track, OvalTrack) else "room",
             "track": track_state(track), "vehicle": asdict(vehicle)}
    arrays = dict(times=trajectory.times, poses=trajectory.poses, velocities=trajectory.velocities,
                  actions=trajectory.actions, scene_json=np.array(json.dumps(scene)),
                  metadata_json=np.array(json.dumps(trajectory.metadata)))
    if trajectory.lidar is not None:
        scans = trajectory.lidar
        arrays.update(lidar_times=scans.times, lidar_poses=scans.poses, lidar_ranges=scans.ranges,
                      lidar_valid=scans.valid, lidar_directions=scans.directions)
    np.savez_compressed(path, **arrays)
    return path


def load_recording(path):
    from .lidar import LidarRecording
    from .simulation import Trajectory
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"])) if "metadata_json" in data else {}
        if "scene_json" in data:
            scene = json.loads(str(data["scene_json"]))
            if scene.get("version") != 1:
                raise ValueError("Unsupported recording scene version")
        else:
            sidecar = path.with_suffix(".json")
            if not sidecar.exists():
                raise ValueError("Recording has no embedded scene or JSON sidecar; regenerate it with this version")
            metadata = json.loads(sidecar.read_text())
            if "track" not in metadata or "vehicle" not in metadata:
                raise ValueError("This older evaluation has no scene data; rerun evaluation from its checkpoint")
            scene = metadata
        track = (Track(**scene["track"]) if scene["track_kind"] == "room" else restore_track(scene["track"]))
        vehicle = Vehicle(**scene["vehicle"])
        scans = None
        if "lidar_ranges" in data:
            if "lidar_directions" not in data:
                raise ValueError("Recording lacks LiDAR directions; regenerate it with this version")
            scans = LidarRecording(data["lidar_times"], data["lidar_poses"], data["lidar_ranges"],
                                   data["lidar_valid"].astype(bool), data["lidar_directions"])
        trajectory = Trajectory(data["times"], data["poses"], data["velocities"], metadata, data["actions"], scans)
    _validate(trajectory)
    return trajectory, track, vehicle


def _validate(trajectory):
    t, poses = trajectory.times, trajectory.poses
    if (t.ndim != 1 or len(t) < 1 or poses.shape != (len(t), 7) or not np.isfinite(t).all()
            or not np.isfinite(poses).all() or np.any(np.diff(t) <= 0)):
        raise ValueError("Playback needs finite, increasing times and one XYZ/XYZW pose per time")
    scans = trajectory.lidar
    if scans is not None:
        if (scans.times.ndim != 1 or len(scans.times) < 1 or np.any(np.diff(scans.times) <= 0)
                or not np.isfinite(scans.times).all() or scans.poses.shape != (len(scans.times), 7)
                or not np.isfinite(scans.poses).all() or scans.ranges.ndim != 2
                or scans.ranges.shape[0] != len(scans.times) or scans.valid.shape != scans.ranges.shape
                or scans.directions.shape != (scans.ranges.shape[1], 3)):
            raise ValueError("Recording LiDAR times, poses, directions and scan shapes do not match")


def _make_renderer(track, *, headless, fps, width, height, device):
    import warp as wp
    wp.init()
    if not wp.is_cuda_available():
        raise RuntimeError("Warp 1.17 playback requires an NVIDIA CUDA driver/GPU for its OpenGL buffers. "
                           "Simulation and the saved NPZ still work on CPU; view it on your GPU machine "
                           "or select a Colab GPU runtime and use --video")
    # Select EGL BEFORE warp.render imports OpenGL bindings.
    import pyglet
    pyglet.options["headless"] = headless and sys.platform == "linux"
    import warp.render
    center = (np.array(track.origin) + np.array([track.length, track.width])/2
              if isinstance(track, MapTrack) else np.zeros(2))
    span = max(track.length, track.width)
    position = np.array([center[0]+.8*span, center[1]-.9*span, .95*span])
    direction = np.array([*center, .15]) - position
    direction /= np.linalg.norm(direction)
    try:
        renderer = wp.render.OpenGLRenderer(title="WarpTracer", up_axis="Y", fps=fps,
            screen_width=width, screen_height=height, near_plane=.01, far_plane=10*span,
            camera_pos=tuple(_to_gl(position)), camera_front=tuple(_to_gl(direction)), camera_up=tuple(_to_gl((0., 0., 1.))),
            draw_grid=False, draw_sky=False, draw_axis=False, show_info=False,
            enable_backface_culling=False, background_color=(.08, .10, .13),
            headless=headless, device=device)
        print(f"Warp OpenGL: {pyglet.gl.gl_info.get_renderer()}", flush=True)
        return renderer
    except Exception as error:
        mode = ("An EGL/OpenGL context is required for offscreen video" if headless else
                "A desktop OpenGL display is required; use --video on a headless machine")
        raise RuntimeError(f"{mode}. Warp renderer initialization failed: {error}") from error


def _draw_track(renderer, track):
    identity = _quat_to_gl((0., 0., 0., 1.))
    if isinstance(track, (OvalTrack, MapTrack)):
        for name, (vertices, faces) in track.mesh_parts().items():
            color = (.28, .30, .33) if name == "road" else (.60, .60, .60)
            renderer.render_mesh(f"track/{name}", _to_gl(vertices), faces, colors=color, smooth_shading=False)
    else:
        renderer.render_box("floor", tuple(_to_gl((0, 0, -.03))), identity,
                            _extent_to_gl(((track.length+2*track.barrier_thickness)/2,
                             (track.width+2*track.barrier_thickness)/2, .03)), color=(.28, .30, .33))
        for i, (position, dimensions) in enumerate(track.barriers()):
            renderer.render_box(f"walls/{i}", tuple(_to_gl(position)), identity,
                                _extent_to_gl(tuple(np.array(dimensions)/2)), color=(.6, .6, .6))


def _sample_index(times, timestamp):
    return int(np.clip(np.searchsorted(times, timestamp, side="right")-1, 0, len(times)-1))


def _draw_frame(renderer, trajectory, vehicle, timestamp):
    renderer.begin_frame(timestamp)
    pose = trajectory.poses[_sample_index(trajectory.times, timestamp)]
    renderer.render_box("vehicle", tuple(_to_gl(pose[:3])), _quat_to_gl(pose[3:]),
                        _extent_to_gl(tuple(np.array(vehicle.dimensions)/2)), color=(.12, .49, .93))
    scans = trajectory.lidar
    if scans is not None:
        index = _sample_index(scans.times, timestamp)
        points = scans.points(index)
        origin = scans.poses[index, :3]
        missing = ~np.isfinite(points).all(axis=1)
        # Warp retains point/line instancers when passed an empty collection.
        # Keep counts stable and move invalid samples outside the clipping range.
        points[missing] = (0., 0., -1.e10)
        displayed = _to_gl(points[::max(1, (len(points)+179)//180)]).astype(np.float32)
        renderer.render_points("lidar/returns", displayed, radius=.018, colors=(1., .82, .27))
        ends = points[::max(1, (len(points)+29)//30)]
        origins = np.broadcast_to(origin, ends.shape).copy()
        origins[missing[::max(1, (len(points)+29)//30)]] = (0., 0., -1.e10)
        lines = _to_gl(np.stack((origins, ends), axis=1).reshape(-1, 3)).astype(np.float32)
        renderer.render_line_list("lidar/rays", lines, np.arange(len(lines)),
                                 color=(.24, .70, .51), radius=.006)
    renderer.end_frame()


def play(trajectory, track, vehicle, *, video=None, fps=30, width=960, height=640, device=None):
    """Open a native replay, or render a fixed-camera MP4 offscreen."""
    _validate(trajectory)
    if type(fps) is not int or fps <= 0 or min(width, height) < 16 or width % 2 or height % 2:
        raise ValueError("Require positive integer FPS and even video dimensions >= 16")
    if video is not None and Path(video).suffix.lower() != ".mp4":
        raise ValueError("Video output must have the .mp4 extension")
    renderer = _make_renderer(track, headless=video is not None, fps=fps, width=width, height=height, device=device)
    try:
        _draw_track(renderer, track)
        start, end = float(trajectory.times[0]), float(trajectory.times[-1])
        if video is not None:
            import imageio.v2 as imageio
            import warp as wp
            path = Path(video)
            path.parent.mkdir(parents=True, exist_ok=True)
            pixels = wp.empty((renderer.screen_height, renderer.screen_width, 3), dtype=wp.uint8, device=device)
            count = int(np.ceil((end-start)*fps)) + 1
            with imageio.get_writer(path, fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=None) as writer:
                for frame in range(count):
                    _draw_frame(renderer, trajectory, vehicle, min(start+frame/fps, end))
                    if not renderer.get_pixels(pixels, split_up_tiles=False, use_uint8=True):
                        raise RuntimeError("Warp could not read the rendered frame")
                    writer.append_data(pixels.numpy())
                    if frame and frame % 300 == 0:
                        print(f"Warp video: {frame}/{count} frames", flush=True)
            return path
        import pyglet
        state = {"time": start, "paused": False}
        def key(symbol, modifiers):
            keys = pyglet.window.key
            if symbol == keys.SPACE:
                state["paused"] = not state["paused"]
            elif symbol in (keys.LEFT, keys.RIGHT):
                state["time"] = np.clip(state["time"] + (-1 if symbol == keys.LEFT else 1)/fps, start, end)
                state["paused"] = True
            elif symbol == keys.R:
                state["time"], state["paused"] = start, False
            else:
                return
            return pyglet.event.EVENT_HANDLED
        renderer.register_key_press_callback(key)
        print("Warp viewer: Space pause/resume; Left/Right seek; R restart; drag mouse to look; WASD move; Esc close.", flush=True)
        while renderer.is_running():
            began = time.perf_counter()
            _draw_frame(renderer, trajectory, vehicle, state["time"])
            if not state["paused"]:
                state["time"] = min(state["time"]+1/fps, end)
            time.sleep(max(0., 1/fps-(time.perf_counter()-began)))
    finally:
        renderer.close()


def main(argv=None):
    parser = argparse.ArgumentParser(prog="warptracer view", description="View a recording using Warp, or export an MP4")
    parser.add_argument("recording", type=Path)
    parser.add_argument("--video", nargs="?", const=True, type=Path, help="Export MP4 offscreen; defaults beside the recording")
    parser.add_argument("--device", help="Rendering device; defaults to available CUDA, otherwise CPU")
    args = parser.parse_args(argv)
    try:
        trajectory, track, vehicle = load_recording(args.recording)
        video = args.recording.with_suffix(".mp4") if args.video is True else args.video
        result = play(trajectory, track, vehicle, video=video, device=args.device)
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        parser.error(str(error))
    if result:
        print(f"Warp video: {result}", flush=True)
