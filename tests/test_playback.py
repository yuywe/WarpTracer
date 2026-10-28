"""Portable recordings, renderer contracts, timing, encoding and GPU smoke test."""
from dataclasses import asdict
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from warptracer import playback
from warptracer.lidar import LidarRecording
from warptracer.scene import Track, Vehicle
from warptracer.simulation import Trajectory
from warptracer.terrain import OvalTrack
from warptracer.maps import MapTrack


def recording():
    poses = np.array([[0, 0, .1, 0, 0, 0, 1], [1, 0, .1, 0, 0, .7071068, .7071068]], np.float32)
    scans = LidarRecording(np.array([0., .1]), poses.copy(), np.array([[1., 2.], [3., 4.]]),
                           np.array([[True, True], [True, False]]), np.array([[1., 0, 0], [0, 1., 0]], np.float32))
    return Trajectory(np.array([0., .1]), poses, np.zeros((2, 6)), {"example": True}, np.zeros((2, 3)), scans)


def map_track():
    free = np.zeros((80, 80), bool); free[5:-5, 5:-5] = True
    theta = np.arange(80) * (2*np.pi/80)
    route = np.column_stack((1.2*np.cos(theta), 1.2*np.sin(theta)))
    return MapTrack(free, .05, (-2., -2.), route)


@pytest.mark.parametrize("track", [Track(), OvalTrack(), map_track()])
def test_recording_embeds_scene_and_scans_without_sidecars_or_pickle(tmp_path, track):
    source = recording()
    path = playback.save_recording(source, track, Vehicle(), tmp_path / "run")
    restored, rebuilt, vehicle = playback.load_recording(path)
    assert type(rebuilt) is type(track) and vehicle == Vehicle()
    assert not path.with_suffix(".json").exists()
    np.testing.assert_array_equal(restored.poses, source.poses)
    np.testing.assert_array_equal(restored.lidar.directions, source.lidar.directions)
    assert restored.metadata == source.metadata
    with np.load(path, allow_pickle=False) as arrays:
        assert all(arrays[key].dtype != object for key in arrays.files)


def test_old_demo_recording_loads_json_scene(tmp_path):
    source = recording(); path = tmp_path / "old.npz"
    np.savez(path, times=source.times, poses=source.poses, velocities=source.velocities, actions=source.actions)
    with pytest.raises(ValueError, match="JSON sidecar"):
        playback.load_recording(path)
    path.with_suffix(".json").write_text(json.dumps({"track_kind": "room", "track": asdict(Track()), "vehicle": asdict(Vehicle())}))
    restored, track, _ = playback.load_recording(path)
    assert isinstance(track, Track) and restored.lidar is None


class Renderer:
    screen_height, screen_width = 32, 48
    def __init__(self): self.frames, self.boxes, self.points, self.lines, self.closed = [], [], [], [], False
    def begin_frame(self, t): self.frames.append(t)
    def render_box(self, *args, **kwargs): self.boxes.append((args, kwargs))
    def render_mesh(self, *args, **kwargs): pass
    def render_points(self, name, points, **kwargs): self.points.append(points.copy())
    def render_line_list(self, name, vertices, indices, **kwargs): self.lines.append(vertices.copy())
    def end_frame(self): pass
    def close(self): self.closed = True
    def get_pixels(self, target, **kwargs):
        value = int(255 * self.frames[-1]/.1)
        data = np.zeros(target.shape, np.uint8); data[:, :, 0] = value
        target.assign(data)
        return True


def test_frame_uses_xyzw_half_extents_fresh_scan_and_hides_invalid_returns():
    renderer = Renderer(); data = recording()
    playback._draw_frame(renderer, data, Vehicle(), .1)
    args, _ = renderer.boxes[-1]
    np.testing.assert_allclose(args[1], playback._to_gl(data.poses[-1, :3]))
    np.testing.assert_allclose(args[2], playback._quat_to_gl(data.poses[-1, 3:]))
    np.testing.assert_allclose(args[3], playback._extent_to_gl(np.array(Vehicle().dimensions)/2))
    np.testing.assert_allclose(renderer.points[-1][0], playback._to_gl(data.lidar.points(1)[0]), atol=1e-6)
    assert (np.abs(renderer.points[-1][1]) > 1.e9).any()
    assert (np.abs(renderer.lines[-1][2:]) > 1.e9).any(axis=1).all()


def test_video_encodes_first_and_terminal_frames_and_closes_renderer(tmp_path, monkeypatch):
    renderer = Renderer()
    monkeypatch.setattr(playback, "_make_renderer", lambda *a, **k: renderer)
    path = playback.play(recording(), Track(), Vehicle(), video=tmp_path / "video.mp4", fps=30, device="cpu")
    assert renderer.closed and renderer.frames == pytest.approx([0., 1/30, 2/30, .1])
    import imageio.v2 as imageio
    with imageio.get_reader(path) as reader:
        first = reader.get_data(0); last = reader.get_data(3)
    assert first.shape == (32, 48, 3) and first[:, :, 0].mean() < 5 and last[:, :, 0].mean() > 245


def test_renderer_closes_after_video_read_failure(tmp_path, monkeypatch):
    renderer = Renderer(); renderer.get_pixels = lambda *a, **k: False
    monkeypatch.setattr(playback, "_make_renderer", lambda *a, **k: renderer)
    with pytest.raises(RuntimeError, match="read the rendered frame"):
        playback.play(recording(), Track(), Vehicle(), video=tmp_path / "failed.mp4", device="cpu")
    assert renderer.closed


def test_native_pause_seek_and_restart_controls(monkeypatch):
    renderer = Renderer()
    symbols = SimpleNamespace(SPACE=1, LEFT=2, RIGHT=3, R=4)
    monkeypatch.setitem(sys.modules, "pyglet", SimpleNamespace(window=SimpleNamespace(key=symbols), event=SimpleNamespace(EVENT_HANDLED=True)))
    monkeypatch.setattr(playback.time, "sleep", lambda _: None)
    monkeypatch.setattr(playback, "_make_renderer", lambda *a, **k: renderer)
    renderer.register_key_press_callback = lambda cb: setattr(renderer, "callback", cb)
    renderer.is_running = lambda: len(renderer.frames) < 4
    def end_frame():
        if len(renderer.frames) == 1: renderer.callback(symbols.SPACE, 0)
        elif len(renderer.frames) == 2: renderer.callback(symbols.RIGHT, 0)
        elif len(renderer.frames) == 3: renderer.callback(symbols.R, 0)
    renderer.end_frame = end_frame
    playback.play(recording(), Track(), Vehicle(), fps=30)
    assert renderer.closed and renderer.frames == pytest.approx([0, 0, 1/30, 1/30])


def test_cli_view_dispatch_and_optional_video_path(tmp_path, monkeypatch):
    from warptracer.cli import main
    path = playback.save_recording(recording(), Track(), Vehicle(), tmp_path / "run.npz")
    calls = []
    monkeypatch.setattr(playback, "play", lambda *a, **kw: calls.append(kw))
    main(["view", str(path), "--video"])
    assert calls[0]["video"] == path.with_suffix(".mp4")
    main(["view", str(path)])
    assert calls[1]["video"] is None


def test_recording_and_training_import_do_not_import_graphics(tmp_path):
    pytest.importorskip("torch")
    script = "import sys; from warptracer import playback; import warptracer.train; assert 'pyglet' not in sys.modules; assert 'warp.render' not in sys.modules; assert 'viser' not in sys.modules"
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


@pytest.mark.skipif(not wp.is_cuda_available(), reason="Warp OpenGL buffers require a working CUDA driver")
def test_native_warp_offscreen_video(tmp_path):
    path = playback.save_recording(recording(), OvalTrack(), Vehicle(), tmp_path / "native.npz")
    subprocess.run([sys.executable, "-m", "warptracer", "view", str(path), "--video"],
                   check=True, capture_output=True, text=True, timeout=120)
    assert path.with_suffix(".mp4").stat().st_size > 1000
