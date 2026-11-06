"""With CUDA available, every physics path stays on it: no CPU kernels, no host readbacks."""
import json

import numpy as np
from PIL import Image
import pytest
import warp as wp

from warptracer.execution import TransitionRunner
from warptracer.layers import figure_eight_bridge
from warptracer.lidar import LidarConfig
from warptracer.maps import load_map
from warptracer.scene import Track
from warptracer.simulation import Simulation
from warptracer.terrain import OvalTrack

pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA driver unavailable")


@pytest.fixture
def spy(monkeypatch):
    """Devices of kernels launched and arrays read back while the test steps physics."""
    launched, readbacks = [], []
    launch, numpy = wp.launch, wp.array.numpy

    def spy_launch(kernel, *args, **kwargs):
        launched.append((str(wp.get_device(kwargs.get("device"))), kernel.key))
        return launch(kernel, *args, **kwargs)

    def spy_numpy(self):
        readbacks.append(str(self.device))
        return numpy(self)

    monkeypatch.setattr(wp, "launch", spy_launch)
    monkeypatch.setattr(wp.array, "numpy", spy_numpy)
    return launched, readbacks


@pytest.fixture
def png_track(tmp_path):
    free = np.zeros((240, 160), bool)
    free[10:230, 10:150] = True
    free[50:190, 50:110] = False
    Image.fromarray(np.flipud(free.T).astype(np.uint8) * 255).save(tmp_path / "ring.png")
    (tmp_path / "ring.json").write_text(json.dumps({"image": "ring.png", "resolution": .05, "origin": [-6., -4.]}))
    return load_map(tmp_path / "ring.json")


def assert_on_cuda(sim, advance, spy):
    launched, readbacks = spy
    sim.set_action(throttle=.5)
    launched.clear()
    readbacks.clear()
    for _ in range(8):
        advance()
    wp.synchronize_device(sim.device)
    assert sim.device.is_cuda and sim.state.body_q.device.is_cuda
    assert [key for device, key in launched if device != "cuda:0"] == []
    assert readbacks == []


@pytest.mark.parametrize("track", [Track(length=12, width=6), OvalTrack(), figure_eight_bridge()],
                         ids=["room", "oval", "bridge"])
def test_lean_physics_defaults_to_cuda(track, spy):
    sim = Simulation(track=track, scenario="drive")
    assert_on_cuda(sim, sim.step, spy)
    assert spy[0]


def test_png_map_physics_defaults_to_cuda(png_track, spy):
    sim = Simulation(track=png_track, scenario="drive")
    assert_on_cuda(sim, sim.step, spy)
    assert spy[0]


@pytest.mark.parametrize("integrator", ["unfused", "fused"])
def test_transitions_with_lidar_stay_on_cuda(integrator, spy):
    sim = Simulation(track=figure_eight_bridge(), scenario="drive",
                     lidar=LidarConfig(beams=64, frequency=60))
    runner = TransitionRunner(sim, backend="eager", substeps=4, integrator=integrator)
    assert_on_cuda(sim, runner.advance, spy)
    assert spy[0]


def test_ppo_environment_stays_on_cuda(spy):
    import torch
    from warptracer.rl_env import EpisodeConfig, RacingEnv
    env = RacingEnv(EpisodeConfig(num_envs=8), track=figure_eight_bridge(), backend="eager")
    env.reset()
    actions = torch.zeros(env.actions.shape, device=env.torch_device, dtype=torch.float32)
    assert_on_cuda(env.sim, lambda: env.step(actions), spy)
    assert spy[0]
