"""Device-resident closed-track episodes with terminal observations and masked respawns.

Public tensors borrow stable Warp buffers. Copy observations/rewards/flags if
retaining them across step(). CUDA work follows the current PyTorch stream.
"""
from contextlib import contextmanager
from dataclasses import dataclass, asdict

import numpy as np
import torch
import warp as wp

from .execution import _centerline_path
from .lidar import LidarConfig
from .simulation import Simulation
from .terrain import OvalTrack, road_rotation, road_gradient
from .maps import MapTrack, track_state
from .map_kernels import route_projection

wp.set_module_options({"enable_backward": False})


@dataclass(frozen=True)
class EpisodeConfig:
    num_envs: int = 1024
    seed: int = 0
    max_steps: int = 3600  # 60 seconds at 60 Hz
    stall_steps: int = 600
    start_speed: float = .5
    max_speed: float = 6.0
    lidar_scale: float = 10.0
    progress_scale: float = 5.0
    collision_penalty: float = 5.0
    lap_bonus: float = 5.0
    time_penalty: float = .01  # per simulated second

    def __post_init__(self):
        for name in ("num_envs", "max_steps", "stall_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**31:
            raise ValueError("seed must be an integer in [0, 2**31)")
        values = [self.start_speed, self.max_speed, self.lidar_scale, self.progress_scale,
                  self.collision_penalty, self.lap_bonus, self.time_penalty]
        if not np.isfinite(values).all() or min(values) < 0 or min(values[1:6]) <= 0:
            raise ValueError("Episode scales must be finite and positive; speed/time penalty may be zero")
        if self.start_speed > self.max_speed:
            raise ValueError("start_speed must not exceed max_speed")


@wp.kernel
def _commands(actions: wp.array2d(dtype=float), commands: wp.array(dtype=wp.vec4),
              contacts: wp.array(dtype=int), motion: wp.array(dtype=wp.vec4),
              steering_max: float, speed_max: float):
    i = wp.tid()
    steering = wp.clamp(actions[i, 0], -1.0, 1.0)
    longitudinal = wp.clamp(actions[i, 1], -1.0, 1.0)
    if not wp.isfinite(steering) or not wp.isfinite(longitudinal):
        steering, longitudinal = 0.0, -1.0
    if motion[i][0] >= speed_max:
        longitudinal = wp.min(longitudinal, 0.0)
    commands[i] = wp.vec4(wp.max(longitudinal, 0.0), wp.max(-longitudinal, 0.0),
                          steering * steering_max, -1.0)
    contacts[i] = 0


@wp.kernel
def _score(poses: wp.array(dtype=wp.transform), motion: wp.array(dtype=wp.vec4),
           path: wp.array(dtype=wp.transform), contacts: wp.array(dtype=int), previous: wp.array(dtype=float),
           progress: wp.array(dtype=float), lengths: wp.array(dtype=int), stalls: wp.array(dtype=int),
           episode_return: wp.array(dtype=float), rewards: wp.array(dtype=float),
           terminated: wp.array(dtype=int), truncated: wp.array(dtype=int), done: wp.array(dtype=int),
           reason: wp.array(dtype=int), finished_return: wp.array(dtype=float),
           finished_length: wp.array(dtype=int), finished_laps: wp.array(dtype=float),
           route_arc: wp.array(dtype=wp.vec2), route_index: wp.array(dtype=int), map_mode: int, route_length: float,
           a: float, b: float, ia: float, ib: float, oa: float, ob: float,
           max_steps: int, stall_steps: int, progress_scale: float,
           collision_penalty: float, lap_bonus: float, time_penalty: float):
    i = wp.tid()
    p = wp.transform_get_translation(poses[i])
    m = motion[i]
    finite = wp.isfinite(p[0]) and wp.isfinite(p[1]) and wp.isfinite(p[2])
    for k in range(4):
        finite = finite and wp.isfinite(m[k])
    if not finite:
        # Do not send NaN origins into the ray marcher. This episode is already
        # terminal; its ignored bootstrap observation may use a safe placeholder.
        poses[i] = path[i % path.shape[0]]
        motion[i] = wp.vec4()
        p = wp.transform_get_translation(poses[i])
    angle = wp.atan2(p[1] / b, p[0] / a)
    if map_mode != 0:
        route_s, nearest = route_projection(p, path, route_arc, route_index[i])
        angle = 2.0 * wp.pi * route_s / route_length
        route_index[i] = nearest
    delta = wp.atan2(wp.sin(angle - previous[i]), wp.cos(angle - previous[i]))
    if not finite:
        delta = 0.0
    previous[i] = angle
    progress[i] += delta
    # Continuous signed local arc progress; no waypoint-quantization reward.
    metric = wp.sqrt(a * a * wp.sin(angle) ** 2.0 + b * b * wp.cos(angle) ** 2.0)
    if map_mode != 0:
        metric = route_length / (2.0 * wp.pi)
    distance = delta * metric
    lengths[i] += 1
    stalls[i] = 0 if wp.abs(m[0]) > .05 else stalls[i] + 1
    outer = p[0] * p[0] / (oa * oa) + p[1] * p[1] / (ob * ob)
    inner = p[0] * p[0] / (ia * ia) + p[1] * p[1] / (ib * ib)
    off_track = map_mode == 0 and (outer > 1.001 or inner < .999)
    failure = not finite or contacts[i] > 0 or off_track or stalls[i] >= stall_steps
    success = progress[i] >= 2.0 * wp.pi and not failure
    term = failure or success
    timeout = lengths[i] >= max_steps and not term
    r = progress_scale * distance - time_penalty / 60.0
    if failure:
        r = -collision_penalty
    elif success:
        r += lap_bonus
    rewards[i] = r
    episode_return[i] += r
    terminated[i] = int(term)
    truncated[i] = int(timeout)
    done[i] = int(term or timeout)
    # 1 collision/off-track, 2 lap, 3 time limit, 4 stall, 5 nonfinite state.
    code = int(0)
    if failure:
        code = 1
        if stalls[i] >= stall_steps:
            code = 4
        if not finite:
            code = 5
    elif success:
        code = 2
    elif timeout:
        code = 3
    reason[i] = code
    finished_return[i] = 0.0
    finished_length[i] = 0
    finished_laps[i] = 0.0
    if term or timeout:
        finished_return[i] = episode_return[i]
        finished_length[i] = lengths[i]
        finished_laps[i] = progress[i] / (2.0 * wp.pi)
        if not finite:
            finished_laps[i] = 0.0


@wp.kernel
def _respawn(active: wp.array(dtype=int), path: wp.array(dtype=wp.transform),
             poses: wp.array(dtype=wp.transform), velocities: wp.array(dtype=wp.spatial_vector),
             next_poses: wp.array(dtype=wp.transform), next_velocities: wp.array(dtype=wp.spatial_vector),
             motion: wp.array(dtype=wp.vec4), controls: wp.array(dtype=wp.vec3),
             commands: wp.array(dtype=wp.vec4), contacts: wp.array(dtype=int), collision: wp.array(dtype=int),
             previous: wp.array(dtype=float), progress: wp.array(dtype=float),
             lengths: wp.array(dtype=int), stalls: wp.array(dtype=int), episode_return: wp.array(dtype=float),
             episodes: wp.array(dtype=int), route_arc: wp.array(dtype=wp.vec2), route_index: wp.array(dtype=int),
             map_mode: int, route_length: float, seed: int, a: float, b: float,
             elevation: float, frequency: float, start_speed: float):
    i = wp.tid()
    if active[i] != 0:
        # Each car owns its RNG sequence. Another car's reset cannot perturb it.
        rng = wp.rand_init(seed ^ episodes[i], i)
        index = wp.min(int(wp.randf(rng) * float(path.shape[0])), path.shape[0] - 1)
        pose = path[index]
        p = wp.transform_get_translation(pose)
        forward = wp.quat_rotate(wp.transform_get_rotation(pose), wp.vec3(1.0, 0.0, 0.0))
        heading = wp.atan2(forward[1], forward[0]) + .1 * (2.0 * wp.randf(rng) - 1.0)
        pose = wp.transform(p, road_rotation(heading, road_gradient(p[0], elevation, frequency)))
        velocity = wp.spatial_vector(wp.vec3(start_speed * wp.cos(heading), start_speed * wp.sin(heading),
            start_speed * wp.cos(heading) * road_gradient(p[0], elevation, frequency)), wp.vec3())
        poses[i] = pose
        next_poses[i] = pose
        velocities[i] = velocity
        next_velocities[i] = velocity
        motion[i] = wp.vec4(start_speed, 0.0, 0.0, heading)
        controls[i] = wp.vec3()
        commands[i] = wp.vec4(0.0, 0.0, 0.0, -1.0)
        contacts[i] = 0
        collision[i] = 0
        previous[i] = wp.atan2(p[1] / b, p[0] / a)
        if map_mode != 0:
            previous[i] = 2.0 * wp.pi * route_arc[index][0] / route_length
            route_index[i] = index
        progress[i] = 0.0
        lengths[i] = 0
        stalls[i] = 0
        episode_return[i] = 0.0
        episodes[i] += 1


@wp.kernel
def _observations(ranges: wp.array2d(dtype=float), valid: wp.array2d(dtype=int),
                  poses: wp.array(dtype=wp.transform), motion: wp.array(dtype=wp.vec4),
                  controls: wp.array(dtype=wp.vec3), observations: wp.array2d(dtype=float),
                  beams: int, lidar_scale: float, speed_scale: float, steering_max: float,
                  elevation: float, frequency: float):
    i, j = wp.tid()
    value = float(1.0)
    if j < beams:
        if valid[i, j] != 0:
            value = wp.clamp(ranges[i, j] / lidar_scale, 0.0, 1.0)
    else:
        k = j - beams
        m = motion[i]
        value = m[0] / speed_scale
        if k == 1:
            value = m[1] / speed_scale
        elif k == 2:
            value = m[2] / 3.0
        elif k == 3:
            value = controls[i][2] / steering_max
        elif k >= 4:
            p = wp.transform_get_translation(poses[i])
            grade = road_gradient(p[0], elevation, frequency)
            value = grade * wp.cos(m[3]) / .2
            if k == 5:
                value = -grade * wp.sin(m[3]) / .2
        value = wp.clamp(value, -5.0, 5.0)
    if not wp.isfinite(value):
        value = 0.0
    observations[i, j] = value


class RacingEnv:
    """A Torch-native vector environment. step() auto-resets only finished cars.

    final_observation is acquired BEFORE reset, including fresh terminal LiDAR.
    observation contains post-reset scans for finished cars, and next-state
    scans for survivors. terminated and truncated remain separate.
    With auto_reset=False, terminal states are retained for recording; the caller
    must stop stepping at the episode end or explicitly reset().
    """
    def __init__(self, config=None, *, device=None, track=None, backend="graph", auto_reset=True):
        self.config = config or EpisodeConfig()
        self.track = track or OvalTrack()
        self.map_mode = isinstance(self.track, MapTrack)
        self.route_length = self.track.route_length if self.map_mode else 1.0
        self.axes = ((1., 1., 1., 1., 1., 1.) if self.map_mode else
                     (*self.track.center_axes, *self.track.inner_axes, self.track.length/2, self.track.width/2))
        if backend not in ("graph", "eager"):
            raise ValueError("backend must be graph or eager")
        if type(auto_reset) is not bool:
            raise ValueError("auto_reset must be a boolean")
        wp.init()
        self.device = wp.get_device(device or ("cuda:0" if wp.is_cuda_available() else "cpu"))
        self.torch_device = torch.device(str(self.device))
        self.num_envs = self.config.num_envs
        self.backend = backend
        self.auto_reset = auto_reset
        self.obs_dim, self.action_dim = 114, 2
        self.graph = None
        self._capture_stream = None
        with self._scope():
            self.sim = Simulation(track=self.track, scenario="drive", engine="lean", device=self.device,
                                  num_envs=self.num_envs, lidar=LidarConfig(backend="grid", frequency=60))
            height = self.sim.drive.ride_height(self.sim.vehicle)
            path = self.track.path(height) if self.map_mode else _centerline_path(self.track, height)
            self.path = wp.array(path, dtype=wp.transform, device=self.device)
            arc = (np.column_stack((self.track.arc, self.track.segment_lengths)) if self.map_mode
                   else np.zeros((1, 2), np.float32))
            self.route_arc = wp.array(arc, dtype=wp.vec2, device=self.device)
            self.route_index = wp.zeros(self.num_envs, dtype=int, device=self.device)
            self.actions_w = wp.zeros((self.num_envs, 2), dtype=float, device=self.device)
            self.obs_w = wp.zeros((self.num_envs, self.obs_dim), dtype=float, device=self.device)
            self.final_obs_w = wp.zeros_like(self.obs_w)
            self.buffers = {}
            for name in ("previous", "progress", "episode_return", "reward", "finished_return", "finished_laps"):
                self.buffers[name] = wp.zeros(self.num_envs, dtype=float, device=self.device)
            for name in ("lengths", "stalls", "episodes", "terminated", "truncated", "done", "reason", "finished_length"):
                self.buffers[name] = wp.zeros(self.num_envs, dtype=int, device=self.device)
            self.all_active = wp.ones(self.num_envs, dtype=int, device=self.device)
            self.actions = wp.to_torch(self.actions_w)
            self.observation = wp.to_torch(self.obs_w)
            self.final_observation = wp.to_torch(self.final_obs_w)
            self.tensors = {name: wp.to_torch(value) for name, value in self.buffers.items()}
            self.reset()
            if backend == "graph":
                self._operations()  # load kernels before capture
                wp.synchronize_device(self.device)
                self.reset()
                self._capture_graph()
                self.reset()

    def _capture_graph(self):
        # Torch's default (null) CUDA stream supports launches but not capture.
        # Use a retained Warp-owned stream for capture only; graph replay still
        # follows the current Torch stream, including the default stream.
        caller_stream = None
        if self.device.is_cuda:
            caller_stream = wp.get_stream(self.device)
            self._capture_stream = wp.Stream(self.device)
            self._capture_stream.wait_stream(caller_stream)
        with wp.ScopedStream(self._capture_stream, sync_enter=False):
            with wp.ScopedCapture(device=self.device, stream=self._capture_stream,
                                  force_module_load=False) as capture:
                self._operations()
            self.graph = capture.graph
        if caller_stream is not None:
            caller_stream.wait_stream(self._capture_stream)

    @contextmanager
    def _scope(self):
        if self.device.is_cuda:
            torch_stream = torch.cuda.current_stream(self.torch_device)
            stream = wp.get_stream(self.device)
            # Nested scopes must reuse the existing external-stream wrapper:
            # destroying a duplicate wrapper unregisters that handle in Warp.
            if int(stream.cuda_stream or 0) != torch_stream.cuda_stream:
                stream = wp.stream_from_torch(torch_stream)
            with wp.ScopedStream(stream):
                yield stream
        else:
            with wp.ScopedDevice(self.device):
                yield None

    def _spawn(self, active):
        s, c, b = self.sim, self.config, self.buffers
        wp.launch(_respawn, dim=self.num_envs, inputs=[active, self.path,
            s.state.body_q, s.state.body_qd, s.next_state.body_q, s.next_state.body_qd,
            s.lean_motion, s.applied_controls, s.device_commands, s.wall_contact_substeps, s.collision,
            b["previous"], b["progress"], b["lengths"], b["stalls"], b["episode_return"], b["episodes"],
            self.route_arc, self.route_index, int(self.map_mode), self.route_length,
            c.seed, *self.axes[:2], self.track.elevation, self.track.frequency, c.start_speed], device=self.device)

    def _observe(self, output):
        s, c = self.sim, self.config
        wp.launch(_observations, dim=output.shape, inputs=[s.lidar.sensor.values, s.lidar.sensor.valid,
            s.state.body_q, s.lean_motion, s.applied_controls, output, 108, c.lidar_scale, c.max_speed,
            s.drive.max_steering, self.track.elevation, self.track.frequency], device=self.device)

    def _operations(self):
        s, c, b = self.sim, self.config, self.buffers
        wp.launch(_commands, dim=self.num_envs, inputs=[self.actions_w, s.device_commands,
                  s.wall_contact_substeps, s.lean_motion, s.drive.max_steering, c.max_speed], device=self.device)
        s._fused_physics_steps(4)
        wp.launch(_score, dim=self.num_envs, inputs=[s.state.body_q, s.lean_motion, self.path, s.wall_contact_substeps,
            b["previous"], b["progress"], b["lengths"], b["stalls"], b["episode_return"], b["reward"],
            b["terminated"], b["truncated"], b["done"], b["reason"], b["finished_return"],
            b["finished_length"], b["finished_laps"], self.route_arc, self.route_index,
            int(self.map_mode), self.route_length, *self.axes, c.max_steps, c.stall_steps,
            c.progress_scale, c.collision_penalty, c.lap_bonus, c.time_penalty], device=self.device)
        s.lidar.update(s.state, 0, 0.0)
        self._observe(self.final_obs_w)
        if self.auto_reset:
            self._spawn(b["done"])
            # The second scan marches only respawned cars; surviving scans are kept.
            s.lidar.update(s.state, 0, 0.0, active=b["done"])
        self._observe(self.obs_w)

    def reset(self):
        with self._scope():
            for array in self.buffers.values():
                array.zero_()
            self.actions_w.zero_()
            self.route_index.zero_()
            self._spawn(self.all_active)
            self.sim.lidar.reset_statistics()
            self.sim.lidar.update(self.sim.state, 0, 0.0)
            self._observe(self.obs_w)
            wp.copy(self.final_obs_w, self.obs_w)
            self.sim.steps = 0
        return self.observation

    def step(self, actions):
        if actions.shape != self.actions.shape or actions.device != self.torch_device or actions.dtype != torch.float32:
            raise ValueError("Actions must be float32 (num_envs, 2) on the environment device")
        with self._scope() as stream:
            self.actions.copy_(actions.detach())
            if self.graph is None:
                self._operations()
            else:
                wp.capture_launch(self.graph, stream=stream)
        self.sim.steps += 4
        self.sim.invalidate_command_cache()
        return self.observation, self.tensors["reward"], self.tensors["terminated"], self.tensors["truncated"]

    def state_dict(self):
        arrays = {**self.buffers, "poses": self.sim.state.body_q, "velocities": self.sim.state.body_qd,
                  "motion": self.sim.lean_motion, "controls": self.sim.applied_controls,
                  "commands": self.sim.device_commands, "contacts": self.sim.wall_contact_substeps,
                  "collision": self.sim.collision, "obs": self.obs_w, "final_obs": self.final_obs_w}
        if self.map_mode:
            arrays["route_index"] = self.route_index
        return {"config": asdict(self.config), "track": track_state(self.track), "steps": self.sim.steps,
                "auto_reset": self.auto_reset,
                "arrays": {k: wp.to_torch(v).detach().cpu().clone() for k, v in arrays.items()}}

    def load_state_dict(self, state):
        if state["config"] != asdict(self.config) or state["track"] != track_state(self.track):
            raise ValueError("Checkpoint episode/track settings do not match this environment")
        if state.get("auto_reset", True) != self.auto_reset:
            raise ValueError("Checkpoint auto-reset setting does not match this environment")
        current = self.state_dict()["arrays"]
        if set(current) != set(state["arrays"]):
            raise ValueError("Checkpoint environment buffers are incomplete")
        with self._scope():
            for name, tensor in state["arrays"].items():
                if name not in current or tensor.shape != current[name].shape:
                    raise ValueError("Checkpoint environment buffers do not match")
                if name in self.buffers:
                    target = self.buffers[name]
                elif name == "route_index":
                    target = self.route_index
                else:
                    target = {"poses": self.sim.state.body_q, "velocities": self.sim.state.body_qd,
                        "motion": self.sim.lean_motion, "controls": self.sim.applied_controls,
                        "commands": self.sim.device_commands, "contacts": self.sim.wall_contact_substeps,
                        "collision": self.sim.collision, "obs": self.obs_w, "final_obs": self.final_obs_w}[name]
                wp.to_torch(target).copy_(tensor.to(self.torch_device))
            wp.copy(self.sim.next_state.body_q, self.sim.state.body_q)
            wp.copy(self.sim.next_state.body_qd, self.sim.state.body_qd)
            self.sim.lidar.reset_statistics()
            self.sim.lidar.update(self.sim.state, 0, 0.0)
        self.sim.steps = state["steps"]
