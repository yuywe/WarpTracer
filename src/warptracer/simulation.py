"""Fixed-step lean physics, independent of display and recording libraries."""
from dataclasses import asdict, dataclass
import time

import numpy as np
import warp as wp

from .scene import Track, Vehicle
from .terrain import OvalTrack
from .maps import MapTrack, track_state
from .layers import LayeredTrack, SEAM_TOLERANCE
from .driving import DRIVING_SCENARIOS, DriveConfig, scripted_driver, write_command
from .lidar import LidarRecording, MountedLidar
from .lean import LeanState, bicycle_params, integrate_bicycle


@dataclass
class Trajectory:
    times: np.ndarray
    poses: np.ndarray
    velocities: np.ndarray
    metadata: dict
    actions: np.ndarray  # applied throttle, brake, steering (rad), aligned with poses
    lidar: LidarRecording | None = None


class Simulation:
    def __init__(self, track=None, vehicle=None, scenario="drive", device=None, physics_hz=240, drive=None, lidar=None, num_envs=1):
        if not isinstance(physics_hz, int) or physics_hz < 120:
            raise ValueError("physics_hz must be an integer >= 120")
        if lidar is not None and physics_hz % lidar.frequency:
            raise ValueError("LiDAR frequency must divide physics_hz")
        if scenario not in DRIVING_SCENARIOS:
            raise ValueError(f"scenario must be one of {DRIVING_SCENARIOS}")
        if not isinstance(num_envs, int) or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        self.num_envs = num_envs
        self.track = track or (Track(length=8, width=8) if scenario == "circle" else Track(length=12, width=6))
        self.vehicle = vehicle or Vehicle()
        self.drive = drive or DriveConfig()
        self.oval = isinstance(self.track, OvalTrack)
        self.png_map = isinstance(self.track, MapTrack)
        self.layered = isinstance(self.track, LayeredTrack)
        self.road_parameters = (wp.vec4(self.track.lane_width, self.track.elevation,
                                       self.track.frequency, self.drive.ride_height(self.vehicle))
                                if self.oval else wp.vec4())
        self.wall_padding = (self.track.length / 2 * (1 - np.cos(np.pi / self.track.segments))
                             + self.vehicle.height * self.track.elevation * self.track.frequency / 4
                             if self.oval else 0.0)
        self.drive.validate_vehicle(self.vehicle)
        self.scenario, self.physics_hz = scenario, physics_hz
        self.dt = 1.0 / physics_hz
        self.params = bicycle_params(self.vehicle, self.drive, self.dt)
        wp.init()
        self.device = wp.get_device(device or ("cuda:0" if wp.is_cuda_available() else "cpu"))
        if min(self.track.length, self.track.width) <= np.hypot(self.vehicle.length, self.vehicle.width):
            raise ValueError("Enclosure must accommodate the car at all headings")
        self.state = LeanState(self.device, num_envs)
        x = -self.track.length / 3 if scenario in ("accelerate-brake", "s-turn") else 0.0
        y = -min(1.3, self.track.width / 4) if scenario == "circle" else 0.0
        # Keep the full chassis inside small enclosures, with clearance so
        # the first stationary step does not count as a wall contact.
        limit_x = (self.track.length - self.vehicle.length) / 2
        limit_y = (self.track.width - self.vehicle.width) / 2
        x = float(np.clip(x, -0.99 * limit_x, 0.99 * limit_x))
        y = float(np.clip(y, -0.99 * limit_y, 0.99 * limit_y))
        self.initial_pose = np.array([[x, y, self.drive.ride_height(self.vehicle), 0, 0, 0, 1]], dtype=np.float32)
        if self.oval:
            padding = self.wall_padding
            radius = np.hypot(self.vehicle.length, self.vehicle.width) / 2 + padding
            a, b = self.track.length / 2, self.track.width / 2
            ia, ib = self.track.inner_axes
            if radius * (a / b + ia / ib) >= self.track.lane_width:
                raise ValueError("Oval lane must accommodate the car's conservative footprint")
            self.initial_pose = self.track.pose(-np.pi / 2, self.drive.ride_height(self.vehicle))[None]
        if self.png_map:
            self.track.validate_vehicle(self.vehicle)
            self.initial_pose = self.track.path(self.drive.ride_height(self.vehicle))[:1]
            self.map_clearance = wp.array(self.track.clearance, dtype=float, device=self.device)
        if self.layered:
            self.track.validate_vehicle(self.vehicle)
            self.initial_pose = self.track.path(self.drive.ride_height(self.vehicle))[:1]
            self.layer_heights = wp.array(self.track.heights, dtype=float, device=self.device)
            self.layer_free = wp.array(self.track.free.astype(np.int32), dtype=int, device=self.device)
            self.layer_clearance = wp.array(self.track.clearance, dtype=float, device=self.device)
            self.initial_layers = np.full(num_envs, self.track.route_layers[0], np.int32)
            self.layers = wp.zeros(num_envs, dtype=int, device=self.device)
        self.initial_pose = np.repeat(self.initial_pose, num_envs, axis=0)
        self.lean_motion = wp.zeros(num_envs, dtype=wp.vec4, device=self.device)
        self.collision = wp.zeros(num_envs, dtype=int, device=self.device)
        self.wall_contact_substeps = wp.zeros(num_envs, dtype=int, device=self.device)
        self.device_commands = wp.zeros(num_envs, dtype=wp.vec4, device=self.device)
        self._uploaded_command = None
        self.applied_controls = wp.zeros(num_envs, dtype=wp.vec3, device=self.device)
        self.lidar = MountedLidar(self.track, self.vehicle, lidar, self.device, batch_size=num_envs,
                                  wheelbase=self.drive.wheelbase) if lidar is not None else None
        self.lidar_stride = physics_hz // lidar.frequency if lidar is not None else 0
        self.reset()

    def reset(self):
        self.state.body_q.assign(self.initial_pose)
        self.state.body_qd.zero_()
        motion = np.zeros((self.num_envs, 4), np.float32)
        # Initial headings can differ, including tilted centerline spawns.
        qx, qy, qz, qw = self.initial_pose[:, 3:].T
        motion[:, 3] = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        self.lean_motion.assign(motion)
        self.collision.zero_()
        self.wall_contact_substeps.zero_()
        if self.layered:
            self.layers.assign(self.initial_layers)
        self.steps = 0
        self.command = (0.0, 0.0, 0.0)
        self.target_speed = -1.0
        self.applied_controls.zero_()
        self._uploaded_command = None
        self._upload_command()
        if self.lidar is not None:
            self.lidar.reset_statistics()
            self.lidar.update(self.state, 0.0)

    def set_action(self, throttle=0.0, brake=0.0, steering=0.0):
        """Forward throttle/brake in [0,1], steering in radians. Brake takes priority."""
        if not np.isfinite((throttle, brake, steering)).all():
            raise ValueError("Controls must be finite")
        self.command = (float(np.clip(throttle, 0, 1)), float(np.clip(brake, 0, 1)),
                        float(np.clip(steering, -self.drive.max_steering, self.drive.max_steering)))
        self.target_speed = -1.0
        self._upload_command()

    def set_target_speed(self, speed, steering=0.0):
        """Optional proportional controller; nonnegative target speed in m/s."""
        if not np.isfinite(speed) or speed < 0:
            raise ValueError("Target speed must be finite and nonnegative")
        if not np.isfinite(steering):
            raise ValueError("Steering must be finite")
        self.command = (0.0, 0.0, float(np.clip(steering, -self.drive.max_steering, self.drive.max_steering)))
        self.target_speed = float(speed)
        self._upload_command()

    def _upload_command(self):
        payload = (*self.command, self.target_speed)
        if payload != self._uploaded_command:
            wp.launch(write_command, dim=self.num_envs, inputs=[self.device_commands, wp.vec4(*payload)],
                      device=self.device)
            self._uploaded_command = payload

    def set_commands(self, commands):
        """Upload per-car [throttle, brake, steering, target_speed] commands.

        Shape is (num_envs, 4). A target of -1 selects direct throttle/brake;
        a nonnegative target selects speed control. Values are checked/clamped.
        For zero-copy GPU controllers, write device_commands on the same stream
        and invalidate_command_cache() before calling scalar setters again.
        """
        values = np.array(commands, dtype=np.float32, copy=True)
        if values.shape != (self.num_envs, 4) or not np.isfinite(values).all():
            raise ValueError("Commands must be finite with shape (num_envs, 4)")
        if np.any((values[:, 3] < 0) & (values[:, 3] != -1)):
            raise ValueError("Target speeds must be -1 (manual) or nonnegative")
        values[:, :2] = np.clip(values[:, :2], 0, 1)
        values[:, 2] = np.clip(values[:, 2], -self.drive.max_steering, self.drive.max_steering)
        self.device_commands.assign(values)
        self.invalidate_command_cache()

    def invalidate_command_cache(self):
        self._uploaded_command = None

    def _integrate(self, substeps):
        """Advance physics in place inside one capture-safe kernel launch."""
        state = [self.state.body_q, self.lean_motion, self.device_commands, self.applied_controls,
                 self.collision, self.state.body_qd, self.wall_contact_substeps]
        if self.png_map:
            from .map_kernels import integrate_map
            kernel, track = integrate_map, [self.map_clearance, wp.vec2(*self.track.origin), self.track.resolution,
                                            self.params]
        elif self.layered:
            from .layer_kernels import integrate_layers
            kernel, track = integrate_layers, [self.layers, self.layer_heights, self.layer_free, self.layer_clearance,
                                               wp.vec2(*self.track.origin), self.track.resolution,
                                               self.drive.ride_height(self.vehicle), SEAM_TOLERANCE, self.params]
        else:
            kernel, track = integrate_bicycle, [self.params, self.track.length, self.track.width, int(self.oval),
                                                self.road_parameters, self.wall_padding]
        wp.launch(kernel, dim=self.num_envs, inputs=state + track + [substeps], device=self.device)

    def step(self):
        self._integrate(1)
        self.steps += 1
        if self.lidar is not None and self.steps % self.lidar_stride == 0:
            self.lidar.update(self.state, self.steps * self.dt)

    def snapshot(self):
        q, qd = self.state.body_q.numpy(), self.state.body_qd.numpy()
        if self.num_envs == 1:
            return q[0].copy(), qd[0].copy()
        return q.copy(), qd.copy()

    def run(self, duration=4.0, record_fps=30, record=True, controller=None, runner=None):
        """Record from reset using a callback per physics step or a TransitionRunner."""
        if self.num_envs != 1:
            raise ValueError("Recorded run() supports one car; use TransitionRunner for batches")
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("duration must be finite and positive")
        if not isinstance(record_fps, int) or record_fps <= 0 or self.physics_hz % record_fps:
            raise ValueError("record_fps must be a positive divisor of physics_hz")
        total_steps = round(duration * self.physics_hz)
        if total_steps < 1:
            raise ValueError("duration is shorter than one physics step")
        stride = self.physics_hz // record_fps
        if runner is not None:
            if runner.sim is not self or controller is not None:
                raise ValueError("Runner must own this simulation and cannot be combined with a callback controller")
            if total_steps % runner.substeps or (record and stride % runner.substeps):
                raise ValueError("Duration and recording intervals must span whole transitions")
        advance = self.step if runner is None else runner.advance
        reset = self.reset if runner is None else runner.reset
        step_size = 1 if runner is None else runner.substeps
        # Compile/warm up outside the measurement, then restore the initial state.
        reset()
        for _ in range(2):
            advance()
        wp.synchronize_device(self.device)
        reset()
        poses, velocities, actions = [], [], []
        controller = controller or scripted_driver
        times = []
        scan_times, scan_poses, scan_ranges, scan_valid = [], [], [], []

        def sample_lidar():
            if self.lidar is not None and (not scan_times or self.lidar.timestamp > scan_times[-1]):
                q, ranges, valid = self.lidar.snapshot()
                scan_times.append(self.lidar.timestamp)
                scan_poses.append(q)
                scan_ranges.append(ranges)
                scan_valid.append(valid)

        def sample():
            q, qd = self.snapshot()
            poses.append(q)
            velocities.append(qd)
            times.append(self.steps * self.dt)
            actions.append(self.applied_controls.numpy()[0].copy())

        sample()
        sample_lidar()
        start = time.perf_counter()
        for i in range(step_size, total_steps + 1, step_size):
            if runner is None:
                controller(self, self.steps * self.dt)
            advance()
            if record and self.lidar is not None and i % self.lidar_stride == 0:
                sample_lidar()
            if record and i % stride == 0:
                sample()
        wp.synchronize_device(self.device)
        elapsed = time.perf_counter() - start
        if times[-1] < total_steps * self.dt:
            sample()
        sample_lidar()
        lidar_recording = None
        if self.lidar is not None:
            lidar_recording = LidarRecording(np.asarray(scan_times), np.asarray(scan_poses),
                                            np.asarray(scan_ranges), np.asarray(scan_valid),
                                            self.lidar.rays.directions.copy())
        poses, velocities = np.asarray(poses), np.asarray(velocities)
        if not np.isfinite(poses).all() or not np.isfinite(velocities).all():
            raise RuntimeError("Simulation produced non-finite state")
        metadata = {
            "scenario": self.scenario, "device": str(self.device),
            "physics_hz": self.physics_hz, "record_fps": record_fps if record else 0,
            "simulated_seconds": total_steps * self.dt, "elapsed_seconds": elapsed,
            "steps_per_second": total_steps / elapsed,
            "timing_includes_pose_recording": record,
            "track": track_state(self.track), "vehicle": asdict(self.vehicle),
            "track_kind": "png" if self.png_map else "layered" if self.layered else ("oval" if self.oval else "room"),
            "drive": asdict(self.drive),
            "lidar": ({**asdict(self.lidar.config), "mount_position": self.lidar.mount_position,
                       "algorithm": self.lidar.algorithm, "march_diagnostics": self.lidar.diagnostics(),
                       "frame": "X forward, Y left, Z up", "pose_layout": "x,y,z,qx,qy,qz,qw",
                       "invalid_range": self.lidar.config.far,
                       "self_vehicle_excluded": True} if self.lidar is not None else None),
            "action_layout": "throttle,brake,steering_rad",
            "controller": runner.controller if runner is not None else "callback",
            "backend": runner.backend if runner is not None else "eager",
            "integrator": runner.integrator if runner is not None else "unfused",
            "wall_contact_substeps": int(self.wall_contact_substeps.numpy()[0]),
            "pose_layout": "x,y,z,qx,qy,qz,qw",
            "velocity_layout": "vx,vy,vz,wx,wy,wz",
        }
        return Trajectory(np.asarray(times), poses, velocities, metadata, np.asarray(actions), lidar_recording)
