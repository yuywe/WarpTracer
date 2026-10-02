"""Fixed-step lean/Newton physics, independent of display and recording libraries."""
from dataclasses import asdict, dataclass
import time
from types import SimpleNamespace

import numpy as np
import warp as wp

from .scene import Track, Vehicle, build_model
from .driving import DRIVING_SCENARIOS, DriveConfig, apply_tire_forces, scripted_driver, write_command
from .lidar import LidarRecording, MountedLidar
from .lean import LeanState, integrate_bicycle, integrate_bicycle_fused


@dataclass
class Trajectory:
    times: np.ndarray
    poses: np.ndarray
    velocities: np.ndarray
    metadata: dict
    actions: np.ndarray  # applied throttle, brake, steering (rad), aligned with poses
    lidar: LidarRecording | None = None


class Simulation:
    def __init__(self, track=None, vehicle=None, scenario="drop", device=None, physics_hz=240, drive=None, lidar=None, engine="newton", num_envs=1):
        if not isinstance(physics_hz, int) or physics_hz <= 0:
            raise ValueError("physics_hz must be a positive integer")
        if lidar is not None and physics_hz % lidar.frequency:
            raise ValueError("LiDAR frequency must divide physics_hz")
        if engine not in ("newton", "lean"):
            raise ValueError("engine must be newton or lean")
        if engine == "lean" and scenario not in DRIVING_SCENARIOS:
            raise ValueError("Lean physics supports driving scenarios; drop/impact need Newton")
        if not isinstance(num_envs, int) or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        if engine == "newton" and num_envs != 1:
            raise ValueError("Batched environments require lean physics")
        self.num_envs = num_envs
        self.engine = engine
        self.driving = scenario in DRIVING_SCENARIOS
        default_track = Track(length=8, width=8) if scenario == "circle" else Track(length=12, width=6)
        self.track = track or (default_track if self.driving else Track())
        self.vehicle = vehicle or Vehicle()
        self.drive = drive or DriveConfig()
        if self.driving:
            self.drive.validate_vehicle(self.vehicle)
            if physics_hz < 120:
                raise ValueError("Driving requires physics_hz >= 120")
        self.scenario, self.physics_hz = scenario, physics_hz
        self.dt = 1.0 / physics_hz
        if engine == "newton":
            import newton
            self.model, self.body, self.initial_velocity = build_model(self.track, self.vehicle, scenario, device, self.drive)
            self.solver = newton.solvers.SolverXPBD(self.model, iterations=8, angular_damping=0.05)
            self.pipeline = newton.CollisionPipeline(self.model)
            self.contacts = self.pipeline.contacts()
            self.control = self.model.control()
            self.state, self.next_state = self.model.state(), self.model.state()
        else:
            wp.init()
            chosen_device = wp.get_device(device or ("cuda:0" if wp.is_cuda_available() else "cpu"))
            if min(self.track.length, self.track.width) <= np.hypot(self.vehicle.length, self.vehicle.width):
                raise ValueError("Lean enclosure must accommodate the car at all headings")
            self.model = SimpleNamespace(device=chosen_device)
            self.body = 0
            self.state, self.next_state = LeanState(chosen_device, num_envs), LeanState(chosen_device, num_envs)
            x = -self.track.length / 3 if scenario in ("accelerate-brake", "s-turn") else 0.0
            y = -min(1.3, self.track.width / 4) if scenario == "circle" else 0.0
            # Keep the full chassis inside small enclosures, with clearance so
            # the first stationary step does not count as a wall contact.
            limit_x = (self.track.length - self.vehicle.length) / 2
            limit_y = (self.track.width - self.vehicle.width) / 2
            x = float(np.clip(x, -0.99 * limit_x, 0.99 * limit_x))
            y = float(np.clip(y, -0.99 * limit_y, 0.99 * limit_y))
            self.initial_pose = np.array([[x, y, self.drive.ride_height(self.vehicle), 0, 0, 0, 1]], dtype=np.float32)
            self.initial_pose = np.repeat(self.initial_pose, num_envs, axis=0)
            self.lean_motion = wp.zeros(num_envs, dtype=wp.vec4, device=chosen_device)
            self.collision = wp.zeros(num_envs, dtype=int, device=chosen_device)
            self.wall_contact_substeps = wp.zeros(num_envs, dtype=int, device=chosen_device)
        self.device_commands = wp.zeros(num_envs, dtype=wp.vec4, device=self.model.device)
        self._uploaded_command = None
        self.applied_controls = wp.zeros(num_envs, dtype=wp.vec3, device=self.model.device)
        self.lidar = MountedLidar(self.track, self.vehicle, lidar, self.model.device, batch_size=num_envs) if lidar is not None else None
        self.lidar_stride = physics_hz // lidar.frequency if lidar is not None else 0
        self.reset()

    def reset(self):
        if self.engine == "newton":
            import newton
            for state in (self.state, self.next_state):
                newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, state)
                state.body_qd.assign(self.initial_velocity[None])
                state.clear_forces()
        else:
            for state in (self.state, self.next_state):
                state.body_q.assign(self.initial_pose)
                state.body_qd.zero_()
            self.lean_motion.zero_()
            self.collision.zero_()
            self.wall_contact_substeps.zero_()
        self.steps = 0
        self.command = (0.0, 0.0, 0.0)
        self.target_speed = -1.0
        self.applied_controls.zero_()
        self._uploaded_command = None
        self._upload_command()
        if self.lidar is not None:
            self.lidar.update(self.state, self.body, 0.0)

    def set_action(self, throttle=0.0, brake=0.0, steering=0.0):
        """Forward throttle/brake in [0,1], steering in radians. Brake takes priority."""
        if not self.driving:
            raise ValueError("Use a driving scenario to apply vehicle controls")
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
        if not self.driving or not np.isfinite(steering):
            raise ValueError("A driving scenario and finite steering are required")
        self.command = (0.0, 0.0, float(np.clip(steering, -self.drive.max_steering, self.drive.max_steering)))
        self.target_speed = float(speed)
        self._upload_command()

    def _upload_command(self):
        payload = (*self.command, self.target_speed)
        if payload != self._uploaded_command:
            wp.launch(write_command, dim=self.num_envs, inputs=[self.device_commands, wp.vec4(*payload)],
                      device=self.model.device)
            self._uploaded_command = payload

    def set_commands(self, commands):
        """Upload per-car [throttle, brake, steering, target_speed] commands.

        Shape is (num_envs, 4). A target of -1 selects direct throttle/brake;
        a nonnegative target selects speed control. Values are checked/clamped.
        For zero-copy GPU controllers, write device_commands on the same stream
        and invalidate_command_cache() before calling scalar setters again.
        """
        values = np.array(commands, dtype=np.float32, copy=True)
        if not self.driving or values.shape != (self.num_envs, 4) or not np.isfinite(values).all():
            raise ValueError("Commands must be finite (num_envs, 4) in a driving scenario")
        if np.any((values[:, 3] < 0) & (values[:, 3] != -1)):
            raise ValueError("Target speeds must be -1 (manual) or nonnegative")
        values[:, :2] = np.clip(values[:, :2], 0, 1)
        values[:, 2] = np.clip(values[:, 2], -self.drive.max_steering, self.drive.max_steering)
        self.device_commands.assign(values)
        self.invalidate_command_cache()

    def invalidate_command_cache(self):
        self._uploaded_command = None

    def _lean_inputs(self):
        d, v = self.drive, self.vehicle
        return [
            self.state.body_q, self.lean_motion, self.device_commands,
            self.applied_controls, self.collision, self.next_state.body_q, self.next_state.body_qd,
            self.wall_contact_substeps,
            self.dt, v.mass, v.length, v.width, d.wheelbase, v.friction, d.lateral_stiffness,
            d.rolling_drag, d.max_acceleration, d.max_braking, d.steering_rate, d.speed_gain,
            self.track.length, self.track.width,
        ]

    def _fused_physics_steps(self, substeps):
        inputs = self._lean_inputs()
        # In-place output keeps the same pointers across every graph replay.
        inputs[5], inputs[6] = self.state.body_q, self.state.body_qd
        wp.launch(integrate_bicycle_fused, dim=self.num_envs, inputs=inputs + [substeps],
                  device=self.model.device)

    def _physics_step(self):
        """One capture-safe substep; callers manage time and sensor scheduling."""
        if self.engine == "lean":
            wp.launch(integrate_bicycle, dim=self.num_envs, inputs=self._lean_inputs(),
                      device=self.model.device)
            self.state, self.next_state = self.next_state, self.state
            return
        self.state.clear_forces()
        if self.driving:
            d, v = self.drive, self.vehicle
            wp.launch(apply_tire_forces, dim=1, inputs=[
                self.state.body_q, self.state.body_qd, self.state.body_f, self.applied_controls,
                self.body, self.device_commands, self.dt, v.mass, v.height,
                v.friction, d.wheelbase, d.track_width, d.support_length, d.spring_stiffness,
                d.spring_damping, d.lateral_stiffness, d.rolling_drag, d.max_acceleration,
                d.max_braking, d.steering_rate, d.speed_gain,
            ], device=self.model.device)
        self.pipeline.collide(self.state, self.contacts)
        self.solver.step(self.state, self.next_state, self.control, self.contacts, self.dt)
        self.state, self.next_state = self.next_state, self.state

    def step(self):
        self._physics_step()
        self.steps += 1
        if self.lidar is not None and self.steps % self.lidar_stride == 0:
            self.lidar.update(self.state, self.body, self.steps * self.dt)

    def snapshot(self):
        q, qd = self.state.body_q.numpy(), self.state.body_qd.numpy()
        if self.num_envs == 1:
            return q[self.body].copy(), qd[self.body].copy()
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
        wp.synchronize_device(self.model.device)
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
        wp.synchronize_device(self.model.device)
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
            "engine": self.engine, "scenario": self.scenario, "device": str(self.model.device),
            "physics_hz": self.physics_hz, "record_fps": record_fps if record else 0,
            "simulated_seconds": total_steps * self.dt, "elapsed_seconds": elapsed,
            "steps_per_second": total_steps / elapsed,
            "timing_includes_pose_recording": record,
            "track": asdict(self.track), "vehicle": asdict(self.vehicle),
            "drive": asdict(self.drive) if self.driving else None,
            "lidar": ({**asdict(self.lidar.config), "mount_position": self.lidar.mount_position,
                       "frame": "X forward, Y left, Z up", "pose_layout": "x,y,z,qx,qy,qz,qw",
                       "invalid_range": self.lidar.config.far,
                       "self_vehicle_excluded": True} if self.lidar is not None else None),
            "action_layout": "throttle,brake,steering_rad",
            "controller": runner.controller if runner is not None else "callback",
            "backend": runner.backend if runner is not None else "eager",
            "integrator": runner.integrator if runner is not None else "unfused",
            "wall_contact_substeps": int(self.wall_contact_substeps.numpy()[0]) if self.engine == "lean" else None,
            "pose_layout": "x,y,z,qx,qy,qz,qw",
            "velocity_layout": "vx,vy,vz,wx,wy,wz",
        }
        return Trajectory(np.asarray(times), poses, velocities, metadata, np.asarray(actions), lidar_recording)
