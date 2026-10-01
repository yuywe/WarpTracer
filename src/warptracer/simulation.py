"""Fixed-step Newton physics, independent of display and recording libraries."""
from dataclasses import asdict, dataclass
import time

import newton
import numpy as np
import warp as wp

from .scene import Track, Vehicle, build_model
from .driving import DRIVING_SCENARIOS, DriveConfig, apply_tire_forces, scripted_driver


@dataclass
class Trajectory:
    times: np.ndarray
    poses: np.ndarray
    velocities: np.ndarray
    metadata: dict
    actions: np.ndarray  # applied throttle, brake, steering (rad), aligned with poses


class Simulation:
    def __init__(self, track=None, vehicle=None, scenario="drop", device=None, physics_hz=240, drive=None):
        if not isinstance(physics_hz, int) or physics_hz <= 0:
            raise ValueError("physics_hz must be a positive integer")
        self.driving = scenario in DRIVING_SCENARIOS
        default_track = Track(length=8, width=8) if scenario == "circle" else Track(length=12, width=6)
        self.track = track or (default_track if self.driving else Track())
        self.vehicle = vehicle or Vehicle()
        self.drive = drive or DriveConfig()
        if self.driving:
            self.drive.validate_vehicle(self.vehicle)
            if physics_hz < 120:
                raise ValueError("Driving support springs require physics_hz >= 120")
        self.scenario, self.physics_hz = scenario, physics_hz
        self.dt = 1.0 / physics_hz
        self.model, self.body, self.initial_velocity = build_model(self.track, self.vehicle, scenario, device, self.drive)
        self.solver = newton.solvers.SolverXPBD(self.model, iterations=8, angular_damping=0.05)
        self.pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.pipeline.contacts()
        self.control = self.model.control()
        self.state, self.next_state = self.model.state(), self.model.state()
        self.applied_controls = wp.zeros(1, dtype=wp.vec3, device=self.model.device)
        self.reset()

    def reset(self):
        for state in (self.state, self.next_state):
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, state)
            # Newton spatial velocity order is linear XYZ, then angular XYZ.
            state.body_qd.assign(self.initial_velocity[None])
            state.clear_forces()
        self.steps = 0
        self.command = (0.0, 0.0, 0.0)
        self.target_speed = -1.0
        self.applied_controls.zero_()

    def set_action(self, throttle=0.0, brake=0.0, steering=0.0):
        """Forward throttle/brake in [0,1], steering in radians. Brake takes priority."""
        if not self.driving:
            raise ValueError("Use a driving scenario to apply vehicle controls")
        if not np.isfinite((throttle, brake, steering)).all():
            raise ValueError("Controls must be finite")
        self.command = (float(np.clip(throttle, 0, 1)), float(np.clip(brake, 0, 1)),
                        float(np.clip(steering, -self.drive.max_steering, self.drive.max_steering)))
        self.target_speed = -1.0

    def set_target_speed(self, speed, steering=0.0):
        """Optional proportional controller; nonnegative target speed in m/s."""
        if not np.isfinite(speed) or speed < 0:
            raise ValueError("Target speed must be finite and nonnegative")
        self.set_action(steering=steering)
        self.target_speed = float(speed)

    def step(self):
        self.state.clear_forces()
        if self.driving:
            d, v = self.drive, self.vehicle
            wp.launch(apply_tire_forces, dim=1, inputs=[
                self.state.body_q, self.state.body_qd, self.state.body_f, self.applied_controls,
                self.body, wp.vec3(*self.command), self.target_speed, self.dt, v.mass, v.height,
                v.friction, d.wheelbase, d.track_width, d.support_length, d.spring_stiffness,
                d.spring_damping, d.lateral_stiffness, d.rolling_drag, d.max_acceleration,
                d.max_braking, d.steering_rate, d.speed_gain,
            ], device=self.model.device)
        self.pipeline.collide(self.state, self.contacts)
        self.solver.step(self.state, self.next_state, self.control, self.contacts, self.dt)
        self.state, self.next_state = self.next_state, self.state
        self.steps += 1

    def snapshot(self):
        return self.state.body_q.numpy()[self.body].copy(), self.state.body_qd.numpy()[self.body].copy()

    def run(self, duration=4.0, record_fps=30, record=True, controller=None):
        """Run from reset. Optional controller(sim, time) sets inputs before each step."""
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("duration must be finite and positive")
        if not isinstance(record_fps, int) or record_fps <= 0 or self.physics_hz % record_fps:
            raise ValueError("record_fps must be a positive divisor of physics_hz")
        total_steps = round(duration * self.physics_hz)
        if total_steps < 1:
            raise ValueError("duration is shorter than one physics step")
        stride = self.physics_hz // record_fps
        # Compile/warm up outside the measurement, then restore the initial state.
        self.reset()
        for _ in range(2):
            self.step()
        wp.synchronize_device(self.model.device)
        self.reset()
        poses, velocities, actions = [], [], []
        controller = controller or scripted_driver
        times = []

        def sample():
            q, qd = self.snapshot()
            poses.append(q)
            velocities.append(qd)
            times.append(self.steps * self.dt)
            actions.append(self.applied_controls.numpy()[0].copy())

        sample()
        start = time.perf_counter()
        for i in range(1, total_steps + 1):
            controller(self, self.steps * self.dt)
            self.step()
            if record and i % stride == 0:
                sample()
        wp.synchronize_device(self.model.device)
        elapsed = time.perf_counter() - start
        if times[-1] < total_steps * self.dt:
            sample()
        poses, velocities = np.asarray(poses), np.asarray(velocities)
        if not np.isfinite(poses).all() or not np.isfinite(velocities).all():
            raise RuntimeError("Simulation produced non-finite state")
        metadata = {
            "scenario": self.scenario, "device": str(self.model.device),
            "physics_hz": self.physics_hz, "record_fps": record_fps if record else 0,
            "simulated_seconds": total_steps * self.dt, "elapsed_seconds": elapsed,
            "steps_per_second": total_steps / elapsed,
            "timing_includes_pose_recording": record,
            "track": asdict(self.track), "vehicle": asdict(self.vehicle),
            "drive": asdict(self.drive) if self.driving else None,
            "action_layout": "throttle,brake,steering_rad",
            "pose_layout": "x,y,z,qx,qy,qz,qw",
            "velocity_layout": "vx,vy,vz,wx,wy,wz",
        }
        return Trajectory(np.asarray(times), poses, velocities, metadata, np.asarray(actions))
