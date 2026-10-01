"""Fixed-step Newton physics, independent of display and recording libraries."""
from dataclasses import asdict, dataclass
import time

import newton
import numpy as np
import warp as wp

from .scene import Track, Vehicle, build_model


@dataclass
class Trajectory:
    times: np.ndarray
    poses: np.ndarray
    velocities: np.ndarray
    metadata: dict


class Simulation:
    def __init__(self, track=None, vehicle=None, scenario="drop", device=None, physics_hz=240):
        if not isinstance(physics_hz, int) or physics_hz <= 0:
            raise ValueError("physics_hz must be a positive integer")
        self.track, self.vehicle = track or Track(), vehicle or Vehicle()
        self.scenario, self.physics_hz = scenario, physics_hz
        self.dt = 1.0 / physics_hz
        self.model, self.body, self.initial_velocity = build_model(self.track, self.vehicle, scenario, device)
        self.solver = newton.solvers.SolverXPBD(self.model, iterations=8, angular_damping=0.05)
        self.pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.pipeline.contacts()
        self.control = self.model.control()
        self.state, self.next_state = self.model.state(), self.model.state()
        self.reset()

    def reset(self):
        for state in (self.state, self.next_state):
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, state)
            # Newton spatial velocity order is linear XYZ, then angular XYZ.
            state.body_qd.assign(self.initial_velocity[None])
            state.clear_forces()
        self.steps = 0

    def step(self):
        self.state.clear_forces()
        self.pipeline.collide(self.state, self.contacts)
        self.solver.step(self.state, self.next_state, self.control, self.contacts, self.dt)
        self.state, self.next_state = self.next_state, self.state
        self.steps += 1

    def snapshot(self):
        return self.state.body_q.numpy()[self.body].copy(), self.state.body_qd.numpy()[self.body].copy()

    def run(self, duration=4.0, record_fps=30, record=True):
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
        poses, velocities = [], []
        times = []

        def sample():
            q, qd = self.snapshot()
            poses.append(q)
            velocities.append(qd)
            times.append(self.steps * self.dt)

        sample()
        start = time.perf_counter()
        for i in range(1, total_steps + 1):
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
            "pose_layout": "x,y,z,qx,qy,qz,qw",
            "velocity_layout": "vx,vy,vz,wx,wy,wz",
        }
        return Trajectory(np.asarray(times), poses, velocities, metadata)
