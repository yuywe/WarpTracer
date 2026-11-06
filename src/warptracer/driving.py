"""Driving parameters and scripted scenarios. No wheel meshes, joints, or wheel spin states."""
from dataclasses import dataclass
import math

import numpy as np
import warp as wp


DRIVING_SCENARIOS = ("drive", "accelerate-brake", "circle", "s-turn")


@dataclass(frozen=True)
class DriveConfig:
    wheelbase: float = 0.32
    ride_clearance: float = 0.04346  # chassis underside above the road, meters
    lateral_stiffness: float = 25.0  # N per m/s of sideways slip, per contact
    rolling_drag: float = 0.15      # N per m/s, per contact
    max_acceleration: float = 2.0   # requested m/s² before the grip limit
    max_braking: float = 4.0
    max_steering: float = 0.418     # radians; positive turns left
    steering_rate: float = 1.5     # rad/s
    speed_gain: float = 3.0        # optional proportional speed controller

    def __post_init__(self):
        values = tuple(vars(self).values())
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("Driving parameters must be finite and positive")
        if self.max_steering >= math.pi / 2:
            raise ValueError("max_steering must be less than pi/2")

    def ride_height(self, vehicle):
        return vehicle.height / 2 + self.ride_clearance

    def validate_vehicle(self, vehicle):
        if self.wheelbase > vehicle.length:
            raise ValueError("Axles must fit inside the chassis footprint")


@wp.kernel
def write_command(commands: wp.array(dtype=wp.vec4), value: wp.vec4):
    commands[wp.tid()] = value


def scripted_driver(sim, time):
    """Deterministic speed/steering requests; vehicle state stays on its device."""
    if sim.scenario == "accelerate-brake":
        sim.set_target_speed(1.8 if 0.5 <= time < 2.5 else 0.0)
    elif sim.scenario == "circle":
        sim.set_target_speed(1.0 if time >= 0.5 else 0.0, steering=0.25)
    elif sim.scenario == "s-turn":
        moving = 0.5 <= time < 8.5
        steering = 0.25 * math.sin(2 * math.pi * (time - 0.5) / 4) if moving else 0.0
        sim.set_target_speed(0.9 if moving else 0.0, steering=steering)
