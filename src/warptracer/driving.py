"""Minimal flat-ground tire forces. No wheel meshes, joints, or wheel spin states."""
from dataclasses import dataclass
import math

import numpy as np
import warp as wp


DRIVING_SCENARIOS = ("drive", "accelerate-brake", "circle", "s-turn")


@dataclass(frozen=True)
class DriveConfig:
    wheelbase: float = 0.32
    track_width: float = 0.22
    support_length: float = 0.05
    spring_stiffness: float = 1200.0  # N/m per contact
    spring_damping: float = 25.0     # N s/m per contact
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
        return vehicle.height / 2 + self.support_length - vehicle.mass * 9.81 / (4 * self.spring_stiffness)

    def validate_vehicle(self, vehicle):
        if self.wheelbase > vehicle.length or self.track_width > vehicle.width:
            raise ValueError("Tire contact points must fit inside the chassis footprint")
        if self.ride_height(vehicle) <= vehicle.height / 2 + 0.005:
            raise ValueError("Support springs must hold the chassis clear of the floor")


@wp.kernel
def apply_tire_forces(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    controls: wp.array(dtype=wp.vec3),
    body: int, command: wp.vec3, target_speed: float,
    dt: float, mass: float, height: float, friction: float,
    wheelbase: float, track_width: float, support_length: float,
    spring: float, damping: float, lateral_stiffness: float, drag: float,
    max_acceleration: float, max_braking: float, steering_rate: float, speed_gain: float,
):
    # Newton spatial arrays store linear components first, angular components last.
    pose = body_q[body]
    rotation = wp.transform_get_rotation(pose)
    center = wp.transform_get_translation(pose)  # centered box => origin is COM
    linear = wp.spatial_top(body_qd[body])
    angular = wp.spatial_bottom(body_qd[body])
    forward = wp.quat_rotate(rotation, wp.vec3(1.0, 0.0, 0.0))
    up = wp.quat_rotate(rotation, wp.vec3(0.0, 0.0, 1.0))
    throttle = command[0]
    brake = command[1]
    if target_speed >= 0.0:
        requested_acceleration = speed_gain * (target_speed - wp.dot(linear, forward))
        throttle = wp.clamp(requested_acceleration / max_acceleration, 0.0, 1.0)
        brake = wp.clamp(-requested_acceleration / max_braking, 0.0, 1.0)
        if target_speed == 0.0:
            throttle = 0.0
            brake = 1.0
    if brake > 0.0:
        throttle = 0.0
    previous_steering = controls[0][2]
    steering = previous_steering + wp.clamp(command[2] - previous_steering,
                                           -steering_rate * dt, steering_rate * dt)
    controls[0] = wp.vec3(throttle, brake, steering)
    force = wp.vec3(0.0)
    torque = wp.vec3(0.0)
    # Analytic contacts against z=0 only; no tire support when overturned.
    if up[2] > 0.5:
        for i in range(4):
            axle = float(1 - 2 * (i // 2))
            side = float(1 - 2 * (i % 2))
            anchor = wp.transform_point(pose, wp.vec3(axle * wheelbase * 0.5,
                                                      side * track_width * 0.5, -height * 0.5))
            compression = support_length - anchor[2]
            if compression > 0.0:
                anchor_velocity = linear + wp.cross(angular, anchor - center)
                normal = wp.clamp(spring * compression - damping * anchor_velocity[2],
                                  0.0, mass * 9.81)
                point = wp.vec3(anchor[0], anchor[1], 0.0)
                arm = point - center
                point_velocity = linear + wp.cross(angular, arm)
                angle = float(0.0)
                if i < 2:
                    angle = steering
                wheel_forward = wp.quat_rotate(rotation, wp.vec3(wp.cos(angle), wp.sin(angle), 0.0))
                wheel_forward = wp.normalize(wp.vec3(wheel_forward[0], wheel_forward[1], 0.0))
                wheel_left = wp.vec3(-wheel_forward[1], wheel_forward[0], 0.0)
                longitudinal_speed = wp.dot(point_velocity, wheel_forward)
                lateral_speed = wp.dot(point_velocity, wheel_left)
                # Smooth braking near zero speed; cap the per-step stopping impulse.
                braking = brake * max_braking * mass * 0.25 * wp.clamp(longitudinal_speed / 0.2, -1.0, 1.0)
                stopping_force = wp.abs(longitudinal_speed) * mass * 0.25 / dt
                braking = wp.clamp(braking, -stopping_force, stopping_force)
                longitudinal = throttle * max_acceleration * mass * 0.25 - braking - drag * longitudinal_speed
                lateral = -lateral_stiffness * lateral_speed
                # Both components share the same friction budget.
                tangent = wp.vec2(longitudinal, lateral)
                scale = wp.min(1.0, friction * normal / wp.max(wp.length(tangent), 1.0e-6))
                tire_force = scale * (longitudinal * wheel_forward + lateral * wheel_left) + wp.vec3(0.0, 0.0, normal)
                force += tire_force
                torque += wp.cross(arm, tire_force)
    body_f[body] = body_f[body] + wp.spatial_vector(force, torque)


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
