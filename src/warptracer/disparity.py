"""Reactive LiDAR navigation, with all scan processing on the simulation device.

Disparities are extended toward their farther side by half the vehicle width
plus a margin. See Nathan Otterness' 2019 disparity extender description.
This prototype adds conservative braking and a side-clearance stop.
"""
from dataclasses import dataclass, asdict

import numpy as np
import warp as wp


@dataclass(frozen=True)
class DisparityConfig:
    max_speed: float = 1.5
    disparity_threshold: float = 0.5
    safety_margin: float = 0.15
    stop_margin: float = 0.3
    search_degrees: float = 80.0
    range_cap: float = 6.0

    def __post_init__(self):
        if not np.isfinite(list(asdict(self).values())).all():
            raise ValueError("Disparity parameters must be finite")
        if min(self.max_speed, self.disparity_threshold, self.stop_margin, self.range_cap) <= 0:
            raise ValueError("Speed, threshold, stop margin and range cap must be positive")
        if self.safety_margin < 0 or not 0 < self.search_degrees < 90:
            raise ValueError("Require nonnegative safety margin and search angle in (0, 90)")


@wp.func
def usable_range(ranges: wp.array2d(dtype=float), valid: wp.array2d(dtype=int),
                 car: int, ray: int, cap: float):
    value = ranges[car, ray]
    # The sensor mask combines misses and near-clipped returns. Treat unknown
    # space as blocked; interpreting a near-clipped obstacle as free is unsafe.
    if valid[car, ray] == 0 or not wp.isfinite(value) or value <= 0.0:
        return float(0.0)
    return wp.min(value, cap)


@wp.kernel
def extend_disparities(ranges: wp.array2d(dtype=float), valid: wp.array2d(dtype=int),
                       filtered: wp.array2d(dtype=float), beams: int,
                       increment: float, radius: float, threshold: float, cap: float):
    car, ray = wp.tid()
    distance = usable_range(ranges, valid, car, ray, cap)
    # Read only the original scan: overlapping extensions take their minimum
    # and cannot cascade or race with neighboring threads.
    for edge in range(beams - 1):
        left = usable_range(ranges, valid, car, edge, cap)
        right = usable_range(ranges, valid, car, edge + 1, cap)
        if wp.abs(left - right) > threshold:
            close = wp.min(left, right)
            span = int(wp.ceil(wp.atan2(radius, close) / increment))
            if left < right and ray > edge and ray <= edge + span:
                distance = wp.min(distance, close)
            if right < left and ray <= edge and ray > edge - span:
                distance = wp.min(distance, close)
    filtered[car, ray] = distance


@wp.kernel
def choose_command(ranges: wp.array2d(dtype=float), valid: wp.array2d(dtype=int),
                   filtered: wp.array2d(dtype=float), commands: wp.array(dtype=wp.vec4),
                   goals: wp.array(dtype=wp.vec2), turns: wp.array(dtype=float), beams: int, first: float, increment: float,
                   search: float, geometry_cap: float, max_speed: float, max_steering: float,
                   radius: float, half_length: float, mount_x: float, mount_y: float,
                   stop_margin: float, braking: float, reaction_time: float, turn_clearance: float):
    car = wp.tid()
    best_score = float(-1.0e10)
    angle = float(0.0)
    best_distance = float(0.0)
    clearance = geometry_cap
    left_clear = geometry_cap
    right_clear = geometry_cap
    left_space = float(0.0)
    right_space = float(0.0)
    for ray in range(beams):
        a = first + float(ray) * increment
        # Target scoring is capped, but turn hysteresis needs actual geometry.
        d = usable_range(ranges, valid, car, ray, geometry_cap)
        if a > 0.5 and a < 1.57:
            left_space += d
        if a < -0.5 and a > -1.57:
            right_space += d
        if wp.abs(a) <= search:
            candidate = filtered[car, ray]
            score = candidate - 0.1 * wp.abs(a)
            # A symmetric scan has a deterministic left-turn tie break.
            if score > best_score + 1.0e-5 or (wp.abs(score - best_score) <= 1.0e-5 and a > angle):
                best_score = score
                angle = a
                best_distance = candidate
        x, y = d * wp.cos(a) + mount_x, d * wp.sin(a) + mount_y
        if wp.abs(a) < 0.35 and d == 0.0:
            clearance = 0.0
        if wp.cos(a) > 0.0 and wp.abs(y) <= radius:
            clearance = wp.min(clearance, wp.max(0.0, x - half_length))
        if x >= -half_length and x <= half_length + stop_margin:
            if a > 0.0:
                left_clear = wp.min(left_clear, y)
            else:
                right_clear = wp.min(right_clear, -y)
    # Smooth room corners need not create disparities. Begin turning while
    # there is still room for the steering-limited car's turning circle, and
    # hold that direction until the forward corridor opens (hysteresis).
    turn = turns[car]
    if clearance > 1.5 * turn_clearance:
        turn = 0.0
    if clearance < turn_clearance and turn == 0.0:
        turn = 1.0
        if right_space > left_space:
            turn = -1.0
    if turn != 0.0:
        # Select a real beam inside the search sector and recheck its filtered
        # distance. Never retain the old target's clearance after an override.
        nearest = float(1.0e10)
        for ray in range(beams):
            a = first + float(ray) * increment
            error = wp.abs(a - turn * search)
            if wp.abs(a) <= search and error < nearest:
                nearest = error
                angle = a
                best_distance = filtered[car, ray]
    turns[car] = turn
    steering = wp.clamp(angle, -max_steering, max_steering)
    available = wp.max(0.0, clearance - stop_margin)
    target_clearance = best_distance + mount_x * wp.cos(angle) + mount_y * wp.sin(angle) - half_length
    available = wp.min(available, wp.max(0.0, target_clearance - stop_margin))
    # Account for one scan period before braking. Existing speed control and
    # steering-rate limits still act at every physics substep.
    delay = braking * reaction_time
    speed = wp.min(max_speed / (1.0 + 2.0 * wp.abs(steering) / max_steering),
                   wp.sqrt(delay * delay + 2.0 * braking * available) - delay)
    if target_clearance <= stop_margin or (steering > 0.0 and left_clear < radius) or (steering < 0.0 and right_clear < radius):
        speed = 0.0
        steering = 0.0
    commands[car] = wp.vec4(0.0, 0.0, steering, speed)
    goals[car] = wp.vec2(angle, best_distance)


class DisparityController:
    """Forward, horizontal LiDAR only; one independent controller per car.

    filtered and goals are borrowed device arrays for inspection. update() must
    run on the same stream as the scanner and vehicle. No CPU scan copies occur.
    """
    def __init__(self, sim, config=None):
        self.config = config or DisparityConfig()
        if not sim.driving or sim.lidar is None:
            raise ValueError("Disparity control requires a driving car with LiDAR")
        lc = sim.lidar.config
        if lc.beams < 3 or not 180 <= lc.fov_degrees < 360 or lc.mount_rpy != (0.0, 0.0, 0.0):
            raise ValueError("Disparity control requires >=3 beams, 180 <= FOV < 360, and an unrotated mount")
        self.sim = sim
        self.first = -np.deg2rad(lc.fov_degrees) / 2
        self.increment = np.deg2rad(lc.fov_degrees) / (lc.beams - 1)
        angles = self.first + np.arange(lc.beams) * self.increment
        if not np.any(np.abs(angles) < .35):
            raise ValueError("Disparity control requires LiDAR coverage within 20 degrees of forward")
        if not np.any(np.abs(angles) <= np.deg2rad(self.config.search_degrees)):
            raise ValueError("Disparity search sector must contain at least one LiDAR beam")
        self.filtered = wp.empty((sim.num_envs, lc.beams), dtype=float, device=sim.model.device)
        self.goals = wp.empty(sim.num_envs, dtype=wp.vec2, device=sim.model.device)
        self.turns = wp.zeros(sim.num_envs, dtype=float, device=sim.model.device)

    def reset(self):
        self.turns.zero_()
        self.filtered.zero_()
        self.goals.zero_()

    def update(self):
        sim, cfg = self.sim, self.config
        ranges, valid = sim.lidar.result.values, sim.lidar.result.valid
        beams = sim.lidar.config.beams
        radius = sim.vehicle.width / 2 + cfg.safety_margin
        cap = min(cfg.range_cap, sim.lidar.config.far)
        wp.launch(extend_disparities, dim=(sim.num_envs, beams), inputs=[
            ranges, valid, self.filtered, beams, self.increment, radius,
            cfg.disparity_threshold, cap], device=sim.model.device)
        wp.launch(choose_command, dim=sim.num_envs, inputs=[
            ranges, valid, self.filtered, sim.device_commands, self.goals, self.turns,
            beams, self.first, self.increment, np.deg2rad(cfg.search_degrees), sim.lidar.config.far,
            cfg.max_speed, sim.drive.max_steering, radius, sim.vehicle.length / 2,
            *sim.lidar.mount_position[:2], cfg.stop_margin,
            min(sim.drive.max_braking, sim.vehicle.friction * 9.81) * .5,
            1.0 / sim.lidar.config.frequency,
            2.0 * sim.drive.wheelbase / np.tan(sim.drive.max_steering) + sim.vehicle.length / 2 + cfg.stop_margin
            + cfg.max_speed * (sim.drive.max_steering / sim.drive.steering_rate
                               + 1.0 / sim.lidar.config.frequency)], device=sim.model.device)
        sim.invalidate_command_cache()

    def __call__(self, sim, time):
        if sim is not self.sim:
            raise ValueError("Controller belongs to a different simulation")
        if sim.steps % sim.lidar_stride == 0:
            if sim.steps == 0:
                self.reset()
            self.update()
