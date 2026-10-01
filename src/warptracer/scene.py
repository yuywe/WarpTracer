"""Shared geometry for physics and playback. Units: meters, kilograms, seconds."""
from dataclasses import dataclass

import newton
import numpy as np
import warp as wp


@dataclass(frozen=True)
class Track:
    """Flat rectangle; length and width measure the clear space inside the walls."""
    length: float = 6.0
    width: float = 4.0
    barrier_height: float = 0.3
    barrier_thickness: float = 0.12

    def __post_init__(self):
        dimensions = (self.length, self.width, self.barrier_height, self.barrier_thickness)
        if not np.isfinite(dimensions).all() or min(dimensions) <= 0:
            raise ValueError("Track dimensions must be finite and positive")

    def barriers(self):
        """Position and full dimensions for four fixed, axis-aligned box walls."""
        length, width = self.length, self.width
        thickness, height = self.barrier_thickness, self.barrier_height
        for side in (-1, 1):
            yield (0, side * (width + thickness) / 2, height / 2), (length + 2 * thickness, thickness, height)
            yield (side * (length + thickness) / 2, 0, height / 2), (thickness, width, height)


@dataclass(frozen=True)
class Vehicle:
    length: float = 0.52
    width: float = 0.26
    height: float = 0.12
    mass: float = 3.2
    friction: float = 0.6

    def __post_init__(self):
        if not np.isfinite((self.length, self.width, self.height, self.mass, self.friction)).all():
            raise ValueError("Vehicle parameters must be finite")
        if min(self.length, self.width, self.height, self.mass) <= 0 or self.friction < 0:
            raise ValueError("Vehicle dimensions/mass must be positive; friction nonnegative")

    @property
    def dimensions(self):
        return self.length, self.width, self.height


def build_model(track, vehicle, scenario="drop", device=None):
    """One free body; all track shapes are static and do not add dynamic bodies."""
    if scenario not in ("drop", "wall-impact"):
        raise ValueError("scenario must be 'drop' or 'wall-impact'")
    wp.init()
    device = wp.get_device(device or ("cuda:0" if wp.is_cuda_available() else "cpu"))
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, -9.81))
    surface = newton.ModelBuilder.ShapeConfig(mu=vehicle.friction, gap=0.005)
    builder.add_ground_plane(cfg=surface, label="floor")
    for i, (position, size) in enumerate(track.barriers()):
        builder.add_shape_box(
            -1, xform=wp.transform(position, wp.quat_identity()),
            hx=size[0] / 2, hy=size[1] / 2, hz=size[2] / 2,
            cfg=surface, color=(0.6, 0.6, 0.6), label=f"barrier_{i}",
        )
    if scenario == "drop":
        position = (0.0, 0.0, 0.55)
        rotation = wp.quat_rpy(0.18, -0.12, 0.08)
        velocity = (0.5, 0.0, 0.0, 0.0, 0.0, 0.0)
    else:
        position = (0.0, -track.width / 2 + vehicle.width / 2 + 0.6, vehicle.height / 2 + 0.015)
        rotation = wp.quat_identity()
        # An initial sideways velocity, not an actuator or tire model.
        velocity = (0.0, -3.0, 0.0, 0.0, 0.0, 0.0)
    body = builder.add_body(xform=wp.transform(position, rotation), label="chassis")
    chassis_cfg = newton.ModelBuilder.ShapeConfig(
        density=float(vehicle.mass / np.prod(vehicle.dimensions)), mu=vehicle.friction, gap=0.005,
    )
    builder.add_shape_box(body, hx=vehicle.length / 2, hy=vehicle.width / 2,
                          hz=vehicle.height / 2, cfg=chassis_cfg, color=(0.08, 0.42, 0.95), label="chassis_box")
    model = builder.finalize(device=device)
    return model, body, np.asarray(velocity, dtype=np.float32)
