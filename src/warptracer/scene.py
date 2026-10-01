"""Shared geometry for physics and playback. Units: meters, kilograms, seconds."""
from dataclasses import dataclass

import newton
import numpy as np
import warp as wp


@dataclass(frozen=True)
class Track:
    straight_length: float = 6.0
    bend_radius: float = 2.5
    lane_width: float = 1.6
    barrier_height: float = 0.3
    barrier_thickness: float = 0.12
    segments_per_turn: int = 24

    def __post_init__(self):
        dimensions = (self.straight_length, self.bend_radius, self.lane_width,
                      self.barrier_height, self.barrier_thickness)
        if not np.isfinite(dimensions).all() or min(dimensions) <= 0:
            raise ValueError("Track dimensions must be finite and positive")
        if self.lane_width >= 2 * self.bend_radius:
            raise ValueError("The inner bend radius must be positive")
        if not isinstance(self.segments_per_turn, int) or self.segments_per_turn < 8:
            raise ValueError("Use at least eight segments per turn")

    def loop(self, radius=None):
        """Counterclockwise stadium boundary, without a duplicate endpoint."""
        radius = self.bend_radius if radius is None else radius
        n = self.segments_per_turn + 1
        right = np.linspace(-np.pi / 2, np.pi / 2, n)
        left = np.linspace(np.pi / 2, 3 * np.pi / 2, n)
        x = np.r_[self.straight_length / 2 + radius * np.cos(right),
                  -self.straight_length / 2 + radius * np.cos(left)]
        y = radius * np.sin(np.r_[right, left])
        return np.column_stack((x, y))

    def barriers(self):
        """Position, full dimensions, yaw, color for each fixed box collider."""
        for boundary, radius in enumerate((self.bend_radius - self.lane_width / 2,
                                            self.bend_radius + self.lane_width / 2)):
            points = self.loop(radius)
            for i, (a, b) in enumerate(zip(points, np.roll(points, -1, axis=0))):
                delta = b - a
                midpoint = (a + b) / 2
                # Overlap neighboring blocks slightly to close seams on bends.
                size = (float(np.linalg.norm(delta)) + self.barrier_thickness,
                        self.barrier_thickness, self.barrier_height)
                color = (0.92, 0.22, 0.18) if (i + boundary) % 2 else (0.9, 0.92, 0.94)
                yield (float(midpoint[0]), float(midpoint[1]), self.barrier_height / 2), size, float(np.arctan2(delta[1], delta[0])), color

    def road_mesh(self):
        inner = self.loop(self.bend_radius - self.lane_width / 2)
        outer = self.loop(self.bend_radius + self.lane_width / 2)
        n = len(inner)
        vertices = np.column_stack((np.vstack((inner, outer)), np.full(2 * n, 0.002)))
        triangles = []
        for i in range(n):
            j = (i + 1) % n
            triangles.extend(((i, i + n, j + n), (i, j + n, j)))
        return vertices.astype(np.float32), np.asarray(triangles, dtype=np.uint32)


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
    for i, (position, size, yaw, color) in enumerate(track.barriers()):
        builder.add_shape_box(
            -1, xform=wp.transform(position, wp.quat_from_axis_angle(wp.vec3(0, 0, 1), yaw)),
            hx=size[0] / 2, hy=size[1] / 2, hz=size[2] / 2,
            cfg=surface, color=color, label=f"barrier_{i}",
        )
    if scenario == "drop":
        position = (0.0, -track.bend_radius, 0.55)
        rotation = wp.quat_rpy(0.18, -0.12, 0.08)
        velocity = (0.5, 0.0, 0.0, 0.0, 0.0, 0.0)
    else:
        position = (0.0, -track.bend_radius, vehicle.height / 2 + 0.015)
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
