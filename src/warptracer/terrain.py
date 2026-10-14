"""Analytic oval road, shared by lean motion, LiDAR and replay geometry."""
from dataclasses import dataclass

import numpy as np
import warp as wp


@wp.func
def road_height(x: float, elevation: float, frequency: float):
    return 0.5 * elevation * (1.0 + wp.sin(frequency * x))


@wp.func
def road_gradient(x: float, elevation: float, frequency: float):
    return 0.5 * elevation * frequency * wp.cos(frequency * x)


@wp.func
def road_rotation(heading: float, gradient: float):
    normal = wp.normalize(wp.vec3(-gradient, 0.0, 1.0))
    forward = wp.normalize(wp.vec3(wp.cos(heading), wp.sin(heading), gradient * wp.cos(heading)))
    left = wp.cross(normal, forward)
    return wp.quat_from_matrix(wp.mat33(
        forward[0], left[0], normal[0],
        forward[1], left[1], normal[1],
        forward[2], left[2], normal[2]))


@dataclass(frozen=True)
class OvalTrack:
    """Elliptical ring with a gentle sine-shaped rise along world X.

    length/width are the outer road dimensions; lane_width is the difference
    between inner and outer ellipse axes. No overhangs or banking are added.
    """
    length: float = 16.0
    width: float = 10.0
    lane_width: float = 2.0
    elevation: float = 0.4
    barrier_height: float = 1.0
    barrier_thickness: float = 0.12
    segments: int = 192
    road_strips: int = 8

    def __post_init__(self):
        values = (self.length, self.width, self.lane_width, self.elevation,
                  self.barrier_height, self.barrier_thickness)
        if not np.isfinite(values).all() or min(values[:3] + values[4:]) <= 0 or self.elevation < 0:
            raise ValueError("Oval dimensions must be finite and positive; elevation nonnegative")
        if self.length < self.width or self.lane_width >= self.width / 2:
            raise ValueError("Oval requires length >= width and lane_width < width/2")
        if self.barrier_thickness >= self.width / 2 - self.lane_width:
            raise ValueError("Oval inner barrier thickness must leave an open center")
        if not isinstance(self.segments, int) or self.segments < 32 or self.segments % 4:
            raise ValueError("Oval segments must be a multiple of four, at least 32")
        if not isinstance(self.road_strips, int) or self.road_strips < 1:
            raise ValueError("Oval road_strips must be a positive integer")
        if self.elevation * self.frequency / 2 > .2:
            raise ValueError("Oval grade must be <= 20%; use gentler elevation")

    @property
    def frequency(self):
        return 2 * np.pi / self.length

    @property
    def inner_axes(self):
        return self.length / 2 - self.lane_width, self.width / 2 - self.lane_width

    @property
    def center_axes(self):
        return (self.length - self.lane_width) / 2, (self.width - self.lane_width) / 2

    def height(self, x, y=0):
        return .5 * self.elevation * (1 + np.sin(self.frequency * np.asarray(x)))

    def gradient(self, x):
        return .5 * self.elevation * self.frequency * np.cos(self.frequency * np.asarray(x))

    def pose(self, angle, ride_height):
        """Centerline spawn with heading tangent to the loop (counterclockwise)."""
        a, b = self.center_axes
        x, y = a * np.cos(angle), b * np.sin(angle)
        heading = np.arctan2(b * np.cos(angle), -a * np.sin(angle))
        gradient = float(self.gradient(x))
        rotation = road_rotation(float(heading), gradient)
        z = self.height(x) + ride_height * np.sqrt(1 + gradient * gradient)
        return np.array([x, y, z, *rotation], dtype=np.float32)

    def mesh_parts(self):
        """Three meshes: road, inner barrier and outer barrier; no asset files.

        The same sampled vertices/faces are used for replay and raycasting.
        Closed barrier solids follow the road's sampled elevation.
        """
        theta = np.arange(self.segments) * (2 * np.pi / self.segments)

        def ring(a, b, height=0):
            x, y = a * np.cos(theta), b * np.sin(theta)
            return np.column_stack((x, y, self.height(x) + height))

        def connect(rings, pairs):
            vertices = np.concatenate(rings).astype(np.float32)
            faces = []
            for u, v in pairs:
                for i in range(self.segments):
                    j = (i + 1) % self.segments
                    a, b = u * self.segments + i, u * self.segments + j
                    c, d = v * self.segments + j, v * self.segments + i
                    faces.extend(((a, b, c), (a, c, d)))
            return vertices, np.asarray(faces, dtype=np.int32)

        outer = np.array([self.length / 2, self.width / 2])
        inner = np.array(self.inner_axes)
        road_rings = [ring(*(inner + f * (outer - inner)))
                      for f in np.linspace(0, 1, self.road_strips + 1)]
        parts = {"road": connect(road_rings, [(i + 1, i) for i in range(self.road_strips)])}
        for name, axes, direction in (("inner_barrier", inner, -1), ("outer_barrier", outer, 1)):
            outside = axes + direction * self.barrier_thickness
            rings = [ring(*axes), ring(*axes, self.barrier_height),
                     ring(*outside, self.barrier_height), ring(*outside)]
            parts[name] = connect(rings, [(0, 1), (1, 2), (2, 3), (3, 0)])
        return parts

    def mesh(self):
        vertices, faces = [], []
        offset = 0
        for v, f in self.mesh_parts().values():
            vertices.append(v)
            faces.append(f + offset)
            offset += len(v)
        return np.concatenate(vertices), np.concatenate(faces)
