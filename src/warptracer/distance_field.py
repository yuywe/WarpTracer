"""EDT clearance-guided 3D ray marching over one terrain layer.

Distance transforms are prepared once on the CPU, then bilinearly sampled in
Warp. No cell traversal, tile hierarchy, sensor mesh or BVH is constructed.
XY distance fields guide jumps; height-gradient bounds limit ground/roof jumps.
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt
import warp as wp

# Model-free rollouts do not use environment adjoints.
wp.set_module_options({"enable_backward": False})

from racesense3d.core import Scan, validate_poses


def _gradient_bound(values, cell):
    """Maximum gradient norm of any bilinear patch (attained at a corner)."""
    dx = np.diff(values.astype(np.float64), axis=0) / cell
    dy = np.diff(values.astype(np.float64), axis=1) / cell
    bound = max(np.hypot(x, y).max() for x in (dx[:, :-1], dx[:, 1:])
                for y in (dy[:-1, :], dy[1:, :]))
    return float(bound) * (1 + 1e-5) + 1e-6


def _signed_distance(mask, cell):
    """Positive outside, negative inside; zero contour between sample centers.

    Half-cell correction reduces the EDT center-to-center boundary bias. The
    interpolant is normalized by its measured gradient bound, not assumed to
    be an exact continuous signed distance. Empty masks have no zero surface.
    """
    if not mask.any():
        return np.full(mask.shape, 1e6, np.float32)
    outside = distance_transform_edt(~mask, sampling=cell)
    inside = distance_transform_edt(mask, sampling=cell)
    values = np.where(mask, -inside + .5 * cell, outside - .5 * cell).astype(np.float32)
    return values / max(1., _gradient_bound(values, cell))


@dataclass(frozen=True)
class DistanceField:
    heights: np.ndarray      # Vertex samples [nx, ny], meters; arrays use XY order.
    surface: np.ndarray      # Binary road/base mask on the same samples.
    walls: np.ndarray        # Binary finite-height obstacle footprint.
    origin: tuple
    cell_size: float
    wall_height: float = 1.0

    def __post_init__(self):
        h = np.array(self.heights, dtype=np.float32, copy=True)
        if h.ndim != 2 or min(h.shape) < 2 or not np.isfinite(h).all():
            raise ValueError("Distance field requires finite 2D height samples, at least 2x2")
        masks = []
        for value in (self.surface, self.walls):
            value = np.asarray(value)
            if value.shape != h.shape or np.any((value != 0) & (value != 1)):
                raise ValueError("Distance-field masks must be binary and match height samples")
            masks.append(np.array(value, dtype=bool, copy=True))
        if (not np.isfinite(self.cell_size) or self.cell_size <= 0 or np.shape(self.origin) != (2,)
                or not np.isfinite(self.origin).all() or not np.isfinite(self.wall_height) or self.wall_height <= 0):
            raise ValueError("Cell size/wall height must be positive; origin must be finite XY")
        # Masks need an empty border so their entire zero contour is inside the
        # sampled domain. A filled border would make clipping hide a wall face.
        for mask in masks:
            if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
                raise ValueError("Distance-field masks require an empty outer sample border")
        for name, value in zip(("heights", "surface", "walls"), (h, *masks)):
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        object.__setattr__(self, "origin", tuple(float(x) for x in self.origin))

    @classmethod
    def oval(cls, track, cell_size=.025):
        if not np.isfinite(cell_size) or not 0 < cell_size <= track.barrier_thickness / 4:
            raise ValueError("Oval grid cell size must be finite, positive and <= barrier_thickness/4")
        half = np.array([track.length / 2, track.width / 2]) + track.barrier_thickness + 3 * cell_size
        size = 2 * np.ceil(half / cell_size).astype(int) + 1
        if np.prod(size, dtype=np.int64) > 4_000_000:
            raise ValueError("Oval distance field exceeds four million samples; increase cell size")
        origin = -(size - 1) * cell_size / 2
        x = origin[0] + np.arange(size[0]) * cell_size
        y = origin[1] + np.arange(size[1]) * cell_size
        xx, yy = np.meshgrid(x, y, indexing="ij")

        def inside(a, b):
            return (xx / a) ** 2 + (yy / b) ** 2 <= 1

        a, b = track.length / 2, track.width / 2
        ia, ib = track.inner_axes
        thickness = track.barrier_thickness
        road = inside(a, b) & ~inside(ia, ib)
        walls = ((inside(ia, ib) & ~inside(ia - thickness, ib - thickness)) |
                 (inside(a + thickness, b + thickness) & ~inside(a, b)))
        heights = np.broadcast_to(track.height(x)[:, None], (len(x), len(y)))
        return cls(heights, road | walls, walls, tuple(origin), cell_size, track.barrier_height)


@wp.func
def _sample(grid: wp.array2d(dtype=wp.vec3), origin: wp.vec2, cell: float, p: wp.vec3):
    fx = (p[0] - origin[0]) / cell
    fy = (p[1] - origin[1]) / cell
    i = wp.clamp(int(wp.floor(fx)), 0, grid.shape[0] - 2)
    j = wp.clamp(int(wp.floor(fy)), 0, grid.shape[1] - 2)
    x = wp.clamp(fx - float(i), 0.0, 1.0)
    y = wp.clamp(fy - float(j), 0.0, 1.0)
    return ((1.0 - x) * (1.0 - y) * grid[i, j] + x * (1.0 - y) * grid[i + 1, j] +
            (1.0 - x) * y * grid[i, j + 1] + x * y * grid[i + 1, j + 1])


@wp.func
def _closing_distance(gap: float, closing_rate: float):
    distance = float(1.0e30)
    if gap <= 0.0:
        distance = 0.0
    elif closing_rate > 1.0e-10:
        distance = wp.max(0.0, gap) / closing_rate
    return distance


@wp.func
def _march(grid: wp.array2d(dtype=wp.vec3), origin: wp.vec2, cell: float,
           wall_height: float, slope: float, z_bounds: wp.vec2,
           start: wp.vec3, direction: wp.vec3, far: float, tolerance: float, max_steps: int,
           cache_samples: bool):
    enter = float(0.0)
    leave = far
    # All zero surfaces lie inside this XYZ bounding box, including barrier roofs.
    for axis in range(3):
        low = z_bounds[0]
        high = z_bounds[1]
        if axis < 2:
            low = origin[axis]
            high = low + cell * float(grid.shape[axis] - 1)
        if wp.abs(direction[axis]) < 1.0e-10:
            if start[axis] < low or start[axis] > high:
                leave = -1.0
        else:
            a = (low - start[axis]) / direction[axis]
            b = (high - start[axis]) / direction[axis]
            enter = wp.max(enter, wp.min(a, b))
            leave = wp.min(leave, wp.max(a, b))
    t = enter
    hit = float(1.0e30)
    steps = int(0)
    horizontal = wp.sqrt(direction[0] * direction[0] + direction[1] * direction[1])
    floor_closing = slope * horizontal - direction[2]
    roof_closing = slope * horizontal + direction[2]
    vertical_scale = wp.sqrt(1.0 + slope * slope)
    cached_i = int(-1)
    cached_j = int(-1)
    node00 = wp.vec3()
    node10 = wp.vec3()
    node01 = wp.vec3()
    node11 = wp.vec3()
    while t <= leave and steps < max_steps:
        p = start + t * direction
        sample = wp.vec3()
        if cache_samples:
            fx = (p[0] - origin[0]) / cell
            fy = (p[1] - origin[1]) / cell
            i = wp.clamp(int(wp.floor(fx)), 0, grid.shape[0] - 2)
            j = wp.clamp(int(wp.floor(fy)), 0, grid.shape[1] - 2)
            x = wp.clamp(fx - float(i), 0.0, 1.0)
            y = wp.clamp(fy - float(j), 0.0, 1.0)
            # Convergence often revisits a patch. Keep its four immutable samples
            # in this ray's registers; interpolate afresh at every position.
            if i != cached_i or j != cached_j:
                node00 = grid[i, j]
                node10 = grid[i + 1, j]
                node01 = grid[i, j + 1]
                node11 = grid[i + 1, j + 1]
                cached_i = i
                cached_j = j
            sample = ((1.0 - x) * (1.0 - y) * node00 + x * (1.0 - y) * node10 +
                      (1.0 - x) * y * node01 + x * y * node11)
        else:
            sample = _sample(grid, origin, cell, p)
        # Packed channels: normalized wall distance, normalized road distance, height.
        wall_xy = sample[0]
        road_xy = sample[1]
        gap = p[2] - sample[2]
        wall_signed = wp.max(wall_xy, wp.max(-gap, gap - wall_height) / vertical_scale)
        floor_distance = wp.max(road_xy, wp.abs(gap) / vertical_scale)
        distance = wp.min(wp.abs(wall_signed), floor_distance)
        steps += 1
        if distance <= tolerance:
            hit = t
            break
        # Direction-aware bounds avoid tiny ground-distance steps for nearly
        # horizontal beams. A floor cannot close faster than slope*XYspeed-dz.
        ground_step = _closing_distance(gap, floor_closing)
        if gap < 0.0:
            ground_step = _closing_distance(-gap, roof_closing)
        ground_step = wp.max(ground_step, _closing_distance(road_xy, horizontal))
        wall_step = _closing_distance(wall_xy, horizontal)
        if wall_signed < 0.0:
            # Inside a solid: seek the closest exit, not an internal cell face.
            wall_step = wp.min(_closing_distance(-wall_xy, horizontal), wp.min(
                _closing_distance(gap, floor_closing),
                _closing_distance(wall_height - gap, roof_closing)))
        elif gap < 0.0:
            wall_step = wp.max(wall_step, _closing_distance(-gap, roof_closing))
        elif gap > wall_height:
            wall_step = wp.max(wall_step, _closing_distance(gap - wall_height, floor_closing))
        step = wp.min(ground_step, wall_step)
        # No forced minimum jump: it could skip a thin surface. Bounds must
        # either converge, prove a miss, or explicitly exhaust the budget.
        t += step
    exhausted = steps >= max_steps and hit > 1.0e29 and t <= leave
    return wp.vec3(hit, float(steps), float(exhausted))


def _cast_kernel(cache_samples):
    @wp.kernel(enable_backward=False, module="unique")
    def cast(grid: wp.array2d(dtype=wp.vec3), origin: wp.vec2, cell: float,
              wall_height: float, slope: float, z_bounds: wp.vec2,
              tolerance: float, max_steps: int,
              origins: wp.array(dtype=wp.vec3), rotations: wp.array(dtype=wp.mat33),
              directions: wp.array(dtype=wp.vec3), factors: wp.array(dtype=float), near: float, far: float,
              values: wp.array2d(dtype=float), valid: wp.array2d(dtype=int),
              iteration_maxima: wp.array2d(dtype=int), limit_counts: wp.array2d(dtype=int)):
        b, r = wp.tid()
        factor = factors[r]
        result = _march(grid, origin, cell, wall_height, slope, z_bounds, origins[b],
                        rotations[b] * directions[r], far / factor, tolerance, max_steps, wp.static(cache_samples))
        distance = result[0] * factor
        values[b, r] = far
        valid[b, r] = 0
        if distance >= near and distance < far:
            values[b, r] = distance
            valid[b, r] = 1
        iteration_maxima[b, r] = wp.max(iteration_maxima[b, r], int(result[1]))
        limit_counts[b, r] += int(result[2])
    return cast


_CAST = {cached: _cast_kernel(cached) for cached in (False, True)}


class GridScene:
    """Shared EDT grids; the grid backend now means distance-field marching."""
    algorithm = "edt-sphere-tracing"

    def __init__(self, field, device=None):
        wp.init()
        self.device = wp.get_device(device or ("cuda:0" if wp.is_cuda_available() else "cpu"))
        self.field = field
        wall_sdf = _signed_distance(field.walls, field.cell_size)
        road_sdf = _signed_distance(field.surface, field.cell_size)
        self.grid = wp.array(np.stack([wall_sdf, road_sdf, field.heights], axis=-1), dtype=wp.vec3, device=self.device)
        self.slope = _gradient_bound(field.heights, field.cell_size)
        low, high = float(field.heights.min()), float(field.heights.max())
        if field.walls.any():
            high += field.wall_height
        self.z_bounds = wp.vec2(low - .01, high + .01)

    def sensor(self, rays, batch_size=1, near=.02, far=30., *, tolerance=.002, max_steps=512, cache_samples=False):
        return GridSensor(self, rays, batch_size, near, far, tolerance, max_steps, cache_samples)


class GridSensor:
    """Borrowed Scan buffers plus explicit, device-resident convergence counters."""
    def __init__(self, scene, rays, batch_size, near, far, tolerance, max_steps, cache_samples):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not np.isfinite((near, far, tolerance)).all() or not 0 <= near < far or tolerance <= 0:
            raise ValueError("Require finite 0 <= near < far and positive march tolerance")
        if not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("March max_steps must be a positive integer")
        self.scene, self.rays, self.batch_size = scene, rays, batch_size
        self.near, self.far, self.tolerance, self.max_steps = float(near), float(far), float(tolerance), max_steps
        self.cache_samples = bool(cache_samples)
        self.directions = wp.array(rays.directions, dtype=wp.vec3, device=scene.device)
        self.factors = wp.array(rays.factors, dtype=float, device=scene.device)
        self.origins = wp.empty(batch_size, dtype=wp.vec3, device=scene.device)
        self.rotations = wp.empty(batch_size, dtype=wp.mat33, device=scene.device)
        shape = (batch_size, len(rays.directions))
        self.values = wp.empty(shape, dtype=float, device=scene.device)
        self.valid = wp.empty(shape, dtype=int, device=scene.device)
        self.iteration_maxima = wp.zeros(shape, dtype=int, device=scene.device)
        self.limit_counts = wp.zeros(shape, dtype=int, device=scene.device)
        self.result = Scan(self.values, self.valid, (batch_size,) + rays.shape)

    def reset_statistics(self):
        self.iteration_maxima.zero_()
        self.limit_counts.zero_()

    def diagnostics(self):
        counts = self.iteration_maxima.numpy()
        return {"algorithm": self.scene.algorithm, "tolerance_m": self.tolerance,
                "sample_cache": self.cache_samples,
                "max_steps": self.max_steps, "max_iterations_observed": int(counts.max()),
                "mean_per_ray_max_iterations": float(counts.mean()),
                "iteration_limit_events": int(self.limit_counts.numpy().sum())}

    def scan_device(self, origins, rotations):
        if (origins.device != self.scene.device or rotations.device != self.scene.device or
                origins.shape != (self.batch_size,) or rotations.shape != (self.batch_size,) or
                origins.dtype != wp.vec3 or rotations.dtype != wp.mat33):
            raise ValueError("Device poses must match the sensor's device, batch size and pose types")
        field = self.scene.field
        wp.launch(_CAST[self.cache_samples], dim=self.values.shape, inputs=[self.scene.grid, wp.vec2(*field.origin), field.cell_size,
            field.wall_height, self.scene.slope, self.scene.z_bounds, self.tolerance, self.max_steps,
            origins, rotations, self.directions, self.factors, self.near, self.far],
            outputs=[self.values, self.valid, self.iteration_maxima, self.limit_counts], device=self.scene.device)
        return self.result

    def scan(self, positions, rotations=None):
        positions, rotations = validate_poses(positions, rotations)
        if len(positions) != self.batch_size:
            raise ValueError("Pose count must match sensor batch size")
        self.origins.assign(positions)
        self.rotations.assign(rotations)
        return self.scan_device(self.origins, self.rotations)
