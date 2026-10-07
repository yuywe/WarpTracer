"""3D rays against a tiled height field and extruded occupancy cells.

No triangle mesh or BVH is constructed. Terrain is bilinear within each cell;
obstacles occupy the interval from terrain height to terrain + obstacle height.
The surface mask allows holes. Arrays use [x, y], in world meters.
"""
from dataclasses import dataclass

import numpy as np
import warp as wp

from racesense3d.core import Scan, validate_poses


@wp.func
def _height(heights: wp.array2d(dtype=float), i: int, j: int, x: float, y: float):
    a = heights[i, j]
    b = heights[i + 1, j] - a
    c = heights[i, j + 1] - a
    d = heights[i + 1, j + 1] - a - b - c
    return a + b * x + c * y + d * x * y


@wp.func
def _root(a: float, b: float, c: float, length: float):
    """Nearest root in a cell interval; stable quadratic solution."""
    hit = float(1.0e30)
    if wp.abs(a) < 1.0e-10:
        if wp.abs(b) > 1.0e-10:
            t = -c / b
            if t >= -1.0e-6 and t <= length + 1.0e-6:
                hit = wp.max(0.0, wp.min(t, length))
    else:
        discriminant = b * b - 4.0 * a * c
        if discriminant >= 0.0:
            sign = float(1.0)
            if b < 0.0:
                sign = -1.0
            q = -0.5 * (b + sign * wp.sqrt(discriminant))
            if wp.abs(q) > 1.0e-20:
                first = q / a
                second = c / q
                if first >= -1.0e-6 and first <= length + 1.0e-6:
                    hit = wp.max(0.0, wp.min(first, length))
                if second >= -1.0e-6 and second <= length + 1.0e-6:
                    hit = wp.min(hit, wp.max(0.0, wp.min(second, length)))
            elif wp.abs(b) < 1.0e-10 and wp.abs(c) < 1.0e-10:
                hit = 0.0
    return hit


@wp.func
def _boundary(p: float, direction: float, low: float, high: float):
    distance = float(1.0e30)
    if direction > 1.0e-10:
        distance = (high - p) / direction
    elif direction < -1.0e-10:
        distance = (low - p) / direction
    return wp.max(0.0, distance)


@wp.func
def _index(p: float, direction: float, origin: float, cell: float):
    bias = float(0.0)
    if direction > 0.0:
        bias = 1.0e-4
    elif direction < 0.0:
        bias = -1.0e-4
    return int(wp.floor((p - origin) / cell + bias))


@wp.func
def _query(heights: wp.array2d(dtype=float), surface: wp.array2d(dtype=int),
           obstacles: wp.array2d(dtype=float), bounds: wp.array2d(dtype=wp.vec2),
           origin: wp.vec2, cell: float, tile: int, start: wp.vec3, direction: wp.vec3,
           far: float, use_tiles: int):
    nx = obstacles.shape[0]
    ny = obstacles.shape[1]
    enter = float(0.0)
    leave = far
    # Clip to the grid in XY. Vertical rays stay in one cell.
    for axis in range(2):
        low = origin[axis]
        high = low + cell * float(nx)
        if axis == 1:
            high = low + cell * float(ny)
        if wp.abs(direction[axis]) < 1.0e-10:
            if start[axis] < low or start[axis] >= high:
                leave = -1.0
        else:
            a = (low - start[axis]) / direction[axis]
            b = (high - start[axis]) / direction[axis]
            enter = wp.max(enter, wp.min(a, b))
            leave = wp.min(leave, wp.max(a, b))
    hit = float(1.0e30)
    t = enter
    previous_solid = bool(False)
    check_entry = enter > 0.0
    if enter == 0.0 and leave > 0.0:
        # An origin exactly on an exposed cell face is already a surface hit.
        before_i = _index(start[0], -direction[0], origin[0], cell)
        before_j = _index(start[1], -direction[1], origin[1], cell)
        after_i = _index(start[0], direction[0], origin[0], cell)
        after_j = _index(start[1], direction[1], origin[1], cell)
        check_entry = before_i != after_i or before_j != after_j
        if check_entry and before_i >= 0 and before_i < nx and before_j >= 0 and before_j < ny:
            bx = wp.clamp((start[0] - origin[0]) / cell - float(before_i), 0.0, 1.0)
            by = wp.clamp((start[1] - origin[1]) / cell - float(before_j), 0.0, 1.0)
            bh = _height(heights, before_i, before_j, bx, by)
            bo = obstacles[before_i, before_j]
            previous_solid = bo > 0.0 and start[2] >= bh and start[2] <= bh + bo
    iterations = int(0)
    while t < leave and iterations < nx + ny + 4:
        iterations += 1
        p = start + t * direction
        i = _index(p[0], direction[0], origin[0], cell)
        j = _index(p[1], direction[1], origin[1], cell)
        i = wp.clamp(i, 0, nx - 1)
        j = wp.clamp(j, 0, ny - 1)
        x = wp.clamp((p[0] - origin[0]) / cell - float(i), 0.0, 1.0)
        y = wp.clamp((p[1] - origin[1]) / cell - float(j), 0.0, 1.0)
        base = _height(heights, i, j, x, y)
        obstacle = obstacles[i, j]
        solid = obstacle > 0.0 and p[2] >= base and p[2] <= base + obstacle
        # A cell boundary is a wall only when crossing into/out of solid space.
        # Internal boundaries between occupied cells must remain invisible.
        if check_entry and solid != previous_solid:
            hit = t
            break
        bi = i // tile
        bj = j // tile
        tile_exit = wp.min(leave, t + wp.min(
            _boundary(p[0], direction[0], origin[0] + float(bi * tile) * cell,
                      origin[0] + float(wp.min((bi + 1) * tile, nx)) * cell),
            _boundary(p[1], direction[1], origin[1] + float(bj * tile) * cell,
                      origin[1] + float(wp.min((bj + 1) * tile, ny)) * cell)))
        z_exit = start[2] + tile_exit * direction[2]
        limits = bounds[bi, bj]
        if use_tiles != 0 and (limits[0] > limits[1] or
                wp.min(p[2], z_exit) > limits[1] + 1.0e-6 or
                wp.max(p[2], z_exit) < limits[0] - 1.0e-6):
            # Conservative bounds prove this entire interval cannot hit anything.
            previous_solid = False
            check_entry = True
            t = tile_exit
            continue
        end = wp.min(leave, t + wp.min(
            _boundary(p[0], direction[0], origin[0] + float(i) * cell, origin[0] + float(i + 1) * cell),
            _boundary(p[1], direction[1], origin[1] + float(j) * cell, origin[1] + float(j + 1) * cell)))
        length = end - t
        if surface[i, j] != 0 or obstacle > 0.0:
            h00 = heights[i, j]
            hx = heights[i + 1, j] - h00
            hy = heights[i, j + 1] - h00
            hxy = heights[i + 1, j + 1] - h00 - hx - hy
            dx = direction[0] / cell
            dy = direction[1] / cell
            a = -hxy * dx * dy
            b = direction[2] - hx * dx - hy * dy - hxy * (x * dy + y * dx)
            c = p[2] - base
            distance = _root(a, b, c, length)
            if obstacle > 0.0:
                distance = wp.min(distance, _root(a, b, c - obstacle, length))
            if distance < 1.0e29:
                hit = t + distance
                break
        p_end = start + end * direction
        ex = wp.clamp(x + direction[0] * length / cell, 0.0, 1.0)
        ey = wp.clamp(y + direction[1] * length / cell, 0.0, 1.0)
        base_end = _height(heights, i, j, ex, ey)
        previous_solid = obstacle > 0.0 and p_end[2] >= base_end and p_end[2] <= base_end + obstacle
        check_entry = True
        if end >= leave:
            # Only an XY grid exit exposes a final side, not the far cutoff.
            if previous_solid and leave < far:
                hit = leave
            break
        t = end
    return hit


@wp.kernel
def _cast(heights: wp.array2d(dtype=float), surface: wp.array2d(dtype=int),
          obstacles: wp.array2d(dtype=float), bounds: wp.array2d(dtype=wp.vec2),
          origin: wp.vec2, cell: float, tile: int, use_tiles: int,
          origins: wp.array(dtype=wp.vec3), rotations: wp.array(dtype=wp.mat33),
          directions: wp.array(dtype=wp.vec3), factors: wp.array(dtype=float), near: float, far: float,
          values: wp.array2d(dtype=float), valid: wp.array2d(dtype=int)):
    b, r = wp.tid()
    factor = factors[r]
    distance = _query(heights, surface, obstacles, bounds, origin, cell, tile,
                      origins[b], rotations[b] * directions[r], far / factor, use_tiles) * factor
    values[b, r] = far
    valid[b, r] = 0
    # Match mesh semantics: a too-near first hit blocks geometry behind it.
    if distance >= near and distance < far:
        values[b, r] = distance
        valid[b, r] = 1


@dataclass(frozen=True)
class HeightField:
    heights: np.ndarray       # [nx+1, ny+1] vertex elevations, meters
    surface: np.ndarray       # [nx, ny] cells with a terrain surface
    obstacles: np.ndarray     # [nx, ny] solid heights above the terrain, meters
    origin: tuple             # XY position of the lower grid corner
    cell_size: float
    tile_size: int = 8

    def __post_init__(self):
        h = np.array(self.heights, dtype=np.float32, copy=True)
        raw_surface = np.asarray(self.surface)
        if np.any((raw_surface != 0) & (raw_surface != 1)):
            raise ValueError("Height field surface mask must be binary")
        s = np.array(raw_surface, dtype=np.int32, copy=True)
        o = np.array(self.obstacles, dtype=np.float32, copy=True)
        if (h.ndim != 2 or min(h.shape) < 2 or s.shape != (h.shape[0] - 1, h.shape[1] - 1)
                or o.shape != s.shape or not np.isfinite(h).all() or not np.isfinite(o).all()
                or np.any(o < 0) or np.any((s != 0) & (s != 1))):
            raise ValueError("Height field requires finite vertex heights, a binary cell surface mask and nonnegative obstacle heights")
        if (not np.isfinite(self.cell_size) or self.cell_size <= 0 or
                np.shape(self.origin) != (2,) or not np.isfinite(self.origin).all()):
            raise ValueError("Grid cell size must be positive and origin a finite XY pair")
        if not isinstance(self.tile_size, int) or self.tile_size < 1:
            raise ValueError("Grid tile size must be a positive integer")
        for name, value in (("heights", h), ("surface", s), ("obstacles", o)):
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        object.__setattr__(self, "origin", tuple(float(x) for x in self.origin))

    @classmethod
    def oval(cls, track, cell_size=0.025):
        if not np.isfinite(cell_size) or cell_size <= 0:
            raise ValueError("Grid cell size must be finite and positive")
        # Resolve every barrier with at least two cells at the narrowest location.
        if cell_size > track.barrier_thickness / 4:
            raise ValueError("Oval grid cell size must be <= barrier_thickness/4")
        half = np.array([track.length / 2, track.width / 2]) + track.barrier_thickness + 2 * cell_size
        size = 2 * np.ceil(half / cell_size).astype(int)
        origin = -size * cell_size / 2
        if np.prod(size, dtype=np.int64) > 4_000_000:
            raise ValueError("Oval grid exceeds four million cells; choose a larger cell size")
        x = origin[0] + np.arange(size[0] + 1) * cell_size
        y = origin[1] + np.arange(size[1] + 1) * cell_size
        xx, yy = np.meshgrid(x[:-1] + cell_size / 2, y[:-1] + cell_size / 2, indexing="ij")

        def inside(a, b):
            return (xx / a) ** 2 + (yy / b) ** 2 <= 1

        a, b = track.length / 2, track.width / 2
        ia, ib = track.inner_axes
        thickness = track.barrier_thickness
        road = inside(a, b) & ~inside(ia, ib)
        walls = ((inside(ia, ib) & ~inside(ia - thickness, ib - thickness)) |
                 (inside(a + thickness, b + thickness) & ~inside(a, b)))
        heights = np.broadcast_to(track.height(x)[:, None], (len(x), len(y)))
        return cls(heights, road | walls, walls * track.barrier_height, tuple(origin), cell_size)


class GridScene:
    """Immutable terrain shared by independent sensor poses, like mesh Scene."""
    def __init__(self, field, device=None, *, use_tiles=True):
        wp.init()
        self.device = wp.get_device(device or ("cuda:0" if wp.is_cuda_available() else "cpu"))
        self.field = field
        self.use_tiles = int(use_tiles)
        self.heights = wp.array(field.heights, dtype=float, device=self.device)
        self.surface = wp.array(field.surface, dtype=int, device=self.device)
        self.obstacles = wp.array(field.obstacles, dtype=float, device=self.device)
        nx, ny = field.surface.shape
        tile = field.tile_size
        active = (field.surface != 0) | (field.obstacles > 0)
        low = np.minimum.reduce([field.heights[:-1, :-1], field.heights[1:, :-1],
                                 field.heights[:-1, 1:], field.heights[1:, 1:]])
        high = np.maximum.reduce([field.heights[:-1, :-1], field.heights[1:, :-1],
                                  field.heights[:-1, 1:], field.heights[1:, 1:]]) + field.obstacles
        dims = ((nx + tile - 1) // tile, (ny + tile - 1) // tile)
        shape = (dims[0] * tile, dims[1] * tile)
        lower, upper = np.full(shape, np.inf, np.float32), np.full(shape, -np.inf, np.float32)
        lower[:nx, :ny] = np.where(active, low, np.inf)
        upper[:nx, :ny] = np.where(active, high, -np.inf)
        lower = lower.reshape(dims[0], tile, dims[1], tile).min(axis=(1, 3))
        upper = upper.reshape(dims[0], tile, dims[1], tile).max(axis=(1, 3))
        self.bounds = wp.array(np.stack([lower, upper], axis=-1), dtype=wp.vec2, device=self.device)

    def sensor(self, rays, batch_size=1, near=0.02, far=30.0):
        return GridSensor(self, rays, batch_size, near, far)


class GridSensor:
    """Reused device buffers with the same scan interface as the mesh sensor."""
    def __init__(self, scene, rays, batch_size, near, far):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not np.isfinite((near, far)).all() or not 0 <= near < far:
            raise ValueError("Require finite 0 <= near < far")
        self.scene, self.rays, self.batch_size = scene, rays, batch_size
        self.near, self.far = float(near), float(far)
        self.directions = wp.array(rays.directions, dtype=wp.vec3, device=scene.device)
        self.factors = wp.array(rays.factors, dtype=float, device=scene.device)
        self.origins = wp.empty(batch_size, dtype=wp.vec3, device=scene.device)
        self.rotations = wp.empty(batch_size, dtype=wp.mat33, device=scene.device)
        self.values = wp.empty((batch_size, len(rays.directions)), dtype=float, device=scene.device)
        self.valid = wp.empty((batch_size, len(rays.directions)), dtype=int, device=scene.device)
        self.result = Scan(self.values, self.valid, (batch_size,) + rays.shape)

    def scan_device(self, origins, rotations):
        if (origins.device != self.scene.device or rotations.device != self.scene.device or
                origins.shape != (self.batch_size,) or rotations.shape != (self.batch_size,) or
                origins.dtype != wp.vec3 or rotations.dtype != wp.mat33):
            raise ValueError("Device poses must match the sensor's device, batch size and pose types")
        field = self.scene.field
        wp.launch(_cast, dim=self.values.shape, inputs=[
            self.scene.heights, self.scene.surface, self.scene.obstacles, self.scene.bounds,
            wp.vec2(*field.origin), field.cell_size, field.tile_size, self.scene.use_tiles,
            origins, rotations, self.directions, self.factors, self.near, self.far],
            outputs=[self.values, self.valid], device=self.scene.device)
        return self.result

    def scan(self, positions, rotations=None):
        positions, rotations = validate_poses(positions, rotations)
        if len(positions) != self.batch_size:
            raise ValueError("Pose count must match sensor batch size")
        self.origins.assign(positions)
        self.rotations.assign(rotations)
        return self.scan_device(self.origins, self.rotations)
