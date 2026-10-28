"""Stacked 2.5D layers: a floor plus elevated decks (bridges) that cross over it.

Each layer is one height field with a drivable mask on the same node grid. A
car belongs to exactly one layer and leaves it only where another layer is
drivable at the same height (a seam, e.g. a ramp foot). Physics reads the grids;
LiDAR and replay use a triangle mesh built from them, so a deck occludes the
road underneath it. Node i sits at origin + (i + 0.5) * resolution, as in maps.py.
"""
import base64
import io
import zlib

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

# Layers whose surfaces differ by less than this are the same road (a seam).
SEAM_TOLERANCE = 0.03


def _pack(array):
    """Compressed .npy bytes as ASCII, so grids fit in JSON recordings."""
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return base64.b64encode(zlib.compress(buffer.getvalue(), 9)).decode("ascii")


def _unpack(text):
    return np.load(io.BytesIO(zlib.decompress(base64.b64decode(text))), allow_pickle=False)


class LayeredTrack:
    """Layer 0 is the floor; higher layers are decks. Walls bound each layer.

    heights/free are [layers, nx, ny]. route is an ordered closed XYZ centerline
    and route_layers gives the layer each route sample drives on. Collision uses
    the same conservative chassis circle as MapTrack, per layer.
    """
    elevation = 0.0
    frequency = 0.0

    def __init__(self, heights, free, resolution, origin, route, route_layers, *,
                 wall_height=.3, deck_thickness=.08, name="layered", spec=None):
        self.heights = np.array(heights, dtype=np.float32, copy=True)
        self.free = np.array(free, dtype=bool, copy=True)
        self.resolution = float(resolution)
        self.origin = tuple(float(x) for x in origin)
        self.barrier_height = float(wall_height)
        self.deck_thickness = float(deck_thickness)
        self.name = str(name)
        self.spec = spec
        if (self.heights.ndim != 3 or self.free.shape != self.heights.shape or len(self.heights) < 1
                or min(self.heights.shape[1:]) < 8 or not np.isfinite(self.heights).all()
                or not self.free.any(axis=(1, 2)).all() or not np.isfinite(self.resolution)
                or self.resolution <= 0 or len(self.origin) != 2 or not np.isfinite(self.origin).all()
                or min(self.barrier_height, self.deck_thickness) <= 0):
            raise ValueError("Layered track requires [layers, nx, ny] finite heights and nonempty free masks, "
                             "positive resolution, wall height and deck thickness")
        if self.free[:, [0, -1], :].any() or self.free[:, :, [0, -1]].any():
            raise ValueError("Layer free space cannot touch the grid edge")
        self.layer_count = len(self.heights)
        self.length, self.width = np.array(self.heights.shape[1:]) * self.resolution
        self.route = np.array(route, dtype=np.float32, copy=True)
        self.route_layers = np.array(route_layers, dtype=np.int32, copy=True)
        if (self.route.ndim != 2 or self.route.shape[1] != 3 or len(self.route) < 4
                or not np.isfinite(self.route).all() or self.route_layers.shape != (len(self.route),)
                or self.route_layers.min() < 0 or self.route_layers.max() >= self.layer_count):
            raise ValueError("Route must be finite XYZ samples, each with a valid layer index")
        self.segment_lengths = np.linalg.norm(np.roll(self.route, -1, axis=0) - self.route, axis=1)
        if self.segment_lengths.min() < 1e-5 or self.segment_lengths.max() > .151:
            raise ValueError("Route must be densely sampled (segments <= 0.15 m)")
        self.route_length = float(self.segment_lengths.sum())
        # A node is open on layer L where another layer is drivable at L's height:
        # walls stop there so the car can cross the seam. Everything else is wall.
        self.seam = np.zeros_like(self.free)
        for layer in range(self.layer_count):
            for other in range(self.layer_count):
                if other != layer:
                    self.seam[layer] |= self.free[other] & (
                        np.abs(self.heights[other] - self.heights[layer]) < SEAM_TOLERANCE)
        walls = ~self.free & ~self.seam
        self.clearance = np.stack([distance_transform_edt(~w) * self.resolution - self.resolution / np.sqrt(2)
                                   for w in walls]).astype(np.float32)
        for array in (self.heights, self.free, self.seam, self.clearance, self.route,
                      self.route_layers, self.segment_lengths):
            array.flags.writeable = False

    def _grid(self, points):
        """Fractional node coordinates of XY (or XYZ) points."""
        return (np.asarray(points, dtype=float)[..., :2] - self.origin) / self.resolution - .5

    def clearance_at(self, layer, points):
        f = self._grid(points)
        ij = np.floor(f + .5).astype(int)
        inside = ((ij >= 0) & (ij < self.free.shape[1:])).all(axis=-1)
        clipped = np.clip(ij, 0, np.array(self.free.shape[1:]) - 1)
        values = self.clearance[layer, clipped[..., 0], clipped[..., 1]]
        return np.where(inside, values - np.linalg.norm((f - ij) * self.resolution, axis=-1), -1.)

    def height_at(self, layer, points):
        """Bilinear surface height, matching the physics kernel."""
        f = self._grid(points)
        upper = np.array(self.heights.shape[1:]) - 2
        i = np.clip(np.floor(f).astype(int), 0, upper)
        t = np.clip(f - i, 0, 1)
        h = self.heights[layer]
        x, y, tx, ty = i[..., 0], i[..., 1], t[..., 0], t[..., 1]
        return ((1 - tx) * (1 - ty) * h[x, y] + tx * (1 - ty) * h[x + 1, y]
                + (1 - tx) * ty * h[x, y + 1] + tx * ty * h[x + 1, y + 1])

    def gradient_at(self, layer, points):
        p = np.asarray(points, dtype=float)[..., :2]
        d = self.resolution
        dx, dy = np.array([d, 0.]), np.array([0., d])
        return np.stack(((self.height_at(layer, p + dx) - self.height_at(layer, p - dx)) / (2 * d),
                         (self.height_at(layer, p + dy) - self.height_at(layer, p - dy)) / (2 * d)), axis=-1)

    def validate_vehicle(self, vehicle):
        radius = np.hypot(vehicle.length, vehicle.width) / 2
        minimum = min(float(self.clearance_at(layer, self.route[self.route_layers == layer]).min())
                      for layer in np.unique(self.route_layers))
        if minimum <= radius + .02:
            raise ValueError(f"Route clearance {minimum:.3f} m is too small for the car's "
                             f"{radius:.3f} m collision radius plus 0.02 m spawn margin")
        return minimum

    def path(self, ride_height):
        """Centerline poses resting on each sample's layer, tilted to its surface."""
        delta = np.roll(self.route, -1, axis=0) - np.roll(self.route, 1, axis=0)
        heading = np.arctan2(delta[:, 1], delta[:, 0])
        poses = np.zeros((len(self.route), 7), np.float32)
        for layer in np.unique(self.route_layers):
            rows = self.route_layers == layer
            p = self.route[rows]
            g = self.gradient_at(layer, p)
            poses[rows, :2] = p[:, :2]
            poses[rows, 2] = self.height_at(layer, p) + ride_height * np.sqrt(1 + (g * g).sum(axis=1))
            poses[rows, 3:] = surface_quaternion(heading[rows], g)
        return poses

    def route_progress(self, poses):
        """Net laps along the route; XYZ nearest samples tell the bridge from the road below."""
        index = cKDTree(self.route).query(np.asarray(poses)[:, :3])[1]
        steps = np.diff(index)
        n = len(self.route)
        steps = (steps + n // 2) % n - n // 2
        return float(steps.sum() / n)

    def mesh_parts(self):
        """Replay geometry: road surfaces, railings and deck structure."""
        return _layer_mesh(self)

    def mesh(self):
        """LiDAR geometry: replay parts inside a closed room (floor, walls, ceiling)
        so every ray returns, as it would indoors. Replay omits the room."""
        parts = list(self.mesh_parts().values()) + [self._room()]
        vertices, faces, offset = [], [], 0
        for v, f in parts:
            vertices.append(v)
            faces.append(f + offset)
            offset += len(v)
        return np.concatenate(vertices), np.concatenate(faces)

    def _room(self):
        x0, y0 = self.origin
        x1, y1 = x0 + self.length, y0 + self.width
        z0, z1 = float(self.heights.min()) - .03, float(self.heights.max()) + 1.5
        corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        vertices, faces = [], []
        for a, b in zip(corners, np.roll(corners, -1, axis=0)):
            base = len(vertices)
            vertices += [[*a, z0], [*b, z0], [*b, z1], [*a, z1]]
            faces += [[base, base + 1, base + 2], [base, base + 2, base + 3]]
        for z in (z0, z1):
            base = len(vertices)
            vertices += [[*c, z] for c in corners]
            faces += [[base, base + 1, base + 2], [base, base + 2, base + 3]]
        return np.array(vertices, np.float32), np.array(faces, np.int32)

    def state_dict(self):
        state = {"kind": "layered", "name": self.name}
        if self.spec is not None:
            return {**state, "generator": self.spec}
        return {**state, "heights": _pack(self.heights), "free": _pack(self.free),
                "resolution": self.resolution, "origin": list(self.origin), "route": _pack(self.route),
                "route_layers": _pack(self.route_layers), "wall_height": self.barrier_height,
                "deck_thickness": self.deck_thickness}

    @classmethod
    def from_state_dict(cls, state):
        if "generator" in state:
            return figure_eight_bridge(**state["generator"])
        return cls(_unpack(state["heights"]), _unpack(state["free"]), state["resolution"], state["origin"],
                   _unpack(state["route"]), _unpack(state["route_layers"]), wall_height=state["wall_height"],
                   deck_thickness=state["deck_thickness"], name=state["name"])


def surface_quaternion(heading, gradient):
    """XYZW rotation: X along the heading projected onto the surface, Z along its normal."""
    heading, g = np.asarray(heading, dtype=float), np.asarray(gradient, dtype=float)
    normal = np.stack((-g[..., 0], -g[..., 1], np.ones_like(heading)), axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    c, s = np.cos(heading), np.sin(heading)
    forward = np.stack((c, s, g[..., 0] * c + g[..., 1] * s), axis=-1)
    forward /= np.linalg.norm(forward, axis=-1, keepdims=True)
    left = np.cross(normal, forward)
    return Rotation.from_matrix(np.stack((forward, left, normal), axis=-1)).as_quat()


def _quads(corners):
    """[n, 4, 3] corner loops to vertices/faces, dropping degenerate triangles.

    A skirt tapering to zero height at a ramp foot keeps only its one real triangle.
    """
    vertices = np.asarray(corners, np.float32).reshape(-1, 3)
    base = 4 * np.arange(len(vertices) // 4, dtype=np.int32)[:, None]
    faces = np.concatenate((base + [0, 1, 2], base + [0, 2, 3]))
    t = vertices[faces]
    area = np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
    return vertices, faces[area > 1e-8].astype(np.int32)


def _layer_mesh(track):
    h, free, res = track.heights, track.free, track.resolution
    nx, ny = h.shape[1:]
    xs = track.origin[0] + (np.arange(nx) + .5) * res
    ys = track.origin[1] + (np.arange(ny) + .5) * res
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    # A cell (i, j) spans nodes (i..i+1, j..j+1); it is road if all four are.
    corner = [(slice(0, -1), slice(0, -1)), (slice(1, None), slice(0, -1)),
              (slice(1, None), slice(1, None)), (slice(0, -1), slice(1, None))]
    road = np.stack([np.logical_and.reduce([f[c] for c in corner]) for f in free])
    match = np.zeros((track.layer_count,) * 2 + road.shape[1:], bool)
    for a in range(track.layer_count):
        for b in range(track.layer_count):
            if a != b:
                match[a, b] = road[b] & np.logical_and.reduce(
                    [np.abs(h[a][c] - h[b][c]) < SEAM_TOLERANCE for c in corner])
    ground = h[0]

    def loop(cells, top):
        """Corner loops of whole cells at per-node heights."""
        i, j = np.nonzero(cells)
        nodes = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
        return np.stack([np.stack((X[n], Y[n], top[n]), axis=-1) for n in nodes], axis=1)

    road_quads, wall_quads, structure_quads = [], [], []
    for layer in range(track.layer_count):
        surface = h[layer]
        elevated = (surface - ground) > SEAM_TOLERANCE
        # Seam cells are drawn once, by the lowest matching layer.
        drawn = road[layer] & ~match[layer, :layer].any(axis=0)
        road_quads.append(loop(drawn, surface))
        opening = match[layer].any(axis=0)
        span = np.zeros_like(road[layer])
        if layer > 0:
            # Deck cells over lower road are bridges with an underside; the rest
            # of an elevated layer stands on solid embankment down to the floor.
            span = road[layer] & np.logical_or.reduce([road[k] for k in range(layer)]) & \
                np.logical_and.reduce([elevated[c] for c in corner])
            structure_quads.append(loop(span, surface - track.deck_thickness))
        for axis in (0, 1):
            a = (slice(0, -1), slice(None)) if axis == 0 else (slice(None), slice(0, -1))
            b = (slice(1, None), slice(None)) if axis == 0 else (slice(None), slice(1, None))
            ra, rb = road[layer][a], road[layer][b]
            # Edge nodes shared by the two cells; cell (i, j) and its +axis neighbor.
            i, j = np.nonzero(ra != rb)
            if axis == 0:
                n0, n1 = (i + 1, j), (i + 1, j + 1)
            else:
                n0, n1 = (i, j + 1), (i + 1, j + 1)
            inside_a = ra[i, j]
            outer = (np.where(inside_a, i + (axis == 0), i), np.where(inside_a, j + (axis == 1), j))
            inner = (np.where(inside_a, i, i + (axis == 0)), np.where(inside_a, j, j + (axis == 1)))
            keep = ~opening[outer]
            n0 = (n0[0][keep], n0[1][keep])
            n1 = (n1[0][keep], n1[1][keep])
            inner = (inner[0][keep], inner[1][keep])

            def edge(z0, z1):
                return np.stack((np.stack((X[n0], Y[n0], z0[n0]), -1), np.stack((X[n1], Y[n1], z0[n1]), -1),
                                 np.stack((X[n1], Y[n1], z1[n1]), -1), np.stack((X[n0], Y[n0], z1[n0]), -1)), axis=1)
            wall_quads.append(edge(surface, surface + track.barrier_height))
            if layer > 0:
                over = span[inner]
                slab = edge(surface - track.deck_thickness, surface)
                skirt = edge(np.minimum(ground, surface), surface)
                structure_quads.append(np.where(over[:, None, None], slab, skirt))
                # Abutments: where a deck stops spanning and becomes embankment.
                sa, sb = span[a], span[b]
                i2, j2 = np.nonzero(ra & rb & (sa != sb))
                if axis == 0:
                    m0, m1 = (i2 + 1, j2), (i2 + 1, j2 + 1)
                else:
                    m0, m1 = (i2, j2 + 1), (i2 + 1, j2 + 1)
                low, high = np.minimum(ground, surface), surface - track.deck_thickness
                structure_quads.append(np.stack(
                    (np.stack((X[m0], Y[m0], low[m0]), -1), np.stack((X[m1], Y[m1], low[m1]), -1),
                     np.stack((X[m1], Y[m1], high[m1]), -1), np.stack((X[m0], Y[m0], high[m0]), -1)), axis=1))
    parts = {"road": _quads(np.concatenate(road_quads)), "walls": _quads(np.concatenate(wall_quads))}
    if structure_quads:
        parts["structure"] = _quads(np.concatenate(structure_quads))
    return parts


def figure_eight_bridge(radius=3.0, separation=6.0, road_width=2.0, bridge_height=.7, plateau=3.0,
                        overlap=1.0, resolution=.05, wall_height=.3, deck_thickness=.08):
    """Two loops joined by crossing straights; one straight bridges the other.

    Loop centers sit at (+-separation, 0). The bridge rises from the far point
    of the left loop, holds bridge_height for `plateau` meters over the
    crossing, and lands at the far point of the right loop. Those far points
    are the seams; each layer's road extends `overlap` meters past them.
    """
    spec = dict(radius=radius, separation=separation, road_width=road_width, bridge_height=bridge_height,
                plateau=plateau, overlap=overlap, resolution=resolution, wall_height=wall_height,
                deck_thickness=deck_thickness)
    c, R = float(separation), float(radius)
    if not (0 < R < c and road_width > 0 and bridge_height > deck_thickness > 0 and plateau >= 0
            and overlap > 0 and resolution > 0):
        raise ValueError("Figure eight needs 0 < radius < separation and positive sizes, with "
                         "bridge_height above deck_thickness")
    if road_width >= R:
        raise ValueError("Road width must be less than the loop radius")
    L = np.sqrt(c * c - R * R)
    beta = np.arctan2(L, R)
    tangent = np.array([L / c, R / c])        # crossing-line direction (cos a, sin a)
    t1, t2 = L * tangent, L * tangent * [1, -1]
    arc = R * (2 * np.pi - 2 * beta)
    total = 4 * L + 2 * arc
    s = np.arange(int(np.ceil(total / .05))) * (total / int(np.ceil(total / .05)))
    # Order: ground straight T2 -> T3 through the crossing, left loop counterclockwise,
    # bridge straight T4 -> T1, right loop clockwise, back to T2.
    xy = np.empty((len(s), 2))
    d = np.array([-tangent[0], tangent[1]])
    leg = s < 2 * L
    xy[leg] = t2 + s[leg, None] * d
    left = (s >= 2 * L) & (s < 2 * L + arc)
    angle = beta + (s[left] - 2 * L) / R
    xy[left] = np.column_stack((-c + R * np.cos(angle), R * np.sin(angle)))
    bridge = (s >= 2 * L + arc) & (s < 4 * L + arc)
    xy[bridge] = -t1 + (s[bridge, None] - 2 * L - arc) * tangent
    right = s >= 4 * L + arc
    angle = np.pi - beta - (s[right] - 4 * L - arc) / R
    xy[right] = np.column_stack((c + R * np.cos(angle), R * np.sin(angle)))
    # Signed distance along the route from the middle of the bridge straight.
    middle = 3 * L + arc
    u = (s - middle + total / 2) % total - total / 2
    half = L + arc / 2
    ramp = half - plateau / 2
    if ramp <= 0:
        raise ValueError("Plateau is longer than the bridge half of the route")
    rise = np.clip((np.abs(u) - plateau / 2) / ramp, 0, 1)
    z = bridge_height * .5 * (1 + np.cos(np.pi * rise))
    z[np.abs(u) >= half] = 0
    route = np.column_stack((xy, z))
    route_layers = (np.abs(u) < half).astype(np.int32)
    margin = road_width / 2 + wall_height + 1.0
    low = route[:, :2].min(axis=0) - margin
    shape = np.ceil((route[:, :2].max(axis=0) + margin - low) / resolution).astype(int)
    nodes = low + (np.stack(np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij"), -1) + .5) * resolution
    points = nodes.reshape(-1, 2)

    def project(extra):
        """Distance and route height at the nearest point of this layer's route stretch."""
        on_layer = (np.abs(u) >= half - extra) if layer == 0 else (np.abs(u) < half + extra)
        index = np.flatnonzero(on_layer)
        distance, nearest = cKDTree(xy[index]).query(points)
        nearest = index[nearest]
        # Refine onto the adjacent segment so heights vary smoothly along the road.
        best_d, best_z = distance, z[nearest]
        for step in (-1, 1):
            a, b = nearest, (nearest + step) % len(s)
            ab = xy[b] - xy[a]
            t = np.clip(((points - xy[a]) * ab).sum(1) / (ab * ab).sum(1), 0, 1)
            dist = np.linalg.norm(points - (xy[a] + t[:, None] * ab), axis=1)
            better = dist < best_d
            best_d = np.where(better, dist, best_d)
            best_z = np.where(better, z[a] + t * (z[b] - z[a]), best_z)
        return best_d, best_z

    heights, free = [], []
    for layer in (0, 1):
        # The road stops `overlap` past the seam, but its rounded end reaches
        # farther; heights keep following the route so they still match there.
        heights.append(project(overlap + road_width)[1].reshape(shape))
        free.append((project(overlap)[0] < road_width / 2).reshape(shape))
    return LayeredTrack(np.stack(heights), np.stack(free), resolution, low, route, route_layers,
                        wall_height=wall_height, deck_thickness=deck_thickness, name="figure-eight-bridge",
                        spec=spec)
