"""PNG occupancy maps, closed routes and portable checkpoint geometry.

Images are top-left row/column; runtime arrays are XY with Y pointing up.
Only preprocessing uses Pillow/scikit-image. Physics and scans share one map.
"""
from collections import deque
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import convolve, distance_transform_edt, gaussian_filter1d, label
from skimage.morphology import skeletonize

from .distance_field import DistanceField
from .terrain import OvalTrack

_ROUTE_ERROR = ("Cannot infer one closed route from this PNG. Use a single closed track, "
                "or set centerline to an ordered x,y CSV in the map JSON")
# A found loop rarely covers every skeleton pixel (a little noise is normal
# even on a clean track); below this fraction, treat it as a genuinely
# ambiguous/branching layout rather than route noise.
_MIN_LOOP_COVERAGE = 0.9


def _prune_skeleton_spurs(skeleton):
    """Erode dead-end skeleton pixels until none remain.

    Real-world track PNGs rarely skeletonize to a perfect single-pixel ring:
    a slightly uneven edge (often right at the start/finish line) leaves a
    short dangling branch. Shaving pixels with at most one neighbor, repeated
    until stable, removes those branches without touching the ring itself,
    since every ring pixel always keeps two neighbors.
    """
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    skeleton = skeleton.copy()
    while True:
        counts = convolve(skeleton.astype(np.uint8), kernel, mode="constant", cval=0)
        endpoints = skeleton & (counts <= 1)
        if not endpoints.any():
            return skeleton
        skeleton &= ~endpoints


def _skeleton_neighbors(pixels, x, y):
    adjacent = set()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            q = x + dx, y + dy
            # Do not add a diagonal shortcut around a connected right angle.
            if (dx or dy) and q in pixels and not (dx and dy and
                    ((x + dx, y) in pixels or (x, y + dy) in pixels)):
                adjacent.add(q)
    return adjacent


def _longest_loop_through(pixels, start):
    """Longest simple path from start, through two of its neighbors, and back.

    Unlike a single fixed-direction walk, this tolerates leftover junctions
    elsewhere in the skeleton (e.g. a short spur too tangled for pruning to
    remove outright): it searches every pair of the start's own neighbors and
    keeps the longest connecting path, rather than failing the moment any
    pixel in the whole skeleton has more than two neighbors.
    """
    neighbors = sorted(_skeleton_neighbors(pixels, *start))
    best = None
    for i in range(len(neighbors)):
        for j in range(i + 1, len(neighbors)):
            source, target = neighbors[i], neighbors[j]
            parent = {source: source}
            queue = deque([source])
            while queue:
                current = queue.popleft()
                if current == target:
                    break
                for neighbor in _skeleton_neighbors(pixels, *current):
                    if neighbor in parent or neighbor == start:
                        continue
                    parent[neighbor] = current
                    queue.append(neighbor)
            if target not in parent:
                continue
            path, node = [start], target
            while node != source:
                path.append(node)
                node = parent[node]
            path.append(source)
            if best is None or len(path) > len(best):
                best = path
    return best


def _automatic_route(free):
    skeleton = _prune_skeleton_spurs(skeletonize(free))
    labels, components = label(skeleton, structure=np.ones((3, 3), dtype=int))
    if components == 0:
        raise ValueError(_ROUTE_ERROR)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    pixels = {tuple(p) for p in np.argwhere(labels == int(np.argmax(sizes)))}
    if len(pixels) < 8:
        raise ValueError(_ROUTE_ERROR)
    # Only junctions (more than two neighbors) can make a walk choose wrong;
    # try those first, padded with other pixels for extra search diversity.
    junctions = sorted(p for p in pixels if len(_skeleton_neighbors(pixels, *p)) > 2)
    candidates = (junctions + [p for p in sorted(pixels) if p not in junctions])[:32]
    best = None
    for start in candidates:
        loop = _longest_loop_through(pixels, start)
        if loop is not None and (best is None or len(loop) > len(best)):
            best = loop
    if best is None or len(best) < _MIN_LOOP_COVERAGE * len(pixels):
        raise ValueError(_ROUTE_ERROR + (" (multiple disconnected loops found)" if best else ""))
    return gaussian_filter1d(np.array(best, dtype=float), 1.5, axis=0, mode="wrap")


def _resample(points, spacing=.1):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 4 or not np.isfinite(points).all():
        raise ValueError("Centerline must contain at least four finite x,y points in route order")
    if np.linalg.norm(points[0] - points[-1]) < 1e-6:
        points = points[:-1]
    delta = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(delta, axis=1)
    if np.any(lengths < 1e-6) or lengths.sum() < 2:
        raise ValueError("Centerline must be a loop at least two meters long without duplicate neighbors")
    arc = np.r_[0, np.cumsum(lengths)]
    count = max(32, int(np.ceil(arc[-1] / spacing)))
    if count > 100_000:
        raise ValueError("Route exceeds 100,000 samples; check meters per pixel")
    samples = np.linspace(0, arc[-1], count, endpoint=False)
    closed = np.vstack((points, points[0]))
    return np.column_stack([np.interp(samples, arc, closed[:, k]) for k in range(2)]).astype(np.float32)


class MapTrack:
    """Flat occupancy track; occupied/unknown cells become finite-height walls.

    Collision uses a conservative circle enclosing the entire box chassis.
    One image pixel is one grid sample; no sensor triangle mesh is built.
    """
    elevation = 0.0
    frequency = 0.0

    def __init__(self, free, resolution, origin, route, *, wall_height=1., name="map"):
        self.free = np.array(free, dtype=bool, copy=True)
        self.resolution = float(resolution)
        self.origin = tuple(float(x) for x in origin)
        self.barrier_height = float(wall_height)
        self.name = str(name)
        if (self.free.ndim != 2 or min(self.free.shape) < 8 or self.free.size > 4_000_000
                or not self.free.any() or not np.isfinite(self.resolution) or self.resolution <= 0
                or len(self.origin) != 2 or not np.isfinite(self.origin).all()
                or not np.isfinite(self.barrier_height) or self.barrier_height <= 0):
            raise ValueError("Map requires free space, 8x8 to four million samples, positive resolution/wall height and finite XY origin")
        if any((self.free[0].any(), self.free[-1].any(), self.free[:, 0].any(), self.free[:, -1].any())):
            raise ValueError("Map must have an occupied/unknown border; free space cannot touch the PNG edge")
        self.length, self.width = np.array(self.free.shape) * self.resolution
        self.route = np.array(route, dtype=np.float32, copy=True)
        if (self.route.ndim != 2 or self.route.shape[1] != 2 or len(self.route) < 4
                or not np.isfinite(self.route).all()):
            raise ValueError("Map route must contain finite XY points")
        self.segment_lengths = np.linalg.norm(np.roll(self.route, -1, axis=0) - self.route, axis=1)
        if self.segment_lengths.min() < 1e-5 or self.segment_lengths.max() > .151:
            raise ValueError("Map route must be densely sampled (segments <= 0.15 m)")
        self.arc = np.r_[0, np.cumsum(self.segment_lengths[:-1])].astype(np.float32)
        self.route_length = float(self.segment_lengths.sum())
        # EDT measures center-to-center. Subtract a cell half-diagonal to bound
        # distance to the entire occupied pixel. Runtime also subtracts the
        # distance to the queried sample, preserving a conservative lower bound.
        self.clearance = (distance_transform_edt(self.free) * self.resolution
                          - self.resolution / np.sqrt(2)).astype(np.float32)
        for array in (self.free, self.route, self.arc, self.segment_lengths, self.clearance):
            array.flags.writeable = False

    def clearance_at(self, points):
        p = np.asarray(points)
        grid = (p - self.origin) / self.resolution - .5
        ij = np.floor(grid + .5).astype(int)
        inside = ((ij >= 0) & (ij < self.free.shape)).all(axis=-1)
        clipped = np.clip(ij, 0, np.array(self.free.shape) - 1)
        values = self.clearance[clipped[..., 0], clipped[..., 1]]
        return np.where(inside, values - np.linalg.norm((grid - ij) * self.resolution, axis=-1), -1.)

    def validate_vehicle(self, vehicle):
        radius = np.hypot(vehicle.length, vehicle.width) / 2
        # Check densely between route points, including the closing segment.
        points = np.concatenate([self.route + f * (np.roll(self.route, -1, axis=0) - self.route)
                                 for f in np.linspace(0, 1, 5, endpoint=False)])
        minimum = float(self.clearance_at(points).min())
        if minimum <= radius + .02:
            raise ValueError(f"Route clearance {minimum:.3f} m is too small for the car's "
                             f"{radius:.3f} m collision radius plus 0.02 m spawn margin. "
                             "Check meters per pixel, clean the PNG, or supply a safer centerline")
        return minimum

    def path(self, ride_height):
        delta = np.roll(self.route, -1, axis=0) - np.roll(self.route, 1, axis=0)
        heading = np.arctan2(delta[:, 1], delta[:, 0])
        poses = np.zeros((len(self.route), 7), np.float32)
        poses[:, :2], poses[:, 2] = self.route, ride_height
        poses[:, 5], poses[:, 6] = np.sin(heading / 2), np.cos(heading / 2)
        return poses

    def distance_field(self):
        # Add an empty sample border required by the bounded ray marcher.
        shape = np.array(self.free.shape) + 4
        surface = np.zeros(shape, bool)
        walls = np.zeros(shape, bool)
        surface[2:-2, 2:-2] = True
        walls[2:-2, 2:-2] = ~self.free
        origin = np.array(self.origin) - 1.5 * self.resolution
        return DistanceField(np.zeros(shape, np.float32), surface, walls,
                             tuple(origin), self.resolution, self.barrier_height)

    def mesh_parts(self):
        # Replay-only side walls use the same bilinear wall zero contour as the
        # sensor (marching squares); no mesh is used by physics or LiDAR.
        from skimage.measure import find_contours
        field = self.distance_field()
        from .distance_field import _signed_distance
        sdf = _signed_distance(field.walls, field.cell_size)
        vertices, faces = [], []
        for contour in find_contours(sdf, 0):
            points = np.asarray(field.origin) + contour * self.resolution
            for a, b in zip(points[:-1], points[1:]):
                offset = len(vertices)
                vertices.extend([[*a, 0], [*b, 0], [*b, self.barrier_height], [*a, self.barrier_height]])
                faces.extend([[offset, offset+1, offset+2], [offset, offset+2, offset+3]])
        x, y = self.origin
        road = np.array([[x, y, 0], [x+self.length, y, 0],
                         [x+self.length, y+self.width, 0], [x, y+self.width, 0]], np.float32)
        return {"road": (road, np.array([[0, 1, 2], [0, 2, 3]], np.int32)),
                "walls": (np.array(vertices, np.float32), np.array(faces, np.int32))}

    def state_dict(self):
        return {"kind": "png", "name": self.name, "shape": list(self.free.shape),
                "free_bits": np.packbits(self.free).tobytes().hex(), "resolution": self.resolution,
                "origin": list(self.origin), "wall_height": self.barrier_height,
                "route": self.route.tolist()}

    @classmethod
    def from_state_dict(cls, state):
        shape = tuple(state["shape"])
        free = np.unpackbits(np.frombuffer(bytes.fromhex(state["free_bits"]), dtype=np.uint8))
        return cls(free[:int(np.prod(shape))].reshape(shape), state["resolution"], state["origin"],
                   state["route"], wall_height=state["wall_height"], name=state["name"])


def load_map(path):
    """Load a JSON map description, or a PNG with a same-stem JSON sidecar."""
    path = Path(path)
    if path.suffix.lower() == ".png":
        path = path.with_suffix(".json")
    config = json.loads(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("Map JSON must be an object with image and resolution")
    allowed = {"image", "resolution", "origin", "free_threshold", "negate", "wall_height", "centerline", "reverse"}
    if set(config) - allowed:
        raise ValueError(f"Unknown map settings: {sorted(set(config) - allowed)}")
    resolution = config.get("resolution")
    if type(resolution) not in (float, int) or not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("Map JSON requires resolution: meters per pixel, e.g. 0.05")
    threshold = config.get("free_threshold", .95)
    if not isinstance(threshold, (int, float)) or not 0 < threshold < 1:
        raise ValueError("free_threshold must be between zero and one")
    for key in ("negate", "reverse"):
        if type(config.get(key, False)) is not bool:
            raise ValueError(f"{key} must be true or false")
    for key in ("image", "centerline"):
        if key in config and not isinstance(config[key], str):
            raise ValueError(f"{key} must be a relative file path string")
    with Image.open(path.parent / config["image"]) as image:
        if image.format != "PNG" or min(image.size) < 8 or image.width * image.height > 4_000_000:
            raise ValueError("Map image must be a PNG, 8x8 to four million pixels")
        rgba = np.array(image.convert("RGBA"))
    intensity = np.asarray(Image.fromarray(rgba).convert("L"), dtype=float) / 255
    if config.get("negate", False):
        intensity = 1 - intensity
    free = np.flipud((intensity >= threshold) & (rgba[:, :, 3] == 255)).T.copy()
    origin = np.asarray(config.get("origin", [0., 0.]), dtype=float)
    if origin.shape != (2,) or not np.isfinite(origin).all():
        raise ValueError("origin must be finite [x, y] for the bottom-left pixel corner")
    if config.get("centerline"):
        route = np.loadtxt(path.parent / config["centerline"], delimiter=",")
    else:
        route = origin + (_automatic_route(free) + .5) * resolution
        area = np.sum(route[:, 0] * np.roll(route[:, 1], -1) - route[:, 1] * np.roll(route[:, 0], -1))
        if area < 0:  # default counterclockwise; supplied CSV keeps its order
            route = route[::-1]
    route = _resample(route)
    if config.get("reverse", False):
        route = route[::-1].copy()
    return MapTrack(free, resolution, origin, route, wall_height=config.get("wall_height", 1.), name=path.stem)


def track_state(track):
    return track.state_dict() if hasattr(track, "state_dict") else asdict(track)


def restore_track(state):
    if state.get("kind") == "layered":
        from .layers import LayeredTrack
        return LayeredTrack.from_state_dict(state)
    return MapTrack.from_state_dict(state) if state.get("kind") == "png" else OvalTrack(**state)


def main(argv=None):
    import argparse
    from .scene import Vehicle
    parser = argparse.ArgumentParser(description="Validate a PNG track and preview its route")
    parser.add_argument("map", type=Path, help="Map JSON or PNG with a JSON sidecar")
    parser.add_argument("--resolution", type=float, help="Create a new PNG sidecar with this meters-per-pixel scale")
    parser.add_argument("--output", type=Path, help="Preview PNG; defaults to outputs/maps/<name>.png")
    args = parser.parse_args(argv)
    if args.resolution is not None:
        if args.map.suffix.lower() != ".png" or not np.isfinite(args.resolution) or args.resolution <= 0:
            parser.error("--resolution requires a PNG and a positive meters-per-pixel value")
        sidecar = args.map.with_suffix(".json")
        if sidecar.exists():
            parser.error(f"{sidecar} already exists; edit its resolution instead")
        try:
            sidecar.write_text(json.dumps({"image": args.map.name, "resolution": args.resolution}, indent=2) + "\n")
        except OSError as error:
            parser.error(str(error))
    try:
        track = load_map(args.map)
        minimum = track.validate_vehicle(Vehicle())
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    preview = Image.fromarray(np.flipud(track.free.T).astype(np.uint8) * 190).convert("RGB")
    draw = ImageDraw.Draw(preview)
    points = (track.route - track.origin) / track.resolution - .5
    points[:, 1] = preview.height - 1 - points[:, 1]
    draw.line([tuple(p) for p in np.vstack((points, points[0]))], fill=(40, 220, 110), width=2)
    p, q = points[0], points[min(8, len(points)-1)]
    draw.line([tuple(p), tuple(q)], fill=(255, 110, 50), width=4)
    direction = q - p
    direction /= np.linalg.norm(direction)
    left = np.array([-direction[1], direction[0]])
    draw.polygon([tuple(q), tuple(q - 7*direction + 4*left), tuple(q - 7*direction - 4*left)],
                 fill=(255, 110, 50))
    output = args.output or Path("outputs/maps") / f"{track.name}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    preview.save(output)
    print(f"Map {track.name}: {track.length:.2f} x {track.width:.2f} m; "
          f"route {track.route_length:.2f} m; minimum conservative clearance {minimum:.3f} m")
    print(f"Route preview: {output}")
