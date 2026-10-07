"""Compare both representations using identical, fully 3D sensor poses."""
import numpy as np
import warp as wp

from racesense3d.core import Scene
from .driving import DriveConfig
from .heightfield import GridScene, HeightField
from .lidar import LidarConfig
from .scene import Vehicle
from .terrain import road_rotation


def oval_scan_comparison(track, device, *, beams=108, cell_size=.025):
    """Measure discretization error, independently of navigation trajectories.

    Sample 48 positions around each of three lane offsets, with road-aligned and
    +/-15-degree pitched sensors. Include sky misses and road/roof intersections.
    This measures approximation error; it is not an exact parity check.
    """
    config = LidarConfig(beams=beams)
    rays = config.rays()
    vehicle = Vehicle()
    ride_height = DriveConfig().ride_height(vehicle)
    mount = np.array([.12, 0, vehicle.height / 2 + .025])
    origins, rotations = [], []
    a, b = track.center_axes
    for offset in (-.5, 0, .5):
        for angle in np.linspace(-np.pi / 2, 3 * np.pi / 2, 48, endpoint=False):
            x, y = (a + offset) * np.cos(angle), (b + offset) * np.sin(angle)
            body_position = np.array([x, y, track.height(x) + ride_height *
                                      np.sqrt(1 + track.gradient(x) ** 2)])
            # Recompute tilt at the offset position; keep the centerline tangent.
            heading = np.arctan2(b * np.cos(angle), -a * np.sin(angle))
            rotation = np.array(wp.quat_to_matrix(road_rotation(float(heading), float(track.gradient(x))))).reshape(3, 3)
            for pitch in (0, -np.pi / 12, np.pi / 12):
                origins.append(body_position + rotation @ mount)
                rotations.append(rotation @ np.array(wp.quat_to_matrix(wp.quat_rpy(0., float(pitch), 0.))).reshape(3, 3))
    positions = np.asarray(origins, np.float32)
    rotations = np.asarray(rotations, np.float32)
    mesh = Scene(*track.mesh(), device=device)
    grid = GridScene(HeightField.oval(track, cell_size), device=device)
    outputs = []
    for scene in (mesh, grid):
        outputs.append(scene.sensor(rays, batch_size=len(positions), near=config.near,
                                    far=config.far).scan(positions, rotations).numpy())
    mesh_ranges, mesh_valid = outputs[0]
    grid_ranges, grid_valid = outputs[1]
    common = mesh_valid & grid_valid
    error = np.abs(mesh_ranges[common] - grid_ranges[common])
    mismatch = mesh_valid != grid_valid
    stats = {name: float(np.percentile(error, percentile)) if len(error) else None
             for name, percentile in (("median", 50), ("p95", 95), ("p99", 99), ("p999", 99.9), ("max", 100))}
    locations = np.argwhere(common)
    worst = []
    for index in np.argsort(error)[-5:][::-1]:
        pose = int(locations[index, 0])
        # LiDAR has one elevation row; reshape to a flat beam index for reporting.
        beam = int(locations[index, -1])
        worst.append({"pose_index": int(pose), "beam": beam, "origin": positions[pose].tolist(),
                      "direction": (rotations[pose] @ rays.directions[beam]).tolist(),
                      "mesh_range_m": float(mesh_ranges[tuple(locations[index])]),
                      "grid_range_m": float(grid_ranges[tuple(locations[index])])})
    return {
        "scope": "Identical sensor poses; 48 angles, three lane offsets, pitches 0/+15/-15 degrees",
        "reference": "mesh", "candidate": "grid", "grid_cell_size_m": cell_size,
        "sensor_poses": len(positions), "rays": mesh_valid.size,
        "common_valid_rays": int(common.sum()), "valid_mismatch_count": int(mismatch.sum()),
        "valid_mismatch_fraction": float(mismatch.mean()),
        "grid_only_hits": int((grid_valid & ~mesh_valid).sum()),
        "mesh_only_hits": int((mesh_valid & ~grid_valid).sum()),
        "absolute_range_error_m": stats,
        "common_hits_within_5cm_fraction": float((error <= .05).mean()) if len(error) else None,
        "common_hit_errors_over_10cm": int((error > .1).sum()),
        "common_hit_errors_over_1m": int((error > 1).sum()), "worst_common_hits": worst,
        "note": "Rasterized walls and bilinear terrain approximate the mesh; grazing/edge hits may differ substantially. Common-hit errors exclude validity mismatches, which are reported separately.",
    }
