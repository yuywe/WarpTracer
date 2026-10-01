"""Vehicle mount and track adapter for the existing racesense3d LiDAR caster.

src/racesense3d is copied unchanged from sensor-prototype commit 51b2d1c.
"""
from dataclasses import dataclass

import numpy as np
import warp as wp

from racesense3d import box, from_quads, lidar


@dataclass(frozen=True)
class LidarConfig:
    beams: int = 1080
    fov_degrees: float = 270.0
    near: float = 0.021
    far: float = 30.0
    frequency: int = 30
    # None places the sensor 2.5 cm above the chassis top, 12 cm forward of COM.
    mount_position: tuple | None = None
    mount_rpy: tuple = (0.0, 0.0, 0.0)

    def __post_init__(self):
        if not isinstance(self.beams, int) or self.beams < 1:
            raise ValueError("LiDAR beams must be a positive integer")
        if not isinstance(self.frequency, int) or self.frequency < 1:
            raise ValueError("LiDAR frequency must be a positive integer")
        if not np.isfinite((self.fov_degrees, self.near, self.far)).all():
            raise ValueError("LiDAR angles and limits must be finite")
        if not 0 < self.fov_degrees <= 360 or not 0 <= self.near < self.far:
            raise ValueError("Require 0 < FOV <= 360 and 0 <= near < far")
        for value in (self.mount_position, self.mount_rpy):
            if value is not None and (np.shape(value) != (3,) or not np.isfinite(value).all()):
                raise ValueError("Mount position and roll/pitch/yaw must be finite triples")
        if self.mount_rpy is None:
            raise ValueError("Mount roll/pitch/yaw must be a finite triple")
        object.__setattr__(self, "mount_rpy", tuple(float(x) for x in self.mount_rpy))
        if self.mount_position is not None:
            object.__setattr__(self, "mount_position", tuple(float(x) for x in self.mount_position))

    def rays(self):
        half_fov = np.deg2rad(self.fov_degrees) / 2
        angles = np.array([0.0]) if self.beams == 1 else np.linspace(
            -half_fov, half_fov, self.beams, endpoint=self.fov_degrees < 360)
        return lidar(angles)


@wp.kernel
def _mount_pose(body_q: wp.array(dtype=wp.transform), body: int, mount: wp.transform,
                poses: wp.array(dtype=wp.transform), origins: wp.array(dtype=wp.vec3),
                rotations: wp.array(dtype=wp.mat33)):
    pose = wp.transform_multiply(body_q[body], mount)
    poses[0] = pose
    origins[0] = wp.transform_get_translation(pose)
    rotations[0] = wp.quat_to_matrix(wp.transform_get_rotation(pose))


@dataclass
class LidarRecording:
    times: np.ndarray
    poses: np.ndarray       # sensor-to-world transforms, XYZ + XYZW
    ranges: np.ndarray      # [scan, beam], meters along unit rays
    valid: np.ndarray       # [scan, beam], bool; misses/filtered hits have far range
    directions: np.ndarray  # [beam, 3], sensor frame: X forward, Y left, Z up

    def points(self, index):
        """Valid returns in world coordinates; invalid beam positions are NaN."""
        pose = self.poses[index]
        local = self.ranges[index, :, None] * self.directions
        t = 2 * np.cross(pose[3:6], local)
        world = local + pose[6] * t + np.cross(pose[3:6], t) + pose[:3]
        world[~self.valid[index]] = np.nan
        return world


class MountedLidar:
    """Reusable, device-resident scanner. The host vehicle is excluded from its mesh."""
    def __init__(self, track, vehicle, config, device):
        self.config = config
        self.rays = config.rays()
        self.mount_position = config.mount_position or (0.12, 0.0, vehicle.height / 2 + 0.025)
        self.mount = wp.transform(self.mount_position, wp.quat_rpy(*config.mount_rpy))
        # Cover every possible floor hit within range while the car is in the enclosure.
        x, y = track.length / 2 + config.far, track.width / 2 + config.far
        quads = [[(-x, -y, 0), (x, -y, 0), (x, y, 0), (-x, y, 0)]]
        for position, dimensions in track.barriers():
            center, half_size = np.asarray(position), np.asarray(dimensions) / 2
            quads.extend(box(center - half_size, center + half_size))
        self.scene = from_quads(quads, device=device)
        self.sensor = self.scene.sensor(self.rays, near=config.near, far=config.far)
        self.poses = wp.empty(1, dtype=wp.transform, device=device)
        self.timestamp = 0.0

    @property
    def result(self):
        """Borrowed Scan buffers; the next scan overwrites them on the same device."""
        return self.sensor.result

    def update(self, state, body, timestamp):
        wp.launch(_mount_pose, dim=1, inputs=[state.body_q, body, self.mount],
                  outputs=[self.poses, self.sensor.origins, self.sensor.rotations],
                  device=self.scene.device)
        self.sensor.scan_device(self.sensor.origins, self.sensor.rotations)
        self.timestamp = float(timestamp)

    def snapshot(self):
        """Explicit host copies for recordings/inspection only."""
        values, valid = self.result.numpy()
        return self.poses.numpy()[0].copy(), values.reshape(-1).copy(), valid.reshape(-1).copy()
