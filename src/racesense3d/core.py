"""Shared closest-hit ray caster; no graphics context or physics engine."""
from dataclasses import dataclass
import numpy as np
import warp as wp
from .sensors import Rays

@wp.kernel
def _cast(mesh: wp.uint64, origins: wp.array(dtype=wp.vec3),
          rotations: wp.array(dtype=wp.mat33), directions: wp.array(dtype=wp.vec3),
          factors: wp.array(dtype=float), near: float, far: float,
          values: wp.array2d(dtype=float), valid: wp.array2d(dtype=int)):
    b, r = wp.tid()
    direction = rotations[b] * directions[r]
    factor = factors[r]
    hit = wp.mesh_query_ray(mesh, origins[b], direction, far / factor)
    values[b, r] = far
    valid[b, r] = 0
    if hit.result:
        value = hit.t * factor
        # A too-near surface still blocks farther geometry.
        if value >= near and value < far:
            values[b, r] = value
            valid[b, r] = 1

@dataclass
class Scan:
    values: wp.array
    valid: wp.array
    shape: tuple

    def numpy(self):
        """Explicit device synchronization and host copy, for inspection."""
        wp.synchronize_device(self.values.device)
        return (self.values.numpy().reshape(self.shape),
                self.valid.numpy().reshape(self.shape).astype(bool))

    def points(self, rays, positions, rotations=None):
        """World-frame point cloud [batch, *sensor_shape, 3]; invalid = NaN."""
        values, valid = self.numpy()
        p, R = validate_poses(positions, rotations)
        if (len(p),) + rays.shape != self.shape:
            raise ValueError("Pose or ray layout does not match scan")
        t = values.reshape(len(p), -1) / rays.factors
        local = t[..., None] * rays.directions
        points = np.einsum('bij,brj->bri', R, local) + p[:, None]
        points[~valid.reshape(len(p), -1)] = np.nan
        return points.reshape(self.shape + (3,))


def validate_poses(positions, rotations=None):
    p = np.atleast_2d(np.asarray(positions, dtype=np.float32))
    if p.ndim != 2 or p.shape[1] != 3 or not len(p) or not np.isfinite(p).all():
        raise ValueError("Positions must be finite [batch, 3]")
    R = np.array(np.broadcast_to(np.eye(3) if rotations is None else rotations,
                                (len(p), 3, 3)), dtype=np.float32, copy=True)
    if (not np.isfinite(R).all()
        or not np.allclose(R.transpose(0, 2, 1) @ R, np.eye(3), atol=1e-5)
        or not np.allclose(np.linalg.det(R), 1, atol=1e-5)):
        raise ValueError("Rotations must be proper orthonormal matrices")
    return p, R


class Scene:
    """Immutable, two-sided triangle mesh shared by independent sensor poses."""
    def __init__(self, vertices, triangles, device=None):
        wp.init()
        self.device = wp.get_device(device or ('cuda:0' if wp.is_cuda_available() else 'cpu'))
        v = np.asarray(vertices, dtype=np.float32)
        f = np.asarray(triangles)
        if (v.ndim != 2 or v.shape[1] != 3 or not len(v) or not np.isfinite(v).all()
            or f.ndim != 2 or f.shape[1] != 3 or not len(f)
            or not np.issubdtype(f.dtype, np.integer) or f.min() < 0 or f.max() >= len(v)):
            raise ValueError("Invalid vertices or triangle indices")
        tri = v[f]
        if np.any(np.linalg.norm(np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]), axis=1) == 0):
            raise ValueError("Degenerate triangle")
        self.mesh = wp.Mesh(points=wp.array(v, dtype=wp.vec3, device=self.device),
                            indices=wp.array(f.astype(np.int32).ravel(), dtype=int, device=self.device))

    def sensor(self, rays: Rays, batch_size=1, near=0.02, far=30.0):
        return Sensor(self, rays, batch_size, near, far)


class Sensor:
    """Fixed-size reusable buffers. Results remain on the selected device.

    scan() accepts host poses and uploads them. scan_device() consumes resident
    Warp pose buffers without a host copy. Returned Scan buffers are borrowed:
    the next scan overwrites them. One sensor instance is not thread-safe.
    """
    def __init__(self, scene, rays, batch_size, near, far):
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not np.isfinite([near, far]).all() or not 0 <= near < far:
            raise ValueError("Require finite 0 <= near < far")
        self.scene, self.rays, self.batch_size = scene, rays, batch_size
        self.near, self.far = float(near), float(far)
        d = scene.device
        self.directions = wp.array(rays.directions, dtype=wp.vec3, device=d)
        self.factors = wp.array(rays.factors, dtype=float, device=d)
        self.origins = wp.empty(batch_size, dtype=wp.vec3, device=d)
        self.rotations = wp.empty(batch_size, dtype=wp.mat33, device=d)
        shape = (batch_size, len(rays.directions))
        self.result = Scan(wp.empty(shape, dtype=float, device=d),
                           wp.empty(shape, dtype=int, device=d), (batch_size,) + rays.shape)

    def scan(self, positions=((0, 0, 0),), rotations=None):
        p, R = validate_poses(positions, rotations)
        if len(p) != self.batch_size:
            raise ValueError("Pose batch does not match sensor batch_size")
        self.origins.assign(p)
        self.rotations.assign(R)
        return self.scan_device(self.origins, self.rotations)

    def scan_device(self, origins, rotations):
        """Caller must supply finite positions and proper rotations.

        Pose arrays must live on the scene device and remain alive until launch
        completion. Ordering/synchronization with other frameworks is caller-owned.
        """
        for a, dtype in ((origins, wp.vec3), (rotations, wp.mat33)):
            if a.shape != (self.batch_size,) or a.dtype != dtype or a.device != self.scene.device:
                raise ValueError("Pose buffers have incompatible shape, dtype, or device")
        wp.launch(_cast, dim=self.result.values.shape,
                  inputs=[self.scene.mesh.id, origins, rotations, self.directions,
                          self.factors, self.near, self.far],
                  outputs=[self.result.values, self.result.valid], device=self.scene.device)
        return self.result
