"""Sensor-local rays. Meters, radians, right-handed frames."""
from dataclasses import dataclass
import numpy as np

CAMERA_TO_BODY = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float32)

@dataclass(frozen=True)
class Rays:
    directions: np.ndarray
    factors: np.ndarray
    shape: tuple

    def __post_init__(self):
        d = np.array(self.directions, dtype=np.float32, copy=True)
        f = np.array(self.factors, dtype=np.float32, copy=True)
        if (d.ndim != 2 or d.shape[1] != 3 or len(d) == 0
            or f.shape != (len(d),) or not np.isfinite(d).all()
            or not np.isfinite(f).all() or np.any(f <= 0)
            or not np.allclose(np.linalg.norm(d, axis=1), 1, atol=1e-6)
            or not self.shape or any(not isinstance(x, (int, np.integer)) or x <= 0 for x in self.shape)
            or np.prod(self.shape) != len(d)):
            raise ValueError("Invalid unit rays, measurement factors, or sensor shape")
        d.setflags(write=False)
        f.setflags(write=False)
        object.__setattr__(self, 'directions', d)
        object.__setattr__(self, 'factors', f)
        object.__setattr__(self, 'shape', tuple(self.shape))


def lidar(azimuth=None, elevation=(0.0,)):
    """X forward, Y left, Z up. Layout (elevation, azimuth), angles radians.

    Default: one 1080-beam ring over 270 degrees. For full 360-degree scans,
    use linspace(-pi, pi, N, endpoint=False) to avoid duplicate rays.
    """
    if azimuth is None:
        azimuth = np.linspace(-3*np.pi/4, 3*np.pi/4, 1080)
    a, e = np.atleast_1d(azimuth), np.atleast_1d(elevation)
    if a.ndim != 1 or e.ndim != 1 or np.any(np.abs(e) > np.pi/2):
        raise ValueError("Expected 1D angles; elevation within [-pi/2, pi/2]")
    a, e = np.meshgrid(a, e)
    d = np.stack([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)], -1)
    return Rays(d.reshape(-1, 3), np.ones(a.size), a.shape)


def camera(width=320, height=240, hfov_degrees=90.0, *, fx=None, fy=None, cx=None, cy=None):
    """Pinhole optical frame: X right, Y down, Z forward; layout (H, W).

    Pixel centers use integer coordinates. Default principal point is
    ((W-1)/2, (H-1)/2); default square pixels derive focal length from HFOV.
    Explicit fx/fy/cx/cy support calibrated intrinsics. No distortion.
    """
    if (not isinstance(width, int) or not isinstance(height, int)
        or width <= 0 or height <= 0 or not 0 < hfov_degrees < 179):
        raise ValueError("Invalid dimensions or field of view")
    fx = width/(2*np.tan(np.deg2rad(hfov_degrees)/2)) if fx is None else fx
    fy = fx if fy is None else fy
    cx = (width-1)/2 if cx is None else cx
    cy = (height-1)/2 if cy is None else cy
    if not np.isfinite([fx, fy, cx, cy]).all() or fx <= 0 or fy <= 0:
        raise ValueError("Invalid intrinsics")
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    d = np.stack([(u-cx)/fx, (v-cy)/fy, np.ones_like(u)], -1).reshape(-1, 3)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return Rays(d, d[:, 2], (height, width))
