"""Small analytic fixtures and a demonstration scene; no external assets."""
import numpy as np
from .core import Scene


def from_quads(quads, device=None):
    q = np.asarray(quads, dtype=np.float32)
    if q.ndim != 3 or q.shape[1:] != (4, 3):
        raise ValueError("Expected [N,4,3] planar quads")
    triangles = np.array([[0,1,2], [0,2,3]])[None] + 4*np.arange(len(q))[:,None,None]
    return Scene(q.reshape(-1,3), triangles.reshape(-1,3), device)


def wall_x(x, span=20):
    return [(x,-span,-span), (x,span,-span), (x,span,span), (x,-span,span)]


GROUND = [(-20,-20,0), (20,-20,0), (20,20,0), (-20,20,0)]


def box(lo, hi):
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo,hi]).all() or np.any(hi <= lo):
        raise ValueError("Expected finite ordered 3D box bounds")
    faces = []
    for axis in range(3):
        a, b = [i for i in range(3) if i != axis]
        for side in (lo[axis], hi[axis]):
            face = []
            for j,k in ((0,0),(1,0),(1,1),(0,1)):
                p = lo.copy()
                p[axis], p[a], p[b] = side, (lo,hi)[j][a], (lo,hi)[k][b]
                face.append(p)
            faces.append(face)
    return faces


def demo_scene(device=None):
    ramp = [(4,1,0), (7,1,1.2), (7,3,1.2), (4,3,0)]
    return from_quads([GROUND, wall_x(8), ramp] + box([3,-1,0], [4,0.5,1.5]), device)
