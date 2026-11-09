"""
Geometric rules for the design candidates
"""
import numpy as np

LOWER = np.array([0.03, -2.0, -2.0, -2.0, -2.0])
UPPER = np.array([0.97, 2.0, 2.0, 2.0, 2.0])


def _cross(o, a, b):
    return (a[..., 0] - o[..., 0]) * (b[..., 1] - o[..., 1]) - (a[..., 1] - o[..., 1]) * (b[..., 0] - o[..., 0])

def _segments_cross(p1, p2, q1, q2):
    d1, d2 = _cross(q1, q2, p1), _cross(q1, q2, p2)
    d3, d4 = _cross(p1, p2, q1), _cross(p1, p2, q2)
    return (d1 * d2 < 0) & (d3 * d4 < 0)

def quads(x):
    x = np.atleast_2d(x)
    n = len(x)
    zero, one, half = np.zeros(n), np.ones(n), np.full(n, 0.5)
    x1, y3, y4, y7, y8 = x.T
    top = np.stack([np.c_[zero, y3], np.c_[x1, zero], np.c_[one, y7], np.c_[half, one]], axis=1)
    bot = np.stack([np.c_[zero, y4], np.c_[x1, zero], np.c_[one, y8], np.c_[half, -one]], axis=1)
    return top, bot

def is_valid(x, w):
    x = np.atleast_2d(x)
    w = np.broadcast_to(np.asarray(w, dtype=float), (len(x),))
    top, bot = quads(x)
    ok = (x[:, 1] - x[:, 2] > 2 * w) & (x[:, 3] - x[:, 4] > 2 * w)
    for q in (top, bot):  # simple polygon: opposite edges do not cross
        ok &= ~_segments_cross(q[:, 0], q[:, 1], q[:, 2], q[:, 3])
        ok &= ~_segments_cross(q[:, 1], q[:, 2], q[:, 3], q[:, 0])
    for i in range(4):
        for j in range(4):
            ok &= ~_segments_cross(top[:, i], top[:, (i + 1) % 4], bot[:, j], bot[:, (j + 1) % 4])
    return ok & np.all((x >= LOWER) & (x <= UPPER), axis=1)

