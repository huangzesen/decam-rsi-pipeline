"""Forced S/N photometry on the stored (2x decimated) matched-filter S/N maps."""
import math
import numpy as np
from detect import gnomonic, poly2


class Forced:
    def __init__(self, results):
        self.maps = {}
        for e, o in results.items():
            ms = o.get("maps") or []
            if not ms:
                continue
            geo = []
            for g, m in ms:
                cu = np.array([math.cos(g["dc"]) * math.cos(g["rc"]), math.cos(g["dc"]) * math.sin(g["rc"]), math.sin(g["dc"])])
                geo.append((g, m, cu, math.cos(g["rad"] * 1.1 + 1e-4)))
            self.maps[e] = geo

    def has(self, e):
        return e in self.maps

    def sample(self, e, u):
        """u (N,3) unit vectors -> forced S/N (nan where off-chip / no map)."""
        u = np.atleast_2d(u)
        out = np.full(len(u), np.nan)
        if e not in self.maps:
            return out
        ra = np.arctan2(u[:, 1], u[:, 0]); de = np.arcsin(np.clip(u[:, 2], -1, 1))
        for g, m, cu, cmin in self.maps[e]:
            near = np.flatnonzero(u @ cu > cmin)
            if len(near) == 0:
                continue
            xi, eta = gnomonic(ra[near], de[near], g["rc"], g["dc"])
            A = poly2(xi, eta)
            x = A @ g["cx"]; y = A @ g["cy"]
            inside = (x >= 2) & (x <= g["nx"] - 3) & (y >= 2) & (y <= g["ny"] - 3)
            if not inside.any():
                continue
            near, x, y = near[inside], x[inside] / 2.0, y[inside] / 2.0
            x0 = np.clip(np.floor(x).astype(int), 0, m.shape[1] - 2); y0 = np.clip(np.floor(y).astype(int), 0, m.shape[0] - 2)
            fx, fy = x - x0, y - y0
            v = ((1 - fx) * (1 - fy) * m[y0, x0].astype(np.float32) + fx * (1 - fy) * m[y0, x0 + 1] +
                 (1 - fx) * fy * m[y0 + 1, x0] + fx * fy * m[y0 + 1, x0 + 1])
            out[near] = v
        return out
