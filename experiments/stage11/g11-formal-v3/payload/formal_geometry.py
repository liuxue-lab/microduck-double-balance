"""Real mesh projection and finite-radius release envelope; no AABB/rim box."""
import hashlib
import numpy as np


class HeadSurface:
    def __init__(self, triangles, closest, height, sphere):
        self.triangles = np.asarray(triangles, dtype=np.float64)
        if self.triangles.ndim != 3 or self.triangles.shape[1:] != (3, 3) or not np.isfinite(self.triangles).all():
            raise ValueError('Invalid actual head triangles')
        self.closest, self.height, self.sphere = closest, height, sphere
        self.projected = self.triangles.copy()
        self.projected[:, :, 2] = 0.0
        self.sha256 = hashlib.sha256(self.triangles.tobytes()).hexdigest()
        self.spawn, self.witness, self.normal, self.face = sphere(
            np.zeros(2), self.triangles, radius=.020, clearance=.0005)

    def measure(self, point):
        p = np.asarray(point, dtype=np.float64)
        if p.shape != (3,) or not np.isfinite(p).all():
            return dict(valid=False, lost=True)
        points, d2 = self.closest(np.array([p[0], p[1], 0.]), self.projected)
        i = int(d2.argmin())
        xy = points[i, :2]
        projected_distance = float(np.sqrt(d2[i]))
        upper_height = self.height(xy, self.triangles)
        points3, gap2 = self.closest(p, self.triangles)
        face = int(gap2.argmin())
        distance = float(np.sqrt(gap2[face]))
        # Same radius extension and -radius recovery depth as the old task;
        # substitute the exact mesh projection/upper envelope for its rectangle.
        lost = projected_distance > .020 or p[2] < upper_height-.020
        return dict(valid=True, lost=bool(lost), projected_distance_m=projected_distance,
                    upper_height_m=upper_height, surface_distance_m=distance,
                    surface_gap_m=distance-.020, nearest_face=face,
                    nearest_point=points3[face].tolist())
