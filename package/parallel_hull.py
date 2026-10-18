"""Candidate: deterministic parallel evaluation of the frozen convex query.

The per-vertex and per-face arithmetic is unchanged. Ordered block reductions
retain the first strictly smaller squared distance and the original min order.
This module is a standalone geometry probe, not installed into any simulator.
"""
import warp as wp
from precise_pose_kernel import relative_vertex

wp.set_module_options({'enable_backward': False, 'fast_math': False, 'fuse_fp': False})
BLOCK = wp.constant(128)


@wp.kernel
def transform_vertices(vertices: wp.array[wp.vec3], positions: wp.array2d[wp.vec3d],
                       rotations: wp.array2d[wp.mat33d], relative: wp.array2d[wp.vec3d],
                       distances: wp.array2d[wp.float64]):
    w, j = wp.tid()
    v = relative_vertex(vertices[j], rotations[w, 1], positions[w, 1], positions[w, 0])
    relative[w, j] = v
    distances[w, j] = wp.dot(v, v)


@wp.kernel
def vertex_blocks(distances: wp.array2d[wp.float64], count: int,
                  block_distance: wp.array2d[wp.float64], block_id: wp.array2d[int]):
    w, b = wp.tid()
    best = wp.float64(1.0e300)
    index = int(-1)
    for j in range(b * BLOCK, wp.min((b + 1) * BLOCK, count)):
        value = distances[w, j]
        if value < best:
            best = value
            index = j
    block_distance[w, b] = best
    block_id[w, b] = index


@wp.kernel
def vertex_winner(relative: wp.array2d[wp.vec3d], block_distance: wp.array2d[wp.float64],
                  block_id: wp.array2d[int], blocks: int, point: wp.array[wp.vec3d],
                  winner_distance: wp.array[wp.float64], winner_id: wp.array[int]):
    w = wp.tid()
    best = wp.float64(1.0e300)
    index = int(-1)
    for b in range(blocks):
        value = block_distance[w, b]
        if value < best:
            best = value
            index = block_id[w, b]
    p = wp.vec3d()
    if index >= 0:
        p = relative[w, index]
    point[w] = p
    winner_distance[w] = best
    winner_id[w] = index


@wp.kernel
def support_dots(relative: wp.array2d[wp.vec3d], point: wp.array[wp.vec3d],
                 dots: wp.array2d[wp.float64]):
    w, j = wp.tid()
    p = point[w]
    dots[w, j] = wp.dot(relative[w, j] - p, p)


@wp.kernel
def support_blocks(dots: wp.array2d[wp.float64], count: int,
                   partial: wp.array2d[wp.float64]):
    w, b = wp.tid()
    best = wp.float64(1.0e300)
    for j in range(b * BLOCK, wp.min((b + 1) * BLOCK, count)):
        best = wp.min(best, dots[w, j])
    partial[w, b] = best


@wp.kernel
def vertex_certificate(partial: wp.array2d[wp.float64], blocks: int,
                       point: wp.array[wp.vec3d], winner_distance: wp.array[wp.float64],
                       winner_id: wp.array[int], positions: wp.array2d[wp.vec3d],
                       radius: float, fast: wp.array[int], result: wp.array2d[wp.float64]):
    w = wp.tid()
    min_dot = wp.float64(1.0e300)
    for b in range(blocks):
        min_dot = wp.min(min_dot, partial[w, b])
    length = wp.sqrt(winner_distance[w])
    valid = int(0)
    if winner_id[w] >= 0 and min_dot >= wp.float64(0.0) and length > wp.float64(0.0):
        valid = 1
        p = point[w]
        s = positions[w, 0]
        # Keep the original sign multiplication and expression association.
        normal = wp.float64(1.0) * p / length
        result[w, 0] = wp.float64(winner_id[w])
        result[w, 1] = min_dot
        result[w, 2] = length
        result[w, 3] = wp.float64(1.0) * length - wp.float64(radius)
        for k in range(3):
            result[w, 4 + k] = normal[k]
            result[w, 7 + k] = wp.float64(s[k]) + (normal[k] * wp.float64(radius) + p[k]) * wp.float64(.5)
        result[w, 10] = wp.float64(1.0)
        result[w, 11] = wp.float64(-1.0)
        result[w, 12] = wp.float64(0.0)
        result[w, 13] = wp.float64(0.0)
        result[w, 14] = wp.float64(0.0)
        result[w, 15] = wp.float64(0.0)
    fast[w] = valid


@wp.kernel
def evaluate_faces(relative: wp.array2d[wp.vec3d], facets: wp.array[wp.vec3i],
                   fast: wp.array[int], face_distance: wp.array2d[wp.float64],
                   face_point: wp.array2d[wp.vec3d], face_bary: wp.array2d[wp.vec2d],
                   face_feature: wp.array2d[int], face_inside: wp.array2d[int],
                   face_nondegenerate: wp.array2d[int]):
    w, j = wp.tid()
    if fast[w] == 0:
        ids = facets[j]
        a = relative[w, ids[0]]
        b = relative[w, ids[1]]
        c = relative[w, ids[2]]
        best = wp.float64(1.0e300)
        q_best = wp.vec3d()
        bary = wp.vec2d()
        feature = int(0)
        inside = int(1)
        nondegenerate = int(1)
        ab = b - a
        ac = c - a
        n = wp.cross(ab, ac)
        den = wp.dot(n, n)
        if wp.dot(n, -a) > wp.float64(0.0):
            inside = 0
        if den <= wp.float64(0.0):
            nondegenerate = 0
        else:
            u = wp.dot(wp.cross(-a, ac), n) / den
            beta = wp.dot(wp.cross(ab, -a), n) / den
            if u >= wp.float64(0.0) and beta >= wp.float64(0.0) and u + beta <= wp.float64(1.0):
                q = a + u * ab + beta * ac
                d2 = wp.dot(q, q)
                if d2 < best:
                    best = d2
                    q_best = q
                    feature = 2
                    bary = wp.vec2d(u, beta)
        for edge in range(3):
            start = a
            end = b
            if edge == 1:
                start = b
                end = c
            elif edge == 2:
                start = c
                end = a
            delta = end - start
            d2 = wp.dot(delta, delta)
            if d2 <= wp.float64(0.0):
                nondegenerate = 0
            else:
                t = wp.clamp(-wp.dot(start, delta) / d2, wp.float64(0.0), wp.float64(1.0))
                q = start + t * delta
                distance = wp.dot(q, q)
                if distance < best:
                    best = distance
                    q_best = q
                    feature = 1
                    if t == wp.float64(0.0) or t == wp.float64(1.0):
                        feature = 0
                    bary = wp.vec2d(t, wp.float64(0.0))
                    if edge == 1:
                        bary = wp.vec2d(wp.float64(1.0) - t, t)
                    elif edge == 2:
                        bary = wp.vec2d(wp.float64(0.0), wp.float64(1.0) - t)
        face_distance[w, j] = best
        face_point[w, j] = q_best
        face_bary[w, j] = bary
        face_feature[w, j] = feature
        face_inside[w, j] = inside
        face_nondegenerate[w, j] = nondegenerate


@wp.kernel
def face_blocks(face_distance: wp.array2d[wp.float64], face_inside: wp.array2d[int],
                face_nondegenerate: wp.array2d[int], fast: wp.array[int], count: int,
                block_distance: wp.array2d[wp.float64], block_id: wp.array2d[int],
                block_inside: wp.array2d[int], block_nondegenerate: wp.array2d[int]):
    w, b = wp.tid()
    if fast[w] == 0:
        best = wp.float64(1.0e300)
        index = int(-1)
        inside = int(1)
        nondegenerate = int(1)
        for j in range(b * BLOCK, wp.min((b + 1) * BLOCK, count)):
            value = face_distance[w, j]
            if value < best:
                best = value
                index = j
            if face_inside[w, j] == 0:
                inside = 0
            if face_nondegenerate[w, j] == 0:
                nondegenerate = 0
        block_distance[w, b] = best
        block_id[w, b] = index
        block_inside[w, b] = inside
        block_nondegenerate[w, b] = nondegenerate


@wp.kernel
def face_winner(block_distance: wp.array2d[wp.float64], block_id: wp.array2d[int],
                block_inside: wp.array2d[int], block_nondegenerate: wp.array2d[int],
                blocks: int, fast: wp.array[int], face_point: wp.array2d[wp.vec3d],
                face_bary: wp.array2d[wp.vec2d], face_feature: wp.array2d[int],
                point: wp.array[wp.vec3d], winner_distance: wp.array[wp.float64],
                winner_id: wp.array[int], bary: wp.array[wp.vec2d], feature: wp.array[int],
                inside_result: wp.array[int], valid_result: wp.array[int]):
    w = wp.tid()
    if fast[w] == 0:
        best = wp.float64(1.0e300)
        index = int(-1)
        inside = int(1)
        nondegenerate = int(1)
        for b in range(blocks):
            value = block_distance[w, b]
            if value < best:
                best = value
                index = block_id[w, b]
            if block_inside[w, b] == 0:
                inside = 0
            if block_nondegenerate[w, b] == 0:
                nondegenerate = 0
        # Preserve the previous nearest vertex if no face produces a candidate.
        if index >= 0:
            point[w] = face_point[w, index]
            bary[w] = face_bary[w, index]
            feature[w] = face_feature[w, index]
        else:
            bary[w] = wp.vec2d()
            feature[w] = 0
        winner_distance[w] = best
        winner_id[w] = index
        inside_result[w] = inside
        valid_result[w] = nondegenerate


@wp.kernel
def finish_faces(partial: wp.array2d[wp.float64], blocks: int, fast: wp.array[int],
                 point: wp.array[wp.vec3d], winner_distance: wp.array[wp.float64],
                 winner_id: wp.array[int], bary: wp.array[wp.vec2d], feature: wp.array[int],
                 inside: wp.array[int], nondegenerate: wp.array[int],
                 positions: wp.array2d[wp.vec3d], radius: float, result: wp.array2d[wp.float64]):
    w = wp.tid()
    if fast[w] == 0:
        min_dot = wp.float64(1.0e300)
        for b in range(blocks):
            min_dot = wp.min(min_dot, partial[w, b])
        length = wp.sqrt(winner_distance[w])
        valid = int(0)
        if nondegenerate[w] == 1 and winner_id[w] >= 0 and length > wp.float64(0.0):
            valid = 1
        sign = wp.float64(1.0)
        if inside[w] == 1:
            sign = wp.float64(-1.0)
        result[w, 0] = wp.float64(-1.0)
        result[w, 1] = min_dot
        result[w, 2] = length
        result[w, 3] = sign * length - wp.float64(radius)
        if valid == 1:
            p = point[w]
            s = positions[w, 0]
            normal = sign * p / length
            for k in range(3):
                result[w, 4 + k] = normal[k]
                result[w, 7 + k] = wp.float64(s[k]) + (normal[k] * wp.float64(radius) + p[k]) * wp.float64(.5)
        result[w, 10] = wp.float64(valid)
        result[w, 11] = wp.float64(winner_id[w])
        result[w, 12] = wp.float64(feature[w])
        result[w, 13] = wp.float64(inside[w])
        result[w, 14] = bary[w][0]
        result[w, 15] = bary[w][1]


class ParallelHull:
    """Bounded scratch buffers; one logical geometry query per supplied pose."""
    kernels_per_query_batch = 12

    def __init__(self, vertices, facets, capacity, device):
        self.vertices, self.facets, self.capacity, self.device = vertices, facets, capacity, device
        self.nv, self.nf = vertices.shape[0], facets.shape[0]
        self.vb, self.fb = (self.nv + 127)//128, (self.nf + 127)//128
        def a(shape, dtype): return wp.empty(shape, dtype=dtype, device=device)
        self.relative = a((capacity, self.nv), wp.vec3d)
        self.distances = a((capacity, self.nv), wp.float64)
        self.dots = a((capacity, self.nv), wp.float64)
        self.vdist = a((capacity, self.vb), wp.float64)
        self.vid = a((capacity, self.vb), int)
        self.support = a((capacity, self.vb), wp.float64)
        self.point = a(capacity, wp.vec3d)
        self.wdist = a(capacity, wp.float64)
        self.wid = a(capacity, int)
        self.fast = a(capacity, int)
        self.fdist = a((capacity, self.nf), wp.float64)
        self.fpoint = a((capacity, self.nf), wp.vec3d)
        self.fbary = a((capacity, self.nf), wp.vec2d)
        self.ffeature = a((capacity, self.nf), int)
        self.finside = a((capacity, self.nf), int)
        self.fvalid = a((capacity, self.nf), int)
        self.bdist = a((capacity, self.fb), wp.float64)
        self.bid = a((capacity, self.fb), int)
        self.binside = a((capacity, self.fb), int)
        self.bvalid = a((capacity, self.fb), int)
        self.bary = a(capacity, wp.vec2d)
        self.feature = a(capacity, int)
        self.inside = a(capacity, int)
        self.valid = a(capacity, int)

    def launch(self, positions, rotations, radius, result, count):
        if not 1 <= count <= self.capacity: raise ValueError('Geometry batch outside allocated capacity')
        def run(kernel, dim, args): wp.launch(kernel, dim=dim, inputs=args, device=self.device)
        run(transform_vertices, (count, self.nv), [self.vertices, positions, rotations, self.relative, self.distances])
        run(vertex_blocks, (count, self.vb), [self.distances, self.nv, self.vdist, self.vid])
        run(vertex_winner, count, [self.relative, self.vdist, self.vid, self.vb, self.point, self.wdist, self.wid])
        run(support_dots, (count, self.nv), [self.relative, self.point, self.dots])
        run(support_blocks, (count, self.vb), [self.dots, self.nv, self.support])
        run(vertex_certificate, count, [self.support, self.vb, self.point, self.wdist, self.wid, positions, radius, self.fast, result])
        run(evaluate_faces, (count, self.nf), [self.relative, self.facets, self.fast, self.fdist, self.fpoint, self.fbary, self.ffeature, self.finside, self.fvalid])
        run(face_blocks, (count, self.fb), [self.fdist, self.finside, self.fvalid, self.fast, self.nf, self.bdist, self.bid, self.binside, self.bvalid])
        run(face_winner, count, [self.bdist, self.bid, self.binside, self.bvalid, self.fb, self.fast, self.fpoint, self.fbary, self.ffeature, self.point, self.wdist, self.wid, self.bary, self.feature, self.inside, self.valid])
        run(support_dots, (count, self.nv), [self.relative, self.point, self.dots])
        run(support_blocks, (count, self.vb), [self.dots, self.nv, self.support])
        run(finish_faces, count, [self.support, self.vb, self.fast, self.point, self.wdist, self.wid, self.bary, self.feature, self.inside, self.valid, positions, radius, result])
