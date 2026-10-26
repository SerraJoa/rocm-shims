"""Approximate o_voxel.convert.mesh_to_flexible_dual_grid: the voxels touched by the surface,
found by densely sampling each triangle (points closer than half a voxel). Returns (coords,)."""
import numpy as np
import torch


def mesh_to_flexible_dual_grid(vertices, faces, grid_size, aabb, **kw):
    V = vertices.detach().cpu().double().numpy(); F = faces.detach().cpu().long().numpy()
    lo = aabb[0].double().numpy(); hi = aabb[1].double().numpy()
    cell = (hi - lo) / grid_size
    T = V[F]                                            # (n, 3, 3)
    longest = np.linalg.norm(T - np.roll(T, 1, axis=1), axis=2).max(1)
    n = np.clip(np.ceil(longest / (0.35 * cell.min())).astype(int), 1, 512)
    pts = [V]
    for k in np.unique(n):
        sel = T[n == k]
        i, j = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing="ij")
        m = (i + j) <= k
        a, b = (i[m] / k)[:, None], (j[m] / k)[:, None]
        w = np.stack([1 - a - b, a, b], 1)[..., 0]     # (p, 3) barycentric weights
        pts.append(np.einsum("pk,nkd->npd", w, sel).reshape(-1, 3))
    P = np.vstack(pts)
    c = np.clip(np.floor((P - lo) / cell), 0, grid_size - 1).astype(np.int64)
    return (torch.from_numpy(np.unique(c, axis=0)).int(),)
