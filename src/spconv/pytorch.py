"""Pure-PyTorch subset of spconv.pytorch: SparseConvTensor and SubMConv3d (submanifold convolution, stride 1).
Weights use the spconv 2.x layout (out, kD, kH, kW, in); cross-correlation as in spconv:
out[o] = sum_k W[k] @ in[o + (k - center) * dilation]. Verified against a dense F.conv3d (see tests)."""
import enum
import itertools

import torch
import torch.nn as nn


class ConvAlgo(enum.Enum):
    Native = 0
    MaskImplicitGemm = 1
    MaskSplitImplicitGemm = 2


class SparseConvTensor:
    def __init__(self, features, indices, spatial_shape, batch_size, grid=None, voxel_num=None,
                 indice_dict=None, benchmark=False, **kw):
        self._features = features
        self.indices = indices
        self.spatial_shape = [int(s) for s in spatial_shape]
        self.batch_size = int(batch_size)
        self.grid = grid
        self.voxel_num = voxel_num
        self.indice_dict = {} if indice_dict is None else indice_dict
        self.benchmark = benchmark
        self.benchmark_record = {}
        self.thrust_allocator = None
        self._timer = None
        self.force_algo = None
        self.int8_scale = None

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, v):
        self._features = v

    def replace_feature(self, f):
        return SparseConvTensor(f, self.indices, self.spatial_shape, self.batch_size, self.grid,
                                self.voxel_num, self.indice_dict)

    def dense(self, channels_first=True):
        f = self._features.reshape(self._features.shape[0], -1)
        out = torch.zeros(self.batch_size, *self.spatial_shape, f.shape[1], dtype=f.dtype, device=f.device)
        i = self.indices.long()
        out[i[:, 0], i[:, 1], i[:, 2], i[:, 3]] = f
        return out.permute(0, 4, 1, 2, 3).contiguous() if channels_first else out


def _neighbours(indices, kernel, dilation):
    """Neighbour row for every kernel offset (-1 if absent), via hashed coordinates. Returns (K, N)."""
    c = indices.long()
    M = int(c[:, 1:].max()) + 2 * max(kernel) * max(dilation) + 2

    def key(x):
        return ((x[:, 0] * M + x[:, 1]) * M + x[:, 2]) * M + x[:, 3]
    off = max(kernel) * max(dilation)
    base = c.clone(); base[:, 1:] += off
    k = key(base)
    order = torch.argsort(k); ks = k[order]
    res = []
    for d in itertools.product(*[range(n) for n in kernel]):
        delta = torch.tensor([0] + [(di - n // 2) * dl for di, n, dl in zip(d, kernel, dilation)], device=c.device)
        q = key(base + delta)
        pos = torch.searchsorted(ks, q).clamp(max=len(ks) - 1)
        hit = ks[pos] == q
        res.append(torch.where(hit, order[pos], torch.full_like(pos, -1)))
    return torch.stack(res, 0)


class SubMConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, indice_key=None, algo=None, **kw):
        super().__init__()
        t = lambda x: tuple(x) if isinstance(x, (list, tuple)) else (x, x, x)  # noqa: E731
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.dilation = t(kernel_size), t(dilation)
        self.indice_key = indice_key
        self.weight = nn.Parameter(torch.zeros(out_channels, *self.kernel_size, in_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        f = x.features
        cache_key = ("_rocm_shims_nbr", self.kernel_size, self.dilation)
        nb = x.indice_dict.get(cache_key)
        if nb is None or nb.shape[1] != f.shape[0]:
            nb = _neighbours(x.indices, self.kernel_size, self.dilation)
            x.indice_dict[cache_key] = nb
        W = self.weight.reshape(self.out_channels, -1, self.in_channels).to(f.dtype)   # (O, K, I)
        out = torch.zeros(f.shape[0], self.out_channels, dtype=f.dtype, device=f.device)
        fz = torch.cat([f, torch.zeros(1, f.shape[1], dtype=f.dtype, device=f.device)], 0)
        for k in range(nb.shape[0]):
            idx = nb[k]
            if (idx < 0).all():
                continue
            out += fz[idx] @ W[:, k].t()     # index -1 picks the zero row
        if self.bias is not None:
            out = out + self.bias.to(f.dtype)
        return x.replace_feature(out)


class SparseConv3d(SubMConv3d):
    def __init__(self, *a, stride=1, padding=0, **kw):
        if stride not in (1, (1, 1, 1), [1, 1, 1]):
            raise NotImplementedError("rocm-shims: strided SparseConv3d is not implemented yet")
        super().__init__(*a, **kw)


class SparseInverseConv3d(nn.Module):
    def __init__(self, *a, **kw):
        raise NotImplementedError("rocm-shims: SparseInverseConv3d is not implemented yet")
