"""Pure-PyTorch stand-in for torch_scatter via Tensor.scatter_reduce:
scatter(reduce=...), scatter_sum/add/mean/max/min."""
import torch


def _prep(src, index, dim, dim_size):
    dim = dim % src.dim()
    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() else 0
    idx = index
    if idx.dim() == 1 and src.dim() > 1:   # 1D index along dim, as torch_scatter broadcasts it
        shape = [1] * src.dim(); shape[dim] = -1
        idx = idx.view(shape)
    while idx.dim() < src.dim():
        idx = idx.unsqueeze(-1)
    shape = list(src.shape); shape[dim] = dim_size
    return dim, idx.expand_as(src), shape


def scatter(src, index, dim=-1, out=None, dim_size=None, reduce="sum"):
    dim, idx, shape = _prep(src, index, dim, dim_size)
    red = {"sum": "sum", "add": "sum", "mean": "mean", "max": "amax", "min": "amin", "mul": "prod"}[reduce]
    res = torch.zeros(shape, dtype=src.dtype, device=src.device) if out is None else out
    return res.scatter_reduce(dim, idx, src, reduce=red, include_self=out is not None)


def scatter_sum(src, index, dim=-1, out=None, dim_size=None):
    return scatter(src, index, dim, out, dim_size, "sum")


scatter_add = scatter_sum


def scatter_mean(src, index, dim=-1, out=None, dim_size=None):
    return scatter(src, index, dim, out, dim_size, "mean")


def scatter_max(src, index, dim=-1, out=None, dim_size=None):
    """Returns (values, argmax) like torch_scatter; argmax is the position along dim."""
    v = scatter(src, index, dim, out, dim_size, "max")
    d, idx, _ = _prep(src, index, dim, dim_size)
    pos = torch.arange(src.shape[d], device=src.device).view([-1 if i == d else 1 for i in range(src.dim())])
    pos = pos.expand_as(src)
    equal = src == v.gather(d, idx)
    arg = torch.full_like(v, src.shape[d], dtype=torch.long).scatter_reduce(
        d, idx, torch.where(equal, pos, torch.full_like(pos, src.shape[d])), reduce="amin", include_self=True)
    return v, arg


def scatter_min(src, index, dim=-1, out=None, dim_size=None):
    v, arg = scatter_max(-src, index, dim, None if out is None else -out, dim_size)
    return -v, arg
