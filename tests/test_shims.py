"""Checks every shim against a reference computation: python tests/test_shims.py (or pytest)."""
import torch
import torch.nn.functional as F

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
ok = lambda a, b, t=2e-4: torch.allclose(a.float(), b.float(), atol=t, rtol=t)


def ingenua(q, k, v, causal=False):   # (B, S, H, D)
    s = torch.einsum("bqhd,bkhd->bhqk", q, k) / q.shape[-1] ** 0.5
    if causal:
        Lq, Lk = q.shape[1], k.shape[1]
        s = s.masked_fill(torch.arange(Lk, device=q.device)[None] > torch.arange(Lq, device=q.device)[:, None] + Lk - Lq, -1e9)
    return torch.einsum("bhqk,bkhd->bqhd", s.softmax(-1), v)


import flash_attn as fa
q, k, v = (torch.randn(2, 37, 4, 16, device=dev) for _ in range(3))
assert ok(fa.flash_attn_func(q, k, v), ingenua(q, k, v)), "flash_attn_func"
assert ok(fa.flash_attn_func(q, k, v, causal=True), ingenua(q, k, v, True)), "causal"
cu = torch.tensor([0, 10, 37], device=dev, dtype=torch.int32)
qv, kv_, vv = q[0], k[0], v[0]
ref = torch.cat([ingenua(qv[None, :10], kv_[None, :10], vv[None, :10])[0], ingenua(qv[None, 10:], kv_[None, 10:], vv[None, 10:])[0]])
assert ok(fa.flash_attn_varlen_qkvpacked_func(torch.stack([qv, kv_, vv], 1), cu, 27), ref), "varlen"
print("flash_attn ok")

import xformers.ops as xo
m = xo.fmha.BlockDiagonalMask.from_seqlens([10, 27])
assert ok(xo.memory_efficient_attention(qv[None], kv_[None], vv[None], m)[0], ref), "xformers bloc"
print("xformers ok")

import torch_scatter as ts
src = torch.randn(9, 3, device=dev); idx = torch.tensor([0, 2, 2, 1, 0, 2, 3, 3, 1], device=dev)
ref = torch.stack([src[idx == i].mean(0) for i in range(4)])
assert ok(ts.scatter_mean(src, idx, dim=0), ref), "scatter_mean"
mx, am = ts.scatter_max(src, idx, dim=0)
assert ok(mx, torch.stack([src[idx == i].max(0).values for i in range(4)])), "scatter_max"
print("torch_scatter ok")

import spconv.pytorch as sp
R = 12
occ = torch.rand(2, R, R, R, device=dev) < 0.3
ind = torch.nonzero(occ).int()
fin = torch.randn(len(ind), 5, device=dev)
conv = sp.SubMConv3d(5, 7, 3).to(dev)
with torch.no_grad():
    conv.weight.normal_(); conv.bias.normal_()
    out = conv(sp.SparseConvTensor(fin, ind, [R, R, R], 2)).features
    dens = torch.zeros(2, 5, R, R, R, device=dev)
    dens[ind[:, 0].long(), :, ind[:, 1].long(), ind[:, 2].long(), ind[:, 3].long()] = fin
    W = conv.weight.permute(0, 4, 1, 2, 3)                      # (O, I, k, k, k) as F.conv3d expects
    rd = F.conv3d(dens, W, conv.bias, padding=1)
    ref = rd[ind[:, 0].long(), :, ind[:, 1].long(), ind[:, 2].long(), ind[:, 3].long()]
assert ok(out, ref, 1e-3), "SubMConv3d"
print("spconv ok")

import o_voxel.convert as ov
import rotary_emb, dropout_layer_norm, rocm_shims  # noqa: F401
V = torch.tensor([[-0.4, -0.4, 0.0], [0.4, -0.4, 0.0], [0.0, 0.4, 0.0]])
c = ov.mesh_to_flexible_dual_grid(V, torch.tensor([[0, 1, 2]]), 16, torch.tensor([[-0.5] * 3, [0.5] * 3]))[0]
assert len(c) > 20 and (c[:, 2] == 8).all(), "o_voxel"
print("o_voxel ok\nALL OK")
