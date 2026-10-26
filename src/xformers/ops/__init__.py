"""Pure-PyTorch stand-in for xformers.ops on ROCm: memory_efficient_attention via SDPA, BlockDiagonalMask,
LowerTriangularMask and SwiGLU (same weight names w1, w2, w3). Shapes as in xformers: (B, S, H, D)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _BlockDiagonalMask:
    def __init__(self, q_seqlen, kv_seqlen=None):
        self.q_seqlen = list(q_seqlen)
        self.kv_seqlen = list(kv_seqlen) if kv_seqlen is not None else list(q_seqlen)

    @classmethod
    def from_seqlens(cls, q_seqlen, kv_seqlen=None):
        return cls(q_seqlen, kv_seqlen)


class LowerTriangularMask:
    pass


class fmha:   # noqa: N801 (xformers.ops.fmha.BlockDiagonalMask)
    BlockDiagonalMask = _BlockDiagonalMask


def memory_efficient_attention(query, key, value, attn_bias=None, p=0.0, scale=None, **kw):
    if isinstance(attn_bias, _BlockDiagonalMask):
        out = torch.empty_like(query)
        a = c = 0
        for lq, lk in zip(attn_bias.q_seqlen, attn_bias.kv_seqlen):
            out[:, a:a + lq] = memory_efficient_attention(query[:, a:a + lq], key[:, c:c + lk],
                                                          value[:, c:c + lk], scale=scale)
            a += lq; c += lk
        return out
    q, k, v = (t.transpose(1, 2) for t in (query, key, value))
    causal = isinstance(attn_bias, LowerTriangularMask)
    m = attn_bias if isinstance(attn_bias, torch.Tensor) else None
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=m, scale=scale, is_causal=causal and m is None)
    return o.transpose(1, 2)


class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features, out_features=None, bias=True, _pack_weights=True):
        super().__init__()
        out_features = out_features or in_features
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))
