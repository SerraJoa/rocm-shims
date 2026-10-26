"""Pure-PyTorch stand-in for flash_attn on ROCm, backed by torch SDPA.
Shapes follow flash-attn 2: dense (B, S, H, D); packed varlen (T, H, D) with cu_seqlens.
Causal masks are bottom-right aligned (as in flash-attn >= 2.1); GQA/MQA by repeating k/v heads."""
import torch
import torch.nn.functional as F

__version__ = "2.7.4.rocm_shims"


def _sdpa(q, k, v, softmax_scale=None, causal=False):
    """q, k, v: (B, S, H, D) -> (B, Sq, H, D)."""
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    if k.shape[1] != q.shape[1]:
        r = q.shape[1] // k.shape[1]
        k, v = k.repeat_interleave(r, 1), v.repeat_interleave(r, 1)
    m = None
    Lq, Lk = q.shape[2], k.shape[2]
    if causal and Lq > 1:
        i = torch.arange(Lq, device=q.device)[:, None] + Lk - Lq
        m = torch.arange(Lk, device=q.device)[None] <= i
    o = F.scaled_dot_product_attention(q, k, v.to(q.dtype), attn_mask=m, scale=softmax_scale)
    return o.transpose(1, 2)


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **kw):
    return _sdpa(q, k, v, softmax_scale, causal)


def flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False, **kw):
    q, k, v = qkv.unbind(2)
    return _sdpa(q, k, v, softmax_scale, causal)


def flash_attn_kvpacked_func(q, kv, dropout_p=0.0, softmax_scale=None, causal=False, **kw):
    k, v = kv.unbind(2)
    return _sdpa(q, k, v, softmax_scale, causal)


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None,
                           dropout_p=0.0, softmax_scale=None, causal=False, **kw):
    out = torch.empty(q.shape[0], q.shape[1], v.shape[-1], dtype=q.dtype, device=q.device)
    cq, ck = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
    for i in range(len(cq) - 1):
        a, b, c, d = cq[i], cq[i + 1], ck[i], ck[i + 1]
        out[a:b] = _sdpa(q[None, a:b], k[None, c:d], v[None, c:d], softmax_scale, causal)[0]
    return out


def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen=None, dropout_p=0.0, softmax_scale=None,
                                     causal=False, **kw):
    q, k, v = qkv.unbind(1)
    return flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, softmax_scale=softmax_scale, causal=causal)


def flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None,
                                    dropout_p=0.0, softmax_scale=None, causal=False, **kw):
    k, v = kv.unbind(1)
    return flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale=softmax_scale, causal=causal)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None, softmax_scale=None,
                            causal=False, **kw):
    """Simple version: writes new k/v into the cache (B, Smax, H, D) at cache_seqlens, then attends."""
    if k is not None:
        n = k.shape[1]
        for b in range(q.shape[0]):
            s = int(cache_seqlens[b]) if cache_seqlens is not None else 0
            k_cache[b, s:s + n] = k[b]
            v_cache[b, s:s + n] = v[b]
    out = []
    for b in range(q.shape[0]):
        s = int(cache_seqlens[b]) + (k.shape[1] if k is not None else 0) if cache_seqlens is not None \
            else k_cache.shape[1]
        out.append(_sdpa(q[b:b + 1], k_cache[b:b + 1, :s], v_cache[b:b + 1, :s], softmax_scale, causal))
    return torch.cat(out, 0)
