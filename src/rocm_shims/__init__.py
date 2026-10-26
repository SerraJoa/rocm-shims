"""Helpers to run CUDA-only research models on ROCm without editing their code.

- safe_stubs(names) / load_safe(path, classes=[...]): torch.load with weights_only=True; classes the checkpoint
  pickles (e.g. deepspeed objects) are replaced by empty allow-listed classes. Never weights_only=False.
- force_weights_only(): make every torch.load in the process use weights_only=True.
- flash_sdpa: drop-in for transformers' _flash_attention_forward using SDPA.
- enable_transformers_flash(): route attn_implementation="flash_attention_2" to flash_sdpa without flash_attn.
- rms_norm_forward: pure-torch RMSNorm forward, to replace FusedRMSNorm (dropout_layer_norm).
"""
import torch
import torch.nn.functional as F

__version__ = "0.1.0"
__all__ = ["safe_stubs", "load_safe", "force_weights_only", "flash_sdpa", "enable_transformers_flash",
           "rms_norm_forward"]


def safe_stubs(names):
    """Register empty classes with the given dotted names ('module.Class') as safe for weights_only loading."""
    cls = []
    for name in names:
        mod, c = name.rsplit(".", 1)
        cls.append(type(c, (), {"__module__": mod, "__qualname__": c,
                                "__init__": lambda self, *a, **k: None,
                                "__setstate__": lambda self, s: self.__dict__.update(s if isinstance(s, dict) else {})}))
    torch.serialization.add_safe_globals(cls)
    return cls


def load_safe(path, classes=(), **kw):
    if classes:
        safe_stubs(classes)
    kw.setdefault("map_location", "cpu")
    return torch.load(path, weights_only=True, **kw)


def force_weights_only():
    load = torch.load
    torch.load = lambda *a, **k: load(*a, **{**k, "weights_only": True})


def flash_sdpa(q, k, v, attention_mask=None, query_length=None, is_causal=False, dropout=0.0, position_ids=None,
               softmax_scale=None, sliding_window=None, use_top_left_mask=False, softcap=None, deterministic=None,
               **kw):
    """Like transformers' _flash_attention_forward: inputs (B, S, H, D), optional 2D padding mask."""
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    Lq, Lk = q.shape[2], k.shape[2]
    m = None
    if is_causal and Lq > 1:
        i = torch.arange(Lq, device=q.device)[:, None] + (0 if use_top_left_mask else Lk - Lq)
        m = (torch.arange(Lk, device=q.device)[None] <= i)[None, None]
    if attention_mask is not None and attention_mask.dim() == 2:
        pad = attention_mask[:, None, None, -Lk:].bool()
        m = pad if m is None else (m & pad)
    if k.shape[1] != q.shape[1]:
        r = q.shape[1] // k.shape[1]
        k, v = k.repeat_interleave(r, 1), v.repeat_interleave(r, 1)
    o = F.scaled_dot_product_attention(q, k, v.to(q.dtype), attn_mask=m, scale=softmax_scale)
    return o.transpose(1, 2).contiguous()


def enable_transformers_flash():
    import transformers.modeling_flash_attention_utils as fu
    import transformers.modeling_utils as mu
    fu._flash_attention_forward = flash_sdpa
    try:
        from transformers.models.opt import modeling_opt as mo
        if hasattr(mo, "_flash_attention_forward"):
            mo._flash_attention_forward = flash_sdpa
    except ImportError:
        pass
    mu.PreTrainedModel._check_and_enable_flash_attn_2 = classmethod(
        lambda cls, config, *a, **k: (setattr(config, "_attn_implementation", "flash_attention_2"), config)[1])
    mu.PreTrainedModel.to_bettertransformer = lambda self: self


def rms_norm_forward(self, x):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight
