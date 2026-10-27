# rocm-shims

Pure-PyTorch stand-ins for a handful of CUDA-only Python extensions, so that some 3D research models can run on AMD GPUs (ROCm) **without editing the models' code**.

They are **not drivers and not optimized kernels**. Each shim reimplements just the part of an extension's API that a specific model calls, using operations that already run on ROCm: `torch.nn.functional.scaled_dot_product_attention`, gathers and matmuls, and `scatter_reduce`. Expect it to be slower than the original CUDA kernels. The upside is that the models run at all.

## What's inside

| Module | Replaces | Implemented |
|---|---|---|
| `flash_attn` | flash-attention 2 | `flash_attn_func`, `*_qkvpacked_func`, `*_kvpacked_func`, `*_varlen_*`, simple `flash_attn_with_kvcache` |
| `xformers.ops` | xformers | `memory_efficient_attention` (incl. `fmha.BlockDiagonalMask`, `LowerTriangularMask`), `SwiGLU` |
| `rotary_emb` | flash-attention's rotary extension | `apply_rotary` |
| `dropout_layer_norm` | flash-attention's fused norm | empty module; patch `FusedRMSNorm.forward` with `rocm_shims.rms_norm_forward` |
| `spconv.pytorch` | spconv 2.x | `SparseConvTensor`, `SubMConv3d` (submanifold, stride 1, spconv weight layout). Strided and inverse convs are **not** implemented. |
| `torch_scatter` | torch_scatter | `scatter`, `scatter_sum/add/mean/max/min` |
| `o_voxel.convert` | TRELLIS.2 o-voxel | `mesh_to_flexible_dual_grid`, **approximate** (voxels touched by the surface, found by dense sampling) |
| `rocm_shims` | — | `load_safe` / `safe_stubs` (`torch.load(weights_only=True)` with empty allow-listed classes for pickled objects such as deepspeed's), `force_weights_only`, `enable_transformers_flash`, `rms_norm_forward` |

`tests/test_shims.py` checks each shim against a reference computation: naive attention, a dense `F.conv3d` for the sparse convolution, and per-group reductions for scatter.

## Use cases it was written for

The package came out of trying open "artist mesh" generators for a hobby project that turns 3D models into paper and laser-cut parts. It covers what those models needed, not each library's full API.

| Model | Shims it needs | Status |
|---|---|---|
| [LATO.2](https://github.com/LoHhhha/LATO.2) (mesh + image → artist mesh) | `spconv`, `flash_attn` varlen, `torch_scatter`, `o_voxel`; weights loaded with `weights_only=True` | Runs end to end **with this package**, about 5 min per mesh. Geometry is plausible; topology is noisy, which upstream also warns about. Not compared against a CUDA run. |
| [DeepMesh](https://github.com/zhaorw02/DeepMesh) | `flash_attn`, `xformers.ops.SwiGLU`, `rotary_emb`, `dropout_layer_norm` + `rms_norm_forward`, `force_weights_only` | Ran with an earlier copy of the same shims; not yet re-run with the package. |
| [FastMesh](https://huggingface.co/papers/2508.19188) | `enable_transformers_flash` | Ran with an earlier copy; not yet re-run with the package. |
| [BPT](https://github.com/Tencent-Hunyuan/bpt) | `safe_stubs` for the pickled deepspeed objects | Runs **with this package**, about 3 min per mesh (watertight output in the latest test). |
| [MeshAnything V2](https://github.com/buaacyw/MeshAnythingV2) | Needed its own OPT attention class with the older `transformers` it pins. `enable_transformers_flash` may not be enough there. | Not covered by the package. |

Tested environment: Radeon 8060S (Strix Halo, gfx1151), ROCm 7.2, PyTorch 2.9.1, Python 3.12, Linux. Other GPUs and versions have not been tried.

## Usage

Install it **only in a dedicated per-model venv**. The module names (`flash_attn`, `spconv`, `torch_scatter`, ...) deliberately shadow the real packages, so in an environment that has them it would hide them. That is also why it is not published on PyPI.

```bash
# 1. venv reusing an existing ROCm torch (e.g. your ComfyUI venv) + rocm-shims (editable)
ROCM_TORCH_SITE=/path/to/rocm-venv/lib/python3.12/site-packages ./make_venv.sh ~/venvs/mymodel
# 2. the model's own requirements, WITHOUT torch / flash-attn / xformers / spconv / torch-scatter
~/venvs/mymodel/bin/pip install -r requirements-without-cuda-bits.txt
# 3. if pip still pulled CUDA torch / nvidia / triton wheels in, remove them
./make_venv.sh --clean ~/venvs/mymodel
# 4. check
~/venvs/mymodel/bin/python tests/test_shims.py
```

From a small launcher script:

```python
import rocm_shims
rocm_shims.force_weights_only()          # never unpickle arbitrary code from checkpoints
rocm_shims.enable_transformers_flash()   # only if the model asks transformers for flash_attention_2
# ... then runpy / import the model's own inference script unchanged
```

To add a missing function, put it in `src/`, add a reference check to `tests/test_shims.py`, and open a PR.

## Authorship

The code was written by an AI assistant, [Claude](https://www.anthropic.com/claude) (Anthropic, via Claude Code), at the direction of **SerraJoa**, who chose the models, ran them on their hardware and judged the results. The tests check the shims against reference computations, but real-world use is so far limited to the models above. Please review it before relying on it, and report what works on your GPU.

## License

MIT
