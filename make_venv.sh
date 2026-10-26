#!/usr/bin/env bash
# Per-model venv that reuses an existing ROCm PyTorch install (via a .pth file) plus rocm-shims.
#   ROCM_TORCH_SITE=/path/to/site-packages ./make_venv.sh /path/venv   create the venv
#   ./make_venv.sh --clean /path/venv                                   remove CUDA torch/nvidia/triton wheels pip pulled in
# ROCM_TORCH_SITE: site-packages of a working ROCm torch (e.g. your ComfyUI venv). Python version must match.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ "$1" = "--clean" ]; then
    V="$2"
    PKGS=$("$V/bin/pip" list --format=freeze 2>/dev/null | cut -d= -f1 | grep -iE '^(torch|torchvision|torchaudio|triton|pytorch-triton.*|nvidia-.*|cuda-.*)$' || true)
    [ -n "$PKGS" ] && "$V/bin/pip" uninstall -y $PKGS
    "$V/bin/python" -c "import torch; print('torch', torch.__version__, 'GPU', torch.cuda.is_available())"
    exit 0
fi
V="$1"
: "${ROCM_TORCH_SITE:?set ROCM_TORCH_SITE to the site-packages of a ROCm torch install}"
PY="${PYTHON:-python$(basename "$(dirname "$ROCM_TORCH_SITE")" | sed 's/^python//')}"
"$PY" -m venv "$V"
SP=$("$V/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
echo "$ROCM_TORCH_SITE" > "$SP/zz_rocm_torch.pth"     # zz: after the venv's own packages
"$V/bin/pip" -q install -e "$HERE"
"$V/bin/python" -c "import torch, flash_attn, spconv.pytorch, torch_scatter, rocm_shims; print('torch', torch.__version__, 'GPU', torch.cuda.is_available(), '| rocm-shims ok')"
