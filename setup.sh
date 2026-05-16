#!/bin/bash
set -e

echo "=== Cut & Paste: MODNet + MiDaS Pipeline Setup ==="
echo "    Environment: Vast.AI Docker, Python 3.10, CUDA 13"
echo ""

# PyTorch CUDA 12.6 builds run fine on CUDA 13 (NVIDIA backward compatible)
echo "[1/5] Installing PyTorch 2.7.0 (cu126)..."
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

echo "[2/5] Installing ML + server dependencies..."
pip install \
    huggingface_hub \
    scipy \
    pillow \
    timm \
    fastapi \
    "uvicorn[standard]" \
    python-multipart

echo "[3/5] Installing MODNet..."
MODNET_DIR="${MODNET_REPO:-/workspace/MODNet}"
if [ ! -d "$MODNET_DIR" ]; then
    git clone https://github.com/ZHKKKe/MODNet "$MODNET_DIR"
else
    echo "  MODNet repo already exists at $MODNET_DIR"
fi

CKPT="$MODNET_DIR/modnet_photographic_portrait_matting.ckpt"
if [ ! -f "$CKPT" ]; then
    echo "  Downloading MODNet weights..."
    pip install -q gdown
    python - <<'PYEOF'
import gdown, os
out = os.path.join(os.environ.get("MODNET_REPO", "/workspace/MODNet"),
                   "modnet_photographic_portrait_matting.ckpt")
gdown.download(id="1Nf1ZxeJZJL8Qx9KadcYYyEmmlKwHHqNk", output=out, quiet=False)
PYEOF
else
    echo "  MODNet weights already present."
fi

echo "[4/5] MiDaS downloads automatically on first inference via torch.hub."

echo "[5/5] HuggingFace login (needed for SAM3 fallback, optional here)..."
if [ -n "$HF_TOKEN" ]; then
    python -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'])"
else
    echo "  Skipping (HF_TOKEN not set — not needed for MODNet+MiDaS)."
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Start the server:"
echo "  python server.py                  # http://0.0.0.0:8000"
echo ""
echo "On Vast.AI: expose port 8000, then visit http://<vast-ip>:8000/"
echo "First run downloads MiDaS small (~80MB) via torch.hub."
