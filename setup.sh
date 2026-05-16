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

echo "[3/5] Installing rembg (portrait matting, auto-downloads model)..."
pip install "rembg[gpu]"

echo "[4/5] Pre-downloading rembg u2net_human_seg model (~170MB from GitHub)..."
python -c "from rembg import new_session; new_session('u2net_human_seg'); print('rembg model ready.')"

echo "[5/5] MiDaS downloads automatically on first inference via torch.hub."

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
