#!/bin/bash
set -e

echo "=== Cut & Paste: SAM3 Background Replacement Setup ==="
echo "    Environment: Vast.AI Docker, Python 3.10, CUDA 13"
echo ""

# PyTorch CUDA 12.6 builds run fine on CUDA 13 (NVIDIA backward compatible)
echo "Installing PyTorch 2.7.0 (cu126 — works on CUDA 13)..."
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

echo "Installing ML + server dependencies..."
pip install \
    "transformers>=5.0" \
    huggingface_hub \
    scipy \
    pillow \
    fastapi \
    "uvicorn[standard]" \
    python-multipart

echo "Logging in to HuggingFace..."
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: Set your HuggingFace token first:  export HF_TOKEN=hf_..."
    exit 1
fi
python -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'])"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Start the server:"
echo "  python server.py                  # http://0.0.0.0:8000"
echo "  python server.py --port 8080      # custom port"
echo ""
echo "On Vast.AI — expose port 8000, then visit:  http://<vast-ip>:8000/"
echo "The first inference downloads SAM3 (~6.9GB)."
