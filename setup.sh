#!/bin/bash
set -e

echo "=== Cut & Paste: SAM3 Background Replacement Setup ==="
echo ""

# Requirements:
#   - Python 3.12+
#   - CUDA 12.6+ (driver 560+)
#   - ~16GB+ GPU VRAM recommended (model is ~6.9GB in bf16)

# Create conda env if it doesn't exist
if ! conda env list | grep -q "^sam3 "; then
    echo "Creating conda environment 'sam3' with Python 3.12..."
    conda create -n sam3 python=3.12 -y
else
    echo "Conda environment 'sam3' already exists."
fi

echo "Activating sam3 environment..."
eval "$(conda shell.bash hook)"
conda activate sam3

echo "Installing PyTorch 2.7.0 (CUDA 12.6)..."
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

echo "Installing ML dependencies..."
pip install "transformers>=5.0" huggingface_hub scipy pillow

echo "Installing harmonization (libcom / PCTNet)..."
pip install libcom

echo "Installing server dependencies..."
pip install fastapi uvicorn[standard] python-multipart

echo "Logging in to HuggingFace..."
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: Set your HuggingFace token first:  export HF_TOKEN=hf_..."
    exit 1
fi
python -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'])"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Start the web server:"
echo "  conda activate sam3"
echo "  python server.py                  # http://0.0.0.0:8000"
echo "  python server.py --port 7860      # custom port"
echo ""
echo "On Vast.AI — expose port 8000 in the instance settings,"
echo "then visit:  http://<vast-ip>:8000/"
echo ""
echo "The first inference will download the SAM3 model (~6.9GB)."
