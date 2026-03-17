#!/bin/bash

module load python/python-3.11.4-gcc-12.2.0
module load cuda/cuda-12.1.0

python3 -m venv ~/expenv
source ~/expenv/bin/activate

pip3 install --upgrade pip
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip3 install vmas matplotlib

python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python3 -c "import vmas; print('VMAS OK')"

echo ""
echo "Setup complete. Environment is at ~/expenv"
echo "Run 'source ~/expenv/bin/activate' to activate it."
