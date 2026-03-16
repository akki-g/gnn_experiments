#!/bin/bash

module load python/3.11
module load cuda/12.1

python -m venv ~/expenv
source ~/expenv/bin/activate

pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install vmas matplotlib

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python -c "import vmas; print('VMAS OK')"

echo ""
echo "Setup complete. Environment is at ~/marl_env"
echo "Run 'source ~/marl_env/bin/activate' to activate it."