#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=logs/test.out
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0:05:00
#SBATCH --account=cenyioha

module load python/python-3.11.4-gcc-12.2.0
module load cuda/cuda-12.1.0
export OPENBLAS_NUM_THREADS=1
source ~/expenv/bin/activate

python3 -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
python3 -c "import vmas; print('VMAS OK')"


