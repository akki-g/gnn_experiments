#!/bin/bash
#SBATCH --job-name=vmas_ss_mappo
#SBATCH --partition=normal
#SBATCH --account=cenyioha
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-2
# Output dirs must exist before submission.
# Run once: mkdir -p $HOME/gnn_experiments/slurm_logs
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail

module load python/python-3.11.4-gcc-12.2.0
module load cuda/cuda-12.1.0
source ~/expenv/bin/activate

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "${REPO_ROOT}/configs/vmas_simple_spread_mappo.yaml" ]]; then
    REPO_ROOT="${HOME}/gnn_experiments"
fi
cd "${REPO_ROOT}"

mkdir -p runs/vmas_simple_spread

# Confirm GPU assignment (SLURM sets CUDA_VISIBLE_DEVICES for us)
echo "=== Job info ==="
echo "Job      : ${SLURM_JOB_ID:-unset}  array=${SLURM_ARRAY_TASK_ID:-0}"
echo "Node     : $(hostname)"
echo "Submit   : ${SLURM_SUBMIT_DIR:-unset}"
echo "Repo     : $(pwd)"
echo "Git rev  : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Python   : $(command -v python)"
python --version
echo "CUDA_VIS : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Threads  : OMP=${OMP_NUM_THREADS} MKL=${MKL_NUM_THREADS} OPENBLAS=${OPENBLAS_NUM_THREADS}"
echo "Seed     : ${SLURM_ARRAY_TASK_ID:-0}"
echo "Start    : $(date)"
echo "================"

nvidia-smi || true

# Rendering is disabled on batch nodes; render later from a checkpoint locally
# or under xvfb-run.

python -m experiments.train_vmas_mappo \
    --config configs/vmas_simple_spread_mappo.yaml \
    --seed "${SLURM_ARRAY_TASK_ID:-0}" \
    --render off

echo "=== done $(date) ==="
