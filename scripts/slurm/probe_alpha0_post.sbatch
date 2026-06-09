#!/bin/bash
#SBATCH --job-name=probe_alpha0_post
#SBATCH --partition=normal
#SBATCH --account=cenyioha
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-11
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#
# Post-comm R2 probe on alpha=0 checkpoints from the F1 sweep.
# 4 modules (identity broadcast attention graph) x 3 seeds = 12 tasks.
#
# Required: SWEEP_DIR must be set to the run directory that contains the
# alpha/<module>/alpha0p0/seed*/ckpt/final.pt checkpoints.
# Default: runs/pursuit_20260608_192647
#
# Submit:
#   sbatch --export=ALL,SWEEP_DIR=runs/pursuit_20260608_192647 \
#     scripts/slurm/probe_alpha0_post.sbatch
#
# Results: each task prints R2_c1_post / R2_c1_raw / R2_c0_local to stdout.
# To collect: grep -r "R2_c1_post\|branch" slurm_logs/probe_alpha0_post_<JOBID>/

set -euo pipefail

module load python/python-3.11.4-gcc-12.2.0
module load cuda/cuda-12.1.0

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "${REPO_ROOT}/configs/pursuit_base.yaml" ]]; then
    REPO_ROOT="${HOME}/gnn_experiments"
fi
cd "${REPO_ROOT}"

mkdir -p slurm_logs

# Task table: 4 modules x 3 seeds (0,1,2)
MODULES=(identity broadcast attention graph)
SEEDS=(0 1 2)
NUM_MODULES=${#MODULES[@]}   # 4
NUM_SEEDS=${#SEEDS[@]}       # 3
# Total = 12 tasks (array 0-11)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODULE_INDEX=$(( TASK_ID / NUM_SEEDS ))
SEED_INDEX=$(( TASK_ID % NUM_SEEDS ))
COMM_MODULE="${MODULES[$MODULE_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"

SWEEP_DIR="${SWEEP_DIR:-runs/pursuit_20260608_192647}"
CKPT="${SWEEP_DIR}/alpha/${COMM_MODULE}/alpha0p0/seed${SEED}/ckpt/final.pt"
CONFIG="configs/pursuit_base.yaml"

if [[ ! -f "${CKPT}" ]]; then
    echo "[error] Checkpoint not found: ${CKPT}"
    echo "[error] Make sure SWEEP_DIR is set correctly and checkpoints are synced."
    exit 2
fi

# --- venv setup ---
VENV_DIR="${VENV_DIR:-${HOME}/expenv}"
if [[ ! -d "${VENV_DIR}" ]]; then
    python -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"

check_imports() {
    python - <<'PY'
import importlib.util, sys
for mod in ("yaml", "numpy", "torch"):
    if importlib.util.find_spec(mod) is None:
        print(f"[setup] Missing: {mod}"); sys.exit(1)
print("[setup] imports ok")
PY
}

if ! check_imports; then
    LOCK_DIR="${HOME}/.gnn_experiments_pip_install.lock"
    until mkdir "${LOCK_DIR}" 2>/dev/null; do sleep 10; done
    cleanup_lock() { rmdir "${LOCK_DIR}" 2>/dev/null || true; }
    trap cleanup_lock EXIT
    if ! check_imports; then
        python -m pip install --upgrade pip
        python -m pip install -r "${REPO_ROOT}/requirements-hpc.txt"
    fi
    cleanup_lock; trap - EXIT
fi

# --- organize logs on exit ---
organize_logs() {
    local log_root="${SLURM_SUBMIT_DIR:-$PWD}/slurm_logs"
    local job="${SLURM_JOB_NAME:-probe_alpha0_post}"
    local ajid="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
    local atid="${SLURM_ARRAY_TASK_ID:-0}"
    local log_dir="${log_root}/${job}_${ajid}/${COMM_MODULE}"
    mkdir -p "${log_dir}"
    mv "${log_root}/${job}_${ajid}_${atid}.out" \
       "${log_dir}/seed${SEED}_task${atid}.out" 2>/dev/null || true
    mv "${log_root}/${job}_${ajid}_${atid}.err" \
       "${log_dir}/seed${SEED}_task${atid}.err" 2>/dev/null || true
}
trap organize_logs EXIT

echo "=== probe job info ==="
echo "Task     : ${TASK_ID}/11"
echo "Module   : ${COMM_MODULE}"
echo "Seed     : ${SEED}"
echo "Ckpt     : ${CKPT}"
echo "Start    : $(date)"
echo "======================"

python -m experiments.run_probe \
    --config "${CONFIG}" \
    --ckpt   "${CKPT}" \
    --probe-mode post \
    --alpha  0.0 \
    --seed   "${SEED}"

echo "=== done $(date) ==="
