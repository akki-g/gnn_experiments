#!/bin/bash
#SBATCH --job-name=pursuit_probe
#SBATCH --partition=normal
#SBATCH --account=cenyioha
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

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

# Override these at submit time if you want to probe a different checkpoint:
#   sbatch --export=ALL,CKPT=runs/.../final.pt,ALPHA=0.25,SEED=3 scripts/slurm/run_probe_pursuit.sbatch
CONFIG="${CONFIG:-configs/pursuit_base.yaml}"
CKPT="${CKPT:-runs/pursuit_20260608_181436/alpha/broadcast/alpha0p0/seed0/ckpt/final.pt}"
PROBE_MODE="${PROBE_MODE:-post}"
ALPHA="${ALPHA:-0.0}"
SEED="${SEED:-0}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "[setup] Missing config: ${CONFIG}"
    exit 2
fi

if [[ ! -f "${CKPT}" ]]; then
    echo "[setup] Missing checkpoint: ${CKPT}"
    exit 2
fi

# --- venv setup ---
VENV_DIR="${VENV_DIR:-${HOME}/expenv}"
REQ_FILE="${REQ_FILE:-${REPO_ROOT}/requirements-hpc.txt}"

if [[ ! -d "${VENV_DIR}" ]]; then
    echo "[setup] Creating virtualenv at ${VENV_DIR}"
    python -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"

check_hpc_imports() {
    python - <<'PY'
import importlib.util
import sys

required = {
    "yaml": "PyYAML",
    "numpy": "numpy",
    "torch": "torch",
    "vmas": "vmas",
}
missing = [pkg for mod, pkg in required.items() if importlib.util.find_spec(mod) is None]
if missing:
    print("[setup] Missing Python packages:", ", ".join(missing))
    sys.exit(1)
print("[setup] Required Python imports are available.")
PY
}

if ! check_hpc_imports; then
    LOCK_DIR="${HOME}/.gnn_experiments_pip_install.lock"
    until mkdir "${LOCK_DIR}" 2>/dev/null; do
        echo "[setup] Waiting for dependency install lock ${LOCK_DIR}"
        sleep 10
    done
    cleanup_lock() { rmdir "${LOCK_DIR}" 2>/dev/null || true; }
    trap cleanup_lock EXIT

    if ! check_hpc_imports; then
        echo "[setup] Installing dependencies from ${REQ_FILE}"
        python -m pip install --upgrade pip
        python -m pip install -r "${REQ_FILE}"
    fi

    cleanup_lock
    trap - EXIT
fi

echo "[probe] repo=${REPO_ROOT}"
echo "[probe] config=${CONFIG}"
echo "[probe] ckpt=${CKPT}"
echo "[probe] probe_mode=${PROBE_MODE} alpha=${ALPHA} seed=${SEED}"

python -m experiments.run_probe \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --probe-mode "${PROBE_MODE}" \
    --alpha "${ALPHA}" \
    --seed "${SEED}"
