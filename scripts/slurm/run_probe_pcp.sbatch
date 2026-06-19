#!/bin/bash
# ============================================================================
# PCP post-comm probe sweep (single job, loops over all cells/seeds).
#
# Runs experiments/run_probe.py --probe-mode post on every
#   <RUN_DIR>/<cell>/seed<s>/ckpt/final.pt
# and tabulates the decisive contrast R2_post(live) - R2_post(_zm):
#   R2_c1_post  capturer post-comm hidden -> prey   (channel signal)
#   R2_c1_raw   capturer raw obs          -> prey   (~chance; isolation check)
#   R2_c0_local detector raw obs          -> prey   (ceiling)
#
# Channel CARRIES the signal iff live R2_post is high, _zm/identity collapse to
# ~R2_raw, and delta(live - _zm) is large. (High R2_post alone is partly
# architectural via neighbor pooling, so the delta is load-bearing.)
#
# Output: human table -> slurm_logs/%x_%j.out ; CSV -> <RUN_DIR>/probe_results.csv
#
# Submit (defaults to the 2026-06-19 smoke run):
#   sbatch scripts/slurm/run_probe_pcp.sbatch
# Override the run dir / config / probe seed:
#   sbatch --export=ALL,RUN_DIR=runs/pcp_smoke_XXXX,CONFIG=configs/pcp_smoke.yaml,SEED=0 \
#       scripts/slurm/run_probe_pcp.sbatch
# ============================================================================
#SBATCH --job-name=pcp_probe
#SBATCH --partition=normal
#SBATCH --account=cenyioha
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

module load python/python-3.11.4-gcc-12.2.0
module load cuda/cuda-12.1.0

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""   # CPU-only job; probe forces device=cpu regardless
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "${REPO_ROOT}/configs/pcp_smoke.yaml" ]]; then
    REPO_ROOT="${HOME}/gnn_experiments"
fi
cd "${REPO_ROOT}"
mkdir -p slurm_logs

RUN_DIR="${RUN_DIR:-runs/pcp_smoke_20260619_113708}"
CONFIG="${CONFIG:-configs/pcp_smoke.yaml}"
SEED="${SEED:-0}"   # fixed probe rollout seed for cross-cell comparability

if [[ ! -f "${CONFIG}" ]]; then
    echo "[setup] Missing config: ${CONFIG}"; exit 2
fi
if [[ ! -d "${RUN_DIR}" ]]; then
    echo "[setup] Missing run dir: ${RUN_DIR}"; exit 2
fi

# --- venv setup (same pattern as run_probe_pursuit.sbatch) ---
VENV_DIR="${VENV_DIR:-${HOME}/expenv}"
REQ_FILE="${REQ_FILE:-${REPO_ROOT}/requirements-hpc.txt}"

if [[ ! -d "${VENV_DIR}" ]]; then
    echo "[setup] Creating virtualenv at ${VENV_DIR}"
    python -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"

check_hpc_imports() {
    python - <<'PY'
import importlib.util, sys
required = {"yaml": "PyYAML", "numpy": "numpy", "torch": "torch"}
missing = [pkg for mod, pkg in required.items() if importlib.util.find_spec(mod) is None]
if missing:
    print("[setup] Missing Python packages:", ", ".join(missing)); sys.exit(1)
print("[setup] Required Python imports are available.")
PY
}

if ! check_hpc_imports; then
    LOCK_DIR="${HOME}/.gnn_experiments_pip_install.lock"
    until mkdir "${LOCK_DIR}" 2>/dev/null; do
        echo "[setup] Waiting for dependency install lock ${LOCK_DIR}"; sleep 10
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

echo "[pcp-probe] repo=${REPO_ROOT}"
echo "[pcp-probe] run_dir=${RUN_DIR}"
echo "[pcp-probe] config=${CONFIG} probe_seed=${SEED}"

python scripts/pcp_probe_sweep.py "${RUN_DIR}" \
    --config "${CONFIG}" \
    --probe-seed "${SEED}" \
    --out-csv "${RUN_DIR}/probe_results.csv"

echo "[pcp-probe] done. CSV at ${RUN_DIR}/probe_results.csv"
