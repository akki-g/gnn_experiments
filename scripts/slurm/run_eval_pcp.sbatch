#!/bin/bash
# ============================================================================
# PCP large-sample deterministic eval sweep (single CPU job).
#
# The training-time eval uses only render.episodes (=2) single-env episodes, so
# eval_capture is a tiny, high-variance estimate of a rare event. This re-evaluates
# every cell/seed checkpoint with many deterministic episodes (default 100) by
# reusing the trainer's own _evaluate_deterministic_policy, and reports per-cell
# eval_capture + paired-bootstrap deltas (live - _zm, live - identity).
#
# Output: human table -> slurm_logs/%x_%j.out ; CSV -> <RUN_DIR>/eval_results.csv
#
# Submit (defaults to the v2 sweep):
#   sbatch scripts/slurm/run_eval_pcp.sbatch
# Override:
#   sbatch --export=ALL,RUN_DIR=runs/pcp_smoke_v2_XXXX,EPISODES=200 scripts/slurm/run_eval_pcp.sbatch
# ============================================================================
#SBATCH --job-name=pcp_eval
#SBATCH --partition=normal
#SBATCH --account=cenyioha
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

module load python/python-3.11.4-gcc-12.2.0
module load cuda/cuda-12.1.0

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""   # CPU-only job; eval forces device=cpu regardless
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "${REPO_ROOT}/scripts/pcp_eval_sweep.py" ]]; then
    REPO_ROOT="${HOME}/gnn_experiments"
fi
cd "${REPO_ROOT}"
mkdir -p slurm_logs

RUN_DIR="${RUN_DIR:-runs/pcp_smoke_v2_20260623_190304}"
EPISODES="${EPISODES:-100}"

if [[ ! -d "${RUN_DIR}" ]]; then
    echo "[setup] Missing run dir: ${RUN_DIR}"; exit 2
fi

# --- venv setup (same pattern as run_probe_pcp.sbatch) ---
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

echo "[pcp-eval] repo=${REPO_ROOT}"
echo "[pcp-eval] run_dir=${RUN_DIR} episodes=${EPISODES}"

python scripts/pcp_eval_sweep.py "${RUN_DIR}" \
    --episodes "${EPISODES}" \
    --device cpu \
    --out-csv "${RUN_DIR}/eval_results.csv"

echo "[pcp-eval] done. CSV at ${RUN_DIR}/eval_results.csv"
