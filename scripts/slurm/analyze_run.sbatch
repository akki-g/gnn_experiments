#!/bin/bash
# ============================================================================
# Analyze a completed sweep run: plot training metrics + run post-comm probes.
#
# Outputs everything under <RUN_DIR>/analysis/:
#   metrics.png           -- all training metrics (rolling=5)
#   summary.csv           -- ranked per-run summary
#   probe_results.csv     -- per-(cell,seed) R2_post / R2_raw / R2_ceil / branch
#
# Override defaults via --export=ALL,...:
#   RUN_DIR   path to sweep run dir (required; defaults shown below)
#   CONFIG    base config yaml (auto-detected from run dir name if unset)
#   PROBE_SEED fixed probe rollout seed for cross-cell comparability (default 0)
#
# Examples:
#   sbatch scripts/slurm/analyze_run.sbatch \
#       --export=ALL,RUN_DIR=runs/qmix_pcp_smoke_20260627_142223
#
#   sbatch scripts/slurm/analyze_run.sbatch \
#       --export=ALL,RUN_DIR=runs/pcp_smoke_20260619_113708,CONFIG=configs/pcp_smoke.yaml
# ============================================================================
#SBATCH --job-name=analyze_run
#SBATCH --partition=normal
#SBATCH --account=cenyioha
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

module load python/python-3.11.4-gcc-12.2.0

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""   # CPU-only; probes run on CPU regardless
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# ── repo root ─────────────────────────────────────────────────────────────────
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "${REPO_ROOT}/experiments/plot_run_metrics.py" ]]; then
    REPO_ROOT="${HOME}/gnn_experiments"
fi
cd "${REPO_ROOT}"
mkdir -p slurm_logs

# ── required args ─────────────────────────────────────────────────────────────
# Accept RUN_DIR as positional arg $1 or env var (--export=ALL,RUN_DIR=...)
RUN_DIR="${1:-${RUN_DIR:-}}"
if [[ -z "${RUN_DIR}" ]]; then
    echo "[analyze] ERROR: RUN_DIR not set."
    echo "          Usage: sbatch scripts/slurm/analyze_run.sbatch runs/<your_run_dir>"
    exit 2
fi
if [[ ! -d "${RUN_DIR}" ]]; then
    echo "[analyze] ERROR: RUN_DIR not found: ${RUN_DIR}"; exit 2
fi

PROBE_SEED="${PROBE_SEED:-0}"

# ── auto-detect config from run dir name if CONFIG not provided ───────────────
if [[ -z "${CONFIG:-}" ]]; then
    DIRNAME="$(basename "${RUN_DIR}")"
    if [[ "${DIRNAME}" == *qmix_pcp* ]]; then
        CONFIG="configs/qmix_pcp.yaml"
    elif [[ "${DIRNAME}" == *pcp_smoke_v2* ]] || [[ "${DIRNAME}" == *pcp_v2* ]]; then
        CONFIG="configs/pcp_smoke_v2.yaml"
    elif [[ "${DIRNAME}" == *pcp* ]]; then
        CONFIG="configs/pcp_smoke.yaml"
    elif [[ "${DIRNAME}" == *pursuit* ]]; then
        CONFIG="configs/pursuit_base.yaml"
    elif [[ "${DIRNAME}" == *pp_smoke* ]] || [[ "${DIRNAME}" == *pp_* ]]; then
        CONFIG="configs/pp_smoke.yaml"
    else
        echo "[analyze] ERROR: Cannot auto-detect config from run dir name '${DIRNAME}'."
        echo "          Pass CONFIG=configs/your_config.yaml via --export."
        exit 2
    fi
    echo "[analyze] Auto-detected config: ${CONFIG}"
fi

if [[ ! -f "${CONFIG}" ]]; then
    echo "[analyze] ERROR: Config not found: ${CONFIG}"; exit 2
fi

OUT_DIR="${RUN_DIR}/analysis"
mkdir -p "${OUT_DIR}"

echo "[analyze] ============================================================="
echo "[analyze] repo     = ${REPO_ROOT}"
echo "[analyze] run_dir  = ${RUN_DIR}"
echo "[analyze] config   = ${CONFIG}"
echo "[analyze] out_dir  = ${OUT_DIR}"
echo "[analyze] probe_seed = ${PROBE_SEED}"
echo "[analyze] start    = $(date)"
echo "[analyze] ============================================================="

# ── venv setup ────────────────────────────────────────────────────────────────
VENV_DIR="${VENV_DIR:-${HOME}/expenv}"
REQ_FILE="${REQ_FILE:-${REPO_ROOT}/requirements-hpc.txt}"

if [[ ! -d "${VENV_DIR}" ]]; then
    echo "[analyze] Creating virtualenv at ${VENV_DIR}"
    python -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"

check_hpc_imports() {
    python - <<'PY'
import importlib.util, sys
required = {"yaml": "PyYAML", "numpy": "numpy", "torch": "torch",
            "matplotlib": "matplotlib", "pandas": "pandas"}
missing = [pkg for mod, pkg in required.items() if importlib.util.find_spec(mod) is None]
if missing:
    print("[analyze] Missing Python packages:", ", ".join(missing)); sys.exit(1)
print("[analyze] Required Python imports are available.")
PY
}

if ! check_hpc_imports; then
    LOCK_DIR="${HOME}/.gnn_experiments_pip_install.lock"
    until mkdir "${LOCK_DIR}" 2>/dev/null; do
        echo "[analyze] Waiting for install lock ${LOCK_DIR}"; sleep 10
    done
    cleanup_lock() { rmdir "${LOCK_DIR}" 2>/dev/null || true; }
    trap cleanup_lock EXIT
    if ! check_hpc_imports; then
        echo "[analyze] Installing dependencies from ${REQ_FILE}"
        python -m pip install --upgrade pip
        python -m pip install -r "${REQ_FILE}"
    fi
    cleanup_lock
    trap - EXIT
fi

check_hpc_imports

# ── 1. Training metrics plot ──────────────────────────────────────────────────
echo ""
echo "[analyze] ===== Step 1/3: Plotting training metrics ====="
python -m experiments.plot_run_metrics "${RUN_DIR}" \
    --recursive --rolling 5 \
    --output "${OUT_DIR}/metrics.png"
echo "[analyze] Saved ${OUT_DIR}/metrics.png"

# ── 2. Per-run summary table ──────────────────────────────────────────────────
echo ""
echo "[analyze] ===== Step 2/3: Generating summary CSV ====="
python -m experiments.plot_run_metrics "${RUN_DIR}" \
    --recursive --summary \
    --summary-output "${OUT_DIR}/summary.csv"
echo "[analyze] Saved ${OUT_DIR}/summary.csv"

# ── 3. Post-comm probe sweep ──────────────────────────────────────────────────
echo ""
echo "[analyze] ===== Step 3/3: Running post-comm probe sweep ====="
python scripts/pcp_probe_sweep.py "${RUN_DIR}" \
    --config "${CONFIG}" \
    --probe-seed "${PROBE_SEED}" \
    --out-csv "${OUT_DIR}/probe_results.csv"
echo "[analyze] Saved ${OUT_DIR}/probe_results.csv"

echo ""
echo "[analyze] ===== Done $(date) ====="
echo "[analyze] Output files:"
ls -lh "${OUT_DIR}/"
