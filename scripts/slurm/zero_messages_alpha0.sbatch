#!/bin/bash
#SBATCH --job-name=zm_alpha0
#SBATCH --partition=normal
#SBATCH --account=cenyioha
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-14
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#
# zero_messages control arm at alpha=0 only.
# 3 comm modules (broadcast attention graph) x 5 seeds = 15 tasks.
#
# Purpose: test whether the alpha=0 blind-agent distance gain
# (identity ~5.5 → comm ~3.7) persists when messages are zeroed.
#   - If zeroed-comm keeps ~3.7  → gain was CTDE critic leakage, not communication.
#   - If zeroed-comm reverts to  ~5.5 → gain was real comm channel information.
#
# Output layout mirrors the main F1 sweep:
#   runs/zm_alpha0_<DT>/alpha/<module>_zm/alpha0p0/seed*/metrics.csv
#
# Submit:
#   sbatch scripts/slurm/zero_messages_alpha0.sbatch
#
# Compare results:
#   python -m experiments.plot_alpha_sweep runs/zm_alpha0_<DT>/alpha \
#     --mode comm --output runs/zm_alpha0_<DT>/alpha/zm_vs_comm.png

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

BASE_CONFIG="configs/pursuit_base.yaml"

# Task table: 3 modules x 5 seeds = 15 tasks (array 0-14)
MODULES=(broadcast attention graph)
NUM_MODULES=${#MODULES[@]}   # 3
NUM_SEEDS=5

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODULE_INDEX=$(( TASK_ID / NUM_SEEDS ))
SEED=$(( TASK_ID % NUM_SEEDS ))
COMM_MODULE="${MODULES[$MODULE_INDEX]}"

# Shared datetime stamp (same atomic-mv pattern as main sweep)
STAMP_FILE="${SLURM_SUBMIT_DIR:-${REPO_ROOT}}/runs/.zm_alpha0_stamp_${SLURM_ARRAY_JOB_ID:-0}"
mkdir -p "$(dirname "${STAMP_FILE}")"
if [[ ! -f "${STAMP_FILE}" ]]; then
    _TMP="$(mktemp "${STAMP_FILE}.XXXXXX")"
    date +%Y%m%d_%H%M%S > "${_TMP}"
    mv -n "${_TMP}" "${STAMP_FILE}" 2>/dev/null || rm -f "${_TMP}"
fi
SWEEP_DT="$(cat "${STAMP_FILE}" 2>/dev/null || date +%Y%m%d_%H%M%S)"

RUN_LABEL="${COMM_MODULE}_zm_alpha0p0_seed${SEED}"
RUN_DIR="runs/zm_alpha0_${SWEEP_DT}/alpha/${COMM_MODULE}_zm/alpha0p0/seed${SEED}"
JOB_TAG="${SLURM_ARRAY_JOB_ID:-manual}_${TASK_ID}"
GENERATED_CONFIG="${TMPDIR:-/tmp}/${JOB_TAG}_${RUN_LABEL}.yaml"

# --- venv setup ---
VENV_DIR="${VENV_DIR:-${HOME}/expenv}"
if [[ ! -d "${VENV_DIR}" ]]; then
    python -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"

check_imports() {
    python - <<'PY'
import importlib.util, sys
for mod, pkg in (("yaml","PyYAML"),("numpy","numpy"),("torch","torch")):
    if importlib.util.find_spec(mod) is None:
        print(f"[setup] Missing: {pkg}"); sys.exit(1)
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
    local job="${SLURM_JOB_NAME:-zm_alpha0}"
    local ajid="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
    local atid="${SLURM_ARRAY_TASK_ID:-0}"
    local log_dir="${log_root}/${job}_${ajid}/${COMM_MODULE}_zm/alpha0p0"
    mkdir -p "${log_dir}"
    mv "${log_root}/${job}_${ajid}_${atid}.out" \
       "${log_dir}/seed${SEED}_task${atid}.out" 2>/dev/null || true
    mv "${log_root}/${job}_${ajid}_${atid}.err" \
       "${log_dir}/seed${SEED}_task${atid}.err" 2>/dev/null || true
}
cleanup_config() { rm -f "${GENERATED_CONFIG}" 2>/dev/null || true; }
on_exit() { cleanup_config; organize_logs; }
trap on_exit EXIT

# --- generate run config (zero_messages=true, alpha=0, full training budget) ---
python - \
    "${BASE_CONFIG}" "${GENERATED_CONFIG}" \
    "${SEED}" "${COMM_MODULE}" \
    "${RUN_LABEL}" "${RUN_DIR}" <<'PY'
import pathlib, sys, yaml

base_path, out_path, seed, comm_module, run_label, run_dir = sys.argv[1:]

with open(base_path) as f:
    cfg = yaml.safe_load(f) or {}

cfg["seed"] = int(seed)
cfg.setdefault("env", {})["alpha"] = 0.0
cfg["env"]["capture_k"] = 4

COMM_BLOCKS = {
    "broadcast": {"module": "broadcast", "rounds": 1, "topology": "knn", "knn_k": 4, "symmetrize": "or", "zero_messages": True},
    "attention": {"module": "attention", "num_heads": 1, "rounds": 1, "topology": "knn", "knn_k": 4, "symmetrize": "or", "zero_messages": True},
    "graph":     {"module": "graph", "num_heads": 4, "num_hops": 2, "topology": "knn", "knn_k": 4, "symmetrize": "or", "zero_messages": True},
}
cfg["comm"] = COMM_BLOCKS[comm_module]

log = cfg.setdefault("logging", {})
log["log_dir"]           = str(pathlib.Path(run_dir).parent)
log["run_dir"]           = run_dir
log["run_label"]         = run_label
log["save_checkpoints"]  = True
log["save_config"]       = True
log["save_console_log"]  = False
log["save_metrics_json"] = False

cfg.setdefault("render", {})["mode"] = "off"

out = pathlib.Path(out_path)
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

echo "=== job info ==="
echo "Task      : ${TASK_ID}/14"
echo "Module    : ${COMM_MODULE} (zero_messages=true)"
echo "Alpha     : 0.0"
echo "Seed      : ${SEED}"
echo "Run dir   : ${RUN_DIR}"
echo "Start     : $(date)"
echo "================"

nvidia-smi || true

python -m experiments.train_vmas_mappo \
    --config "${GENERATED_CONFIG}" \
    --seed   "${SEED}" \
    --render off

echo "=== done $(date) ==="
