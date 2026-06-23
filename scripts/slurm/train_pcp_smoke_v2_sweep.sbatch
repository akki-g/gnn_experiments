#!/bin/bash
#SBATCH --job-name=pcp_smoke_v2_sweep
#SBATCH --partition=normal
#SBATCH --account=cenyioha
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-14
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#
# PCP smoke sweep v2 — EASED CAPTURE (test learnability).
#
# Iteration after smoke v1 + probe: the probe proved the comm channel carries
# prey location to blind capturers (R2_post broadcast 0.86 / attention 0.76 vs
# _zm ~0.22), but eval_capture was ~0 everywhere because capture_radius=0.6 was
# too tight to act on. v2 eases capture to make it learnable:
#   - capture_radius     : 0.6 -> 1.0  (still < detect_radius=2.0)
#   - entropy_coef_final : 0.001 -> 0.01 (entropy floor; keep capturers exploring)
#
# 15 runs = 5 cells (identity, broadcast, broadcast_zm, attention, attention_zm) x 3 seeds.
#
# Grid (parallel fixed arrays — NOT env-overridable):
#   CELL_MODULE : identity  broadcast  broadcast  attention  attention
#   CELL_ZM     : false     false      true       false      true
#   NUM_SEEDS   : 3
#
# graph comm intentionally skipped (oversmoothing — separate concern).
#
# Index decomposition: CELL_INDEX = TASK_ID / NUM_SEEDS ; SEED = TASK_ID % NUM_SEEDS
#
# Fixed knob assertions (inline Python, SystemExit on mismatch):
#   detect_radius == 2.0 ; capture_radius == 1.0 ; detection_window == 5
#   num_full_sight == 3 ; alpha == 0.0 ; target_speed == 0.0
#   entropy_coef_final == 0.01   (v2 entropy floor)
#
# Output: runs/pcp_smoke_v2_${SWEEP_DT}/${MODULE}${ZM_SUFFIX}/seed${SEED}/metrics.csv
# Wall-time: ~6h per cell. Submit: sbatch scripts/slurm/train_pcp_smoke_v2_sweep.sbatch

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
if [[ ! -f "${REPO_ROOT}/configs/pcp_smoke_v2.yaml" ]]; then
    REPO_ROOT="${HOME}/gnn_experiments"
fi
cd "${REPO_ROOT}"

# ------------------------------------------------------------------ #
# Fixed sweep grid -- intentionally NOT env-overridable.
# 5 cells x 3 seeds = 15 runs total.
# ------------------------------------------------------------------ #
BASE_CONFIG="configs/pcp_smoke_v2.yaml"
NUM_SEEDS=3

# Explicit (module, zm_flag) pairs.  Order is fixed — index must match array formula.
CELL_MODULE_LIST=(identity   broadcast   broadcast   attention   attention)
CELL_ZM_LIST=(   false      false       true        false       true)

NUM_CELLS="${#CELL_MODULE_LIST[@]}"    # 5
TOTAL_TASKS=$(( NUM_CELLS * NUM_SEEDS ))  # 15

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID >= TOTAL_TASKS )); then
    echo "[setup] Invalid SLURM_ARRAY_TASK_ID=${TASK_ID}; total tasks=${TOTAL_TASKS}"
    exit 2
fi

CELL_INDEX=$(( TASK_ID / NUM_SEEDS ))
SEED=$(( TASK_ID % NUM_SEEDS ))

COMM_MODULE="${CELL_MODULE_LIST[$CELL_INDEX]}"
ZM_FLAG="${CELL_ZM_LIST[$CELL_INDEX]}"

if [[ ! -f "${BASE_CONFIG}" ]]; then
    echo "[setup] Missing config: ${BASE_CONFIG}"
    exit 2
fi

# Shared datetime stamp (first task wins via atomic mv -n).
STAMP_FILE="${SLURM_SUBMIT_DIR:-${REPO_ROOT}}/runs/.pcp_smoke_v2_stamp_${SLURM_ARRAY_JOB_ID:-0}"
mkdir -p "$(dirname "${STAMP_FILE}")"
if [[ ! -f "${STAMP_FILE}" ]]; then
    _TMP="$(mktemp "${STAMP_FILE}.XXXXXX")"
    date +%Y%m%d_%H%M%S > "${_TMP}"
    mv -n "${_TMP}" "${STAMP_FILE}" 2>/dev/null || rm -f "${_TMP}"
fi
SWEEP_DT="$(cat "${STAMP_FILE}" 2>/dev/null || date +%Y%m%d_%H%M%S)"

ZM_SUFFIX=""
if [[ "${ZM_FLAG}" == "true" ]]; then
    ZM_SUFFIX="_zm"
fi
RUN_LABEL="${COMM_MODULE}${ZM_SUFFIX}_seed${SEED}"
RUN_DIR="runs/pcp_smoke_v2_${SWEEP_DT}/${COMM_MODULE}${ZM_SUFFIX}/seed${SEED}"
JOB_TAG="${SLURM_ARRAY_JOB_ID:-manual}_${TASK_ID}"
GENERATED_CONFIG="${TMPDIR:-/tmp}/${JOB_TAG}_${RUN_LABEL}.yaml"

# --- venv setup (mirrors train_pcp_smoke_sweep.sbatch) ---
VENV_DIR="${VENV_DIR:-${HOME}/expenv}"
REQ_FILE="${REQ_FILE:-${REPO_ROOT}/requirements-hpc.txt}"
HPC_REQUIREMENTS=(
    "torch==2.12.0"
    "numpy==2.3.5"
    "PyYAML==6.0.3"
)

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
    print("[setup] Missing Python packages:", ", ".join(missing))
    sys.exit(1)
print("[setup] Required Python imports are available.")
PY
}

write_missing_requirements() {
    python - "$1" <<'PY'
import importlib.util, pathlib, sys
out_path = pathlib.Path(sys.argv[1])
required = {"yaml": "PyYAML==6.0.3", "numpy": "numpy==2.3.5", "torch": "torch==2.12.0"}
missing = [spec for mod, spec in required.items() if importlib.util.find_spec(mod) is None]
out_path.write_text("\n".join(missing) + ("\n" if missing else ""))
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
        MISSING_REQ="$(mktemp)"
        write_missing_requirements "${MISSING_REQ}"
        echo "[setup] Installing missing HPC Python dependencies:"
        cat "${MISSING_REQ}"
        python -m pip install --upgrade pip
        if [[ -s "${MISSING_REQ}" ]]; then
            python -m pip install -r "${MISSING_REQ}"
        elif [[ -f "${REQ_FILE}" ]]; then
            python -m pip install -r "${REQ_FILE}"
        else
            python -m pip install "${HPC_REQUIREMENTS[@]}"
        fi
        rm -f "${MISSING_REQ}"
    fi
    cleanup_lock
    trap - EXIT
fi

# --- log organization ---
organize_slurm_logs() {
    local job_name="${SLURM_JOB_NAME:-pcp_smoke_v2_sweep}"
    local array_job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
    local array_task_id="${SLURM_ARRAY_TASK_ID:-0}"
    local log_root="${SLURM_SUBMIT_DIR:-$PWD}/slurm_logs"
    local log_dir="${log_root}/${job_name}_${array_job_id}/${COMM_MODULE}${ZM_SUFFIX}"
    mkdir -p "${log_dir}"
    mv "${log_root}/${job_name}_${array_job_id}_${array_task_id}.out" \
        "${log_dir}/seed${SEED}_task${array_task_id}.out" 2>/dev/null || true
    mv "${log_root}/${job_name}_${array_job_id}_${array_task_id}.err" \
        "${log_dir}/seed${SEED}_task${array_task_id}.err" 2>/dev/null || true
}

cleanup_generated_config() { rm -f "${GENERATED_CONFIG}" 2>/dev/null || true; }
on_exit() { cleanup_generated_config; organize_slurm_logs; }
trap on_exit EXIT

check_hpc_imports

# --- generate run config ---
# IMPORTANT: detect_radius, capture_radius, detection_window, num_full_sight,
# alpha, target_speed, and entropy_coef_final are FIXED v2 knobs — never overridden.
python - \
    "${BASE_CONFIG}" "${GENERATED_CONFIG}" \
    "${SEED}" \
    "${COMM_MODULE}" \
    "${RUN_LABEL}" "${RUN_DIR}" \
    "${ZM_FLAG}" <<'PY'
import pathlib, sys, yaml

base_path, out_path, seed, comm_module, run_label, run_dir, zm_flag = sys.argv[1:]
zero_messages = zm_flag.lower() == "true"

with open(base_path) as f:
    cfg = yaml.safe_load(f) or {}

cfg["seed"] = int(seed)

# Comm block definitions — only the modules swept here.
COMM_BLOCKS = {
    "identity":  {"module": "identity",  "topology": "knn", "knn_k": 4, "symmetrize": "or", "zero_messages": False},
    "broadcast": {"module": "broadcast", "rounds": 1, "topology": "knn", "knn_k": 4, "symmetrize": "or", "zero_messages": zero_messages},
    "attention": {"module": "attention", "num_heads": 1, "rounds": 1, "topology": "knn", "knn_k": 4, "symmetrize": "or", "zero_messages": zero_messages},
}
if comm_module not in COMM_BLOCKS:
    raise SystemExit(f"[config] PCP smoke v2 sweep only supports identity|broadcast|attention, got {comm_module!r}")
cfg["comm"] = COMM_BLOCKS[comm_module]

# ------------------------------------------------------------------ #
# Post-merge assertions: fixed v2 knobs must not be clobbered.
# ------------------------------------------------------------------ #
env = cfg.get("env", {})
algo = cfg.get("algo", {})

assert abs(float(env.get("detect_radius", float("nan"))) - 2.0) < 1e-9, (
    f"env.detect_radius must be 2.0 in PCP smoke v2, got {env.get('detect_radius')}"
)
assert abs(float(env.get("capture_radius", float("nan"))) - 1.0) < 1e-9, (
    f"env.capture_radius must be 1.0 in PCP smoke v2, got {env.get('capture_radius')}"
)
assert int(env.get("detection_window", -1)) == 5, (
    f"env.detection_window must be 5 in PCP smoke v2, got {env.get('detection_window')}"
)
assert int(env.get("num_full_sight", -1)) == 3, (
    f"env.num_full_sight must be 3 in PCP smoke v2, got {env.get('num_full_sight')}"
)
assert abs(float(env.get("alpha", float("nan"))) - 0.0) < 1e-9, (
    f"env.alpha must be 0.0 in PCP smoke v2, got {env.get('alpha')}"
)
assert abs(float(env.get("target_speed", float("nan"))) - 0.0) < 1e-9, (
    f"env.target_speed must be 0.0 in PCP smoke v2, got {env.get('target_speed')}"
)
assert abs(float(algo.get("entropy_coef_final", float("nan"))) - 0.01) < 1e-9, (
    f"algo.entropy_coef_final must be 0.01 (v2 entropy floor), got {algo.get('entropy_coef_final')}"
)

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

echo "=== Job info ==="
echo "Job       : ${SLURM_JOB_ID:-unset}  array=${SLURM_ARRAY_TASK_ID:-0}"
echo "Node      : $(hostname)"
echo "Submit    : ${SLURM_SUBMIT_DIR:-unset}"
echo "Repo      : $(pwd)"
echo "Git rev   : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Python    : $(command -v python)"
python --version
echo "CUDA_VIS  : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Task      : ${TASK_ID}/${TOTAL_TASKS}"
echo "Comm      : ${COMM_MODULE}  ZeroMsg=${ZM_FLAG}  CellIndex=${CELL_INDEX}  Seed=${SEED}"
echo "Run dir   : ${RUN_DIR}"
echo "Config    : ${GENERATED_CONFIG}"
echo "Start     : $(date)"
echo "================"

nvidia-smi || true

python -m experiments.train_vmas_mappo \
    --config "${GENERATED_CONFIG}" \
    --seed "${SEED}" \
    --render off

echo "=== done $(date) ==="
