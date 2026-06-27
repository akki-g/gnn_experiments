#!/bin/bash
#SBATCH --job-name=qmix_pcp_full_sweep
#SBATCH --partition=normal
#SBATCH --account=cenyioha
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-69
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#
# QMIX-on-PCP PUBLISHABLE sweep — the cross-algorithm comm-necessity study.
#
# Mirrors scripts/slurm/train_pcp_v2_full_sweep.sbatch EXACTLY except the learning
# algorithm: same env, same comm slot, same 7 cells x 10 seeds grid. If PCP comm
# necessity (probe dR2_post>0, eval_capture gap CI-excludes-0) reproduces here on an
# off-policy value-decomposition learner, communication necessity is shown to be a
# property of TASK STRUCTURE, not of the policy-gradient algorithm.
#
# RUN THE SMOKE FIRST (train_qmix_pcp_smoke.sbatch) to confirm QMIX learns the signal.
#
# 70 runs = 7 cells x 10 seeds.
#   CELL_MODULE : identity broadcast broadcast attention attention graph graph
#   CELL_ZM     : false    false     true      false     true      false true
#   NUM_SEEDS   : 10
# Index: CELL_INDEX = TASK_ID / NUM_SEEDS ; SEED = TASK_ID % NUM_SEEDS
#
# Fixed knob assertions: detect_radius==2.0, capture_radius==1.0, detection_window==5,
#   num_full_sight==3, alpha==0.0, target_speed==0.0, algo.name==qmix
# Output: runs/qmix_pcp_full_${SWEEP_DT}/${MODULE}${ZM_SUFFIX}/seed${SEED}/metrics.csv
#
# After training, probe + large-sample eval reuse the MAPPO sbatch wrappers unchanged
# (the model factory dispatches on the checkpoint's algo field):
#   sbatch --export=ALL,RUN_DIR=runs/qmix_pcp_full_${SWEEP_DT} scripts/slurm/run_probe_pcp.sbatch
#   sbatch --export=ALL,RUN_DIR=runs/qmix_pcp_full_${SWEEP_DT},EPISODES=200 scripts/slurm/run_eval_pcp.sbatch
#
# Submit: sbatch scripts/slurm/train_qmix_pcp_full_sweep.sbatch

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
if [[ ! -f "${REPO_ROOT}/configs/qmix_pcp.yaml" ]]; then
    REPO_ROOT="${HOME}/gnn_experiments"
fi
cd "${REPO_ROOT}"

# ------------------------------------------------------------------ #
# Fixed sweep grid -- 7 cells x 10 seeds = 70 runs.
# ------------------------------------------------------------------ #
BASE_CONFIG="configs/qmix_pcp.yaml"
NUM_SEEDS=10

CELL_MODULE_LIST=(identity  broadcast  broadcast  attention  attention  graph  graph)
CELL_ZM_LIST=(    false     false      true       false      true       false  true)

NUM_CELLS="${#CELL_MODULE_LIST[@]}"        # 7
TOTAL_TASKS=$(( NUM_CELLS * NUM_SEEDS ))   # 70

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
    echo "[setup] Missing config: ${BASE_CONFIG}"; exit 2
fi

STAMP_FILE="${SLURM_SUBMIT_DIR:-${REPO_ROOT}}/runs/.qmix_pcp_full_stamp_${SLURM_ARRAY_JOB_ID:-0}"
mkdir -p "$(dirname "${STAMP_FILE}")"
if [[ ! -f "${STAMP_FILE}" ]]; then
    _TMP="$(mktemp "${STAMP_FILE}.XXXXXX")"
    date +%Y%m%d_%H%M%S > "${_TMP}"
    mv -n "${_TMP}" "${STAMP_FILE}" 2>/dev/null || rm -f "${_TMP}"
fi
SWEEP_DT="$(cat "${STAMP_FILE}" 2>/dev/null || date +%Y%m%d_%H%M%S)"

ZM_SUFFIX=""
if [[ "${ZM_FLAG}" == "true" ]]; then ZM_SUFFIX="_zm"; fi
RUN_LABEL="${COMM_MODULE}${ZM_SUFFIX}_seed${SEED}"
RUN_DIR="runs/qmix_pcp_full_${SWEEP_DT}/${COMM_MODULE}${ZM_SUFFIX}/seed${SEED}"
JOB_TAG="${SLURM_ARRAY_JOB_ID:-manual}_${TASK_ID}"
GENERATED_CONFIG="${TMPDIR:-/tmp}/${JOB_TAG}_${RUN_LABEL}.yaml"

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

organize_slurm_logs() {
    local job_name="${SLURM_JOB_NAME:-qmix_pcp_full_sweep}"
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

python - \
    "${BASE_CONFIG}" "${GENERATED_CONFIG}" \
    "${SEED}" "${COMM_MODULE}" "${RUN_LABEL}" "${RUN_DIR}" "${ZM_FLAG}" <<'PY'
import pathlib, sys, yaml
base_path, out_path, seed, comm_module, run_label, run_dir, zm_flag = sys.argv[1:]
zero_messages = zm_flag.lower() == "true"
with open(base_path) as f:
    cfg = yaml.safe_load(f) or {}
cfg["seed"] = int(seed)

# Comm slot — IDENTICAL blocks to the MAPPO PCP full sweep (parity principle).
COMM_BLOCKS = {
    "identity":  {"module": "identity",  "topology": "knn", "knn_k": 4, "symmetrize": "or", "zero_messages": False},
    "broadcast": {"module": "broadcast", "rounds": 1, "topology": "knn", "knn_k": 4, "symmetrize": "or", "zero_messages": zero_messages},
    "attention": {"module": "attention", "num_heads": 1, "rounds": 1, "topology": "knn", "knn_k": 4, "symmetrize": "or", "zero_messages": zero_messages},
    "graph":     {"module": "graph", "num_heads": 4, "num_hops": 2, "topology": "knn", "knn_k": 4, "symmetrize": "or", "zero_messages": zero_messages},
}
if comm_module not in COMM_BLOCKS:
    raise SystemExit(f"[config] unsupported comm module {comm_module!r}")
cfg["comm"] = COMM_BLOCKS[comm_module]

env = cfg.get("env", {}); algo = cfg.get("algo", {})
assert abs(float(env.get("detect_radius", float("nan"))) - 2.0) < 1e-9, f"detect_radius != 2.0: {env.get('detect_radius')}"
assert abs(float(env.get("capture_radius", float("nan"))) - 1.0) < 1e-9, f"capture_radius != 1.0: {env.get('capture_radius')}"
assert int(env.get("detection_window", -1)) == 5, f"detection_window != 5: {env.get('detection_window')}"
assert int(env.get("num_full_sight", -1)) == 3, f"num_full_sight != 3: {env.get('num_full_sight')}"
assert abs(float(env.get("alpha", float("nan"))) - 0.0) < 1e-9, f"alpha != 0.0: {env.get('alpha')}"
assert abs(float(env.get("target_speed", float("nan"))) - 0.0) < 1e-9, f"target_speed != 0.0: {env.get('target_speed')}"
assert str(algo.get("name", "")).lower() == "qmix", f"algo.name != qmix: {algo.get('name')}"

log = cfg.setdefault("logging", {})
log["log_dir"] = str(pathlib.Path(run_dir).parent); log["run_dir"] = run_dir; log["run_label"] = run_label
log["save_checkpoints"] = True; log["save_config"] = True; log["save_console_log"] = False; log["save_metrics_json"] = False
cfg.setdefault("render", {})["mode"] = "off"
out = pathlib.Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

echo "=== qmix_pcp_full task ${TASK_ID}/${TOTAL_TASKS}: ${COMM_MODULE}${ZM_SUFFIX} seed${SEED} (git $(git rev-parse --short HEAD 2>/dev/null || echo ?)) ==="
echo "Run dir: ${RUN_DIR}  Start: $(date)"
nvidia-smi || true

python -m experiments.train_qmix --config "${GENERATED_CONFIG}" --seed "${SEED}" --render off

echo "=== done $(date) ==="
