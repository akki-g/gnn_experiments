#!/bin/bash
#SBATCH --job-name=pursuit_comm_sweep
#SBATCH --partition=normal
#SBATCH --account=cenyioha
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-59
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

# Sweep: 4 comm modules x 5 alpha levels x 3 seeds = 60 array tasks
# Output layout: runs/pursuit/alpha/{comm_module}/alpha{alpha}/seed{seed}/metrics.csv

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

# Sweep parameters — override via environment variables if needed.
BASE_CONFIG="${BASE_CONFIG:-configs/pursuit_base.yaml}"
read -r -a ALPHAS <<< "${ALPHAS:-0.0 0.25 0.5 0.75 1.0}"
read -r -a COMM_MODULES <<< "${COMM_MODULES:-identity broadcast attention graph}"
NUM_SEEDS="${NUM_SEEDS:-3}"

# Task decomposition:
#   TASK_ID in [0, NUM_MODULES * NUM_ALPHAS * NUM_SEEDS)
#   outer loop = comm module, middle loop = alpha, inner loop = seed
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
NUM_ALPHAS="${#ALPHAS[@]}"
NUM_MODULES="${#COMM_MODULES[@]}"
TOTAL_TASKS=$((NUM_MODULES * NUM_ALPHAS * NUM_SEEDS))

if (( TASK_ID >= TOTAL_TASKS )); then
    echo "[setup] Invalid SLURM_ARRAY_TASK_ID=${TASK_ID}; total tasks=${TOTAL_TASKS}"
    exit 2
fi

MODULE_INDEX=$(( TASK_ID / (NUM_ALPHAS * NUM_SEEDS) ))
ALPHA_INDEX=$(( (TASK_ID % (NUM_ALPHAS * NUM_SEEDS)) / NUM_SEEDS ))
SEED=$(( TASK_ID % NUM_SEEDS ))

COMM_MODULE="${COMM_MODULES[$MODULE_INDEX]}"
ALPHA="${ALPHAS[$ALPHA_INDEX]}"

if [[ ! -f "${BASE_CONFIG}" ]]; then
    echo "[setup] Missing config: ${BASE_CONFIG}"
    exit 2
fi

ALPHA_LABEL="alpha${ALPHA//./p}"
RUN_LABEL="${COMM_MODULE}_${ALPHA_LABEL}_seed${SEED}"
RUN_DIR="runs/pursuit/alpha/${COMM_MODULE}/${ALPHA_LABEL}/seed${SEED}"
JOB_TAG="${SLURM_ARRAY_JOB_ID:-manual}_${TASK_ID}"
GENERATED_CONFIG="${TMPDIR:-/tmp}/${JOB_TAG}_${RUN_LABEL}.yaml"

# --- venv setup (same lock pattern as alpha sweep) ---
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

# --- log organization (mirrors alpha sweep) ---
organize_slurm_logs() {
    local job_name="${SLURM_JOB_NAME:-pursuit_comm_sweep}"
    local array_job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
    local array_task_id="${SLURM_ARRAY_TASK_ID:-0}"
    local log_root="${SLURM_SUBMIT_DIR:-$PWD}/slurm_logs"
    local log_dir="${log_root}/${job_name}_${array_job_id}/${COMM_MODULE}/${ALPHA_LABEL}"
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
python - \
    "${BASE_CONFIG}" "${GENERATED_CONFIG}" \
    "${ALPHA}" "${SEED}" \
    "${COMM_MODULE}" \
    "${RUN_LABEL}" "${RUN_DIR}" <<'PY'
import pathlib, sys, yaml

base_path, out_path, alpha, seed, comm_module, run_label, run_dir = sys.argv[1:]

with open(base_path) as f:
    cfg = yaml.safe_load(f) or {}

cfg["seed"] = int(seed)
cfg.setdefault("env", {})["alpha"] = float(alpha)

# Inject comm block — no comm_radius so all modules use full all-to-all comm.
COMM_BLOCKS = {
    "identity":  {"module": "identity"},
    "broadcast": {"module": "broadcast", "rounds": 1},
    "attention": {"module": "attention", "num_heads": 1, "rounds": 1},
    "graph":     {"module": "graph", "num_heads": 4, "num_hops": 2},
}
if comm_module not in COMM_BLOCKS:
    print(f"[config] Unknown comm module '{comm_module}'; using identity.", flush=True)
    comm_module = "identity"
cfg["comm"] = COMM_BLOCKS[comm_module]

log = cfg.setdefault("logging", {})
log["log_dir"]          = str(pathlib.Path(run_dir).parent)
log["run_dir"]          = run_dir
log["run_label"]        = run_label
log["save_checkpoints"] = False
log["save_config"]      = True   # keep config.yaml so plotter can read comm.module
log["save_console_log"] = False
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
python - <<'PY'
import importlib.metadata as md, torch, yaml
def version(name):
    try: return md.version(name)
    except md.PackageNotFoundError: return "unknown"
print(f"PyYAML  : {yaml.__version__}")
print(f"NumPy   : {version('numpy')}")
print(f"Torch   : {torch.__version__} cuda_available={torch.cuda.is_available()} cuda={torch.version.cuda}")
PY
echo "CUDA_VIS  : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Threads   : OMP=${OMP_NUM_THREADS} MKL=${MKL_NUM_THREADS} OPENBLAS=${OPENBLAS_NUM_THREADS}"
echo "Task      : ${TASK_ID}/${TOTAL_TASKS}"
echo "Comm      : ${COMM_MODULE}"
echo "Alpha     : ${ALPHA}"
echo "Seed      : ${SEED}"
echo "Run label : ${RUN_LABEL}"
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
