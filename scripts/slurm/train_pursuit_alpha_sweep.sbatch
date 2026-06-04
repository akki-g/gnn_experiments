#!/bin/bash
#SBATCH --job-name=pursuit_alpha_mappo
#SBATCH --partition=normal
#SBATCH --account=cenyioha
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-29
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

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

mkdir -p runs/pursuit_sweep/generated_configs
mkdir -p runs/pursuit_sweep/probes

# Default sweep: 2 pursuit configs x 5 alpha levels x 3 seeds = 30 array tasks.
# Override with, for example:
#   sbatch --array=0-14 --export=ALL,SWEEP_CONFIGS="configs/pursuit_base.yaml" scripts/slurm/train_pursuit_alpha_sweep.sbatch
read -r -a CONFIGS <<< "${SWEEP_CONFIGS:-configs/pursuit_base.yaml configs/pursuit_smoke.yaml}"
read -r -a ALPHAS <<< "${ALPHAS:-0.0 0.25 0.5 0.75 1.0}"
NUM_SEEDS="${NUM_SEEDS:-3}"
RUN_PROBE="${RUN_PROBE:-1}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
NUM_ALPHAS="${#ALPHAS[@]}"
TASKS_PER_CONFIG=$((NUM_ALPHAS * NUM_SEEDS))
CONFIG_INDEX=$((TASK_ID / TASKS_PER_CONFIG))
WITHIN_CONFIG=$((TASK_ID % TASKS_PER_CONFIG))
ALPHA_INDEX=$((WITHIN_CONFIG / NUM_SEEDS))
SEED=$((WITHIN_CONFIG % NUM_SEEDS))
TOTAL_TASKS=$((${#CONFIGS[@]} * TASKS_PER_CONFIG))

if (( TASK_ID >= TOTAL_TASKS )); then
    echo "[setup] Invalid SLURM_ARRAY_TASK_ID=${TASK_ID}; total tasks=${TOTAL_TASKS}"
    exit 2
fi

BASE_CONFIG="${CONFIGS[$CONFIG_INDEX]}"
ALPHA="${ALPHAS[$ALPHA_INDEX]}"
if [[ ! -f "${BASE_CONFIG}" ]]; then
    echo "[setup] Missing config: ${BASE_CONFIG}"
    exit 2
fi

CONFIG_STEM="$(basename "${BASE_CONFIG}" .yaml)"
ALPHA_LABEL="alpha${ALPHA//./p}"
RUN_LABEL="${CONFIG_STEM}_${ALPHA_LABEL}"
LOG_DIR="runs/pursuit_sweep/${CONFIG_STEM}"
JOB_TAG="${SLURM_ARRAY_JOB_ID:-manual}_${TASK_ID}"
GENERATED_CONFIG="runs/pursuit_sweep/generated_configs/${JOB_TAG}_${RUN_LABEL}_seed${SEED}.yaml"
PROBE_LOG="runs/pursuit_sweep/probes/${JOB_TAG}_${RUN_LABEL}_seed${SEED}.log"

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
import importlib.util
import sys

required = {
    "yaml": "PyYAML",
    "numpy": "numpy",
    "torch": "torch",
}
missing = [
    package for module, package in required.items()
    if importlib.util.find_spec(module) is None
]
if missing:
    print("[setup] Missing Python packages:", ", ".join(missing))
    sys.exit(1)
print("[setup] Required Python imports are available.")
PY
}

write_missing_requirements() {
    python - "$1" <<'PY'
import importlib.util
import pathlib
import sys

out_path = pathlib.Path(sys.argv[1])
required = {
    "yaml": "PyYAML==6.0.3",
    "numpy": "numpy==2.3.5",
    "torch": "torch==2.12.0",
}
missing = [
    spec for module, spec in required.items()
    if importlib.util.find_spec(module) is None
]
out_path.write_text("\n".join(missing) + ("\n" if missing else ""))
PY
}

if ! check_hpc_imports; then
    LOCK_DIR="${HOME}/.gnn_experiments_pip_install.lock"
    until mkdir "${LOCK_DIR}" 2>/dev/null; do
        echo "[setup] Waiting for dependency install lock ${LOCK_DIR}"
        sleep 10
    done
    cleanup_lock() {
        rmdir "${LOCK_DIR}" 2>/dev/null || true
    }
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

organize_slurm_logs() {
    local job_name="${SLURM_JOB_NAME:-pursuit_alpha_mappo}"
    local array_job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
    local array_task_id="${SLURM_ARRAY_TASK_ID:-0}"

    local log_root="${SLURM_SUBMIT_DIR:-$PWD}/slurm_logs"
    local log_dir="${log_root}/${job_name}_${array_job_id}_${array_task_id}"

    mkdir -p "${log_dir}"

    mv "${log_root}/${job_name}_${array_job_id}_${array_task_id}.out" "${log_dir}/" 2>/dev/null || true
    mv "${log_root}/${job_name}_${array_job_id}_${array_task_id}.err" "${log_dir}/" 2>/dev/null || true
}

trap organize_slurm_logs EXIT

check_hpc_imports

python - "${BASE_CONFIG}" "${GENERATED_CONFIG}" "${ALPHA}" "${SEED}" "${RUN_LABEL}" "${LOG_DIR}" <<'PY'
import pathlib
import sys

import yaml

base_path, out_path, alpha, seed, run_label, log_dir = sys.argv[1:]
with open(base_path, "r") as f:
    cfg = yaml.safe_load(f) or {}

cfg["seed"] = int(seed)
cfg.setdefault("env", {})["alpha"] = float(alpha)
cfg.setdefault("logging", {})["log_dir"] = log_dir
cfg["logging"]["run_label"] = run_label
cfg.setdefault("render", {})["mode"] = "off"

out = pathlib.Path(out_path)
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

echo "=== Job info ==="
echo "Job      : ${SLURM_JOB_ID:-unset}  array=${SLURM_ARRAY_TASK_ID:-0}"
echo "Node     : $(hostname)"
echo "Submit   : ${SLURM_SUBMIT_DIR:-unset}"
echo "Repo     : $(pwd)"
echo "Git rev  : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Python   : $(command -v python)"
python --version
python - <<'PY'
import importlib.metadata as md
import torch
import yaml

def version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return "unknown"

print(f"PyYAML   : {yaml.__version__}")
print(f"NumPy    : {version('numpy')}")
print(f"Torch    : {torch.__version__} cuda_available={torch.cuda.is_available()} torch_cuda={torch.version.cuda}")
PY
echo "CUDA_VIS : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Threads  : OMP=${OMP_NUM_THREADS} MKL=${MKL_NUM_THREADS} OPENBLAS=${OPENBLAS_NUM_THREADS}"
echo "Task     : ${TASK_ID}/${TOTAL_TASKS}"
echo "Config   : ${BASE_CONFIG}"
echo "Generated: ${GENERATED_CONFIG}"
echo "Alpha    : ${ALPHA}"
echo "Seed     : ${SEED}"
echo "Run label: ${RUN_LABEL}"
echo "Log dir  : ${LOG_DIR}"
echo "Probe    : ${RUN_PROBE}"
echo "Start    : $(date)"
echo "================"

nvidia-smi || true

python -m experiments.train_vmas_mappo \
    --config "${GENERATED_CONFIG}" \
    --seed "${SEED}" \
    --render off

if [[ "${RUN_PROBE}" == "1" ]]; then
    python -m experiments.run_probe \
        --config "${GENERATED_CONFIG}" \
        --alpha "${ALPHA}" \
        --seed "${SEED}" | tee "${PROBE_LOG}"
fi

echo "=== done $(date) ==="
