#!/bin/bash
#SBATCH --job-name=marl_thesis
#SBATCH --output=logs/job_%A_%a.out      # %A = array job ID, %a = task index
#SBATCH --error=logs/job_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-9
#SBATCH --account=cenyioha

# ── Job config map ────────────────────────────────────────────────────
# Index 0  → IPPO baseline
# Index 1  → GNN K=2 r_comm=1.0
# Index 2  → GNN K=2 r_comm=1.5
# Index 3  → GNN K=2 r_comm=2.0
# Index 4  → GNN K=3 r_comm=1.0
# Index 5  → GNN K=3 r_comm=1.5
# Index 6  → GNN K=3 r_comm=2.0
# Index 7  → GNN K=5 r_comm=1.0
# Index 8  → GNN K=5 r_comm=1.5
# Index 9  → GNN K=5 r_comm=2.0

K_VALUES=(0   2   2   2   3   3   3   5   5   5)
R_VALUES=(0.0 1.0 1.5 2.0 1.0 1.5 2.0 1.0 1.5 2.0)

TASK=${SLURM_ARRAY_TASK_ID}

# ── Environment ───────────────────────────────────────────────────────
# Adjust module names to whatever `module avail python` and
# `module avail cuda` show on Newton.
module load python/3.11
module load cuda/12.1

source ~/marl_env/bin/activate

# ── Paths ─────────────────────────────────────────────────────────────
REPO_DIR=$HOME/gnn_marl          # ← change to your actual repo folder name
OUTPUT_DIR=$REPO_DIR/outputs/cluster_results
LOG_DIR=$REPO_DIR/logs

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

cd "$REPO_DIR" || exit 1

# ── Print job info ────────────────────────────────────────────────────
echo "============================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Array Task:   $TASK"
echo "Node:         $SLURMD_NODENAME"
echo "GPU:          $CUDA_VISIBLE_DEVICES"
echo "Start time:   $(date)"
echo "============================="

# ── Dispatch ──────────────────────────────────────────────────────────
if [ "$TASK" -eq 0 ]; then
    echo "Running: IPPO baseline"
    python train.py --algo ippo --output_dir "$OUTPUT_DIR"

else
    K="${K_VALUES[$TASK]}"
    R="${R_VALUES[$TASK]}"
    echo "Running: GNN K=$K r_comm=$R"
    python train.py --algo gnn --k "$K" --r_comm "$R" --output_dir "$OUTPUT_DIR"
fi

echo "============================="
echo "End time: $(date)"
echo "============================="
