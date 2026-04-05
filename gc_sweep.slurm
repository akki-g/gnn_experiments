#!/bin/bash
#SBATCH --job-name=gc_sweep
#SBATCH --output=logs/gc_job_%A_%a.out
#SBATCH --error=logs/gc_job_%A_%a.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-6
#SBATCH --account=cenyioha

# -- Job config map ---
# Guided Coverage 
# Idx 0 - IPPO Baseline (no comm)
# Idx 1 - GNN K=2 r_comm=1.0
# Idx 2  - GNN K=2 r_comm=1.5
# Idx 3  - GNN K=3 r_comm=1.0
# Idx 4  - GNN K=3 r_comm=1.5
# Idx 5  - HetNet r_comm=1.0
# Idx 6  - HetNet r_comm=1.5

ALGOS=(ippo  gnn gnn gnn gnn hetnet hetnet)
K_VALUES=(0  2   2   3   3   0      0)
R_VALUES=(0  1.0 1.5 1.0 1.5 1.0    1.5)

TASK=${SLURM_ARRAY_TASK_ID}

module load python/python-3.11.4-gcc-12.2.0
module load cuda/cuda-12.1.0

source ~/expenv/bin/activate

REPO_DIR=$HOME/gnn_experiments
OUTPUT_DIR=$REPO_DIR/outputs/guided_coverage
LOG_DIR=$REPO_DIR/logs

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

cd "$REPO_DIR" || exit 1


echo "============================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Array Task:   $TASK"
echo "Node:         $SLURMD_NODENAME"
echo "GPU:          $CUDA_VISIBLE_DEVICES"
echo "Start time:   $(date)"
echo "============================="


ALGO="${ALGOS[$TASK]}"
K="${K_VALUES[$TASK]}"
R="${R_VALUES[$TASK]}"

if [ "$ALGO" == "ippo" ]; then
    echo "Running: IPPO baseline (Guided Coverage)"
    python guided_coverage_train.py --algo ippo --output_dir "$OUTPUT_DIR"

elif [ "$ALGO" == "gnn" ]; then
    echo "Running: GNN-MAPPO K=$K r_comm=$R (Guided Coverage)"
    python guided_coverage_train.py --algo gnn --k "$K" --r_comm "$R" --output_dir "$OUTPUT_DIR"

elif [ "$ALGO" == "hetnet" ]; then
    echo "Running: HetNet/MAHAC r_comm=$R (Guided Coverage)"
    python guided_coverage_train.py --algo hetnet --r_comm "$R" --output_dir "$OUTPUT_DIR"
fi

echo "============================="
echo "End time: $(date)"
echo "============================="