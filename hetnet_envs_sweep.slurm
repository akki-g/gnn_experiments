#!/bin/bash
#SBATCH --job-name=hetnet_envs
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err
#SBATCH --partition=normal
#SBATCH --account=cenyioha
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-13

# -- Job config map --------------------------------------------------------
# Index  0  - PP   IPPO
# Index  1  - PP   GNN K=2 r=1.5
# Index  2  - PP   GNN K=3 r=1.5
# Index  3  - PP   GNN K=2 r=2.0
# Index  4  - PP   HetNet r=1.5
# Index  5  - PP   HetNet r=2.0
# Index  6  - PCP  IPPO
# Index  7  - PCP  GNN K=2 r=1.5
# Index  8  - PCP  GNN K=3 r=1.5
# Index  9  - PCP  GNN K=2 r=2.0
# Index 10  - PCP  HetNet r=1.5
# Index 11  - PCP  HetNet r=2.0
# Index 12  - PCP  HetNet r=1.5 s=3/i=3
# Index 13  - PCP  HetNet r=1.5 s=1/i=3

TASK=${SLURM_ARRAY_TASK_ID}

# -- Common params ---------------------------------------------------------
ITERS=1500
N_ENVS=256
ROLLOUT=256
MAX_STEPS=80
SEED=42

# -- Environment -----------------------------------------------------------
module load python/python-3.11.4-gcc-12.2.0
module load cuda/cuda-12.1.0

source ~/expenv/bin/activate

# -- Paths -----------------------------------------------------------------
REPO_DIR=$HOME/gnn_experiments
OUTPUT_DIR=$REPO_DIR/outputs/hetnet_envs
LOG_DIR=$REPO_DIR/logs

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

cd "$REPO_DIR" || exit 1

# -- Print job info --------------------------------------------------------
echo "============================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Array Task:   $TASK"
echo "Node:         $SLURMD_NODENAME"
echo "GPU:          $CUDA_VISIBLE_DEVICES"
echo "Start time:   $(date)"
echo "============================="

# -- Dispatch --------------------------------------------------------------
case "$TASK" in
  0)
    echo "Running: PP IPPO"
    python train_hetnet_envs.py --env pp --algo ippo \
      --n_scouts 2 --n_interc 1 \
      --n_envs $N_ENVS --rollout_length $ROLLOUT --max_steps $MAX_STEPS \
      --iters $ITERS --seed $SEED --output_dir "$OUTPUT_DIR"
    ;;
  1)
    echo "Running: PP GNN K=2 r=1.5"
    python train_hetnet_envs.py --env pp --algo gnn --k 2 --r_comm 1.5 \
      --n_scouts 2 --n_interc 1 \
      --n_envs $N_ENVS --rollout_length $ROLLOUT --max_steps $MAX_STEPS \
      --iters $ITERS --seed $SEED --output_dir "$OUTPUT_DIR"
    ;;
  2)
    echo "Running: PP GNN K=3 r=1.5"
    python train_hetnet_envs.py --env pp --algo gnn --k 3 --r_comm 1.5 \
      --n_scouts 2 --n_interc 1 \
      --n_envs $N_ENVS --rollout_length $ROLLOUT --max_steps $MAX_STEPS \
      --iters $ITERS --seed $SEED --output_dir "$OUTPUT_DIR"
    ;;
  3)
    echo "Running: PP GNN K=2 r=2.0"
    python train_hetnet_envs.py --env pp --algo gnn --k 2 --r_comm 2.0 \
      --n_scouts 2 --n_interc 1 \
      --n_envs $N_ENVS --rollout_length $ROLLOUT --max_steps $MAX_STEPS \
      --iters $ITERS --seed $SEED --output_dir "$OUTPUT_DIR"
    ;;
  4)
    echo "Running: PP HetNet r=1.5"
    python train_hetnet_envs.py --env pp --algo hetnet --r_comm 1.5 \
      --n_scouts 2 --n_interc 1 \
      --n_envs $N_ENVS --rollout_length $ROLLOUT --max_steps $MAX_STEPS \
      --iters $ITERS --seed $SEED --output_dir "$OUTPUT_DIR"
    ;;
  5)
    echo "Running: PP HetNet r=2.0"
    python train_hetnet_envs.py --env pp --algo hetnet --r_comm 2.0 \
      --n_scouts 2 --n_interc 1 \
      --n_envs $N_ENVS --rollout_length $ROLLOUT --max_steps $MAX_STEPS \
      --iters $ITERS --seed $SEED --output_dir "$OUTPUT_DIR"
    ;;
  6)
    echo "Running: PCP IPPO"
    python train_hetnet_envs.py --env pcp --algo ippo \
      --n_scouts 2 --n_interc 2 \
      --n_envs $N_ENVS --rollout_length $ROLLOUT --max_steps $MAX_STEPS \
      --iters $ITERS --seed $SEED --output_dir "$OUTPUT_DIR"
    ;;
  7)
    echo "Running: PCP GNN K=2 r=1.5"
    python train_hetnet_envs.py --env pcp --algo gnn --k 2 --r_comm 1.5 \
      --n_scouts 2 --n_interc 2 \
      --n_envs $N_ENVS --rollout_length $ROLLOUT --max_steps $MAX_STEPS \
      --iters $ITERS --seed $SEED --output_dir "$OUTPUT_DIR"
    ;;
  8)
    echo "Running: PCP GNN K=3 r=1.5"
    python train_hetnet_envs.py --env pcp --algo gnn --k 3 --r_comm 1.5 \
      --n_scouts 2 --n_interc 2 \
      --n_envs $N_ENVS --rollout_length $ROLLOUT --max_steps $MAX_STEPS \
      --iters $ITERS --seed $SEED --output_dir "$OUTPUT_DIR"
    ;;
  9)
    echo "Running: PCP GNN K=2 r=2.0"
    python train_hetnet_envs.py --env pcp --algo gnn --k 2 --r_comm 2.0 \
      --n_scouts 2 --n_interc 2 \
      --n_envs $N_ENVS --rollout_length $ROLLOUT --max_steps $MAX_STEPS \
      --iters $ITERS --seed $SEED --output_dir "$OUTPUT_DIR"
    ;;
  10)
    echo "Running: PCP HetNet r=1.5"
    python train_hetnet_envs.py --env pcp --algo hetnet --r_comm 1.5 \
      --n_scouts 2 --n_interc 2 \
      --n_envs $N_ENVS --rollout_length $ROLLOUT --max_steps $MAX_STEPS \
      --iters $ITERS --seed $SEED --output_dir "$OUTPUT_DIR"
    ;;
  11)
    echo "Running: PCP HetNet r=2.0"
    python train_hetnet_envs.py --env pcp --algo hetnet --r_comm 2.0 \
      --n_scouts 2 --n_interc 2 \
      --n_envs $N_ENVS --rollout_length $ROLLOUT --max_steps $MAX_STEPS \
      --iters $ITERS --seed $SEED --output_dir "$OUTPUT_DIR"
    ;;
  12)
    echo "Running: PCP HetNet r=1.5 s=3/i=3"
    python train_hetnet_envs.py --env pcp --algo hetnet --r_comm 1.5 \
      --n_scouts 3 --n_interc 3 \
      --n_envs $N_ENVS --rollout_length $ROLLOUT --max_steps $MAX_STEPS \
      --iters $ITERS --seed $SEED --output_dir "$OUTPUT_DIR"
    ;;
  13)
    echo "Running: PCP HetNet r=1.5 s=1/i=3"
    python train_hetnet_envs.py --env pcp --algo hetnet --r_comm 1.5 \
      --n_scouts 1 --n_interc 3 \
      --n_envs $N_ENVS --rollout_length $ROLLOUT --max_steps $MAX_STEPS \
      --iters $ITERS --seed $SEED --output_dir "$OUTPUT_DIR"
    ;;
esac

echo "============================="
echo "End time: $(date)"
echo "============================="
