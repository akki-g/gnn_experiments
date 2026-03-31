#!/bin/bash
#SBATCH --job-name=hetnet_sweep
#SBATCH --output=logs/hetnet_%A_%a.out
#SBATCH --error=logs/hetnet_%A_%a.err
#SBATCH --account=cenyioha
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-8

# ===========================================================================
# HetNet Configuration Sweep
# ===========================================================================
# Configs 0-2: Baseline / sanity checks (small, should learn quickly)
# Configs 3-5: Medium complexity (the thesis sweet spot)
# Configs 6-8: Stress tests (push HetNet to its limits)
# ===========================================================================

mkdir -p logs
module load python/python-3.11.4-gcc-12.2.0
module load cuda/cuda-12.1.0
source ~/expenv/bin/activate

TASK=${SLURM_ARRAY_TASK_ID}

case $TASK in

# ===== BASELINE CONFIGS (sanity checks) =====

0)
    # Config 0: Minimal — 2s/2i vs 1 intruder, wide comms
    # Should converge fast. If this fails, something is fundamentally broken.
    RUN_NAME="baseline_2s2i_1int"
    ARGS="--n_scouts 2 --n_interc 2 --n_intruders 1 --n_zones 2 \
          --n_envs 64 --max_steps 200 --rollout_length 200 \
          --n_iterations 2000 --r_comm 2.0 \
          --scout_fov 1.5 --interceptor_fov 0.6 --tag_radius 0.15 \
          --hidden_dim 64 --n_heads 4 --head_dim 16 --n_gat_layers 3 \
          --lr_actor 3e-4 --lr_critic 1e-3 --entropy_coef 0.01"
    ;;

1)
    # Config 1: Same as 0 but NO communication (r_comm=0)
    # IPPO baseline — agents can only use own observations.
    # HetNet should outperform this. If it doesn't, communication isn't helping.
    RUN_NAME="ippo_nocommunication_2s2i"
    ARGS="--n_scouts 2 --n_interc 2 --n_intruders 1 --n_zones 2 \
          --n_envs 64 --max_steps 200 --rollout_length 200 \
          --n_iterations 2000 --r_comm 0.0 \
          --scout_fov 1.5 --interceptor_fov 0.6 --tag_radius 0.15 \
          --hidden_dim 64 --n_heads 4 --head_dim 16 --n_gat_layers 3 \
          --lr_actor 3e-4 --lr_critic 1e-3 --entropy_coef 0.01"
    ;;

2)
    # Config 2: Same as 0 but tight comms (r_comm=0.5)
    # Tests whether partial communication topology matters —
    # agents must move into range to share info.
    RUN_NAME="tight_comm_2s2i"
    ARGS="--n_scouts 2 --n_interc 2 --n_intruders 1 --n_zones 2 \
          --n_envs 64 --max_steps 200 --rollout_length 200 \
          --n_iterations 2000 --r_comm 0.5 \
          --scout_fov 1.5 --interceptor_fov 0.6 --tag_radius 0.15 \
          --hidden_dim 64 --n_heads 4 --head_dim 16 --n_gat_layers 3 \
          --lr_actor 3e-4 --lr_critic 1e-3 --entropy_coef 0.01"
    ;;

# ===== MEDIUM CONFIGS (thesis sweet spot) =====

3)
    # Config 3: 3s/3i vs 2 intruders, skilled adversaries
    # Core thesis config. Enough agents for interesting communication patterns,
    # skilled intruders that evade interceptors forcing scout-interceptor relay.
    RUN_NAME="thesis_3s3i_2int_skilled"
    ARGS="--n_scouts 3 --n_interc 3 --n_intruders 2 --n_zones 2 \
          --n_envs 128 --max_steps 300 --rollout_length 300 \
          --n_iterations 3000 --r_comm 1.5 \
          --scout_fov 1.2 --interceptor_fov 0.4 --tag_radius 0.1 \
          --hidden_dim 64 --n_heads 4 --head_dim 16 --n_gat_layers 3 \
          --lr_actor 2e-4 --lr_critic 8e-4 --entropy_coef 0.01 \
          --ppo_epochs 4"
    ;;

4)
    # Config 4: Extreme FOV asymmetry — interceptors are nearly blind
    # interceptor_fov=0.15 means they can barely see anything.
    # Forces HEAVY reliance on scout communication.
    # This is where HetGAT's type-specific messaging should shine brightest
    # vs a vanilla GCN that would average scout info away.
    RUN_NAME="blind_interceptors_3s3i"
    ARGS="--n_scouts 3 --n_interc 3 --n_intruders 2 --n_zones 2 \
          --n_envs 128 --max_steps 300 --rollout_length 300 \
          --n_iterations 3000 --r_comm 1.5 \
          --scout_fov 1.5 --interceptor_fov 0.15 --tag_radius 0.1 \
          --hidden_dim 64 --n_heads 4 --head_dim 16 --n_gat_layers 3 \
          --lr_actor 2e-4 --lr_critic 8e-4 --entropy_coef 0.015"
    ;;

5)
    # Config 5: Imbalanced team — 2 scouts must guide 4 interceptors
    # Tests S_2 x S_4 typed permutation equivariance.
    # More interceptors than scouts means each scout message is high-value
    # and the attention mechanism must route info efficiently.
    RUN_NAME="imbalanced_2s4i_2int"
    ARGS="--n_scouts 2 --n_interc 4 --n_intruders 2 --n_zones 3 \
          --n_envs 128 --max_steps 300 --rollout_length 300 \
          --n_iterations 3000 --r_comm 1.5 \
          --scout_fov 1.3 --interceptor_fov 0.3 --tag_radius 0.1 \
          --hidden_dim 64 --n_heads 4 --head_dim 16 --n_gat_layers 3 \
          --lr_actor 2e-4 --lr_critic 8e-4 --entropy_coef 0.01"
    ;;

# ===== STRESS TEST CONFIGS (push the limits) =====

6)
    # Config 6: Large team — 4s/4i vs 4 intruders, 4 zones
    # 8 defenders total, fully connected graph has 56 directed edges across
    # 4 edge types. Tests whether HetGAT scales and whether the attention
    # mechanism can handle a noisy, high-degree communication graph.
    # Longer training needed.
    RUN_NAME="large_4s4i_4int"
    ARGS="--n_scouts 4 --n_interc 4 --n_intruders 4 --n_zones 4 \
          --n_envs 128 --max_steps 400 --rollout_length 400 \
          --n_iterations 5000 --r_comm 1.5 \
          --scout_fov 1.2 --interceptor_fov 0.4 --tag_radius 0.1 \
          --world_size 6.0 \
          --hidden_dim 64 --n_heads 4 --head_dim 16 --n_gat_layers 3 \
          --lr_actor 1.5e-4 --lr_critic 5e-4 --entropy_coef 0.015 \
          --ppo_epochs 3"
    ;;

7)
    # Config 7: Adversarial nightmare — expert intruders, tiny tag radius
    # intruder_skill=1.0: full goal-seeking + active evasion of interceptors.
    # tag_radius=0.05: interceptors must get extremely close to tag.
    # interceptor_fov=0.2: nearly blind.
    # This REQUIRES sophisticated scout->interceptor communication —
    # scouts must continuously relay precise intruder positions for
    # interceptors to close the final gap. Tests whether the learned
    # communication protocol can transmit fine-grained spatial info.
    RUN_NAME="adversarial_expert_intruders"
    ARGS="--n_scouts 3 --n_interc 3 --n_intruders 3 --n_zones 3 \
          --n_envs 128 --max_steps 400 --rollout_length 400 \
          --n_iterations 5000 --r_comm 1.5 \
          --scout_fov 1.5 --interceptor_fov 0.2 --tag_radius 0.05 \
          --world_size 5.0 \
          --hidden_dim 64 --n_heads 4 --head_dim 16 --n_gat_layers 3 \
          --lr_actor 1.5e-4 --lr_critic 5e-4 --entropy_coef 0.02 \
          --ppo_epochs 3 --log_std_min -1.0"
    ;;

8)
    # Config 8: Sparse communication + large world — relay chain required
    # r_comm=0.8 in a world_size=8.0 arena means agents can only talk to
    # neighbors within ~10% of the arena width. With 5s/5i spread across
    # the map, multi-hop message passing through the 3 HetGAT layers
    # becomes essential — a scout near an intruder must relay through
    # intermediate defenders to reach a distant interceptor.
    # This is the theoretical ceiling test for GNN depth.
    RUN_NAME="sparse_relay_5s5i_largemap"
    ARGS="--n_scouts 5 --n_interc 5 --n_intruders 3 --n_zones 3 \
          --n_envs 128 --max_steps 500 --rollout_length 400 \
          --n_iterations 5000 --r_comm 0.8 \
          --scout_fov 1.5 --interceptor_fov 0.4 --tag_radius 0.1 \
          --world_size 8.0 \
          --hidden_dim 64 --n_heads 4 --head_dim 16 --n_gat_layers 3 \
          --lr_actor 1e-4 --lr_critic 5e-4 --entropy_coef 0.02 \
          --ppo_epochs 3 --log_std_min -1.0"
    ;;

esac

echo "============================================="
echo "Running config $TASK: $RUN_NAME"
echo "============================================="

python train.py \
    --device cuda \
    --seed 42 \
    --save_interval 200 \
    --log_interval 10 \
    --run_name "${RUN_NAME}_${SLURM_ARRAY_JOB_ID}" \
    $ARGS

echo "Job $TASK ($RUN_NAME) finished at $(date)"