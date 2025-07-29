#!/bin/bash -l
#SBATCH --output projects/collectively_grounded_llms/experiments/train/sft_instruct_fsdp_lora/2025_07_28__14_37_23/test_fold_None/slurm-%x-%j.out
#SBATCH --error projects/collectively_grounded_llms/experiments/train/sft_instruct_fsdp_lora/2025_07_28__14_37_23/test_fold_None/slurm-%x-%j.out
#SBATCH --chdir ./
#SBATCH --job-name sft_instruct_fsdp_lora_2025_07_28__14_37_23
#
#SBATCH --nodes=5
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:4
#SBATCH --partition=gpu
#
# Wall clock limit (max is 24 hours):
#SBATCH --time=08:00:00

source .env
PTMP_PATH="/ptmp/$USER"

export HF_HOME="/ptmp/certuer/huggingface"
export WANDB_ENTITY="chm-ml"
export WANDB_PROJECT="collectively_grounded_llms"
export WANDB_RUN_GROUP="Lama-3-8B-Instruct-lr-1e-4-LoRA | 2025_07_28__14_37_23"
export HUGGING_FACE_HUB_TOKEN="$HUGGINGFACE_TOKEN"
export WANDB_API_KEY="$WANDB_API_KEY"

##########


module purge
module load apptainer/1.2.2


# This is important! The lazy datasets caching mechanism causes trouble for our GPFS file system
# when multiple nodes try to cache at the same location -> That's why we configure every node to
# do the caching separately on its CPU RAM. This is also more performant ...
# Make sure this path is accessible from within the container!
export HF_DATASETS_CACHE=$JOB_SHMTMPDIR

# force crashing on nccl issues like hanging broadcast
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "START TIME: $(date)"

GPUS_PER_NODE=4
NNODES=$SLURM_NNODES
NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)

# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

# OTHER LAUNCHERS CAN BE USED HERE
export LAUNCHER="accelerate launch \
    --config_file projects/coopbot/configs/sft/fsdp_config.yaml \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --num_processes $NUM_PROCESSES \
    --num_machines $NNODES \
"
# Note: it is important to escape `$SLURM_PROCID` since we want the srun on each node to evaluate this variable

export PROGRAM="src/sft.py --config projects/collectively_grounded_llms/experiments/train/sft_instruct_fsdp_lora/2025_07_28__14_37_23/test_fold_None/sft_2025_07_28__14_37_23.yml"

export CMD="$LAUNCHER $PROGRAM"
echo $CMD

srun --jobid $SLURM_JOBID apptainer exec \
    --nv \
    -B .:"$HOME",$HF_DATASETS_CACHE,$PTMP_PATH \
    --pwd /root/llm-strategic-tuning \
    --bind .:/root/llm-strategic-tuning \
    /u/yjiang/projects/coopbot/llm-strategic-tuning/images/strategic_fsdp_v2.sif bash -c "$CMD" 2>&1

echo "END TIME: $(date)"



