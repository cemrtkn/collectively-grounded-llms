#!/bin/bash -l
#SBATCH --output {output_dir}/slurm-%x-%j.out
#SBATCH --error {output_dir}/slurm-%x-%j.out
#SBATCH --chdir ./
#SBATCH --job-name {experiment_name}_{job_id}
#
#SBATCH --nodes={n_nodes}
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task={n_cpu}
#SBATCH --mem=0
#
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:{n_gpu}
#SBATCH --partition={partition}
#
# Wall clock limit (max is 24 hours):
#SBATCH --time={time}

source .env
PTMP_PATH="/ptmp/$USER"

export HF_HOME="{hf_home}"
export WANDB_ENTITY="chm-ml"
export WANDB_PROJECT="{dataset_name}"
export WANDB_RUN_GROUP="{name} | {job_id}"
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
    --config_file configs/fsdp_config.yaml \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --num_processes $NUM_PROCESSES \
    --num_machines $NNODES \
"
# Note: it is important to escape `$SLURM_PROCID` since we want the srun on each node to evaluate this variable

export PROGRAM="{script} --config {config_file}"

export CMD="$LAUNCHER $PROGRAM"
echo $CMD

srun --jobid $SLURM_JOBID apptainer exec \
    --nv \
    -B .:"$HOME",$HF_DATASETS_CACHE,$PTMP_PATH \
    --pwd /root/llm-strategic-tuning \
    --bind .:/root/llm-strategic-tuning \
    {image} bash -c "$CMD" 2>&1

echo "END TIME: $(date)"



