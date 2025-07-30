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

module purge
module load apptainer

source .env

# create huggingface cache directory if it doesn't exist
mkdir -p ~/.cache/huggingface

echo "Runing SFT using the image: {image}"

srun apptainer exec \
	--nv \
    --contain \
    --cleanenv \
    --pwd /root/llm-strategic-tuning \
    --bind .:/root/llm-strategic-tuning \
    --bind ~/.cache/huggingface:/root/.cache/huggingface \
    --bind /ptmp:/ptmp \
    --env HUGGING_FACE_HUB_TOKEN="$HUGGINGFACE_TOKEN" \
    --env WANDB_API_KEY="$WANDB_API_KEY" \
    --env WANDB_ENTITY="chm-ml" \
    --env WANDB_PROJECT="{dataset_name}" \
    --env WANDB_RUN_GROUP="{experiment_name}" \
    --env WANDB_NAME="{job_id}_test_fold_{test_fold}" \
    --env NCCL_DEBUG="INFO" \
    --env NCCL_BLOCKING_WAIT="0" \
    --env HF_HOME="/root/.cache/huggingface" \
	{image} \
    python -m torch.distributed.run \
		--nnodes="$SLURM_NNODES" \
		--nproc-per-node=gpu \
		--rdzv-id="$SLURM_JOBID" \
		--rdzv-endpoint=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1) \
		--rdzv-backend="c10d" \
    {script} --config {config_file}
