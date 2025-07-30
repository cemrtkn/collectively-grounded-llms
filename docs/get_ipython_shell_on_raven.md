# Ipython shell with gpu and apptainer on raven
For development and testing, it is useful to get an ipython shell on raven with our apptainer image and gpu.
## Steps
1. Grap the gpu with slurm: `srun -p gpu --cpus-per-task=8 --gres=gpu:a100:1 --mem=40 --time=01:00:00 --pty bash`
2. Load apptainer: `module load apptainer`
3. Get into your container: `apptainer shell --writable-tmpfs --nv --contain --cleanenv --pwd /root/llm-strategic-tuning --bind .:/root/llm-strategic-tuning --bind ~/.cache/huggingface:/root/.cache/huggingface --env HUGGING_FACE_HUB_TOKEN="$HUGGINGFACE_TOKEN" --env HF_HOME="/root/.cache/huggingface" --env WANDB_API_KEY="$WANDB_API_KEY" --env WANDB_ENTITY="chm-ml" --env WANDB_PROJECT="test_runs" path/to/your/image`
4. Start ipython: `ipython`
5. Test if it worked: `import torch; print(torch.cuda.is_available())`. If the result is `true`, everything worked correctly.