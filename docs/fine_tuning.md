# Fine-tuning with FSDP

This project supports fine-tuning LLMs using FSDP (Fully Sharded Data Parallel) in a k-fold cross-validation setup. Automated SLURM job scripts facilitate the creation and submission of jobs across different train/test fold combinations. For more information on SLURM job script automation, refer to [run_slurm.md](run_slurm.md). 
- Supported models are the Instruct version of `meta-llama/Meta-Llama-3` (8B and 70B sizes)
- Supported Fine-tuning types are: Full and LoRA.

## Initial Setup

For an initial setup, download the pre-trained model weights to `/ptmp/${USER}/huggingface` by executing the following command:

```bash
./scripts/raven/download_model.sh [HuggingFace_model_name]
```

## Placeholders and Arguments:

Beyond the general placeholders specified in [run_slurm.md](run_slurm.md) (`n_gpu`, `time`, `config_file`, etc.), the followings are relevant for fine-tuning:

- `--model_name`: Path to the the Huggingface model (e.g. `meta-llama/Meta-Llama-3-8B-Instruct`)
- `--num_train_epochs`: Number of training epochs.
- `--per_device_train_batch_size`: Training batch size per device.
- `--per_device_eval_batch_size`: Evaluation batch size per device.
- `--lr`: Learning rate.
- `--name`: Name of the run in W&B.
- `--n_folds`: Number of test folds for cross-validation (e.g., `--n_folds 2` will run two test folds) (see [run_slurm.md](run_slurm.md)).
- `--ptmp_dir` (optional): set to `/ptmp/$USER` to save the fine-tuned model in the ptmp directory. More information about the ptmp can be found [here](https://docs.mpcdf.mpg.de/doc/computing/raven-user-guide.html#file-systems).
- `--hf_home`: path to your HF cache (`ptmp/$USER/huggingface`)


Logs and files created, including bash and YAML configurations, are stored in `/experiments/<<dataset_name>>` (see [run_slurm.md](run_slurm.md)).

The Fine-tuned model can be found either in the same directory, or under `/ptmp/$USER/experiments/` if `--ptmp_dir` is set to `/ptmp/$USER`.


### Full fine-tuning
Set `--config_file` to `configs/toy_dataset/train/sft_instruct_fsdp.yaml`


Examples (using fold 0 as test data):

- Llama-3 `8B`:


```bash
python run_slurm.py \
  --n_gpu 20 \
  --config_file configs/toy_dataset/train/sft_instruct_fsdp.yaml \
  --time 00:05:00 \
  --image /u/yjiang/projects/coopbot/llm-strategic-tuning/images/strategic_fsdp_v2.sif \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --num_train_epochs 5 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --n_folds 1 \
  --template scripts/raven/run_slurm_template_accelerate.slurm \
  --hf_home /ptmp/certuer/huggingface \
  --ptmp_dir /ptmp/$USER \
  --name Lama-3-8B-Instruct-lr-1e-4-LoRA \
  --lr 1e-4
```



### LoRA fine-tuning
Set `--config_file` to `projects/coopbot/configs/sft/sft_instruct_fsdp_lora.yml`


Examples (using fold 0 as test data):

- Llama-3 `8B`:
```bash
python run_slurm.py \
  --n_gpu 20 \
  --config_file configs/toy_dataset/train/sft_instruct_fsdp_lora.yaml \
  --time 00:05:00 \
  --image /u/yjiang/projects/coopbot/llm-strategic-tuning/images/strategic_fsdp_v2.sif \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --num_train_epochs 5 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --n_folds 1 \
  --template scripts/raven/run_slurm_template_accelerate.slurm \
  --hf_home /ptmp/certuer/huggingface \
  --ptmp_dir /ptmp/$USER \
  --name Lama-3-8B-Instruct-lr-1e-4-LoRA \
  --lr 1e-4
```8B-Instruct-lr-1e-4-LoRA --lr 1e-4
```




### Additional Information:

<details><summary>Dev Dataset</summary>
<p>

For debugging purposes, you can use the dev dataset by manually updating the `data_path` in the specified config file `--config_file` to:

```bash
data/toy_dataset/processed
```
