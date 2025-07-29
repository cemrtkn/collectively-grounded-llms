# SLURM Training Documentation

This document describes how to start SLURM training runs and customize
training settings using the provided infrastructure.

## Quick Start

### Required Environment Variables
Create a `.env` file in your project root mirroring the `.env_example` file:

```bash
HUGGINGFACE_TOKEN=your_hf_token_here
WANDB_API_KEY=your_wandb_key_here
```

### Change default names

Set default names for your project in `run_slurm.py`:

```python
# Default values for job submission
# PLEASE CHANGE THESE VALUES TO YOUR OWN PROJECT NAME AND JOB NAME
DEFAULT_PROJECT_NAME = "hpc_repo_template"
DEFAULT_JOB_NAME = "v1"
DEFAULT_GROUP_NAME = "hpc_demo"
```

### Basic Usage

```bash
# Submit a training job with a config file
python src/run_slurm.py --config_file examples/example_config.yaml

# Submit with custom settings
python src/run_slurm.py \
    --config_file examples/example_config.yaml \
    --job_name my_experiment \
    --n_gpu 2 \
    --time 02:00:00
```

### Dry Run (Generate scripts without submission)

```bash
python src/run_slurm.py \
    --config_file examples/example_config.yaml \
    --dry
```

## Best Practices

1. **Start small**: Use `--dry` flag to validate configs
2. **Test locally**: Run short experiments before scaling up
3. **Monitor resources**: Check GPU utilization and memory usage
4. **Save frequently**: Use appropriate `save_steps` values
5. **Version configs**: Keep track of successful configurations
6. **Use descriptive names**: Clear job and group names for organization

## Command-Line Options

### Required Arguments

- `--config_file`: Path to YAML configuration file containing training
  parameters

### Resource Management

- `--n_gpu`: Number of GPUs to use (default: 1)
  - Single node: 1-4 GPUs
  - Multi-node: Must be divisible by 4 (e.g., 8, 12, 16)
- `--time`: Wall clock time limit in HH:MM:SS format (default: 00:10:00)
  - Maximum: 24:00:00

### Job Configuration

- `--job_name`: Job name for identification (default: v1)
- `--group_name`: Experiment group name (default: debug)
- `--project_name`: SLURM project to charge (default: YOUR_PROJECT_NAME)
- `--output_dir`: Custom output directory (auto-generated if not
  specified)

### Execution Environment

- `--script`: Training script to run (default: src/train.py)
- `--template`: SLURM template file (default: src/slurm_scripts/template.slurm)
- `--image`: Apptainer image path (default: predefined path)

### Flags

- `--dry`: Generate files without submitting to SLURM

### Dynamic Parameters

Any additional `--parameter value` pairs are passed through to the
training script and can be used for placeholder replacement in config
files.


### Training Script Options

The training script (`src/train.py`) supports additional options:

- `--config`: YAML config file (required)
- `--output_dir`: Override output directory
- `--dataset_path`: Override dataset path
- `--dry_run`: Validate config without training
- `--verbose`: Enable debug logging


## Configuration file

See `examples/example_config.yaml` for a complete working example
that can be run immediately with the provided sample data.

### Core Training Parameters

```yaml
# Required
dataset_path: "path/to/your/dataset.txt"
output_dir: "experiments/<<<group_name>>>/<<<job_name>>>"

# Data processing
max_length: 512  # Maximum sequence length for tokenization
train_test_split: 0.8  # Fraction for training (rest for evaluation)
```

### Model Configuration

```yaml
assistant_config:
  model_path: "gpt2"  # HuggingFace model name or local path
  generation_params:
    do_sample: true
    temperature: 0.7
    max_new_tokens: 50
    pad_token_id: 50256
```

### Training Arguments

Based on HuggingFace `TrainingArguments`. Key options:

```yaml
training_args:
  # Core training settings
  num_train_epochs: 3
  per_device_train_batch_size: 4
  per_device_eval_batch_size: 4
  learning_rate: 1e-5

  # Output and saving
  output_dir: "experiments/<<<group_name>>>/<<<job_name>>>"
  overwrite_output_dir: true
  save_strategy: "epoch"  # "no", "steps", "epoch"
  save_steps: 1000
  save_total_limit: 3

  # Evaluation
  eval_strategy: "steps"  # "no", "steps", "epoch"
  eval_steps: 500
  load_best_model_at_end: true
  metric_for_best_model: "eval_loss"
  greater_is_better: false

  # Logging
  logging_dir: "./logs"
  logging_steps: 10

  # Performance
  dataloader_num_workers: 4
  fp16: true  # Enable mixed precision training
  gradient_accumulation_steps: 1
  warmup_steps: 500
  weight_decay: 0.01
```

### Placeholder Replacement

Use `<<<parameter_name>>>` in config files for dynamic replacement:

```yaml
output_dir: "experiments/<<<group_name>>>/<<<job_name>>>/<<<custom_param>>>"
```

## Environment Configuration

### Container Environment

The system uses Apptainer with these bind mounts:

- `.:/root/llm-strategic-tuning`: Project directory
- `~/.cache/huggingface:/root/.cache/huggingface`: HF cache
- `/ptmp:/ptmp`: Temporary storage

### Weights & Biases Integration

Automatic W&B logging happens to (request access to wandb team):

- Entity: `chm-ml`
- Project: `{project_name}`
- Run group: `{group_name}`
- Run name: `{job_name}/{job_id}`

## Common Usage Examples

### Basic Training

```bash
python src/run_slurm.py \
    --config_file examples/example_config.yaml \
    --job_name basic_training \
    --n_gpu 1 \
    --time 01:00:00
```

### Multi-GPU Training

```bash
python src/run_slurm.py \
    --config_file config/large_model.yaml \
    --job_name multi_gpu_exp \
    --n_gpu 4 \
    --time 12:00:00 \
    --group_name production
```

### Custom Model Training

```bash
python src/run_slurm.py \
    --config_file config/custom.yaml \
    --model_name microsoft/DialoGPT-large \
    --learning_rate 5e-6 \
    --batch_size 8
```

## Output Structure

Jobs create organized output directories:

```
experiments/
└── {group_name}/
    └── {job_name}/
        └── {job_id}/
            ├── {job_id}.yml        # Processed config
            ├── {job_id}.sh         # Generated SLURM script
            ├── slurm-{job_name}-{slurm_id}.out  # SLURM logs
            ├── model_files/        # Trained model
            └── training_config.json # Training metadata
```

## Monitoring and Debugging

### Check Job Status

```bash
watch squeue --me
```

### Troubleshooting

1. **Permission denied**: Check `.env` file and tokens
2. **Out of memory**: Reduce batch size or sequence length
3. **Time limit exceeded**: Increase `--time` parameter
4. **Image not found**: Update `--image` path or build container
