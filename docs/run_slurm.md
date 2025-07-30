# Automated SLURM Job Script Creation and Submission with YAML Configuration Merging

The `run_slurm.py` script is designed to automate the creation and submission of SLURM job scripts with support for Docker and merged YAML configurations. Users can specify a SLURM job submission template, a main configuration file, and additional optional configuration files. The script generates and, optionally, submits the corresponding SLURM job scripts, saving their configurations and output files.

## Requirements

On raven:

```
module load python-waterboa/2024.06 
```

## Customizable Templates and Configurations
- **SLURM Job Submission Template**: Users can create a custom bash script template for SLURM jobs that include placeholders for configurations. 
- **Main Configuration File**: This is the primary YAML file containing the configurations needed for the job.
- **Optional Configuration Files**: These are additional YAML files that can be included in the main configuration file to extend or override the base settings.

## Script Features
- **Configuration Merging**: Combines settings from multiple YAML files, preserving the order and overriding as necessary.
- **Dynamic Script Creation**: Generates a SLURM job script from the template, replacing placeholders with actual values from the configurations.
- **Job Submission**: Submits the generated script to the SLURM cluster, if not in dry-run mode.
- **Output Management**: Stores the generated configurations and output logs in a structured directory hierarchy for easy tracking.

## How to Use 
1. **Create a SLURM Job Submission Template**: Write a bash script with necessary SLURM directives and placeholders for dynamic content (example in `scripts/raven/sft_cluster_template_docker.sh`)
2. **Prepare Configuration Files**: 
   - Create a main YAML configuration file for the job settings. (example in `configs/dev/sft_example.yml`)
   - Optionally, create additional YAML files for specific settings you wish to override or extend from the main configuration. 
3. **Merge Configurations**: Use the `__include` directive in the YAML files to specify the additional configuration files to be merged.
4. **Execute the Script**: `python run_slurm.py --config_file <main_config_file_path> --template <slurm_template_file_path>`
**Additional Options**:
    - `--script`: specifying the script to run. `src/sft.py` is the default.
    - `--dry`: generate scripts without submitting to the cluster for testing purposes.
    - `--n_gpu`: the number of GPUs
    - `--time`: job duration as needed
    - `--n_folds`: number of test folds for parallel job submission in SFT (see below for details)
    - Additional options: if there are placeholders in the config file with the format "<<datatype: value>>", they will be replaced with the value specified in the command line. The valid datatypes now are "str", "int", "float", and "bool". If the datatype is not specified, it will be treated as a string.

    **Example**:
    `python run_slurm.py --config_file configs/dev/sft_example.yml --template scripts/raven/run_slurm_template.sh --dry`

## Output Directory Structure
The script saves each job's configurations and output in a separate directory named using a timestamp-based ID for uniqueness. The structure is as follows:
```
<experiments>/
└── <dataset_name (e.g., sft or simulation)>/
    └── <experiment_group (e.g., train or test) >/
        └── <test_fold>/
            └── <timestamp>/
                ├── <experiment_group>_<timestampt>.yml
                ├── <experiment_group>_<timestampt>.sh
                └── slurm-<job_name>-<run_id>.out
```   



