import os
import re
import sys

# import shutil
import subprocess
from datetime import datetime
import argparse

# from collections import OrderedDict
import yaml


def parse_paths(args):
    """
    Extracts project and experiment names from the given config path.
    """
    config_path = args["config_file"]
    # Extract relevant path parts: ["configs", "<<dataset_name>>", "<<experiment_group>>", "<<experiment_name>>.yml"]
    config_path_parts = config_path.split(os.sep)

    assert config_path_parts[0] == "configs", "Expects config to be at the top level."
    assert config_path_parts[-1].endswith(".yml") or config_path_parts[-1].endswith(
        ".yaml"
    ), "Expects .yml or .yaml file as config."
    assert (
        len(config_path_parts) == 4
    ), "Expects config in configs/dataset_name/experiment_group/experiment_name.yml format."

    script = args["script"]
    script_path_parts = script.split(os.sep)
    assert os.path.exists(script), f"Script {script} does not exist."
    assert script.endswith(".py"), "Expects .py file as script."

    return {
        "dataset_name": config_path_parts[1],
        "experiment_name": config_path_parts[-1].split(".")[0],
        "experiment_group": config_path_parts[2],
        "script_name": script_path_parts[-1].split(".")[0],
    }


def generate_local_job_id():
    """
    Generates a local job ID based on timestamp.
    """
    return datetime.now().strftime("%Y_%m_%d__%H_%M_%S")


def parse_unknown_args(unknown_args, argv):
    # Process unknown arguments to create a dictionary
    dynamic_args = {}
    for arg in unknown_args:
        if arg.startswith("--"):
            key = arg.lstrip("-")
            # Assuming the next item in the list is the value
            if argv.index(arg) + 1 < len(argv):
                value = argv[argv.index(arg) + 1]
                dynamic_args[key] = value

    return dynamic_args


def define_input_output(args):
    # Compute additional arguments
    args.update(parse_paths(args))
    if args["input_dir"] is not None and args["output_dir"] is None:
        args["output_dir"] = args["input_dir"]
    elif args["input_dir"] is None and args["output_dir"] is None:
        if args["exp_dir"] is not None:
            exp_dir = args["exp_dir"]
        else:
            exp_dir = os.path.join("experiments", args["dataset_name"])

        args["output_dir"] = os.path.join(
            exp_dir,
            args["experiment_group"],
            args["experiment_name"],
            args["job_id"],
        )

    if args["n_folds"] is not None:
        args["output_dir"] = os.path.join(args["output_dir"], f"test_fold_{args['test_fold']}")
        if args["input_dir"] is not None:
            args["input_dir"] = os.path.join(args["input_dir"], f"test_fold_{args['test_fold']}")

    if not os.path.exists(args["output_dir"]):
        os.makedirs(args["output_dir"])
    return args


def define_compute_resources(args):
    if args["n_gpu"] > 4:
        assert args["n_gpu"] % 4 == 0
        n_nodes = args["n_gpu"] // 4
        n_gpu = 4
    else:
        n_nodes = 1
        n_gpu = args["n_gpu"]
    partition = "gpu"
    cpu = n_gpu * 18  # 18 cores per GPU

    args = {**args, "n_nodes": n_nodes, "n_gpu": n_gpu, "n_cpu": cpu, "partition": partition}
    return args


def deep_merge_configs(main_config, included_config):
    """
    Deep merges two configurations with the main config retaining its order.
    """
    for key, value in included_config.items():
        if key in main_config and isinstance(main_config[key], dict) and isinstance(value, dict):
            deep_merge_configs(main_config[key], value)
        else:
            main_config[key] = value


def find_include_value(data, target_key):
    if isinstance(data, dict):
        for key, value in data.items():
            if key == target_key:
                return value
            elif isinstance(value, dict):
                result = find_include_value(value, target_key)
                if result is not None:
                    return result
    return None


def replace_placeholder(element, replacements=None):
    if replacements is None:
        replacements = {}

    if isinstance(element, dict):
        for key, value in element.items():
            element[key] = replace_placeholder(value, replacements)
    elif isinstance(element, list):
        return [replace_placeholder(item, replacements) for item in element]
    elif isinstance(element, str):

        def replacement_function(match):
            placeholder_type = match.group(1) if match.group(1) else "str"
            placeholder_variable = match.group(2)
            default_value = match.group(3) if match.group(3) else None
            replacement_value = replacements.get(placeholder_variable, default_value if default_value else match.group(0))

            # Convert replacement value to specified type
            if placeholder_type == "float" and replacement_value is not None:
                # try to convert to float, if not possible, set to None
                try:
                    replacement_value = float(replacement_value)
                except ValueError:
                    replacement_value = None
            elif placeholder_type == "int" and replacement_value is not None:
                try:
                    replacement_value = int(replacement_value)
                except ValueError:
                    replacement_value = None
            elif placeholder_type == "bool" and replacement_value is not None:
                replacement_value = True if str(replacement_value).lower() == "true" else False
            elif placeholder_type == "list_int" and replacement_value is not None:
                try:
                    replacement_value = list(map(int, str(replacement_value).split(",")))
                except ValueError:
                    replacement_value = None

            return replacement_value

        pattern = r"<<(?:(\w+): )?(\w+)(?:\|([^>]+))?>>"
        matches = list(re.finditer(pattern, element))

        # If exactly one match and it spans the entire string, perform type conversion
        if len(matches) == 1 and matches[0].span() == (0, len(element)):
            return replacement_function(matches[0])

        # For strings with multiple placeholders or additional text, replace without type conversion
        def string_replacement_function(match):
            return str(replacement_function(match))
        result = re.sub(pattern, string_replacement_function, element)
        return result

    return element


def load_and_merge_configs(
    config_path,
):
    """
    Loads configuration from the main file and merges included configurations while preserving the order.
    """
    with open(config_path, "r") as file:
        # Load the main configuration with FullLoader to preserve the order
        main_config = yaml.load(file, Loader=yaml.FullLoader)

    # Check if there are included configs and process them
    includes = find_include_value(main_config, "__include")
    if includes is not None:
        for include_path in includes:
            with open(include_path, "r") as inc_file:
                included_config = yaml.load(inc_file, Loader=yaml.FullLoader)
                deep_merge_configs(main_config, included_config)

    return main_config


def copy_config(config, args):
    """
    Copies the given config file to a job-specific directory after merging included configurations.
    Preserves the order of parameters in the main config file.
    """
    dest_filename = f"{args['script_name']}_{args['job_id']}.yml"
    dest_path = os.path.join(args["output_dir"], dest_filename)

    with open(dest_path, "w") as file:
        yaml.dump(config, file, sort_keys=False)  # Prevent sorting keys on dump
    args["copied_config_file"] = dest_path
    return args


def generate_bash_script(args):
    """
    Reads in a bash template file and replaces placeholders with the given config path.
    Writes the modified script to the job-specific directory.
    """
    output_path = os.path.join(args["output_dir"], f"{args['script_name']}_{args['job_id']}.sh")

    with open(args["template"], "r") as file:
        script = file.read().format(**args)

    with open(output_path, "w") as file:
        file.write(script)
    return output_path


def submit_script(script_path):
    """
    Submits the given bash script to sbatch.
    """
    subprocess.run(["sbatch", script_path])


def main():
    parser = argparse.ArgumentParser(
        description="Submit jobs with documentation of the YAML configuration."
    )
    parser.add_argument(
        "--config_file", type=str, required=True, help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--n_folds", type=int, default=None, help="Number of folds to use as test sets for sft."
    )
    parser.add_argument("--input_dir", type=str, default=None, help="Path to the input data.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Path to the output data. Set to None to use input_dir. If both are None, path of config file is used.",
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        default=None,
        help="Path to store the config and output.",
    )
    parser.add_argument(
        "--ptmp_dir",
        type=str,
        default=None,
        help="The path to the PTMP directory to save the model weights under ptmp_dir + train_args.output_dir.",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="scripts/raven/run_slurm_template.slurm",
        help="Path to the bash script template.",
    )
    parser.add_argument(
        "--dry", action="store_true", help="Only create files, do not submit the job."
    )
    parser.add_argument("--n_gpu", type=int, default=1, help="Number of GPUs to use.")
    parser.add_argument(
        "--time", type=str, default="00:10:00", help="Expected runtime in HH:MM:SS format."
    )
    parser.add_argument("--script", type=str, default="src/sft.py", help="Script to run.")
    parser.add_argument(
        "--image",
        type=str,
        default="/u/yjiang/projects/coopbot/llm-strategic-tuning/images/strategic_fsdp_v2.sif",
        help="Apptainer image to use",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from.",
    )
    argv = sys.argv[1:]

    known_args, unknown_args = parser.parse_known_args(sys.argv[1:])
    dynamic_args = parse_unknown_args(unknown_args, argv)
    args_dict = vars(known_args)
    args_dict.update(dynamic_args)

    args_dict = define_compute_resources(args_dict)
    args_dict["job_id"] = generate_local_job_id()
    n_folds = args_dict["n_folds"] if args_dict["n_folds"] is not None else 1
    for test_fold in range(n_folds):
        fold_args_dict = args_dict.copy()
        fold_args_dict["test_fold"] = test_fold if n_folds > 1 else None
        fold_args_dict = define_input_output(fold_args_dict)
        config = load_and_merge_configs(fold_args_dict["config_file"])
        config = replace_placeholder(config, fold_args_dict)
        fold_args_dict = copy_config(config, fold_args_dict)
        fold_args_dict["config_file"] = fold_args_dict["copied_config_file"]
        generated_script = generate_bash_script(fold_args_dict)
        if not args_dict["dry"]:
            submit_script(generated_script)
        else:
            print(f"Generated script at {generated_script} without submission.")


if __name__ == "__main__":
    main()