import yaml
from transformers import TrainingArguments

from src.dataset import EventDatasetConfig
from src.sft_types import TrainingConfig


def load_config(config: TrainingConfig | str) -> TrainingConfig:
    """Load the configuration from a file if necessary.

    Args:
        config: A TrainingConfig object or a path to a yaml file with TrainingConfig contents.

    Returns:
        TrainingConfig: The TrainingConfig object.
    """
    # Guard clause for when we pass in a TrainingConfig object.
    if isinstance(config, TrainingConfig):
        return config

    # Load the configuration from the given file.
    with open(config) as f:
        config_dict = yaml.safe_load(f)

        # The line below is only needed because we use SkipValidation
        # for the TrainingArguments field.
        config_dict["train_args"] = TrainingArguments(**config_dict["train_args"])

        # Typecast the config_dict to a TrainingConfig object.
        training_config: TrainingConfig = TrainingConfig(**config_dict)

        return training_config


def load_dataset_config(config: str, data_path=None):
    with open(config) as f:
        config_dict = yaml.safe_load(f)
        if data_path is not None:
            config_dict["train_dataset_config"]["data_path"] = data_path
        return EventDatasetConfig(**config_dict["train_dataset_config"])


# def set_output_dir(config: TrainingConfig, test_fold: int) -> None:
#     """Set the output directory for the given configuration.

#     Args:
#         config: A TrainingConfig object.
#         test_fold: An integer representing the fold number to use for testing.
#     """
#     if "WANDB_NAME" not in os.environ:
#         os.environ["WANDB_NAME"] = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

#     if config.output_dir_root is not None:
#         config.train_args.output_dir = os.path.join(config.output_dir_root, "models")
