from transformers import PreTrainedTokenizer
from transformers.trainer_pt_utils import LabelSmoother

from src.dataset import DataCollatorWithPadding, ToyDatasetDict
from src.sft_types import TrainingConfig

IGNORE_INDEX = LabelSmoother.ignore_index


def load_dataset_and_collator(
    config: TrainingConfig, tokenizer: PreTrainedTokenizer, test_fold: int = 0
):
    return ToyDatasetDict(
        config.train_dataset_config, tokenizer, test_fold
    ), DataCollatorWithPadding(
        feature_name_to_padding_value={
            "input_ids": tokenizer.pad_token_id,
            "labels": IGNORE_INDEX,
        }
    )
