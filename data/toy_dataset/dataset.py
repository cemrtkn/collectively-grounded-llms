from transformers import PreTrainedTokenizerBase, AutoTokenizer
from datasets import Dataset, DatasetDict, concatenate_datasets
from parser import ToyParser, ToyParserConfig
from typing import List, Dict, Any
from base_types import Conversation
from pydantic import BaseModel
from typing import Optional
import yaml
import os
import sys
from tqdm import tqdm
from transformers.trainer_pt_utils import LabelSmoother


IGNORE_INDEX = LabelSmoother.ignore_index

# from src.sft_types import TrainingConfig
class ToyDataset(Dataset):
    @classmethod
    def from_convs(
        cls,
        convs: List[Conversation],
        tokenizer: PreTrainedTokenizerBase,
        parser: ToyParser,
        mask_untrainable_tokens: bool,
    ):
        texts = parser.parse(convs)
        data = (
            texts_to_training_tensors_instruct(texts, tokenizer, mask_untrainable_tokens)
        )

        result = cls.from_dict(data)
        result.set_format(type="torch", columns=["input_ids", "labels"])
        return result

class CrossvalDatasetDict(DatasetDict):
    """DatasetDict with multiple datasets for each fold in crossvalidation."""

    @classmethod
    def load_from_disk(cls, path: str, test_fold: int = 0) -> DatasetDict:
        """Loads a dataset dictionary from a given path and returns train and test sets.

        Args:
            path (str): The path to load the dataset from.

        Returns:
            CrossvalDatasetDict: The loaded datawiset.
        """
        dataset_dict = super().load_from_disk(path)

        # Merge the folds into train and test sets
        return cls._merge_datasets(dataset_dict=dataset_dict, test_fold=test_fold)

    @classmethod
    def _merge_datasets(cls, dataset_dict: DatasetDict, test_fold: int) -> DatasetDict:
        """Merges all datasets except the test fold into one.

        Args:
            dataset_dict (DatasetDict): The datasets to merge.

        Returns:
            DatasetDict: The merged dataset.
        """
        result_dic = {}

        # Merge all datasets except the test fold
        # if the dataset_dict contains only one fold then set the test dataset to None
        if len(dataset_dict) == 1:
            result_dic["train"] = dataset_dict[str(test_fold)]
        else:
            train_datasets = [ds for i, ds in dataset_dict.items() if i != str(test_fold)]

            # Concatenate training datasets into a single dataset
            if len(train_datasets) > 0:
                train_dataset = concatenate_datasets(train_datasets)
                result_dic["train"] = train_dataset

            # Get the test fold as a dataset
            test_dataset = dataset_dict[str(test_fold)]
            result_dic["test"] = test_dataset

        return DatasetDict(result_dic)

class ToyDatasetConfig(BaseModel):
    mask_untrainable_tokens: bool = True
    data_path: str
    parser_config: ToyParserConfig
    test_fold: Optional[int] = 0

class ToyDatasetDict(CrossvalDatasetDict):
    def __init__(
        self, config: ToyDatasetConfig, tokenizer: PreTrainedTokenizerBase, test_fold: int = 0
    ):
        """Loads the dataset from the config and splits it into train and test.

        Args:
            config (ToyDatasetConfig): The config to load the datasets from.
            tokenizer (PreTrainedTokenizerBase): The tokenizer to use.
            test_fold (int, optional): The fold to use as test. Defaults to 0.
        """
        super().__init__()

        # Instance attributes
        self.config = ToyDatasetConfig(**config)
        self.test_fold = test_fold

        self._load_datasets(tokenizer)

    def _load_datasets(self, tokenizer: PreTrainedTokenizerBase):
        # Load datasets using the paths from the config
        raw_datasets: DatasetDict = self.load_from_disk(self.config.data_path, self.test_fold)
        
        parser = ToyParser(self.config.parser_config)

        for ds_name, ds in raw_datasets.items():
            # New format
            convs = [
                Conversation(**{k: v for k, v in data.items()})
                for data in ds
            ]
            self[ds_name] = ToyDataset.from_convs(
                convs,
                tokenizer,
                parser,
                self.config.mask_untrainable_tokens
            )
        print(f"Loaded {len(convs)} conversations from {ds_name} dataset.")

       

def texts_to_training_tensors_instruct(
    data: Dict[str, List[Any]],
    tokenizer: PreTrainedTokenizerBase,
    mask_untrainable_tokens=True,
    start_target_text="<|start_header_id|>assistant<|end_header_id|>",
) -> dict[str, Any]:
    """Turns a list of texts into tokenized training tensors.
    If mask_untrainable_tokens is set, the labels of all text
    before start_target_text are set to the ignore_token.

    Note: only works for single target at the end of each data point. we don't expect to train on multiple targets in a single data point when training an Instruct model.
    Note: No padding is done here. Padding is done in the collator."""

    # create a copy of data
    result = data.copy()
    input_ids_list = []
    labels_list = []
    
    start_target_sequence = tokenizer(start_target_text, add_special_tokens=False)["input_ids"]

    tokenized_games = tokenizer(data["text"], add_special_tokens=False)["input_ids"]

    # Cloning tokenized_games to labels using a deep copy
    labels = [list(game) for game in tokenized_games]
    if mask_untrainable_tokens:
        for idx, input_list in tqdm(enumerate(tokenized_games), total=len(tokenized_games)):
            target_start_index = find_sequence(input_list, start_target_sequence)

            if target_start_index != len(start_target_sequence) - 1:
                # Set labels before the start index to IGNORE_INDEX
                labels[idx][:target_start_index] = [IGNORE_INDEX] * target_start_index
                input_ids_list.append(input_list)
                labels_list.append(labels[idx])
            else:
                print("Instruction not found in input list.")

    result["input_ids"] = input_ids_list
    result["labels"] = labels_list
    return result


def find_sequence(input_list, start_sequence):
    """Find the last start sequence in a list."""
    last_start_index = -1
    start_sequence_len = len(start_sequence)
    for i in range(len(input_list) - start_sequence_len + 1):
        if input_list[i : i + start_sequence_len] == start_sequence:
            last_start_index = i

    return last_start_index + start_sequence_len


if __name__ == "__main__":
    # Example usage
    with open("../configs//train/sft_instruct_fsdp_lora.yaml", "r") as f:
        config = yaml.safe_load(f)
    #config = TrainingConfig(**config)

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"],
        token=os.getenv("HUGGINGFACE_TOKEN"),  # or pass directly
        local_files_only=False  # default
    )

    if tokenizer.pad_token is None:
        print("Setting pad token to EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    
    dataset_config = config["train_dataset_config"]

    
    dataset_dict = ToyDatasetDict(
        config=dataset_config,
        tokenizer=tokenizer,
        test_fold=0
    )