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
from src.dataset import CrossvalDatasetDict, texts_to_training_tensors_instruct


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
        self.config = config
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


if __name__ == "__main__":
    # Example usage
    with open("/u/certuer/collectively-grounded-llms/configs/toy_dataset/train/sft_instruct_fsdp_lora.yaml", "r") as f:
        config = yaml.safe_load(f)
    print("Loaded config:", config)
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

    dataset_config = ToyDatasetConfig(**dataset_config)
    dataset_dict = ToyDatasetDict(
        config=dataset_config,
        tokenizer=tokenizer,
        test_fold=0
    )