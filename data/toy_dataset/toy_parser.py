import yaml
from datasets import DatasetDict, concatenate_datasets
from pydantic import BaseModel
from typing import List
from data.toy_dataset.toy_base_types import Conversation

class ToyParserConfig(BaseModel):
    fields: dict

class ToyParser():
    def __init__(self, config: ToyParserConfig) -> None:
        self.config = config
        self.field_configs = config.fields


    def parse(self, convs: List[Conversation]):
        """Formats an example dictionary into a model input string using parser_config."""
        text_results ={"text":[]}

        for conv in convs:
            parts = []
            for key in self.field_configs:
                text_template = self.field_configs[key]["text"]
                try:
                    # Format using keys in the example
                    filled_text = text_template.format(**conv.__dict__)
                except KeyError as e:
                    raise ValueError(f"Missing key {e} in example: {conv}")
                
                parts.append(filled_text)

            text_results["text"].append("\n".join(parts))

        return text_results




