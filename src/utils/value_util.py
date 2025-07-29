import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import yaml
from tqdm import tqdm, trange
from transformer_heads.model.model import HeadedModel
from transformer_heads.output import HeadedModelOutput
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import ModelOutput

from src.parser import ParserConfig
from src.sft_types import TrainingConfig


def generate_different_sequences(
    model: PreTrainedModel,
    context: torch.Tensor,
    sequence_bias_add: int,
    sequence_bias_decay: float,
    generation_args: dict,
    num_generations: int,
) -> list[torch.Tensor]:
    """Generates different completions using an LLM

    Uses sequence bias to ensure generation of different completions.
    All tokens that were generated in previous completions get a negative bias.

    Args:
        model: The LLM model to use for generation
        context: The context to generate completions for
        sequence_bias_add: The amount to add to the sequence bias for each token generated
        sequence_bias_decay: The amount to decay the sequence bias for each generation
        generation_args: The arguments to pass to the model.generate method
        num_generations: The number of completions to generate
    Returns:
        list[torch.Tensor]: A list of generated completions
    """
    gen_sequences = []
    sequence_bias = defaultdict(float)
    for _ in range(num_generations):
        gen = model.generate(context, sequence_bias=sequence_bias or None, **generation_args)
        if isinstance(gen, GenerateOutput):
            gen = gen.sequences
        gen = gen[0][context.shape[-1] :]
        gen_sequences.append(gen)
        for key in sequence_bias:
            sequence_bias[key] *= sequence_bias_decay
        if sequence_bias_add != 0:
            for tok in gen:
                sequence_bias[(tok.item(),)] += sequence_bias_add
    return gen_sequences


def batched_prediction(
    model: HeadedModel, input_ids: torch.Tensor, batch_size: int
) -> dict[str, torch.Tensor]:
    """Predicts using a transformer_heads model in batches

    Args:
        model: The transformer_heads model to use for prediction
        input_ids: The input ids to predict on
        batch_size: The batch size to use for prediction
    Returns:
        dict[str, torch.Tensor]: A dictionary of predictions by head
    """
    preds_by_head = defaultdict(list)
    for i in range(0, input_ids.shape[0], batch_size):
        outputs: HeadedModelOutput = model(input_ids[i : i + batch_size])
        for key in outputs.preds_by_head:
            preds_by_head[key].append(outputs.preds_by_head[key])
    return {key: torch.cat(val) for key, val in preds_by_head.items()}


class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == 0:
            control.should_evaluate = True
