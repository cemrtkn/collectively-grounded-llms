import argparse
import os
from contextlib import nullcontext

from transformers import set_seed
from transformers import AutoTokenizer
from transformers import Trainer
import torch.distributed as dist
from torch.profiler import profile

from src.sft_types import TrainingConfig
from src.utils.setup_model import get_model
from src.utils.config import load_config  
from src.utils.dataset import load_dataset_and_collator
from src.utils.value_util import EvaluateFirstStepCallback

import torch

torch.cuda.empty_cache()


def run_sft(config: TrainingConfig):
    """Train a model using the given configuration.

    Especially notable training arguments (TrainingArguments) include...
        - learning_rate: float
        - weight_decay: float
        - num_train_epochs: int
        - batch_size: int
    Also consider using....
        - load_best_model_at_end: bool = True
        - group_by_length: bool = True
        - neftune_noise_alpha: float'

    Args:
        config: A TrainingConfig object or a path to a yaml file containing a
            TrainingConfig object.
    """
    set_seed(config.train_args.seed)

    saving_dir = config.train_args.output_dir
    config.train_args.output_dir = (
        os.path.join(config.ptmp_dir, saving_dir) if config.ptmp_dir else saving_dir
    )
    os.makedirs(config.train_args.output_dir, exist_ok=True)

    print("=" * 8, "Load Model.", "=" * 8)
    model = get_model(config)
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(config.model)
    if tokenizer.pad_token is None:
        print("Setting pad token to EOS token.")
        tokenizer.pad_token = tokenizer.eos_token

    print("=" * 8, "Prepare Dataset.", "=" * 8)

    datasetdict, collator = load_dataset_and_collator(
        config=config, tokenizer=tokenizer, test_fold=config.train_dataset_config.test_fold or 0
    )

    print("=" * 8, "Start Training.", "=" * 8)

    config.train_args.eval_strategy = (
        "no" if "test" not in datasetdict else config.train_args.eval_strategy
    )

    trainer = Trainer(
        model=model,
        args=config.train_args,
        train_dataset=datasetdict["train"],
        eval_dataset=datasetdict["test"] if "test" in datasetdict else None,
        data_collator=collator,
    )

    checkpoint = None
    if config.train_args.resume_from_checkpoint is not None:
        checkpoint = config.train_args.resume_from_checkpoint
        print(f"Resuming from checkpoint: {checkpoint}")

    if config.train_args.logging_first_step and "test" in datasetdict:
        trainer.add_callback(EvaluateFirstStepCallback())

    profiler = nullcontext()
    if config.run_profiler:
        profiler = profile(record_shapes=True, with_stack=True, profile_memory=True)

    with profiler as prof:
        (
            trainer.train(resume_from_checkpoint=checkpoint)
            if checkpoint is not None
            else trainer.train()
        )

    if dist.is_initialized():
        dist.barrier()

    if dist.is_initialized() and dist.get_rank() == 0:
        print("=" * 8, "Save Model.", "=" * 8)

        if config.run_profiler:
            prof.export_memory_timeline(f"memory-profile.html", device="cuda:0")

        print(f"fsdp_plugin: {trainer.accelerator.state.fsdp_plugin}")
        if trainer.is_fsdp_enabled:
            trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
        trainer.save_model()
        tokenizer.save_pretrained(config.train_args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/example_configs/sft.yml")
    args = parser.parse_args()
    config = load_config(args.config)  # Load config to read test_folds.
    run_sft(config=config)
