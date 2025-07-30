import torch
import wandb
from tqdm import trange
from pathlib import Path
from pydantic import BaseModel
import torch.distributed as dist
from typing import List, Optional
from src.parser import ParserConfig
from transformers import TrainerCallback
from torch.nn.utils.rnn import pad_sequence
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_INDEX = LabelSmoother.ignore_index

class Filter(BaseModel):
    types: Optional[List[str]] = None
    rounds: Optional[List[int]] = None
    
class LoggingGroup(BaseModel):
    name: str
    filter: Optional[Filter] = None
    
    
class logpEvalCallback(TrainerCallback):
    def __init__(self, eval_dataset, tokenizer, logging_group_for_custom_eval, saving_dir, parser_config: ParserConfig):
        super().__init__()
        self.tokenizer = tokenizer
        self.saving_dir = Path(saving_dir.replace("models", "logp"))
        self.saving_dir.mkdir(parents=True, exist_ok=True)
        self.logging_group =  logging_group_for_custom_eval

        # assert that the filter criteria are valid for each logging group
        for logging_group in self.logging_group:
            group_name = logging_group.name
            filter_criteria = logging_group.filter
            if filter_criteria:
                if filter_criteria.types:
                    assert all(e in set(eval_dataset['event_name']) for e in filter_criteria.types), f"""Some filter.types keys in group "{group_name}"" are invalid"""
                if filter_criteria.rounds:
                    assert all(e in set(eval_dataset['round_number']) for e in filter_criteria.rounds), f"""Some rounds in group "{group_name}"" are invalid"""
           
        self.eval_dataset = eval_dataset
        self.df = self.eval_dataset.to_pandas()
        
        # Initialize disallowed mask
        
        self.disallowed_mask = {}
        self.event_types_with_allowed_values = {name: evt for name, evt in parser_config.events.items() if evt.allowed_values is not None}
        for event_type_value in self.event_types_with_allowed_values.values():
            allowed_tokens = [
                token_id
                for x in event_type_value.allowed_values
                for token_id in self.tokenizer.encode(x, add_special_tokens=False)
            ]
            self.disallowed_mask[event_type_value.name] = torch.ones(
                len(self.tokenizer.vocab), dtype=torch.bool
            )
            # set only the allowed tokens to False and disallow the rest by keeping them to True
            self.disallowed_mask[event_type_value.name][allowed_tokens] = False

        
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        
        model = model or kwargs['model']
        batch_size = args.per_device_eval_batch_size
        
        distributed = dist.is_initialized()
        if distributed:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1
            
        if rank == 0:
            print(f"\n ##################### \n Evaluation at step {state.global_step}: Eval dataset size = {len(self.eval_dataset)} \n ##################### \n")
        
        
        # 1) shard the dataset for each rank
        dataset_size = len(self.eval_dataset['input_ids'])
        chunk_size = (dataset_size + world_size - 1) // world_size
        start_idx = rank * chunk_size
        end_idx   = min(start_idx + chunk_size, dataset_size)

        # Slice out only what this rank should process
        ids    = self.eval_dataset['input_ids'][start_idx:end_idx]
        labels = self.eval_dataset['labels'][start_idx:end_idx]
        event_names = self.eval_dataset['event_name'][start_idx:end_idx]

        
        # the function texts_to_training_tensors_instruct does not pad the input_ids and labels and returns a list of lists (to use a padding collator while training), therefore we need to pad them here
        padded_input_ids = pad_sequence(
            [torch.tensor(sublist) for sublist in ids],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        padded_labels = pad_sequence(
            [torch.tensor(sublist) for sublist in labels],
            batch_first=True,
            padding_value=LabelSmoother.ignore_index,
        )
        
        # Shift the ids to create the labels for next token prediction
        padded_labels = padded_labels[:, 1:]
        padded_input_ids = padded_input_ids[:, :-1]
        padded_input_ids = padded_input_ids.to(model.device)
        
    
        
        # 2) forward pass on local shard only
        all_log_probs_local = []
        with torch.no_grad():
            for i in trange(0, len(padded_input_ids), batch_size, desc=f"running model for Rank {rank}"):
                logits = model(padded_input_ids[i : i + batch_size]).logits.cpu()
                tgt_mask = padded_labels[i : i + batch_size] != LabelSmoother.ignore_index
                
                for batch_idx in range(logits.shape[0]):
                    tgt_logits = logits[batch_idx][tgt_mask[batch_idx]]
                    tgt_labels = padded_labels[i + batch_idx][tgt_mask[batch_idx]]
                    
                    if len(tgt_logits) > 0:
                        # mask the disallowed tokens (e.g. for "action" event, we only compute the logp for the allowed tokens ["A", "B"])
                        if event_names[i + batch_idx] in self.event_types_with_allowed_values.keys():
                            disallowed_mask = self.disallowed_mask[event_names[i + batch_idx]]
                            tgt_logits[:, disallowed_mask] = -float("inf")
                        
                        log_prob = (
                            tgt_logits.log_softmax(dim=-1)
                            .gather(dim=-1, index=tgt_labels.unsqueeze(-1))
                            .squeeze()
                        )
                        avg_log_prob = float(
                            torch.mean(log_prob[~torch.isinf(log_prob)]).cpu()
                        )
                        all_log_probs_local.append(avg_log_prob)
                
        
        # 3) gather all local_log_probs across ranks
        if distributed:
            # turn the local log_probs into a tensor
            local_len = torch.tensor([len(all_log_probs_local)], dtype=torch.long, device=model.device)
            all_len = [torch.zeros_like(local_len) for _ in range(world_size)]
            dist.all_gather(all_len, local_len)

            # gather the actual data
            max_len = int(max(all_len).item())
            padded_local = all_log_probs_local + [0.0]*(max_len - local_len.item())
            local_tensor = torch.tensor(padded_local, dtype=torch.float, device=model.device)

            gather_list = [torch.zeros_like(local_tensor) for _ in range(world_size)]
            dist.all_gather(gather_list, local_tensor)

            # only rank=0 does final assembling & logging
            if rank == 0:

                assembled = []
                for r in range(world_size):
                    length_r = all_len[r].item()
                    # slicing out the real portion from each rank
                    assembled.extend(gather_list[r][:length_r].tolist())

        # Single-GPU/node scenario
        else:
            assembled = all_log_probs_local
        
        
        # Synchronize all processes to ensure that all log_probs are computed before logging to wandb
        if dist.is_initialized():
            dist.barrier()
        
        # 3) log the results to wandb
        if rank == 0:

            # save the logp values at each step in a csv file
            step = state.global_step
            self.df['logp'] = assembled
            # create the folder if it does not exist
            self.df.to_csv(f"{self.saving_dir}/result_step_{step}.csv", index=False)
            

            for group in self.logging_group:
                group_name = group.name
                filter_criteria = group.filter

                if filter_criteria:
                    rows = self.df.copy()
                    if filter_criteria.types:
                        rows = rows[rows['event_name'].isin(filter_criteria.types)]
                    if filter_criteria.rounds:
                        rows = rows[rows['round_number'].isin(filter_criteria.rounds)]
                else:
                    rows = self.df.copy()
                    
                # compute the mean
                mean = rows['logp'].mean()
                wandb.log({f"logp/{group_name}": mean, 'epoch': state.epoch})
                
        