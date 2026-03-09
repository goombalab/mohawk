import gc

import torch
import torch.distributed as dist

from dataloaders import setup_dataloader
from evals.eval_api import EvalAPI
from utils.distributed import barrier, local_rank, world_size
from utils.logging import logger

class evaluator(EvalAPI):
    def __init__(self, cfg, DataConfig, n_batches, eval_type = "all_tokens", *args, **kwargs):
        super().__init__(*args, **kwargs, name="Perplexity")

        self.cfg = cfg
        self.n_batches = n_batches
        self.eval_type = eval_type
        self.dataloader = setup_dataloader(data_cfg=DataConfig, split="val")
        self.eos_token_id = self.dataloader.tokenizer.eos_token_id
        self.pad_token_id = self.dataloader.tokenizer.pad_token_id

    def is_better(self, current_best, new):
        return new["eval_score"] < current_best["eval_score"]

    @torch.inference_mode()
    def __call__(self, wrapper_model, *args, **kwargs):
        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()

        # Prepare model
        train_state = wrapper_model.model.training
        wrapper_model.model.eval()

        # We evaluate LLM using generated data and ground truth data
        loss_fn = torch.nn.CrossEntropyLoss(
            reduction="none", ignore_index=self.pad_token_id
        )
        total_correct = 0
        total_loss = 0
        num_tokens_seen = 0

        logger.info(f"[EVAL] Evaluating perplexity for {self.n_batches} batches")
        logger.info(f"[EVAL] Using:\n {self.dataloader}")
        for idx, batch in enumerate(self.dataloader):
            if idx >= self.n_batches:
                break
            batch = {k: v.to(local_rank) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            input_ids = batch["input_ids"]
            position_ids = batch["position_ids"] if "position_ids" in batch else None

            # Forward pass with model
            model_outputs = wrapper_model(input_ids=input_ids, position_ids=position_ids)

            # Get logits
            if self.eval_type == "all_tokens":
                vocab_size = model_outputs.logits.shape[-1]
                logits = model_outputs.logits[:, :-1, :].contiguous().view(-1, vocab_size) # chop off the last token because we don't have a label for it
                labels = input_ids[..., 1:].contiguous().view(-1)
            elif self.eval_type == "last_token":
                last_token_idx = (input_ids[...,1:] == self.eos_token_id).to(torch.int).argmax(dim=1) - 1
                logits = model_outputs.logits[:, :-1, :].gather(
                    1, last_token_idx.view(-1,1,1).expand(-1,1,model_outputs.logits.size(-1))
                ).squeeze(1)
                labels = input_ids[...,1:].gather(1, last_token_idx.unsqueeze(1)).squeeze(1)
            else:
                raise ValueError(f"Unknown eval_type: {self.eval_type}")
            

            model_outputs = None


            # Setup outputs
            outputs = {}
            non_pad_mask = labels != self.pad_token_id
            actual_num_tokens = non_pad_mask.sum()

            # Perplexity
            loss = loss_fn(logits, labels).sum()
            outputs["ppl"] = torch.exp(loss / actual_num_tokens).item()
            outputs["loss"] = loss

            # Accuracy
            preds = torch.argmax(logits, dim=-1)
            correct = (preds == labels)[non_pad_mask].sum()
            outputs["accuracy"] = (correct / actual_num_tokens).item()

            # accumulate
            total_loss += outputs["loss"].item()
            total_correct += correct.item()
            num_tokens_seen += actual_num_tokens.item()

            # Release memory
            logits = labels = None

        barrier()
        # Convert to tensors
        total_loss = torch.tensor(total_loss).to(local_rank)
        total_correct = torch.tensor(total_correct).to(local_rank)
        num_tokens_seen = torch.tensor(num_tokens_seen).to(local_rank)

        # Reduce
        if world_size > 1:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_tokens_seen, op=dist.ReduceOp.SUM)

        # Average
        total_loss = total_loss / num_tokens_seen
        total_correct = total_correct / num_tokens_seen
        avg_accuracy = total_correct.item()
        avg_perplexity = torch.exp(total_loss).item()

        # Back to Normal
        wrapper_model.model.train(train_state)

        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()

        # Return
        return {
            "eval_score": avg_perplexity,
            "perplexity": avg_perplexity,
            "accuracy": avg_accuracy,
        }
