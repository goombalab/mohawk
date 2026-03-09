import torch
import torch.distributed as dist

from dataloaders import setup_dataloader
from evals.eval_api import EvalAPI
from utils.distributed import barrier, local_rank, world_size
from utils.logging import logger


class evaluator(EvalAPI):
    def __init__(self, cfg, DataConfig, n_batches, *args, **kwargs):
        super().__init__(*args, **kwargs, name="eval_hstates")
        self.cfg = cfg
        self.n_batches = n_batches
        self.dataloader = setup_dataloader(
            data_cfg=DataConfig, split="val"
        )

    def is_better(self, current_best, new):
        return new["eval_score"] < current_best["eval_score"]

    @torch.inference_mode()
    def __call__(self, wrapper_model, wrapper_teacher, *args, **kwargs):

        logger.info(f"[EVAL] Evaluating hstates for {self.n_batches} batches")

        running_hstate_distance = 0
        num_hstates_seen = 0

        wrapper_model.model.eval()
        torch.cuda.empty_cache()
        for idx, batch in enumerate(self.dataloader):
            if idx >= self.n_batches:
                break
            batch = {k: v.to(local_rank) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            teacher_outputs = wrapper_teacher(
                **batch,
                output_hidden_states=True,
                output_attentions=False,
                use_cache=False,
            )

            for layer_idx in range(len(wrapper_model.module.backbone.layers)):
                input = teacher_outputs.hidden_states[layer_idx].to(local_rank)

                # Forward pass
                wrapper_model.module.forward_fn = "_layer_forward"
                student_outputs = wrapper_model(
                    layer_idx=layer_idx,
                    hidden_states=input,
                    return_mixer_matrix=False,
                    return_hidden_states=True,
                )
                wrapper_model.module.forward_fn = "_forward"
                teacher_hstate = teacher_outputs.hidden_states[layer_idx + 1]
                teacher_hstate = teacher_hstate.to(local_rank)

                # Calculate hstates distance:
                assert student_outputs["hidden_states"].size() == teacher_hstate.size()
                hstates_distance = torch.norm(
                    student_outputs["hidden_states"] - teacher_hstate, p=2, dim=(-1,)
                ).mean()

                running_hstate_distance += hstates_distance.item()
                num_hstates_seen += 1

            # Free memory
            teacher_outputs = teacher_hstate = None

        barrier()

        # Convert to tensors
        running_hstate_distance = torch.tensor(running_hstate_distance).to(local_rank)
        num_hstates_seen = torch.tensor(num_hstates_seen).to(local_rank)

        # Reduce
        if world_size > 1:
            dist.all_reduce(running_hstate_distance, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_hstates_seen, op=dist.ReduceOp.SUM)

        # Average
        avg_hstate_distance = running_hstate_distance / num_hstates_seen

        # Return
        return {
            "eval_score": avg_hstate_distance.item(),
            "hstates_distance": avg_hstate_distance.item(),
        }
