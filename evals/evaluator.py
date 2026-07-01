import os
import torch
import pandas
import numbers
from utils.logging import logger
from utils.distributed import barrier, is_master
from importlib import import_module
from utils.utils import nested_dict_to_table
import gc

class Evaluator:
    def __init__(
            self, 
            cfg,
            dataloader, 
            student_wrapper, 
            teacher_wrapper
            ):

        self.cfg = cfg

        # Keep reference to other objects
        self.dataloader = dataloader
        self.student_wrapper = student_wrapper
        self.teacher_wrapper = teacher_wrapper

        # Load the evaluation objects
        self.eval_objects = [
            import_module(f"evals.{item['Evaluation']}").evaluator(
                cfg=cfg, **item
                )
            for item in (cfg.EvalConfig or [])
            ]

        # Initialize bests
        self.bests = [None] * len(self.eval_objects)

        # Initialize variables
        self.step = 0
    
    def increment_step(self):
        self.step += 1

    def set_step(self, step):
        self.step = step

    def is_my_turn(self, eval_obj, last=False, **kwargs):
        """
        Check if it is time to evaluate
        Args:
            eval_obj: evaluation object
            idx: current index
            train_dataloader: training dataloader
        Returns:
            bool: whether it is time to evaluate
        """
        eval_at_start = eval_obj.eval_at_start and self.step == 0
        eval_at_end = eval_obj.eval_at_end and last
        my_turn = (
            (self.step % eval_obj.frequency == 0)
            and (eval_obj.frequency != torch.inf)
            and (self.step > 0)
        )
        return my_turn or eval_at_start or eval_at_end

    def write_metrics(self, results):
        """
        Write metrics to logger and return them as a dictionary
        Args:
            results: dict of results to write
        Returns:
            metrics: dict of metrics
        """
        metrics = {}
        # WANDB & LOG
        for key, value in results.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                value = value.item()
            if isinstance(value, numbers.Number):
                logger.info(f"[EVAL] {key}: {float(value):.6f}")
                metrics[key] = float(value)
            elif type(value) == pandas.DataFrame:
                logger.info(f"[EVAL] {key}: \n {value.to_string()}")
                rows = {
                    f"Benchmark/{row[0]}": row[1]["Result"] for row in value.iterrows()
                }
                # NOTE: Benchmark_AVG is already included (overriden)
                metrics.update(rows)
            else:
                logger.info(f"[EVAL] {key}: \n {nested_dict_to_table(value)}")
                # logger.info(f"[EVAL] {key}: \n {value}")
        return metrics

    def eval(self, **kwargs):
        """
        Evaluate the model
        Args:
            student_wrapper: student model
            teacher_wrapper: teacher model
        Returns:
            metrics: dict of metrics
        """
        
        metrics = {}
        for _idx_, eval_obj in enumerate(self.eval_objects):

            # Check if evaluation conditions are met
            if not self.is_my_turn(eval_obj, **kwargs):
                continue

            # EVALUATE
            barrier()
            results = eval_obj(self.student_wrapper, wrapper_teacher=self.teacher_wrapper, step=self.step)
            barrier()

            # clean up
            torch.cuda.empty_cache()
            gc.collect()


            # RECORD BEST
            current_best = self.bests[_idx_]
            is_best = current_best is None or eval_obj.is_better(
                current_best=current_best,
                new=results,
                )
            if is_best:
                self.bests[_idx_] = results

            # WANDB & LOG
            metrics.update(self.write_metrics(results))

            # SAVE BEST
            if is_best and eval_obj.save_best:
                eval_score = results["eval_score"]
                logger.info(f"[SAVE] Saving best model with {eval_obj.name} score of {eval_score}")
                self._save_checkpoint(
                    eval_obj=eval_obj,
                    results=results,
                    checkpoint_type="best"
                )

            # SAVE LATEST
            if eval_obj.save_latest:
                eval_score = results["eval_score"]
                logger.info(f"[SAVE] Saving latest model with {eval_obj.name} score of {eval_score}")
                self._save_checkpoint(
                    eval_obj=eval_obj,
                    results=results,
                    checkpoint_type="latest"
                )

        return metrics

    def _save_checkpoint(self, eval_obj, results, checkpoint_type="best"):
        """
        Save a checkpoint (best or latest).
        Args:
            eval_obj: evaluation object
            results: evaluation results
            checkpoint_type: "best" or "latest"
        Returns:
            saved_dir: directory where checkpoint was saved
        """
        # Get the base save directory
        base_save_dir = self.student_wrapper.config.ManagementConfig.paths.save_dir
        
        # Create subdirectory for this checkpoint type
        if checkpoint_type == "best":
            save_dir = os.path.join(base_save_dir, f"best_{eval_obj.name}")
        elif checkpoint_type == "latest":
            save_dir = os.path.join(base_save_dir, f"latest_{eval_obj.name}")
        else:
            raise ValueError(f"Unknown checkpoint_type: {checkpoint_type}")
        
        # Temporarily modify save_dir to save to the subdirectory
        original_save_dir = self.student_wrapper.config.ManagementConfig.paths.save_dir
        try:
            self.student_wrapper.config.ManagementConfig.paths.save_dir = save_dir
            saved_dir = self.student_wrapper.save_weights()
        finally:
            self.student_wrapper.config.ManagementConfig.paths.save_dir = original_save_dir

        if not is_master:
            return None
        if saved_dir is None:
            saved_dir = save_dir
        
        # Save dataloader state
        try:
            dataloader_state = self.dataloader.state_dict()
        except NotImplementedError:
            logger.warning(
                "[SAVE] Dataloader state is not implemented for this loader; "
                f"skipping dataloader_state_dict.pth for {checkpoint_type}_{eval_obj.name}."
            )
        else:
            torch.save(
                dataloader_state,
                f"{saved_dir}/dataloader_state_dict.pth"
            )
        
        # Save txt with results
        txt_path = f"{saved_dir}/{checkpoint_type}_{eval_obj.name}.txt"
        with open(txt_path, "w") as f:
            f.write(str(results))
        
        return saved_dir
