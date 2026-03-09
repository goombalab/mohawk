import torch


class EvalAPI:
    def __init__(
        self,
        name=None,  # name of evaluation
        frequency=torch.inf,  # frequency of evaluation
        save_best=False,  # if True, then save the best model
        save_latest=False,  # if True, then save the latest model on each evaluation
        eval_at_start=False,  # if True, then evaluate at step 0
        eval_at_end=False,  # if True, then evaluate at the end of training
        step=None,  # current step
        *args,
        **kwargs
    ):
        """
        Initialize evaluation.
        """
        self.name = name
        self.frequency = frequency
        self.save_best = save_best
        self.save_latest = save_latest
        self.eval_at_start = eval_at_start
        self.eval_at_end = eval_at_end
        self.step = step

    def is_better(self, current_best, new):
        """
        Returns True if new score is better than current score.
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """
        Execute evaluation.
        """
        raise NotImplementedError
