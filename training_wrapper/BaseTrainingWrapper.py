from utils.logging import logger
import signal

class BaseTrainingWrapper:
    def __init__(
            self, 
            model=None, 
            optimizer=None, 
            scheduler=None, 
            mode="train",
            *args,
            **kwargs,
            ):
        self._model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.mode = mode

        assert self.mode in ["train", "inference"], f"Invalid mode: {self.mode}"

    def backward(self, loss):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return f"TrainingWrapper(model={self._model}, optimizer={self.optimizer}, scheduler={self.scheduler})"

    def state_dict(self):
        return {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "model": self._model.state_dict(),
        }

    def save_weights(self, path):
        """
        Save optimizer state.
        """
        logger.info("[TRAINING_WRAPPER] save_weights not implemented.")
        raise NotImplementedError

    def save_optimizer(self):
        """
        Save optimizer state.
        """
        logger.info("[TRAINING_WRAPPER] save_optimizer not implemented.")
        raise NotImplementedError

    def save_scheduler(self):
        """
        Save scheduler state.
        """
        logger.info("[TRAINING_WRAPPER] save_scheduler not implemented.")
        raise NotImplementedError
