import torch
import torch.distributed as torch_dist

from distill.utils import setup_optimizer, setup_scheduler, setup_gradients
from utils.distributed import is_master, local_rank
from utils.logging import logger

from .BaseTrainingWrapper import BaseTrainingWrapper
from utils.weights_utils import save_weights
from utils.config import Config
from typing import Any, List


class CentralizedTrainingWrapper(BaseTrainingWrapper):
    def __init__(
        self : BaseTrainingWrapper,
        config : Config,
        compile_model : bool = False,
        model_dtype : torch.dtype = torch.float32,
        mixed_precision : bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.config = config
        self.compile_model = compile_model
        self.model_dtype = model_dtype
        self.mixed_precision = mixed_precision
        self.cuda_graph = True
        # 1. Mixed Precision
        self._setup_mixed_precision()
        # 2. Set model to train mode
        self._model.train()
        # 3. Setup model gradients
        self._setup_gradients()
        # 4. Move model to device
        self._model = self._model.to(local_rank)
        # 5. Compile model
        self._compile_model()
        # 6. Setup optimizer and scheduler
        self._setup_optimizer()
    
        # Clean up
        torch.cuda.empty_cache()

    @property
    def model(self):
        return self._model

    @property
    def module(self):
        return self._model

    def __call__(self, *args, **kwargs):
        with torch.amp.autocast(
            device_type=f"cuda:{local_rank}",
            dtype=torch.bfloat16,
            enabled=self.mixed_precision
        ):
            output = self.model(*args, **kwargs)
        return output
    

    def generate(self, *args, **kwargs):
        """
        Generate function for the model.
        """
        with torch.amp.autocast(
            device_type=f"cuda:{local_rank}",
            dtype=torch.bfloat16,
            enabled=self.mixed_precision
        ):
            output = self.model.generate(*args, **kwargs)
        return output
    
    def __getattr__(self, name: str) -> Any:
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self._model, name)

    def __getitem__(self, key: int) -> Any:
        """Forward indexing calls in case the module is an ``nn.Sequential``."""
        return self._model[key]
    
    def backward(self, loss):
        """
        Backward pass with the loss
        """
        if not loss.requires_grad: # loss.grad_fn is None
            return
        if self.mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self):
        """
        Step the optimizer and scheduler.
        """
        assert not self.mode == "inference", "Cannot step optimizer in inference mode."
        # Optimizer step
        if self.mixed_precision:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        # clean up and step
        self.optimizer.zero_grad()
        self.scheduler.step()
        return
    
    def _setup_gradients(self):
        """
        Setup gradients for the model.
        """
        if self.mode == "train":
            setup_gradients(self._model, self.config.OptimizerConfig)
        elif self.mode == "inference":
            logger.info("[TRAINING_WRAPPER] Setting requires_grad=False for inference mode.")
            self._model.requires_grad_(False)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")


    def _compile_model(self):
        """
        Compile the model using torch.compile.
        """
        if not self.compile_model:
            return self._model

        # Set float32 matmul precision
        torch.set_float32_matmul_precision('high')

        kwargs = {"backend": "inductor", "mode": "default", "fullgraph": True, "options": None}
        # if self.mode == "train":
        #     # kwargs["fullgraph"] = True
        #     # kwargs["options"] = {"triton.cudagraphs": True}
        #     pass
        # elif self.mode == "inference":
        #     pass
        # else:
        #     raise ValueError(f"Invalid mode: {self.mode}")
        
        self._model = torch.compile(self._model, **kwargs)
        logger.info(f"[TRAINING_WRAPPER] Model compiled with torch.compile:\n"
                    f"\tBackend: {kwargs['backend']}\n"
                    f"\tMode: {kwargs['mode']}\n"
                    f"\tFullgraph: {kwargs.get('fullgraph', False)}\n"
                    f"\tOptions: {kwargs.get('options', {})}"
                    )
        
    def _setup_optimizer(self):
        assert hasattr(self, "_model"), "Model not initialized."

        if not self.mode == "train":
            return
        
        # setup optimizer and scheduler
        opt_config = self.config.OptimizerConfig
        total_grad_steps = int(self.config.TrainConfig["n_batches"] / self.config.TrainConfig["accumulation_steps"])        
        self.optimizer = setup_optimizer(self._model, opt_config)
        self.scheduler = setup_scheduler(self.optimizer, opt_config, total_grad_steps)

    def _setup_mixed_precision(self):        
        if self.mixed_precision and self.model_dtype == "bfloat16":
            logger.warning(
                "Cannot use mixed precision with bfloat16. Setting mixed_precision=False."
            )
            self.mixed_precision = False
            
        if not self.mixed_precision:
            return
        
        if self.mode == "train":
            assert all([p.dtype == torch.float32 for p in self._model.parameters()]), "Model must be in float32 for mixed precision training."

        # Mixed Precision
        self.scaler = torch.amp.GradScaler('cuda')

    def save_weights(self):
        """
        ONLY MASTER SAVES WEIGHTS
        """
        if not is_master:
            return

        return save_weights(
                model_state=self._model.state_dict(),
                opt_state=self.optimizer.state_dict() if self.optimizer else None,
                scheduler_state=self.scheduler.state_dict() if self.scheduler else None,
                config=self.config,
                tmp_dir=self.config.ManagementConfig.paths.tmp_dir,
                save_dir=self.config.ManagementConfig.paths.save_dir,
            )

    def broadcast(self, value, src=0):
        """
        Broadcast a value to all processes.
        """
        if isinstance(value, torch.Tensor):
            torch_dist.broadcast(value, src=src)
        else:
            lst = [value]
            torch_dist.broadcast_object_list(lst, src=src)
            value = lst[0]
        return value
