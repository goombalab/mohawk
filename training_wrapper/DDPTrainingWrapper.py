import os

import torch
import torch.distributed as torch_dist
from torch.nn.parallel.distributed import DistributedDataParallel, _MixedPrecision

from distill.utils import setup_optimizer, setup_scheduler, setup_gradients
from utils.distributed import is_master, local_rank
from utils.logging import logger

from .BaseTrainingWrapper import BaseTrainingWrapper
from utils.weights_utils import save_weights

class DDPTrainingWrapper(BaseTrainingWrapper):
    def __init__(
        self,
        config,
        compile_model=False,
        model_dtype=torch.float32,
        weights_from_rank0=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.config = config
        self.model_dtype = model_dtype
        self.weights_from_rank0 = weights_from_rank0
        if isinstance(self.model_dtype, str):
            self.model_dtype = getattr(torch, self.model_dtype)
        # 1. Setup gradients (before DDP, if in train mode)
        if self.mode == "train":
            self._setup_gradients()
        # 2. Broadcast weights from rank 0 if needed (before moving to device)
        if self.weights_from_rank0:
            self._broadcast_weights_from_rank0()
        # 3. Move model to device (before DDP)
        self._model = self._model.to(local_rank)
        # 4. Compile model (before DDP)
        self.compile_model = compile_model
        if self.compile_model:
            self._model = self._compile_model(self._model)
        # 5. setup DDP
        self._model = self._setup_ddp()
        # 6. setup train mode
        if self.mode == "train":
            self._setup_train_mode()
        else:
            self._model.eval()

        # Clean up
        torch.cuda.empty_cache()

    @property
    def model(self):
        return self._model

    @property
    def module(self):
        return self._model.module

    def __call__(self, *args, **kwargs):
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            return self.model(*args, **kwargs)
    
    def backward(self, loss):
        """
        Backward pass with the loss
        """
        assert not self.mode == "inference", "Cannot backward in inference mode."
        # Backward pass
        self.scaler.scale(loss).backward()
        # loss.backward()

    def step(self):
        """
        Step the optimizer and scheduler.
        """
        assert not self.mode == "inference", "Cannot step optimizer in inference mode."
        # Optimizer step
        # self.optimizer.step()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        # clean up and step
        self.optimizer.zero_grad()
        self.scheduler.step()
        return

    def _compile_model(self, model):
        """
        Compile the model using torch.compile.
        """
        if not self.compile_model:
            return model
        assert not isinstance(model, DistributedDataParallel), "Model should not be wrapped with DDP when compiling."
        # Set float32 matmul precision
        torch.set_float32_matmul_precision('high')
        # Compile model
        kwargs = {"backend": "inductor", "mode": "default", "fullgraph": False}
        self._model = torch.compile(model, **kwargs)
        logger.info(f"[TRAINING_WRAPPER] Model compiled with torch.compile")
        return self._model

    def _setup_ddp(self):
        assert torch_dist.is_initialized(), "Distributed environment not initialized."
        assert hasattr(self, "_model"), "Model not initialized."
        # Ensure model is on the correct device (check first parameter)
        first_param = next(self._model.parameters(), None)
        if first_param is not None:
            assert first_param.device.type == "cuda", f"Model must be on CUDA device, got {first_param.device}"
            assert first_param.device.index == local_rank, f"Model must be on device {local_rank}, got {first_param.device.index}"
        
        kwargs = {}
        kwargs["device_ids"] = [local_rank]
        kwargs["output_device"] = local_rank

        # Mixed precision scaler (for AMP)
        if self.mode == "train":
            self.scaler = torch.cuda.amp.GradScaler()

        # TODO: Check if this is necessary:
        # kwargs["find_unused_parameters"] = True
        # kwargs["static_graph"] = True
        # kwargs["gradient_as_bucket_view"] = True

        return DistributedDataParallel(self._model, **kwargs)

    def _setup_gradients(self):
        """
        Setup gradients for the model.
        """
        logger.info("[TRAINING_WRAPPER] Setting up gradients.")
        if self.mode == "train":
            setup_gradients(self._model, self.config.OptimizerConfig)
        elif self.mode == "inference":
            self._model.requires_grad_(False)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def _setup_train_mode(self):
        assert hasattr(self, "_model"), "Model not initialized."
        assert isinstance(self._model, DistributedDataParallel), "Model must be wrapped with DDP."

        # Set model to train mode
        self._model.train()
        # setup optimizer and scheduler
        # NOTE: Optimizer MUST be initialized AFTER model and parallelism are set
        total_grad_steps = int(self.config.TrainConfig["n_batches"] / self.config.TrainConfig.get("accumulation_steps", 1))
        self.optimizer = setup_optimizer(self._model, self.config.OptimizerConfig)
        self.scheduler = setup_scheduler(self.optimizer, self.config.OptimizerConfig, total_grad_steps)

    def _broadcast_weights_from_rank0(self):
        """
        Broadcast model weights from rank 0 to all ranks.
        Similar to FSDP's weights_from_rank0 functionality.
        """
        assert torch_dist.is_initialized(), "Distributed environment not initialized."
        assert not isinstance(
            self._model, DistributedDataParallel
        ), "Model should not be wrapped with DDP when broadcasting weights."
        
        if local_rank == 0:
            # Rank 0: get state dict
            state_dict = self._model.state_dict()
        else:
            # Other ranks: create empty state dict with same structure
            state_dict = {k: torch.empty_like(v) for k, v in self._model.state_dict().items()}
        
        # Broadcast all tensors from rank 0
        for key in state_dict.keys():
            torch_dist.broadcast(state_dict[key], src=0)
        
        # Load state dict on all ranks
        self._model.load_state_dict(state_dict)
        del state_dict
        logger.info("[TRAINING_WRAPPER] Broadcasted weights from rank 0 to all ranks.")

    def save_weights(self):
        """
        ONLY MASTER SAVES WEIGHTS
        """
        if not is_master:
            return

        # Get state dict from the underlying module (unwrap DDP)
        model_state = self._model.module.state_dict()
        # Remove 'module.' prefix if present (shouldn't be, but just in case)
        model_state = {k.replace("module.", ""): v for k, v in model_state.items()}
        # Remove '_orig_mod.' prefix if model was compiled
        model_state = {k.replace("_orig_mod.", ""): v for k, v in model_state.items()}
        
        # Get optimizer and scheduler states
        opt_state = self.optimizer.state_dict() if hasattr(self, 'optimizer') and self.optimizer is not None else None
        scheduler_state = self.scheduler.state_dict() if hasattr(self, 'scheduler') and self.scheduler is not None else None
        
        # Get config from the underlying module
        config = getattr(self._model.module, 'config', None)
        if config is None:
            logger.warning("[TRAINING_WRAPPER] Model does not have a config attribute. Using config from wrapper.")
            config = self.config

        return save_weights(
            model_state=model_state,
            opt_state=opt_state,
            scheduler_state=scheduler_state,
            config=config,
            tmp_dir=self.config.ManagementConfig.paths.tmp_dir,
            save_dir=self.config.ManagementConfig.paths.save_dir,
        )
