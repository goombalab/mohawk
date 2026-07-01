from functools import partial

import torch
import torch.distributed as torch_dist
import torch.distributed.fsdp as fsdp_module
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)

from distill.utils import setup_optimizer, setup_scheduler, setup_gradients, load_scheduler
from utils.distributed import barrier, is_master, local_rank, world_size
from utils.logging import logger

from .BaseTrainingWrapper import BaseTrainingWrapper
from utils.weights_utils import save_weights
import os


class FSDPTrainingWrapper(BaseTrainingWrapper):
    def __init__(
        self,
        config,
        compile_model: bool = False,
        mixed_precision: bool = False,
        model_dtype: str = "float32",
        weights_from_rank0: bool = False,
        cpu_offload: bool = False,
        *args,
        **kwargs,
    ):
        BaseTrainingWrapper.__init__(self, *args, **kwargs)
        # 0. Initialize
        self.config = config
        self.compile_model = compile_model
        self.mixed_precision = mixed_precision
        self.model_dtype = model_dtype
        self.weights_from_rank0 = weights_from_rank0
        self.cpu_offload = cpu_offload
        self.use_orig_params = False
        if isinstance(self.model_dtype, str):
            self.model_dtype = getattr(torch, self.model_dtype)
        if self.mixed_precision and self.model_dtype == torch.bfloat16:
            logger.warning("Cannot use mixed precision with bfloat16. Setting mixed_precision=False.")
            self.mixed_precision = False
        # 1. Setup gradients (before FSDP)
        self._setup_gradients()
        # 2. Compile model (before FSDP, after setup_gradients)
        self._compile_model()
        # 3. setup FSDP
        self._model = self._setup_fsdp()
        # 4. setup train mode
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
        # See CentralizedTrainingWrapper.__call__: no_grad in inference mode is
        # belt-and-braces against accidentally building an autograd graph
        # through a teacher when an upstream tensor has requires_grad=True.
        if self.mode == "inference":
            with torch.no_grad():
                return self._model(*args, **kwargs)
        return self._model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """
        Generate method for the model.
        """
        assert hasattr(self, "_model"), "Model not initialized."
        assert isinstance(self._model, FSDP), "Model must be wrapped with FSDP."
        return self._model.generate(*args, **kwargs)

    def no_sync(self):
        if isinstance(self._model, FSDP):
            if not getattr(self, "_logged_no_sync", False):
                logger.info("[TRAINING_WRAPPER] Using FSDP no_sync for gradient accumulation.")
                self._logged_no_sync = True
            return self._model.no_sync()
        return super().no_sync()

    def backward(self, loss):
        """
        Backward pass with the loss
        """
        loss.backward()
        # Gradient Clipping (need to mind)
        # self._model.clip_grad_norm_(max_norm=1.0)

    def step(self):
        """
        Step the optimizer and scheduler.
        """
        assert not self.mode == "inference", "Cannot step optimizer in inference mode."
        # Optimizer step
        self.optimizer.step()
        # clean up and step
        self.optimizer.zero_grad()
        self.scheduler.step()
        return
    
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

    def _compile_model(self):
        """
        Compile the model using torch.compile.
        """
        if not self.compile_model:
            return self._model
                
        assert not isinstance(self._model, FSDP), "Model should not be wrapped with FSDP when compiling."

        if not self.use_orig_params:
            logger.warning(
                "Model compilation requires using original parameters. Setting use_orig_params=True."
            )
            self.use_orig_params = True

        # Set float32 matmul precision
        torch.set_float32_matmul_precision('high')

        # Compile model
        kwargs = {"backend": "inductor", "mode": "default", "fullgraph": False, "options": None}
        if self.mode == "train":
            # kwargs["fullgraph"] = True
            # kwargs["options"] = {"triton.cudagraphs": True}
            pass
        elif self.mode == "inference":
            pass
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        
        self._model = torch.compile(self._model, **kwargs)
        logger.info(f"[TRAINING_WRAPPER] Model compiled with torch.compile:\n"
                    f"\tBackend: {kwargs['backend']}\n"
                    f"\tMode: {kwargs['mode']}\n"
                    f"\tFullgraph: {kwargs.get('fullgraph', False)}\n"
                    f"\tOptions: {kwargs.get('options', {})}"
                    )


    def _setup_fsdp(self):
        assert torch_dist.is_initialized(), "Distributed environment not initialized."
        assert hasattr(self, "_model"), "Model not initialized."
        activation_checkpointing = self.config.TrainConfig.get("activation_checkpointing", True)
        sharding_strategy = self.config.TrainConfig.get(
            "sharding_strategy", "HYBRID_SHARD"
        )
        kwargs = {}

        # DEVICE ID:
        kwargs["device_id"] = torch.cuda.current_device()
        
        # SHARDING STRATEGY:
        kwargs["sharding_strategy"] = fsdp_module.ShardingStrategy[sharding_strategy]

        # MIXED PRECISION:
        if self.mixed_precision:
            kwargs["mixed_precision"] = fsdp_module.MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
                # cast_forward_inputs=True,
            )

        # Wrap for good sharding. Probe common transformer layouts; HF GPT-2
        # exposes its blocks at .transformer.h, our LayeredMambaLM at
        # .backbone.layers, Llama-style models at .model.layers.
        if hasattr(self._model, "backbone") and hasattr(self._model.backbone, "layers"):
            layers = self._model.backbone.layers
        elif hasattr(self._model, "model") and hasattr(self._model.model, "layers"):
            layers = self._model.model.layers
        elif hasattr(self._model, "transformer") and hasattr(self._model.transformer, "h"):
            layers = self._model.transformer.h
        else:
            layers = None

        block_cls = layers[0].__class__ if layers else None
        if block_cls is not None:
            kwargs["auto_wrap_policy"] = partial(
                fsdp_module.wrap.transformer_auto_wrap_policy,
                transformer_layer_cls={block_cls},
            )
        else:
            logger.warning(
                "[TRAINING_WRAPPER] Could not infer transformer block class; "
                "FSDP will wrap the whole module."
            )

        # USE ORIGINAL PARAMS?
        # Cannot flatten params if some are not trainable and some are
        if len(set([p.requires_grad for p in self._model.parameters()])) > 1:
            logger.warning(
                "Cannot flatten params if some are not trainable and some are."
                " Setting use_orig_params=True."
            )
            self.use_orig_params = True
        kwargs["use_orig_params"] = self.use_orig_params
        
        if self.weights_from_rank0:
            # SYNC MODULE STATES
            # If True, then each FSDP module will broadcast module parameters and buffers from rank 0 
            # to ensure that they are replicated across ranks (adding communication overhead to this constructor). 
            # This can help load state_dict checkpoints via load_state_dict in a memory efficient way. 
            kwargs["sync_module_states"] = True

            # PARAM INIT FN
            # If provided, this function will be called on each parameter of the module after the module is replicated.
            # This can be useful for initializing parameters if not all ranks are intialized.
            def init_fn(x: torch.nn.Module):
                if local_rank != 0:
                    device = torch.cuda.current_device()
                    return x.to_empty(device=device, recurse=False)
                else:
                    return x
            kwargs["param_init_fn"] = init_fn
            # kwargs["param_init_fn"] = lambda x: x.to_empty(device=torch.cuda.current_device(), recurse=False) if local_rank != 0 else x
        
        # CPU OFFLOAD
        if self.cpu_offload:
            kwargs["cpu_offload"] = fsdp_module.CPUOffload(offload_params=True)

        # SET FSDP
        self._model = FSDP(module=self._model, **kwargs)

        # ACTIVATION CHECKPOINTING
        if activation_checkpointing and block_cls is not None:
            non_reentrant_wrapper = partial(
                checkpoint_wrapper,
                offload_to_cpu=False,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )
            apply_activation_checkpointing(
                self._model,
                checkpoint_wrapper_fn=non_reentrant_wrapper,
                check_fn=lambda submodule: isinstance(submodule, block_cls),
            )
        elif activation_checkpointing:
            logger.warning(
                "[TRAINING_WRAPPER] activation_checkpointing requested but no "
                "block class was inferred; skipping."
            )

        # Support FSDP generation:
        FSDP._sample = lambda self, *args, **kwargs: type(self.module)._sample(self, *args, **kwargs)
        FSDP.generate = lambda self, *args, **kwargs: type(self.module).generate(self, *args, **kwargs)

        logger.info(f"[TRAINING_WRAPPER] FSDP setup complete:\n"
                    f"\tModel dtype: {self.model_dtype}\n"
                    f"\tMixed Precision: {self.mixed_precision}\n"
                    f"\tSharding Strategy: {sharding_strategy}\n"
                    f"\tCPU Offload: {self.cpu_offload}\n"
                    f"\tUse original params: {self.use_orig_params}\n"
                    f"\tActivation Checkpointing: {activation_checkpointing}"
                    )

        return self._model

    def _setup_train_mode(self):
        assert hasattr(self, "_model"), "Model not initialized."
        assert isinstance(self._model, FSDP), "Optimizer must be initialized *AFTER* FSDP setup."
        # Set model to train mode
        self._model.train()
        # setup optimizer and scheduler
        # NOTE: Optimizer MUST be initialized AFTER model and parallelism are set
        total_grad_steps = int(self.config.TrainConfig["n_batches"] / self.config.TrainConfig["accumulation_steps"])        
        self.optimizer = setup_optimizer(self._model, self.config.OptimizerConfig)
        self.scheduler = setup_scheduler(self.optimizer, self.config.OptimizerConfig, total_grad_steps)

        load_cfg = getattr(self.config, "LoadConfig", None)
        if load_cfg is None:
            return
        if hasattr(load_cfg, "optimizer"):
            path = load_cfg.optimizer.path
            full_osd = torch.load(f=os.path.join(path, "optimizer.pth"), map_location="cpu") if local_rank == 0 else None
            sharded_osd = FSDP.scatter_full_optim_state_dict(full_osd, self._model)
            self.optimizer.load_state_dict(sharded_osd)
        if hasattr(load_cfg, "scheduler"):
            scheduler_state = load_scheduler(load_cfg.scheduler)
            configured_num_steps = self.scheduler.num_steps
            self.scheduler.load_state_dict(scheduler_state)
            if self.scheduler._step_count > configured_num_steps:
                raise ValueError(
                    f"Loaded scheduler step {self.scheduler._step_count} exceeds "
                    f"configured total steps {configured_num_steps}."
                )
            self.scheduler.num_steps = configured_num_steps

        # FSDP.scatter_full_optim_state_dict(self.optimizer, self._model)

    def save_weights(self):
        """
        ALL PROCESSES MATERIALIZE WEIGHTS
        ONLY MASTER SAVES WEIGHTS
        """
        full_state_config = fsdp_module.FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        full_optim_config = fsdp_module.FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        barrier()
        with FSDP.state_dict_type(
            module=self._model,
            state_dict_type=fsdp_module.StateDictType.FULL_STATE_DICT,
            state_dict_config=full_state_config,
            optim_state_dict_config=full_optim_config,
        ):
            # Full model weights
            full_model_state = self._model.state_dict()
            full_model_state = {k.replace("module.", ""): v for k, v in full_model_state.items()} # for torch.fsdp
            full_model_state = {k.replace("_orig_mod.", ""): v for k, v in full_model_state.items()} # for torch.compile

            # Full optimizer state
            full_osd = FSDP.full_optim_state_dict(self._model, self.optimizer)
            barrier()

            if is_master:
                save_weights(
                    model_state=full_model_state,
                    opt_state=full_osd,
                    scheduler_state=self.scheduler.state_dict(),
                    config=self.config,
                    tmp_dir=self.config.ManagementConfig.paths.tmp_dir,
                    save_dir=self.config.ManagementConfig.paths.save_dir,
                )

            barrier()
        return self.config.ManagementConfig.paths.save_dir


    def send_to_master(self, value):
        """
        Send a value to the master process (rank 0).
        For Tensors, it reduces them to rank 0 using SUM operation.
        For other types, it sends them using object gathering.
        """

        if isinstance(value, torch.Tensor):
            # Reduce tensor to rank 0
            torch_dist.reduce(value, dst=0, op=torch_dist.ReduceOp.SUM)
            if local_rank == 0:
                return value  # Only rank 0 gets the reduced value
            else:
                return None
        else:
            # Handle non-tensor data types
            lst = [None] * world_size if local_rank == 0 else None
            torch_dist.gather_object(value, lst, dst=0)
            if local_rank == 0:
                return lst  # Rank 0 gets a list of all values
            else:
                return None
    
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

    def gather(self, value):
        """
        Gather a value from all processes and return as a single concatenated tensor.
        """
        if isinstance(value, torch.Tensor):
            gathered = [torch.zeros_like(value) for _ in range(world_size)]
            torch_dist.all_gather(gathered, value)
            if value.dim() == 0:
                gathered = [v.unsqueeze(0) for v in gathered]
            return torch.cat(gathered, dim=0)
        else:
            lst = [None] * world_size
            torch_dist.all_gather_object(lst, value)
            return torch.tensor(lst)
