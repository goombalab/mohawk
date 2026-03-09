import time
import torch
import torch.distributed as torch_dist

from dataloaders import setup_dataloader

from utils.distributed import local_rank, world_size
from utils.logging import logger, wandb
from evals.evaluator import Evaluator
from utils.utils import set_seed
from utils.config import Config
from distill.distill_steps.registry import Registry
from utils.init_model import get_init_fn, lazy_init

def find_batch_size(cfg):
    """
    Find the batch size that is divisible by the effective batch size and world size.
    """
    effective_batch_size = cfg.TrainConfig.effective_batch_size
    batch_size = cfg.TrainDataConfig.TorchDataLoader.batch_size

    # Make sure the effective batch size is divisible by world size
    bs = effective_batch_size
    while bs % world_size != 0:
        bs += batch_size
    if effective_batch_size != bs:
        logger.warning(
            f"[TRAIN] effective_batch_size {effective_batch_size} is not divisible by world_size {world_size}. "
            f"Increasing effective_batch_size to {bs}."
        )
    effective_batch_size = bs

    # Make sure the batch size is divisible by the effective batch size and world size
    bs = batch_size
    while effective_batch_size % (bs * world_size) != 0:
        bs -= 1
    if batch_size != bs:
        logger.warning(
            f"[TRAIN] Batch size {batch_size} is not divisible by effective_batch_size {effective_batch_size} and world_size {world_size}. "
            f"Reducing batch_size to {bs}."
        )
    batch_size = bs

    assert (effective_batch_size % (batch_size * world_size) == 0)
    assert (effective_batch_size >= world_size)
    cfg.TrainConfig.effective_batch_size = effective_batch_size
    cfg.TrainDataConfig.TorchDataLoader.batch_size = batch_size
    return

def distill(cfg):

    distill_step = Registry(cfg.DistillConfig)

    # Empty cache
    torch.cuda.empty_cache()

    # Set up logging TAG
    TAG = (cfg.DistillConfig.name + "-" + cfg.DistillConfig.type).upper()

    # Set seed
    set_seed(cfg.TrainConfig.seed)
    
    find_batch_size(cfg)

    # Setup dataloaders
    train_dataloader = setup_dataloader(data_cfg=cfg.TrainDataConfig)
    if hasattr(cfg.LoadConfig, "dataloader"):
        train_dataloader.load_state_dict(torch.load(cfg.LoadConfig.dataloader.path + "/dataloader_state_dict.pth"))
    
    dataloader_iter = iter(train_dataloader)
    assert isinstance(next(dataloader_iter), dict), "Dataloader must return a dictionary"

    # Set variables
    batch_size = train_dataloader.batch_size
    
    # Set accumulation steps
    accumulation_steps = cfg.TrainConfig.effective_batch_size // (batch_size * world_size)
    cfg.TrainConfig["accumulation_steps"] = accumulation_steps
            
    # Set n_batches
    cfg.TrainConfig["n_batches"] = int(cfg.TrainConfig.n_tokens / (world_size * batch_size * train_dataloader.max_seq_len))

    # STUDENT
    logger.info("[TRAIN] Loading student model...")
    init_fn = get_init_fn(cfg.TrainConfig.get("init_fn", "lazy")) # lazy_init or eager_init
    student_wrapper = init_fn(
        components_cfg=cfg.ComponentsConfig,
        details_cfg=cfg.TrainConfig,
        load_cfg=cfg.LoadConfig,
        cfg=cfg,
        mode="train",
    )
    
    # TEACHER
    logger.info("[TRAIN] Loading teacher model...")
    teacher_wrapper = lazy_init(
        details_cfg=cfg.TeacherConfig,
        load_cfg=Config.from_dict({"model": [{"path": cfg.TeacherConfig.dir}]}),
        cfg=cfg,
        mode="inference",
    )

    # Evaluator
    evaluator = Evaluator(
        cfg, 
        dataloader=train_dataloader, 
        student_wrapper=student_wrapper, 
        teacher_wrapper=teacher_wrapper
        )

    # Start training
    logger.info(f"[{TAG}] Training started.")
    times, scores = [], []
    scheduler = student_wrapper.scheduler
    for step in range(scheduler._step_count, scheduler.num_steps):

        # Forward & Backward
        start = time.time()
        for _ in range(accumulation_steps):
            batch = next(dataloader_iter)
            score = distill_step(batch, student_wrapper, teacher_wrapper, cfg, tokenizer=train_dataloader.tokenizer)
            scores.extend(score)
        
        # Step
        student_wrapper.step()

        # Record time & score
        duration = torch.tensor(time.time() - start).to(local_rank)
        avg_score = torch.tensor(scores).mean().to(local_rank)
        if torch_dist.is_initialized():
            torch_dist.all_reduce(avg_score, op=torch_dist.ReduceOp.AVG)
            torch_dist.all_reduce(duration, op=torch_dist.ReduceOp.AVG)

        times.append(duration)

        # Print metrics
        if step % 100 == 0:
            avg_step_time = torch.tensor(times).mean().item()
            logger.info(
                f"[{TAG}] Step {step}/{scheduler.num_steps} - AVG_SCORE {avg_score:.6f} - AVG_STEP_TIME {avg_step_time:.4f}s"
            )
            times = []

        # Evaluate
        metrics = evaluator.eval()

        # Record metrics
        metrics["Train/_"+cfg.DistillConfig.type] = metrics["Train/"+cfg.DistillConfig.type] = avg_score.item()
        metrics["Train/lr"] = student_wrapper.optimizer.param_groups[-1]["lr"]
        metrics["Data/samples_seen"] = (step + 1) * world_size * accumulation_steps
        metrics["Data/tokens_seen"] = metrics["Data/samples_seen"] * batch["input_ids"].numel()
        metrics["Data/tokens_seen (M)"] =  metrics["Data/tokens_seen"] / 1e6
        wandb.log(metrics)

        # Get ready for the next step
        scores = []
        evaluator.increment_step()

    metrics = evaluator.eval(last=True)
    wandb.log(metrics)

    wandb.define_metric("Data/tokens_seen")
    # wandb.define_metric("_eval_perplexity", step_metric="Data/tokens_seen")
    wandb.define_metric("Train/_"+cfg.DistillConfig.type, step_metric="Data/tokens_seen")
    logger.info(f'[TRAIN] Training finished.\nWeights saved in {cfg.ManagementConfig.paths.save_dir}')
