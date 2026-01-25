import argparse
import csv
import importlib
import os
import pprint
import random
import shutil
import signal
import sys
import time
import warnings
from datetime import datetime

# Set NCCL debug level before any other imports
os.environ["NCCL_DEBUG"] = "WARNING"

# Set wandb to offline mode before any wandb imports
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_DISABLE_CODE"] = "true"
os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_OFFLINE"] = "true"
os.environ["WANDB_ANONYMOUS"] = "allow"

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import wandb
from dataloaders.build_vocab import Vocab
from loguru import logger
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from torch.utils.tensorboard import SummaryWriter
from utils import logger_tools, metric, other_tools


def prepare_all():
    """
    Parse command line arguments and prepare configuration
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="./configs/intention_w_distill.yaml"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debugging mode")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        default="train",
        help="Choose between 'train' or 'test' mode",
    )
    parser.add_argument(
        "--checkpoint",
        "--ckpt",
        type=str,
        default=None,
        help="Checkpoint path for testing or resuming training",
    )
    parser.add_argument(
        "--local-rank",
        type=int,
        default=0,
        help="Local rank for distributed training (auto-set by torch.distributed.launch)"
    )
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # Load config
    if args.config.endswith(".yaml"):
        cfg = OmegaConf.load(args.config)
        # Only set exp_name from config filename if not already defined in config
        if not hasattr(cfg, "exp_name") or cfg.exp_name is None:
            cfg.exp_name = args.config.split("/")[-1][:-5]
    else:
        raise ValueError(
            "Unsupported config file format. Only .yaml files are allowed."
        )

    # Handle resume from checkpoint
    if args.resume:
        cfg.resume_from_checkpoint = args.resume

    # Debug mode settings
    if args.debug:
        cfg.wandb_project = "debug"
        cfg.exp_name = "debug"
        cfg.solver.max_train_steps = 4

    # Process override arguments
    if args.overrides:
        for arg in args.overrides:
            if "=" in arg:
                key, value = arg.split("=")
                try:
                    value = eval(value)
                except:
                    pass
                if key in cfg:
                    cfg[key] = value
                else:
                    try:
                        # Handle nested config with dot notation
                        keys = key.split(".")
                        cfg_node = cfg
                        for k in keys[:-1]:
                            cfg_node = cfg_node[k]
                        cfg_node[keys[-1]] = value
                    except:
                        raise ValueError(f"Key {key} not found in config.")

    # Set up wandb
    if hasattr(cfg, "wandb_key"):
        os.environ["WANDB_API_KEY"] = cfg.wandb_key

    # Create output directories
    save_dir = os.path.join(cfg.output_dir, cfg.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "sanity_check"), exist_ok=True)

    # Save config
    config_path = os.path.join(save_dir, "sanity_check", f"{cfg.exp_name}.yaml")
    with open(config_path, "w") as f:
        OmegaConf.save(cfg, f)

    # Copy source files for reproducibility
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sanity_check_dir = os.path.join(save_dir, "sanity_check")
    output_dir = os.path.abspath(cfg.output_dir)

    def is_in_output_dir(path):
        return os.path.abspath(path).startswith(output_dir)

    def should_copy_file(file_path):
        if is_in_output_dir(file_path):
            return False
        if "__pycache__" in file_path:
            return False
        if file_path.endswith(".pyc"):
            return False
        return True

    # Copy Python files
    for root, dirs, files in os.walk(current_dir):
        if is_in_output_dir(root):
            continue

        for file in files:
            if file.endswith(".py"):
                full_file_path = os.path.join(root, file)
                if should_copy_file(full_file_path):
                    relative_path = os.path.relpath(full_file_path, current_dir)
                    dest_path = os.path.join(sanity_check_dir, relative_path)
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    try:
                        shutil.copy(full_file_path, dest_path)
                    except Exception as e:
                        print(f"Warning: Could not copy {full_file_path}: {str(e)}")

    return cfg, args


def seed_everything(seed):
    """
    Set random seeds for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@logger.catch
def main_worker(rank, world_size, cfg, args):
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    
    if hasattr(args, 'local_rank'):
        rank = args.local_rank
    
    dist.init_process_group(backend=os.environ.get("TORCH_DISTRIBUTED_BACKEND", "nccl"), rank=rank, world_size=world_size)

    logger_tools.set_args_and_logger(cfg, rank)
    seed_everything(cfg.seed)
    other_tools.print_exp_info(cfg)

    # Initialize trainer
    logger.info(f"Initializing CustomTrainer on GPU {rank}...")
    cfg.rank = rank
    trainer = __import__(
        f"trainer.generative_trainer", fromlist=["something"]
    ).CustomTrainer(cfg, args)
    logger.info(f"CustomTrainer initialized successfully on GPU {rank}")

    # Resume logic
    resume_epoch = 0
    if args.resume:
        # Find the checkpoint path
        if os.path.isdir(args.resume):
            ckpt_path = os.path.join(args.resume, "ckpt.pth")
        else:
            ckpt_path = args.resume
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        trainer.load_checkpoint(checkpoint)
        resume_epoch = checkpoint.get("epoch", 0) + 1  # Start from next epoch
        logger.info(
            f"Resumed from checkpoint {ckpt_path}, starting at epoch {resume_epoch}"
        )

    if args.mode == "train" and not args.resume:
        logger.info("Training from scratch ...")
    elif args.mode == "train" and args.resume:
        logger.info(f"Resuming training from checkpoint {args.resume} ...")
    elif args.mode == "test":
        logger.info("Testing ...")

    # Early stopping initialization
    early_stopping_enabled = cfg.solver.get("early_stopping", {}).get("enabled", False)
    if early_stopping_enabled:
        patience = cfg.solver.early_stopping.get("patience", 10)
        delta = cfg.solver.early_stopping.get("delta", 0.001)
        early_stopping_counter = 0
        # Initialize previous best metrics with current val_best
        best_metrics = trainer.val_best.copy()
        logger.info(f"Early stopping enabled with patience={patience}, delta={delta}")
    else:
        # 初始化变量以避免在广播时出现 NameError
        early_stopping_counter = 0
        patience = float('inf')  # 设置为无穷大，确保不会触发早停

    if args.mode == "train":
        start_time = time.time()
        for epoch in range(resume_epoch, cfg.solver.epochs + 1):
            if cfg.ddp:
                sampler = getattr(trainer.train_loader, 'sampler', None)
                if sampler and hasattr(sampler, 'set_epoch'):
                    sampler.set_epoch(epoch)

            if (epoch) % cfg.val_period == 0:
                # 分布式环境下，同步所有进程
                if cfg.ddp:
                    dist.barrier()
                
                if rank == 0:
                    if cfg.data.test_clip:
                        trainer.test_clip(epoch)
                    else:
                        trainer.val(epoch)

                    # Early stopping check
                    if early_stopping_enabled:
                        # Check all non-gt metrics for improvement
                        all_no_improve = True
                        
                        # Check fgd (lower is better)
                        if trainer.val_best["fgd"]["value"] < best_metrics["fgd"]["value"] - delta:
                            all_no_improve = False
                        
                        # Check bc (higher is better)
                        if trainer.val_best["bc"]["value"] > best_metrics["bc"]["value"] + delta:
                            all_no_improve = False
                        
                        # Check l1div (higher is better)
                        if trainer.val_best["l1div"]["value"] > best_metrics["l1div"]["value"] + delta:
                            all_no_improve = False
                        
                        # Check test_clip_fgd if applicable (lower is better)
                        if "test_clip_fgd" in trainer.val_best and "test_clip_fgd" in best_metrics:
                            if trainer.val_best["test_clip_fgd"]["value"] < best_metrics["test_clip_fgd"]["value"] - delta:
                                all_no_improve = False
                        
                        if all_no_improve:
                            early_stopping_counter += 1
                            logger.info(f"Early stopping counter: {early_stopping_counter}/{patience}")
                            if early_stopping_counter >= patience:
                                logger.info(f"Early stopping triggered after {epoch} epochs")
                                # Save final checkpoint
                                trainer.save_checkpoint(
                                    epoch=epoch,
                                    iteration=epoch * len(trainer.train_loader),
                                    is_best=False,
                                    best_metric_name=None
                                )
                                # 早停触发，跳过后续评估，直接进入广播阶段
                        else:
                            early_stopping_counter = 0
                            # Update best metrics if any improvement
                            best_metrics = trainer.val_best.copy()
                            logger.info("Early stopping counter reset")
                
                # 分布式环境下，广播早停信号给所有进程
                if cfg.ddp:
                    # 主进程决定是否需要早停
                    if rank == 0 and early_stopping_enabled:
                        should_stop = 1 if early_stopping_counter >= patience else 0
                    else:
                        should_stop = 0
                    
                    early_stop_signal = torch.tensor([should_stop], dtype=torch.int, device=trainer.model.device)
                    dist.broadcast(early_stop_signal, src=0)
                    
                    if early_stop_signal.item() == 1:
                        logger.info(f"Received early stopping signal from rank 0, exiting after epoch {epoch}")
                        return  # Exit training loop

            epoch_time = time.time() - start_time
            if trainer.rank == 0:
                logger.info(
                    f"Time info >>>> elapsed: {epoch_time/60:.2f} mins\t"
                    + f"remain: {(cfg.solver.epochs/(epoch+1e-7)-1)*epoch_time/60:.2f} mins"
                )

            if epoch != cfg.solver.epochs:
                if cfg.ddp:
                    trainer.train_loader.sampler.set_epoch(epoch)
                trainer.tracker.reset()
                trainer.train(epoch)

            if cfg.debug and rank == 0:
                trainer.test(epoch)

        # Final cleanup and logging
        if rank == 0:
            for k, v in trainer.val_best.items():
                logger.info(f"Best {k}: {v['value']:.6f} at epoch {v['epoch']}")

            wandb.finish()
    elif args.mode == "test" and not cfg.data.test_clip:
        trainer.test(999)
    elif args.mode == "test" and cfg.data.test_clip:
        trainer.test_clip(999)


if __name__ == "__main__":
    import socket

    if os.environ.get("LOCAL_RANK") is not None or os.environ.get("WORLD_SIZE") is not None:
        master_addr = "127.0.0.1"
        master_port = int(os.environ.get("MASTER_PORT", "29500"))
        logger.info(f"Using torch.distributed.launch environment: MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")
    else:
        master_addr = "127.0.0.1"
        master_port = 29500

        def is_port_in_use(port, host="127.0.0.1"):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind((host, port))
                    return False
                except socket.error:
                    return True

        while is_port_in_use(master_port):
            print(f"Port {master_port} is in use, trying next port...")
            master_port += 1

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["NCCL_SOCKET_IFNAME"] = "lo"

    cfg, args = prepare_all()

    if cfg.ddp:
        if os.environ.get("LOCAL_RANK") is not None or os.environ.get("WORLD_SIZE") is not None:
            world_size = int(os.environ.get("WORLD_SIZE", len(cfg.gpus)))
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            main_worker(local_rank, world_size, cfg, args)
        else:
            mp.set_start_method("spawn", force=True)
            mp.spawn(
                main_worker,
                args=(len(cfg.gpus), cfg, args),
                nprocs=len(cfg.gpus),
            )
    else:
        main_worker(0, 1, cfg, args)
