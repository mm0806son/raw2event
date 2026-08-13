"""Train QKFormer on event data from any of the three modalities.

Trains the QKFormer classifier on the dv/raw/rgb event datasets, from scratch or
fine-tuned from a checkpoint, with optional Weights & Biases tracking.

Example:
    python train_class/train_qkformer.py --data_dir ./data/unified80 \
        --modality dv --T 16 --epochs 96 --seed 0
"""

import datetime
import os
import random
import sys
import time
import math
import yaml
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
from torch.cuda import amp
import torch.distributed as dist
from torchvision import transforms

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
QKFORMER_DIR = os.path.join(SCRIPT_DIR, "QKFormer", "cifar10-dvs")
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Put the upstream QKFormer code on the import path.
sys.path.insert(0, QKFORMER_DIR)
sys.path.insert(0, ROOT_DIR)

from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven import neuron as sj_neuron
from timm.models import create_model
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer


# ---------------------------------------------------------------------------
# SpikingJelly backend compatibility patch
# ---------------------------------------------------------------------------
def _patch_lif_backend_when_cupy_missing():
    """
    QKFormer upstream hardcodes backend='cupy' in many MultiStepLIFNode calls.
    For CPU-only or no-CuPy environments, transparently fall back to backend='torch'.
    """
    if getattr(sj_neuron, "cupy", None) is not None:
        return

    if getattr(sj_neuron.MultiStepLIFNode, "_raw2event_backend_patched", False):
        return

    original_init = sj_neuron.MultiStepLIFNode.__init__

    def _patched_init(self, *args, **kwargs):
        if kwargs.get("backend") == "cupy":
            kwargs["backend"] = "torch"
        return original_init(self, *args, **kwargs)

    sj_neuron.MultiStepLIFNode.__init__ = _patched_init
    sj_neuron.MultiStepLIFNode._raw2event_backend_patched = True
    print("  CuPy unavailable; MultiStepLIFNode backend falls back to torch.")


_patch_lif_backend_when_cupy_missing()

# Upstream QKFormer modules (model.py, utils.py, autoaugment.py).
import model as qkformer_model  # noqa: F401  — registers into the timm registry
import utils
import autoaugment

# Local datasets.
sys.path.insert(0, SCRIPT_DIR)
from train_utils.dataset import (
    EventNpzDataset,
    split_dataset,
    split_dataset_deterministic,
    split_dataset_deterministic_with_val,
    CIFAR10_CLASSES,
)

# ---------------------------------------------------------------------------
# Seeding
#
# The seed used to be hard-coded to 2024, so multi-seed matrix runs with
# SEEDS="0 1 2" only differed in PYTHONHASHSEED
# and produced statistically identical RNG streams. We now expose a real
# `--seed` argparse flag (default 2024 to preserve historical behaviour
# when no flag is passed) and seed Python `random`, NumPy, torch CPU, and
# all CUDA devices from a single helper. The cuDNN determinism toggles
# below are left at module load time because they do not depend on the
# seed value and were already in this file.
# ---------------------------------------------------------------------------
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def _apply_seed(seed: int) -> None:
    """Seed every RNG used in this training script with `seed`.

    Called from `main()` AFTER `parse_args()` so the value comes from
    `--seed`. Under `torch.distributed.run --nproc_per_node=N` each rank
    is a separate Python process that independently runs `main()`, so
    every rank applies the same seed independently — matching the prior
    module-level behaviour of `_seed_ = 2024`.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # Use manual_seed_all so DDP child ranks all get a deterministic
        # CUDA RNG state, not just the current device.
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# WandB configuration
# ---------------------------------------------------------------------------
def load_wandb_config(config_path: str = None) -> dict:
    """
    Load the WandB config, preferring an untracked ``.local`` override.
    """
    if config_path is None:
        config_path = os.path.join(SCRIPT_DIR, "wandb_env.yaml")

    # Prefer the .local override.
    local_path = config_path.replace(".yaml", ".local.yaml")
    if os.path.exists(local_path):
        config_path = local_path

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {
        "enabled": True,
        "project": "raw2event-qkformer",
        "entity": "",
        "mode": "online",
        "tags": [],
    }


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------
def train_one_epoch(
    model,
    criterion,
    optimizer,
    data_loader,
    device,
    epoch,
    print_freq,
    scaler=None,
    T_train=None,
    aug=None,
    trival_aug=None,
    mixup_fn=None,
):
    """Train one epoch; returns (avg_loss, avg_acc1, avg_acc5)."""
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        image = image.float()  # [N, T, C, H, W]
        N, T, C, H, W = image.shape

        if aug is not None:
            image = torch.stack([aug(image[i]) for i in range(N)])

        if trival_aug is not None:
            image = torch.stack([trival_aug(image[i]) for i in range(N)])

        if mixup_fn is not None:
            image, target = mixup_fn(image, target)
            target_for_compu_acc = target.argmax(dim=-1)

        if T_train:
            sec_list = np.random.choice(image.shape[1], T_train, replace=False)
            sec_list.sort()
            image = image[:, sec_list]

        if scaler is not None:
            with amp.autocast():
                output = model(image)
                loss = criterion(output, target)
        else:
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        functional.reset_net(model)

        if mixup_fn is not None:
            acc1, acc5 = utils.accuracy(output, target_for_compu_acc, topk=(1, 5))
        else:
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

        batch_size = image.shape[0]
        loss_s = loss.item()
        if math.isnan(loss_s):
            raise ValueError("loss is NaN")

        metric_logger.update(loss=loss_s, lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))

    metric_logger.synchronize_between_processes()
    return (
        metric_logger.loss.global_avg,
        metric_logger.acc1.global_avg,
        metric_logger.acc5.global_avg,
    )


def evaluate(model, criterion, data_loader, device, print_freq=100, header="Test:"):
    """Evaluate the model; returns (avg_loss, avg_acc1, avg_acc5)."""
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True).float()
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)
            functional.reset_net(model)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    loss = metric_logger.loss.global_avg
    acc1 = metric_logger.acc1.global_avg
    acc5 = metric_logger.acc5.global_avg
    print(f" * Acc@1 = {acc1:.2f}, Acc@5 = {acc5:.2f}, loss = {loss:.4f}")
    return loss, acc1, acc5


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Train QKFormer on event data")

    # ---- Project-specific ----
    parser.add_argument(
        "--modality",
        type=str,
        default="dv",
        choices=["dv", "raw", "rgb"],
        help="event modality",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/unified80",
        help="directory holding the NPZ files",
    )
    parser.add_argument(
        "--finetune", action="store_true", help="fine-tune from pretrained weights"
    )
    parser.add_argument(
        "--pretrained_weights", type=str, default="", help="path to a pretrained .pth"
    )
    parser.add_argument(
        "--wandb_config", type=str, default=None, help="path to a WandB config file"
    )
    parser.add_argument("--no_wandb", action="store_true", help="disable WandB logging")

    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help=(
            "Master RNG seed. Propagates to Python random, numpy, "
            "torch CPU, torch.cuda (manual_seed_all). Default 2024 "
            "preserves historical behaviour when --seed is omitted."
        ),
    )

    parser.add_argument(
        "--wandb_run_id",
        type=str,
        default=os.environ.get("WANDB_RUN_ID", ""),
        help="WandB run id to resume into",
    )
    parser.add_argument(
        "--wandb_resume",
        type=str,
        default=os.environ.get("WANDB_RESUME", "allow"),
        help="WandB resume policy; only used with --wandb_run_id",
    )

    # ---- Data split ----
    parser.add_argument(
        "--split_mode",
        type=str,
        default="deterministic",
        choices=["deterministic", "random"],
        help="deterministic = fixed per-class slices; random = stratified random",
    )
    parser.add_argument(
        "--train_per_class",
        type=int,
        default=0,
        help="per-class training samples for the deterministic split; 0 uses the rest",
    )
    parser.add_argument(
        "--test_per_class",
        type=int,
        default=600,
        help="per-class test samples for the deterministic split",
    )
    parser.add_argument(
        "--val_per_class",
        type=int,
        default=0,
        help=(
            "per-class validation samples; > 0 enables the 3-way split where "
            "val drives checkpoint selection and test stays held out. "
            "0 keeps the 2-way split."
        ),
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="train fraction for the random split",
    )

    # ---- Upstream QKFormer ----
    parser.add_argument(
        "--num-classes", type=int, default=10, help="number of classes"
    )
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", "--batch_size", default=16, type=int)
    parser.add_argument("-j", "--workers", default=4, type=int, help="dataloader workers")
    parser.add_argument("--print-freq", default=64, type=int)
    parser.add_argument(
        "--output-dir", default="./train_class/output", help="base directory for checkpoints and logs"
    )
    parser.add_argument(
        "--resume",
        default="",
        help="resume from a previous run directory or a specific .pth",
    )
    parser.add_argument(
        "--amp", default=True, action="store_true", help="enable mixed-precision training"
    )
    parser.add_argument("--T", default=16, type=int, help="SNN simulation time steps")
    parser.add_argument(
        "--T_train", default=None, type=int, help="time steps randomly sampled during training"
    )

    # ---- Optimizer ----
    parser.add_argument("--opt", default="adamw", type=str)
    parser.add_argument("--opt-eps", default=1e-8, type=float)
    parser.add_argument("--opt-betas", default=None, type=float)
    parser.add_argument("--weight-decay", default=0.06, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)

    # ---- LR schedule ----
    parser.add_argument("--sched", default="cosine", type=str)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--lr-noise", type=float, nargs="+", default=None)
    parser.add_argument("--lr-noise-pct", type=float, default=0.67)
    parser.add_argument("--lr-noise-std", type=float, default=1.0)
    parser.add_argument("--lr-cycle-mul", type=float, default=1.0)
    parser.add_argument("--lr-cycle-limit", type=int, default=1)
    parser.add_argument("--warmup-lr", type=float, default=1e-5)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=96)
    parser.add_argument("--epoch-repeats", type=float, default=0.0)
    parser.add_argument("--start-epoch", default=0, type=int)
    parser.add_argument("--decay-epochs", type=float, default=20)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--cooldown-epochs", type=int, default=10)
    parser.add_argument("--patience-epochs", type=int, default=10)
    parser.add_argument("--decay-rate", type=float, default=0.1)

    # ---- Augmentation ----
    parser.add_argument("--mixup", type=float, default=0.5)
    parser.add_argument("--cutmix", type=float, default=0.0)
    parser.add_argument("--cutmix-minmax", type=float, nargs="+", default=None)
    parser.add_argument("--mixup-prob", type=float, default=1.0)
    parser.add_argument("--mixup-switch-prob", type=float, default=0.5)
    parser.add_argument("--mixup-mode", type=str, default="batch")
    parser.add_argument("--smoothing", type=float, default=0.1)

    # Distributed interface, not currently enabled.
    parser.add_argument("--world-size", default=1, type=int)
    parser.add_argument("--dist-url", default="env://")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(args):
    # ---- Distributed init; skipped on a single device ----
    utils.init_distributed_mode(args)

    # ---- Apply RNG seed ----
    # Each rank runs main() in its own process, so calling _apply_seed
    # here gives every rank an identical RNG starting state — exactly
    # what the previous hard-coded `_seed_ = 2024` produced.
    _apply_seed(args.seed)
    print(f"[seed] using args.seed={args.seed}", flush=True)

    print(f"\n{'=' * 60}")
    print(
        f"  QKFormer | modality: {args.modality} | "
        f"mode: {'finetune' if args.finetune else 'scratch'}"
    )
    print(f"{'=' * 60}\n")
    print(args)

    # ---- Timestamped output directory ----
    mode_tag = "finetune" if args.finetune else "scratch"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # When resuming, recover the previous output directory from the path.
    if args.resume:
        if os.path.isdir(args.resume):
            # A directory was given; look for the latest checkpoint.
            output_dir = args.resume
            args.resume = os.path.join(
                output_dir, f"output_{args.modality}_{mode_tag}_latest.pth"
            )
        else:
            output_dir = os.path.dirname(args.resume)
        print(f"  Resuming; reusing directory {output_dir}")
    else:
        output_dir = os.path.join(
            args.output_dir,
            f"QKFormer_{args.modality}_{mode_tag}_T{args.T}_{timestamp}",
        )
        if utils.is_main_process():
            os.makedirs(output_dir, exist_ok=True)

    if args.distributed:
        dist.barrier()

    if args.distributed:
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device(args.device)

    # ---- Load data and persist the split ----
    print("\nLoading dataset...")
    full_dataset = EventNpzDataset(
        data_dir=args.data_dir,
        modality=args.modality,
        T=args.T,
        target_h=128,
        target_w=128,
    )

    split_info_path = os.path.join(output_dir, "split_info.json")
    import json

    # val_dataset stays None for the legacy 2-way split; only the 3-way
    # split (--val_per_class > 0) populates it. It gates best-ckpt selection.
    val_dataset = None
    if args.resume and os.path.exists(split_info_path):
        # On resume, reuse the stored split indices.
        print(f"  Reading stored split: {split_info_path}")
        with open(split_info_path, "r") as f:
            split_info = json.load(f)
        train_indices = split_info["train_indices"]
        test_indices = split_info["test_indices"]
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
        # Older checkpoints predate the 3-way split and carry no val_indices.
        val_idx = split_info.get("val_indices")
        if val_idx:
            val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
        print(
            f"  Split restored: train={len(train_dataset)}, "
            f"val={len(val_dataset) if val_dataset is not None else 0}, "
            f"test={len(test_dataset)}"
        )
    else:
        # First run: build the split according to --split_mode.
        if args.split_mode == "deterministic":
            if args.val_per_class > 0:
                # 3-way split: val drives checkpoint selection, test stays held
                # out. The test slice matches the 2-way split exactly, so
                # cross-modal evaluation is unaffected.
                train_dataset, val_dataset, test_dataset = (
                    split_dataset_deterministic_with_val(
                        full_dataset,
                        val_per_class=args.val_per_class,
                        test_per_class=args.test_per_class,
                    )
                )
            else:
                train_dataset, test_dataset = split_dataset_deterministic(
                    full_dataset,
                    train_per_class=args.train_per_class,
                    test_per_class=args.test_per_class,
                )
        else:
            train_dataset, test_dataset = split_dataset(
                full_dataset, train_ratio=args.train_ratio, seed=42
            )
        if utils.is_main_process():
            print(f"  Split written to: {split_info_path}")
            with open(split_info_path, "w") as f:
                json.dump(
                    {
                        "split_mode": args.split_mode,
                        "train_per_class": args.train_per_class
                        if args.split_mode == "deterministic"
                        else None,
                        "val_per_class": args.val_per_class
                        if (args.split_mode == "deterministic" and val_dataset is not None)
                        else None,
                        "test_per_class": args.test_per_class
                        if args.split_mode == "deterministic"
                        else None,
                        "train_ratio": args.train_ratio
                        if args.split_mode == "random"
                        else None,
                        "train_indices": train_dataset.indices,
                        "val_indices": (
                            val_dataset.indices if val_dataset is not None else None
                        ),
                        "test_indices": test_dataset.indices,
                    },
                    f,
                )

    if args.distributed:
        dist.barrier()

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True
        )
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset, shuffle=False
        )
        val_sampler = (
            torch.utils.data.distributed.DistributedSampler(
                val_dataset, shuffle=False
            )
            if val_dataset is not None
            else None
        )
    else:
        train_sampler = None
        test_sampler = None
        val_sampler = None

    # Per-DataLoader Generator pinned to args.seed so that the train
    # loader's shuffle order is reproducible when no DistributedSampler
    # is used. Under DDP the DistributedSampler owns the shuffle, so
    # this generator only governs any auxiliary RNG draws DataLoader
    # itself does (negligible, but deterministic now).
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)

    data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.workers,
        drop_last=(len(train_dataset) > args.batch_size),
        pin_memory=True,
        generator=loader_generator,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=args.workers,
        drop_last=False,
        pin_memory=True,
    )
    # Validation loader (only for the 3-way split). Used SOLELY for best-ckpt
    # selection so the test set stays truly held out.
    data_loader_val = None
    if val_dataset is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.workers,
            drop_last=False,
            pin_memory=True,
        )

    # ---- Build the model ----
    print("\nBuilding QKFormer...")
    # Instantiate directly rather than through timm.create_model: recent timm
    # injects extra kwargs such as pretrained_cfg_overlay that the vit_snn
    # constructor rejects.
    from model import vit_snn
    from functools import partial

    model = vit_snn(
        patch_size=16,
        embed_dims=256,
        num_heads=16,
        mlp_ratios=1,
        in_channels=2,
        num_classes=args.num_classes,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=4,
        sr_ratios=1,
        drop_rate=0.0,
        drop_path_rate=0.1,
    )

    # ---- Load pretrained weights when fine-tuning ----
    if args.finetune and args.pretrained_weights:
        print(f"  Loading pretrained weights: {args.pretrained_weights}")
        checkpoint = torch.load(args.pretrained_weights, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("model", checkpoint)

        # The upstream CIFAR10-DVS head has a different class count.
        head_key = "head.weight"
        if head_key in state_dict:
            if state_dict[head_key].shape[0] != args.num_classes:
                print(
                    f"  Head size mismatch (checkpoint={state_dict[head_key].shape[0]}, "
                    f"target={args.num_classes}); skipping the head weights"
                )
                del state_dict[head_key]
                if "head.bias" in state_dict:
                    del state_dict["head.bias"]

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")
    elif args.finetune and not args.pretrained_weights:
        print("  --finetune given without --pretrained_weights; training from scratch.")

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_parameters:,}")
    model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # ---- Loss and optimizer ----
    criterion_train = SoftTargetCrossEntropy().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(args, model)

    scaler = amp.GradScaler() if args.amp else None
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)

    # ---- Resume ----
    # Best-ckpt selection consults VAL acc when a val loader exists (3-way
    # split, --val_per_class > 0); falls back to TEST acc for legacy 2-way runs.
    best_metric_name = "val_acc1" if data_loader_val is not None else "test_acc1"
    max_best_acc1 = 0.0
    test_acc1_at_best = 0.0  # test acc snapshot at the epoch scoring max_best_acc1
    test_acc5_at_best = 0.0
    # Legacy aliases kept so downstream tooling / wandb summaries keep reading them.
    max_test_acc1 = 0.0
    test_acc5_at_max_test_acc1 = 0.0
    start_epoch = args.start_epoch

    if args.resume:
        print(f"  Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        max_test_acc1 = checkpoint.get("max_test_acc1", 0.0)
        test_acc5_at_max_test_acc1 = checkpoint.get("test_acc5_at_max_test_acc1", 0.0)
        max_best_acc1 = checkpoint.get("max_best_acc1", max_test_acc1)
        test_acc1_at_best = checkpoint.get("test_acc1_at_best", max_test_acc1)
        test_acc5_at_best = checkpoint.get("test_acc5_at_best", test_acc5_at_max_test_acc1)

    # ---- WandB ----
    wandb_run = None
    if not args.no_wandb and utils.is_main_process():
        try:
            import wandb

            wb_cfg = load_wandb_config(args.wandb_config)
            if wb_cfg.get("enabled", True):
                run_name = f"{args.modality}_{mode_tag}_T{args.T}_lr{args.lr}"
                wandb_init_kwargs = dict(
                    project=wb_cfg.get("project", "raw2event-qkformer"),
                    entity=wb_cfg.get("entity", None) or None,
                    name=run_name,
                    mode=wb_cfg.get("mode", "online"),
                    tags=wb_cfg.get("tags", []) + [args.modality, mode_tag],
                    config=vars(args),
                    dir=output_dir,
                )
                if args.wandb_run_id:
                    wandb_init_kwargs["id"] = args.wandb_run_id
                    wandb_init_kwargs["resume"] = args.wandb_resume
                    print(f"  WandB resuming: id={args.wandb_run_id}, resume={args.wandb_resume}")
                wandb_run = wandb.init(**wandb_init_kwargs)
                print(f"  WandB connected: {wandb_run.url}")
        except ImportError:
            print("  wandb not installed; skipping online logging.")
        except Exception as e:
            print(f"  WandB init failed: {e}")

    # ---- Augmentation ----
    train_snn_aug = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5)])
    train_trivalaug = autoaugment.SNNAugmentWide()

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.num_classes,
        )

    # ---- Training loop ----
    print(f"\nTraining epochs {start_epoch} to {num_epochs}")
    start_time = time.time()

    for epoch in range(start_epoch, num_epochs):
        save_max = False

        if args.distributed:
            train_sampler.set_epoch(epoch)

        # Disable Mixup for the final epochs, as upstream does.
        if mixup_fn is not None and epoch >= int(num_epochs * 0.78):
            mixup_fn.mixup_enabled = False

        train_loss, train_acc1, train_acc5 = train_one_epoch(
            model,
            criterion_train,
            optimizer,
            data_loader,
            device,
            epoch,
            args.print_freq,
            scaler,
            args.T_train,
            train_snn_aug,
            train_trivalaug,
            mixup_fn,
        )
        lr_scheduler.step(epoch)

        test_loss, test_acc1, test_acc5 = evaluate(
            model, criterion, data_loader_test, device
        )
        # Validation drives best-ckpt selection when present; test stays a
        # pure logging signal so it is never peeked at for model selection.
        if data_loader_val is not None:
            val_loss, val_acc1, val_acc5 = evaluate(
                model, criterion, data_loader_val, device, header="Val:"
            )
            best_signal = val_acc1
        else:
            val_loss = val_acc1 = val_acc5 = None
            best_signal = test_acc1  # legacy 2-way fallback

        # ---- Logging ----
        if wandb_run is not None:
            import wandb

            log_dict = {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/acc1": train_acc1,
                "train/acc5": train_acc5,
                "test/loss": test_loss,
                "test/acc1": test_acc1,
                "test/acc5": test_acc5,
                "lr": optimizer.param_groups[0]["lr"],
            }
            if data_loader_val is not None:
                log_dict.update(
                    {
                        "val/loss": val_loss,
                        "val/acc1": val_acc1,
                        "val/acc5": val_acc5,
                    }
                )
            wandb.log(log_dict)

        # ---- Update the best record; val drives selection, test is logged only ----
        if best_signal > max_best_acc1:
            max_best_acc1 = best_signal
            test_acc1_at_best = test_acc1
            test_acc5_at_best = test_acc5
            # Legacy aliases = test snapshot at the best epoch (== max test acc
            # in the 2-way case, so old behaviour is preserved bit-for-bit).
            max_test_acc1 = test_acc1_at_best
            test_acc5_at_max_test_acc1 = test_acc5_at_best
            save_max = True

        # ---- Checkpointing ----
        checkpoint_data = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "best_metric_name": best_metric_name,
            "max_best_acc1": max_best_acc1,
            "test_acc1_at_best": test_acc1_at_best,
            "test_acc5_at_best": test_acc5_at_best,
            # legacy keys kept for external tooling that reads them
            "max_test_acc1": max_test_acc1,
            "test_acc5_at_max_test_acc1": test_acc5_at_max_test_acc1,
        }

        # Always keep the latest checkpoint.
        latest_path = os.path.join(
            output_dir, f"output_{args.modality}_{mode_tag}_latest.pth"
        )
        utils.save_on_master(checkpoint_data, latest_path)

        # Keep the best checkpoint.
        if save_max:
            best_path = os.path.join(
                output_dir, f"output_{args.modality}_{mode_tag}_best.pth"
            )
            utils.save_on_master(checkpoint_data, best_path)

        if data_loader_val is not None:
            print(
                f"  📈 Epoch {epoch}: Val Acc@1={val_acc1:.2f}% "
                f"(Best Val={max_best_acc1:.2f}%) | Test Acc@1={test_acc1:.2f}% (logging only)"
            )
        else:
            print(
                f"  📈 Epoch {epoch}: Test Acc@1={test_acc1:.2f}% (Best={max_best_acc1:.2f}%)"
            )

    # ---- Done ----
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    print(f"\n{'=' * 60}")
    print("  Training complete.")
    print(f"  Modality: {args.modality} | mode: {mode_tag}")
    print(f"  Best Test Acc@1: {max_test_acc1:.2f}%")
    print(f"  Test Acc@5 at that epoch: {test_acc5_at_max_test_acc1:.2f}%")
    print(f"  Total time: {total_time_str}")
    print(f"  Weights written to: {output_dir}")
    print(f"{'=' * 60}")

    if wandb_run is not None:
        import wandb

        wandb.summary["best_test_acc1"] = max_test_acc1
        wandb.summary["best_test_acc5"] = test_acc5_at_max_test_acc1
        wandb.finish()

    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    main(args)
