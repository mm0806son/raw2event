"""
Train MobileNetV2 on event data from any of the three modalities.

Consumes the dv/raw/rgb NPZ files written by ``process_all_batches``.
``--representation`` selects ``timestack`` (polarity discarded, events summed
into one count map replicated to three channels) or ``stacked_histogram``
(``2*T`` channels, polarity split across T time bins). Training runs from
scratch by default, matching the QKFormer protocol.

Persists its data split, supports resume, and writes best and latest
checkpoints.
"""

import argparse
import datetime
import glob
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import yaml
from torch.cuda import amp
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

import sys

sys.path.insert(0, SCRIPT_DIR)
# Repo root too, so ``tools.*`` helpers are importable from this entry point.
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from train_utils.dataset import (
    split_dataset,
    split_dataset_deterministic,
    split_dataset_deterministic_with_val,
)
from train_utils.event_augmentation import AUGMENTATION_MODES
from train_utils.mobile_dataset import EventCountImageDataset, EventStackedHistogramDataset


# Determinism flags stay module-level so they apply before any import path
# constructs CUDA streams.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def _apply_seed(seed: int) -> None:
    """Seed every RNG used in this trainer with ``seed``.

    Mirrors ``train_qkformer._apply_seed``.
    Called once from main() after argparse, before any randomness
    (dataset shuffle, split, model init) happens.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_wandb_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = os.path.join(SCRIPT_DIR, "wandb_env.yaml")

    local_path = config_path.replace(".yaml", ".local.yaml")
    if os.path.exists(local_path):
        config_path = local_path

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        "enabled": True,
        "project": "raw2event-mobilenetv2",
        "entity": "",
        "mode": "online",
        "tags": [],
    }


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    maxk = min(max(topk), output.shape[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    results = []
    for k in topk:
        k = min(k, output.shape[1])
        correct_k = correct[:k].reshape(-1).float().sum(0)
        results.append(correct_k.mul_(100.0 / batch_size))
    return results


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value: float, n: int = 1):
        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


class SequentialLRScheduler:
    """Warmup, then cosine decay, then a small constant tail."""

    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        cosine_end_epoch: int,
        final_lr_ratio: float,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.cosine_end_epoch = cosine_end_epoch
        self.final_lr_ratio = final_lr_ratio
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.last_epoch = -1

    def _factor(self, epoch: int) -> float:
        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            return float(epoch + 1) / float(self.warmup_epochs)
        if epoch < self.cosine_end_epoch:
            cosine_span = max(self.cosine_end_epoch - self.warmup_epochs, 1)
            cos_inner = math.pi * (epoch - self.warmup_epochs) / cosine_span
            return float((math.cos(cos_inner) + 1.0) / 2.0)
        return self.final_lr_ratio

    def step(self, epoch: int):
        self.last_epoch = epoch
        factor = self._factor(epoch)
        for base_lr, param_group in zip(self.base_lrs, self.optimizer.param_groups):
            param_group["lr"] = base_lr * factor

    def state_dict(self):
        return {
            "base_lrs": self.base_lrs,
            "last_epoch": self.last_epoch,
            "warmup_epochs": self.warmup_epochs,
            "cosine_end_epoch": self.cosine_end_epoch,
            "final_lr_ratio": self.final_lr_ratio,
        }

    def load_state_dict(self, state_dict):
        self.base_lrs = state_dict["base_lrs"]
        self.last_epoch = state_dict["last_epoch"]
        self.warmup_epochs = state_dict["warmup_epochs"]
        self.cosine_end_epoch = state_dict["cosine_end_epoch"]
        self.final_lr_ratio = state_dict["final_lr_ratio"]
        if self.last_epoch >= 0:
            self.step(self.last_epoch)


class MobileNetV2Classifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
        dropout: float = 0.5,
        hidden_dim: int = 128,
        in_channels: int = 3,
    ):
        """MobileNetV2 backbone adapted for variable input channels.

        For ``in_channels != 3`` the first conv (which expects RGB) is
        replaced with a freshly-initialised Conv2d that keeps the same
        kernel / stride / padding / output channels but accepts the new
        channel count. With ``pretrained=False`` (project default for the
        cross-modal experiments) this is lossless; with
        ``pretrained=True`` the first-layer ImageNet weights are discarded,
        which is fine here because the pipeline always trains from scratch.
        """
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v2(weights=weights)

        if in_channels != 3:
            old_conv = backbone.features[0][0]
            new_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=(old_conv.bias is not None),
            )
            backbone.features[0][0] = new_conv

        self.in_channels = in_channels
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(backbone.last_channel, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def build_optimizer(
    model: nn.Module,
    lr: float,
    momentum: float,
    base_weight_decay: float,
    dense_l2: float,
):
    dense_weight = model.classifier[0].weight
    dense_bias = model.classifier[0].bias
    handled = {id(dense_weight), id(dense_bias)}

    other_decay = []
    other_no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in handled:
            continue
        if param.ndim == 1 or name.endswith(".bias"):
            other_no_decay.append(param)
        else:
            other_decay.append(param)

    param_groups = [
        {"params": other_decay, "weight_decay": base_weight_decay},
        {"params": other_no_decay, "weight_decay": 0.0},
        {"params": [dense_weight], "weight_decay": dense_l2},
        {"params": [dense_bias], "weight_decay": 0.0},
    ]
    return torch.optim.SGD(param_groups, lr=lr, momentum=momentum)


def train_one_epoch(
    model, criterion, optimizer, data_loader, device, epoch, print_freq, scaler=None
):
    model.train()
    use_amp = scaler is not None and scaler.is_enabled()

    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    for step, (images, target) in enumerate(data_loader):
        start_time = time.time()
        images = images.to(device, non_blocking=True).float()
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with amp.autocast():
                output = model(images)
                loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(images)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = images.shape[0]
        loss_meter.update(loss.item(), batch_size)
        acc1_meter.update(acc1.item(), batch_size)
        acc5_meter.update(acc5.item(), batch_size)

        if step % max(print_freq, 1) == 0:
            img_per_s = batch_size / max(time.time() - start_time, 1e-6)
            print(
                f"Epoch [{epoch}] Step [{step}/{len(data_loader)}] "
                f"loss={loss_meter.avg:.4f} acc1={acc1_meter.avg:.2f} "
                f"acc5={acc5_meter.avg:.2f} img/s={img_per_s:.2f}"
            )

    return loss_meter.avg, acc1_meter.avg, acc5_meter.avg


@torch.no_grad()
def evaluate(model, criterion, data_loader, device):
    model.eval()

    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    for images, target in data_loader:
        images = images.to(device, non_blocking=True).float()
        target = target.to(device, non_blocking=True)

        output = model(images)
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        loss_meter.update(loss.item(), batch_size)
        acc1_meter.update(acc1.item(), batch_size)
        acc5_meter.update(acc5.item(), batch_size)

    print(
        f" * Acc@1 = {acc1_meter.avg:.2f}, Acc@5 = {acc5_meter.avg:.2f}, loss = {loss_meter.avg:.4f}"
    )
    return loss_meter.avg, acc1_meter.avg, acc5_meter.avg


def resolve_resume_checkpoint(resume_path: str) -> tuple[str, str]:
    if os.path.isdir(resume_path):
        candidates = sorted(glob.glob(os.path.join(resume_path, "output_*_latest.pth")))
        if not candidates:
            raise FileNotFoundError(f"no latest checkpoint under '{resume_path}'")
        return resume_path, candidates[0]
    return os.path.dirname(resume_path), resume_path


def parse_args():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 on event data")
    parser.add_argument(
        "--modality", type=str, default="dv", choices=["dv", "raw", "rgb"]
    )
    parser.add_argument(
        "--data_dir", type=str, default="./data/unified80"
    )
    parser.add_argument("--wandb_config", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--resume", default="", help="resume from a checkpoint file or an output directory")
    parser.add_argument(
        "--frozen_splits",
        default="",
        help=(
            "Path to a frozen train/val/test JSON. When given, training refuses "
            "to start unless the computed split matches it exactly, in order. "
            "Use it whenever two runs must be compared as a matched pair."
        ),
    )

    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("-b", "--batch_size", default=16, type=int)
    parser.add_argument("-j", "--workers", default=4, type=int)
    parser.add_argument("--print_freq", default=20, type=int)
    parser.add_argument("--output_dir", default=os.path.join(SCRIPT_DIR, "output"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use ImageNet-pretrained weights",
    )

    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--dense_dropout", type=float, default=0.5)
    parser.add_argument("--dense_l2", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)

    parser.add_argument("--target_h", type=int, default=128)
    parser.add_argument("--target_w", type=int, default=128)
    parser.add_argument("--output_channels", type=int, default=3)
    parser.add_argument("--input_size", type=int, default=224)
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
        "--train_ratio", type=float, default=0.9, help="train fraction for the random split"
    )
    parser.add_argument("--warmup_epochs", type=int, default=20)
    parser.add_argument("--cosine_end_epoch", type=int, default=80)
    parser.add_argument("--final_lr_ratio", type=float, default=0.01)
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="RNG seed for python / numpy / torch (default 2024 matches the "
        "pre-2026-06 single-seed behaviour of this script)",
    )
    parser.add_argument(
        "--representation",
        type=str,
        default="timestack",
        choices=["timestack", "stacked_histogram"],
        help="Event representation. 'timestack' = polarity-stripped, time-pooled "
        "count image (3 channels via replicate; default, preserves legacy "
        "behaviour). 'stacked_histogram' = RVT-style polarity-split time-binned "
        "histogram (2*T channels, where T = --rep_T).",
    )
    parser.add_argument(
        "--rep_T",
        type=int,
        default=10,
        help="Number of time bins for the stacked_histogram representation "
        "(RVT default = 10). Ignored when --representation=timestack.",
    )
    parser.add_argument(
        "--rep_count_cutoff",
        type=int,
        default=10,
        help="Saturating per-(pixel,bin,polarity) event count (RVT default = 10). "
        "Caps the histogram so a few high-density pixels do not dominate. "
        "Set <=0 to disable; ignored when --representation=timestack.",
    )
    parser.add_argument(
        "--augmentation",
        type=str,
        default="none",
        choices=list(AUGMENTATION_MODES),
        help="Raw-event-layer training augmentation (applied ONLY to the "
        "train split; val/test always use the clean dataset). 'none' = "
        "baseline (default). 'eventdrop' = EventDrop random/time/area drop "
        "(IJCAI 2021). 'flip_h' = horizontal mirror p=0.5. 'eventdrop_flip' "
        "= both. 'noise_inject' = add background/hot-pixel noise events "
        "(sim-completion). 'polarity_rebalance' = random polarity flips "
        "(sim-completion; targets RAW OFF-channel coverage deficit).",
    )
    parser.add_argument(
        "--val_per_class",
        type=int,
        default=60,
        help="Per-class validation samples held aside from train and used only "
        "for best-checkpoint selection. Set 0 for the 2-way split where test "
        "doubles as val, which is best-of-N biased and kept only for protocol "
        "matching.",
    )

    return parser.parse_args()


def main(args):
    _apply_seed(args.seed)

    # Honour rep_count_cutoff <= 0 as "disabled" so the CLI exposes a
    # single integer knob (no None ambiguity).
    if args.rep_count_cutoff is not None and args.rep_count_cutoff <= 0:
        args.rep_count_cutoff = None

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; falling back to CPU")
        args.device = "cpu"

    mode_tag = "imagenet" if args.pretrained else "scratch"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Per-run subdir name encodes the representation so timestack and
    # stacked_histogram variants of the same (modality, seed) live side-
    # by-side without overwriting each other's checkpoints.
    if args.representation == "stacked_histogram":
        rep_tag = f"STH_T{args.rep_T}_C{args.rep_count_cutoff}"
    else:
        rep_tag = "TS1281281"

    # Encode augmentation in the run dir so augmented ckpts never collide with
    # the existing baseline ckpts. "none" keeps the legacy naming (empty
    # suffix) so prior baseline runs/paths are unaffected.
    aug_tag = "" if args.augmentation == "none" else f"_aug-{args.augmentation}"

    if args.resume:
        output_dir, args.resume = resolve_resume_checkpoint(args.resume)
        print(f"  Resuming; reusing directory {output_dir}")
    else:
        output_dir = os.path.join(
            args.output_dir,
            f"MobileNetV2_{args.modality}_{mode_tag}_{rep_tag}{aug_tag}_{timestamp}",
        )
        os.makedirs(output_dir, exist_ok=True)

    device = torch.device(args.device)

    print(f"\n{'=' * 60}")
    print(f"  MobileNetV2 | modality: {args.modality} | pretrained: {args.pretrained}")
    print(f"{'=' * 60}\n")
    print(args)

    print("\nLoading dataset...")

    def build_event_dataset(augmentation: str):
        """Construct the representation dataset with a given augmentation.

        Used twice so the train split can draw from an augmented instance
        while val/test draw from a clean (augmentation='none') instance —
        the two instances share an identical file_list/labels, hence an
        identical deterministic split, so the augmentation never leaks into
        evaluation.
        """
        if args.representation == "stacked_histogram":
            return EventStackedHistogramDataset(
                data_dir=args.data_dir,
                modality=args.modality,
                T=args.rep_T,
                target_h=args.target_h,
                target_w=args.target_w,
                output_size=args.input_size,
                count_cutoff=args.rep_count_cutoff,
                normalize=True,
                augmentation=augmentation,
                aug_seed=args.seed,
            )
        return EventCountImageDataset(
            data_dir=args.data_dir,
            modality=args.modality,
            target_h=args.target_h,
            target_w=args.target_w,
            output_channels=args.output_channels,
            output_size=args.input_size,
            normalize=True,
            augmentation=augmentation,
            aug_seed=args.seed,
        )

    # eval_dataset is always clean; it drives the split and serves val/test.
    eval_dataset = build_event_dataset("none")
    in_channels = 2 * args.rep_T if args.representation == "stacked_histogram" else args.output_channels

    # train_source is augmented only when requested; otherwise reuse the clean
    # instance. Assert the two instances are split-compatible so the shared
    # indices address the same files in both.
    if args.augmentation != "none":
        train_source = build_event_dataset(args.augmentation)
        assert train_source.file_list == eval_dataset.file_list, (
            "augmented and clean dataset file_list mismatch — split indices "
            "would address different files (leakage risk)"
        )
        assert train_source.labels == eval_dataset.labels, (
            "augmented and clean dataset labels mismatch"
        )
    else:
        train_source = eval_dataset

    split_info_path = os.path.join(output_dir, "split_info.json")
    val_dataset = None
    if args.resume and os.path.exists(split_info_path):
        print(f"  Reading stored split: {split_info_path}")
        with open(split_info_path, "r", encoding="utf-8") as f:
            split_info = json.load(f)
        # Guard against silently switching the augmentation protocol on resume:
        # split_info records what the run was trained with. (Legacy records
        # predating augmentation default to "none".)
        recorded_aug = split_info.get("augmentation", "none")
        if recorded_aug != args.augmentation:
            raise ValueError(
                f"resume augmentation mismatch: this run was trained with "
                f"augmentation='{recorded_aug}' but --augmentation="
                f"'{args.augmentation}' was passed. Re-run with "
                f"--augmentation {recorded_aug}, or start a fresh run."
            )
        # train draws from the (possibly augmented) train_source; val/test
        # draw from the clean eval_dataset so resume keeps augmentation off
        # the held-out splits.
        train_dataset = torch.utils.data.Subset(
            train_source, split_info["train_indices"]
        )
        test_dataset = torch.utils.data.Subset(eval_dataset, split_info["test_indices"])
        # Older checkpoints predate the 3-way split and carry no val_indices.
        val_idx = split_info.get("val_indices")
        if val_idx:
            val_dataset = torch.utils.data.Subset(eval_dataset, val_idx)
            print(
                f"  Split restored: train={len(train_dataset)}, val={len(val_dataset)}, "
                f"test={len(test_dataset)}"
            )
        else:
            print(
                f"  Split restored (2-way, no val): train={len(train_dataset)}, "
                f"test={len(test_dataset)}"
            )
    else:
        # First run: build the split according to --split_mode.
        if args.split_mode == "deterministic":
            if args.val_per_class > 0:
                # 3-way split: test stays held out, val drives checkpoint selection.
                train_dataset, val_dataset, test_dataset = (
                    split_dataset_deterministic_with_val(
                        eval_dataset,
                        val_per_class=args.val_per_class,
                        test_per_class=args.test_per_class,
                    )
                )
            else:
                # 2-way split, kept for protocol matching with earlier runs.
                train_dataset, test_dataset = split_dataset_deterministic(
                    eval_dataset,
                    train_per_class=args.train_per_class,
                    test_per_class=args.test_per_class,
                )
        else:
            train_dataset, test_dataset = split_dataset(
                eval_dataset, train_ratio=args.train_ratio, seed=42
            )
        # Re-point the train split at the (possibly augmented) train_source.
        # split_* returns Subsets of eval_dataset; rebuilding with the same
        # indices keeps val/test clean while train gets augmentation.
        if train_source is not eval_dataset:
            train_dataset = torch.utils.data.Subset(
                train_source, train_dataset.indices
            )
        with open(split_info_path, "w", encoding="utf-8") as f:
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
                    "augmentation": args.augmentation,
                    "train_indices": train_dataset.indices,
                    "val_indices": (
                        val_dataset.indices if val_dataset is not None else None
                    ),
                    "test_indices": test_dataset.indices,
                    "representation": (
                        f"stacked_histogram_T{args.rep_T}_C{args.rep_count_cutoff}"
                        if args.representation == "stacked_histogram"
                        else "timestack_1281281"
                    ),
                    "in_channels": in_channels,
                    "input_size": args.input_size,
                    "output_channels": args.output_channels,
                    "rep_T": args.rep_T if args.representation == "stacked_histogram" else None,
                    "rep_count_cutoff": (
                        args.rep_count_cutoff
                        if args.representation == "stacked_histogram"
                        else None
                    ),
                },
                f,
                indent=2,
            )
        print(f"  Split written to: {split_info_path}")

    # ── Frozen-split gate ────────────────────────────────────────────────
    # Placed after BOTH branches on purpose. The split is recomputed from a
    # directory glob on a fresh run and reloaded from split_info.json on
    # resume, and neither path consults the frozen lists — so a recording that
    # failed to generate, or a stale file left in the directory, shifts the
    # split silently and two runs that are reported as a matched pair are
    # trained on different data. The resume path is the more dangerous of the
    # two, because its indices were computed against a directory listing that
    # no longer has to match the current one.
    if getattr(args, "frozen_splits", ""):
        from train_utils import frozen_splits as frozen_splits_mod

        computed = frozen_splits_mod.computed_splits(
            eval_dataset.file_list,
            train_dataset.indices,
            val_dataset.indices if val_dataset is not None else [],
            test_dataset.indices,
            modality=args.modality,
        )
        frozen_splits_mod.assert_bound(
            frozen_splits_mod.load_frozen(args.frozen_splits), computed
        )
        print(
            f"  Split bound to frozen list: {args.frozen_splits} "
            f"(train={len(computed['train'])}, val={len(computed['val'])}, "
            f"test={len(computed['test'])})"
        )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )
    # Validation loader (only when val_dataset is present — 3-way split).
    # Used SOLELY for best-ckpt selection so test stays truly held out.
    val_loader = None
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            drop_last=False,
            pin_memory=(device.type == "cuda"),
        )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    print("\nBuilding MobileNetV2...")
    model = MobileNetV2Classifier(
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        dropout=args.dense_dropout,
        hidden_dim=args.hidden_dim,
        in_channels=in_channels,
    )
    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_parameters:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        model=model,
        lr=args.lr,
        momentum=args.momentum,
        base_weight_decay=args.weight_decay,
        dense_l2=args.dense_l2,
    )
    lr_scheduler = SequentialLRScheduler(
        optimizer=optimizer,
        warmup_epochs=args.warmup_epochs,
        cosine_end_epoch=args.cosine_end_epoch,
        final_lr_ratio=args.final_lr_ratio,
    )
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler(
            "cuda", enabled=(args.amp and device.type == "cuda")
        )
    else:
        scaler = amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    # Best-ckpt selection uses VAL acc when val_loader exists (modern 3-way
    # split, val_per_class > 0); falls back to TEST acc for legacy 2-way
    # split runs to preserve old behaviour.
    best_metric_name = "val_acc1" if val_loader is not None else "test_acc1"
    max_best_acc1 = 0.0
    test_acc1_at_best = 0.0  # test acc snapshot at the epoch that scored max_best_acc1
    test_acc5_at_best = 0.0
    start_epoch = args.start_epoch

    if args.resume:
        print(f"  Resuming from checkpoint: {args.resume}")
        # weights_only=True restricts unpickling to tensors + basic Python
        # types — our checkpoints store only those (model/optimizer/scheduler
        # state_dicts + scalars + vars(args) dict), so this is safe and
        # defends against the pickle-RCE class of attacks.
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        # The augmentation RNG is seeded from args.seed; resuming with a
        # different seed would silently change the augmentation stream.
        ckpt_seed = checkpoint.get("args", {}).get("seed")
        if ckpt_seed is not None and ckpt_seed != args.seed:
            raise ValueError(
                f"resume seed mismatch: checkpoint was trained with seed "
                f"{ckpt_seed} but --seed={args.seed} was passed. Re-run with "
                f"--seed {ckpt_seed} to keep the augmentation stream consistent."
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        # Legacy checkpoints stored "max_test_acc1"; new ones store the
        # name-agnostic "max_best_acc1". Honour both.
        max_best_acc1 = checkpoint.get(
            "max_best_acc1", checkpoint.get("max_test_acc1", 0.0)
        )
        test_acc1_at_best = checkpoint.get(
            "test_acc1_at_best", checkpoint.get("max_test_acc1", 0.0)
        )
        test_acc5_at_best = checkpoint.get(
            "test_acc5_at_best", checkpoint.get("test_acc5_at_max_test_acc1", 0.0)
        )

    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb

            wb_cfg = load_wandb_config(args.wandb_config)
            if wb_cfg.get("enabled", True):
                run_name = (
                    f"mobilenetv2_{args.modality}_{mode_tag}_{rep_tag.lower()}"
                    f"{aug_tag.lower()}_lr{args.lr}"
                )
                wandb_run = wandb.init(
                    project=wb_cfg.get("project", "raw2event-mobilenetv2"),
                    entity=wb_cfg.get("entity", None) or None,
                    name=run_name,
                    mode=wb_cfg.get("mode", "online"),
                    tags=wb_cfg.get("tags", [])
                    + [
                        args.modality,
                        mode_tag,
                        "MobileNetV2",
                        rep_tag,
                        f"aug-{args.augmentation}",
                    ],
                    config=vars(args),
                    dir=output_dir,
                )
                print(f"  WandB connected: {wandb_run.url}")
        except ImportError:
            print("  wandb not installed; skipping online logging.")
        except Exception as exc:
            print(f"  WandB init failed: {exc}")

    print(f"\nTraining epochs {start_epoch} to {args.epochs}")
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        lr_scheduler.step(epoch)

        # Advance the augmentation RNG epoch so each epoch sees a different
        # (but reproducible) draw. No-op for augmentation="none". Subset
        # forwards to the underlying dataset via .dataset.
        if hasattr(train_dataset, "dataset") and hasattr(
            train_dataset.dataset, "set_epoch"
        ):
            train_dataset.dataset.set_epoch(epoch)

        train_loss, train_acc1, train_acc5 = train_one_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            print_freq=args.print_freq,
            scaler=scaler,
        )
        # Always evaluate on test (just for logging; NOT used for best
        # selection when val_loader exists). Evaluate on val too if we
        # have one — val is what actually gates best-ckpt saving.
        test_loss, test_acc1, test_acc5 = evaluate(
            model, criterion, test_loader, device
        )
        if val_loader is not None:
            val_loss, val_acc1, val_acc5 = evaluate(
                model, criterion, val_loader, device
            )
            best_signal = val_acc1
        else:
            val_loss = val_acc1 = val_acc5 = None
            best_signal = test_acc1  # legacy 2-way fallback

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
            if val_loader is not None:
                log_dict.update({
                    "val/loss": val_loss,
                    "val/acc1": val_acc1,
                    "val/acc5": val_acc5,
                })
            wandb.log(log_dict)

        save_max = False
        if best_signal > max_best_acc1:
            max_best_acc1 = best_signal
            test_acc1_at_best = test_acc1
            test_acc5_at_best = test_acc5
            save_max = True

        checkpoint_data = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "best_metric_name": best_metric_name,
            "max_best_acc1": max_best_acc1,
            "test_acc1_at_best": test_acc1_at_best,
            "test_acc5_at_best": test_acc5_at_best,
            # legacy keys kept for any external tooling that reads them
            "max_test_acc1": test_acc1_at_best,
            "test_acc5_at_max_test_acc1": test_acc5_at_best,
        }

        latest_path = os.path.join(
            output_dir, f"output_{args.modality}_{mode_tag}_latest.pth"
        )
        torch.save(checkpoint_data, latest_path)
        if save_max:
            best_path = os.path.join(
                output_dir, f"output_{args.modality}_{mode_tag}_best.pth"
            )
            torch.save(checkpoint_data, best_path)

        if val_loader is not None:
            print(
                f"  📈 Epoch {epoch}: lr={optimizer.param_groups[0]['lr']:.6f} "
                f"Val Acc@1={val_acc1:.2f}% (Best={max_best_acc1:.2f}%)  "
                f"Test Acc@1={test_acc1:.2f}% (held-out)"
            )
        else:
            print(
                f"  📈 Epoch {epoch}: lr={optimizer.param_groups[0]['lr']:.6f} "
                f"Test Acc@1={test_acc1:.2f}% (Best={max_best_acc1:.2f}%, legacy "
                "test-as-val selection)"
            )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    print(f"\n{'=' * 60}")
    print("  Training complete.")
    print(f"  Modality: {args.modality} | pretrained: {mode_tag}")
    if val_loader is not None:
        print(f"  Best Val Acc@1 (used for ckpt selection): {max_best_acc1:.2f}%")
        print(f"  Test Acc@1 @ best-val epoch (held-out):   {test_acc1_at_best:.2f}%")
        print(f"  Test Acc@5 @ best-val epoch (held-out):   {test_acc5_at_best:.2f}%")
    else:
        print(f"  Best Test Acc@1 (test used as val): {max_best_acc1:.2f}%")
        print(f"  Test Acc@5 at that epoch:           {test_acc5_at_best:.2f}%")
    print(f"  Total time: {total_time_str}")
    print(f"  Weights written to: {output_dir}")
    print(f"{'=' * 60}")

    if wandb_run is not None:
        import wandb

        if val_loader is not None:
            wandb.summary["best_val_acc1"] = max_best_acc1
            wandb.summary["test_acc1_at_best_val"] = test_acc1_at_best
            wandb.summary["test_acc5_at_best_val"] = test_acc5_at_best
        else:
            wandb.summary["best_test_acc1"] = max_best_acc1
            wandb.summary["best_test_acc5"] = test_acc5_at_best
        wandb.finish()


if __name__ == "__main__":
    main(parse_args())
