"""Event representation datasets for MobileNetV2.

Two representations, selected by the trainer's ``--representation`` flag:

* ``timestack`` discards polarity and time, accumulating every event into one
  count map replicated to three channels for an ImageNet-shaped CNN input.
* ``stacked_histogram`` follows RVT (Zubic et al., ICCV 2023): ``2*T`` channels
  indexed by ``t_bin * 2 + polarity_pos``, keeping polarity and coarse time.

Both share the same normalization (clip, log1p, per-channel min-max) so their
dynamic ranges are comparable. Input NPZ columns are ``[t_us, x, y, p]``.
"""

import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CLASS_DIR = os.path.dirname(SCRIPT_DIR)
if TRAIN_CLASS_DIR not in sys.path:
    sys.path.insert(0, TRAIN_CLASS_DIR)

from train_utils.dataset import CLASS_TO_IDX, CIFAR10_CLASSES, parse_class_from_filename
from train_utils.event_augmentation import AUGMENTATION_MODES, resolve_augmentation


def minmax_normalize_per_channel(image: torch.Tensor) -> torch.Tensor:
    """Min-max normalize a ``[C, H, W]`` image per channel."""
    image = image.float()
    channels = image.shape[0]
    flat = image.view(channels, -1)
    min_vals = flat.min(dim=1).values.view(channels, 1, 1)
    max_vals = flat.max(dim=1).values.view(channels, 1, 1)
    return (image - min_vals) / (max_vals - min_vals + 1e-8)


def events_to_timestack_image(
    events: np.ndarray,
    target_h: int = 128,
    target_w: int = 128,
    output_channels: int = 3,
    output_size: int = 224,
    normalize: bool = True,
    coord_w: int | None = None,
    coord_h: int | None = None,
) -> torch.Tensor:
    """
    Collapse events into a single count image.

    coord_w / coord_h: optional override of the source coordinate canvas used
        to rescale events onto the target grid. When ``None`` (the default,
        and the baseline / no-augmentation path) the canvas is derived
        per-sample from ``x.max()+1`` / ``y.max()+1`` exactly as before, so
        baseline behaviour is bit-identical. Augmented samples pass the
        sample's PRE-augmentation extent so EventDrop acts as pure occlusion
        (no silent zoom) and the horizontal flip maps as a clean mirror.

    Returns:
        FloatTensor of shape ``[C, output_size, output_size]``.
    """
    if len(events) == 0:
        image = torch.zeros(output_channels, target_h, target_w, dtype=torch.float32)
    else:
        x_coords = events[:, 1].astype(np.float32)
        y_coords = events[:, 2].astype(np.float32)

        original_w = coord_w if coord_w is not None else max(int(x_coords.max()) + 1, 1)
        original_h = coord_h if coord_h is not None else max(int(y_coords.max()) + 1, 1)

        x_scaled = np.floor(x_coords * target_w / original_w).astype(np.int64)
        y_scaled = np.floor(y_coords * target_h / original_h).astype(np.int64)
        x_scaled = np.clip(x_scaled, 0, target_w - 1)
        y_scaled = np.clip(y_scaled, 0, target_h - 1)

        count_grid = np.zeros((target_h, target_w), dtype=np.float32)
        np.add.at(count_grid, (y_scaled, x_scaled), 1.0)

        image = torch.from_numpy(count_grid).unsqueeze(0)

        # Compress the residual cross-modality density difference.
        image = torch.log1p(image)

        if output_channels > 1:
            image = image.repeat(output_channels, 1, 1)

    if normalize:
        image = minmax_normalize_per_channel(image)

    if output_size != target_h or output_size != target_w:
        image = F.interpolate(
            image.unsqueeze(0),
            size=(output_size, output_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    return image.contiguous()


class EventCountImageDataset(Dataset):
    """Read dv/raw/rgb NPZ files and emit single-frame images for MobileNetV2."""

    def __init__(
        self,
        data_dir: str,
        modality: str = "dv",
        target_h: int = 128,
        target_w: int = 128,
        output_channels: int = 3,
        output_size: int = 224,
        normalize: bool = True,
        augmentation: str = "none",
        aug_seed: int = 2024,
    ):
        super().__init__()
        assert modality in ("dv", "raw", "rgb"), f"unsupported modality: {modality}"
        assert augmentation in AUGMENTATION_MODES, (
            f"unsupported augmentation: {augmentation}; valid: {AUGMENTATION_MODES}"
        )

        self.data_dir = data_dir
        self.modality = modality
        self.target_h = target_h
        self.target_w = target_w
        self.output_channels = output_channels
        self.output_size = output_size
        self.normalize = normalize
        self.augmentation = augmentation
        self.aug_seed = aug_seed
        # Current epoch; bumped by set_epoch() so per-epoch augmentation
        # varies while staying reproducible. Only meaningful when
        # augmentation != "none".
        self.epoch = 0

        pattern = os.path.join(data_dir, f"*_filtered_{modality}.npz")
        self.file_list = sorted(
            glob.glob(pattern), key=lambda f: int(os.path.basename(f).split("_")[0])
        )
        if len(self.file_list) == 0:
            raise FileNotFoundError(
                f"no '{modality}' .npz files under '{data_dir}'\n"
                f"search pattern: {pattern}"
            )

        self.labels = []
        for npz_path in self.file_list:
            cls_name = parse_class_from_filename(npz_path)
            self.labels.append(CLASS_TO_IDX[cls_name])

        print(
            f"[EventCountImageDataset] modality={modality}, samples={len(self.file_list)}, "
            f"representation=timestack, shape=({output_channels}x{output_size}x{output_size}), "
            f"augmentation={augmentation}"
        )
        print(
            f"  class distribution: "
            f"{ {CIFAR10_CLASSES[i]: self.labels.count(i) for i in range(10) if self.labels.count(i) > 0} }"
        )

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch so per-epoch augmentation varies reproducibly."""
        self.epoch = epoch

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        npz_path = self.file_list[idx]
        events = np.load(npz_path)["events"]
        events, coord_w, coord_h = resolve_augmentation(
            events,
            self.augmentation,
            self.aug_seed,
            self.epoch,
            idx,
            self.target_w,
            self.target_h,
        )
        image = events_to_timestack_image(
            events=events,
            target_h=self.target_h,
            target_w=self.target_w,
            output_channels=self.output_channels,
            output_size=self.output_size,
            normalize=self.normalize,
            coord_w=coord_w,
            coord_h=coord_h,
        )
        label = self.labels[idx]
        return image, label


def events_to_stacked_histogram(
    events: np.ndarray,
    T: int = 10,
    target_h: int = 128,
    target_w: int = 128,
    output_size: int = 224,
    count_cutoff: int | None = 10,
    normalize: bool = True,
    coord_w: int | None = None,
    coord_h: int | None = None,
) -> torch.Tensor:
    """Stacked-histogram event representation (RVT-style).

    Builds a ``[2*T, output_size, output_size]`` tensor where channel
    ``c = t_bin * 2 + (1 if polarity_positive else 0)`` carries the count
    of events with that polarity in that time bin at each spatial pixel.
    Conventions follow Zubic et al. ICCV 2023 / RVT default (T=10,
    count_cutoff=10).

    Same per-channel min-max + log1p normalisation as
    ``events_to_timestack_image`` so the dynamic range is comparable
    across representation variants.
    """
    out_c = 2 * T
    if len(events) == 0:
        return torch.zeros(out_c, output_size, output_size, dtype=torch.float32)

    t = events[:, 0].astype(np.float64)
    x = events[:, 1].astype(np.float32)
    y = events[:, 2].astype(np.float32)
    p = events[:, 3]

    # Time bin assignment over [t_min, t_max] uniformly.
    t_min, t_max = t.min(), t.max()
    if t_max == t_min:
        t_bins = np.zeros(len(events), dtype=np.int64)
    else:
        t_bins = ((t - t_min) / (t_max - t_min + 1e-6) * T).astype(np.int64)
    t_bins = np.clip(t_bins, 0, T - 1)

    # Spatial bin assignment. coord_w/coord_h override pins the canvas to the
    # sample's pre-augmentation extent (see events_to_timestack_image); None
    # keeps the legacy per-sample dynamic extent so baseline runs are
    # bit-identical.
    original_w = coord_w if coord_w is not None else max(int(x.max()) + 1, 1)
    original_h = coord_h if coord_h is not None else max(int(y.max()) + 1, 1)
    x_scaled = np.clip(np.floor(x * target_w / original_w).astype(np.int64), 0, target_w - 1)
    y_scaled = np.clip(np.floor(y * target_h / original_h).astype(np.int64), 0, target_h - 1)

    # Polarity squashed to {0, 1} so the function is robust to upstream
    # encodings of ``p`` ({-1,+1}, {0,1}, {False, True}).
    p_pos = (p > 0).astype(np.int64)
    channels = t_bins * 2 + p_pos

    hist = np.zeros((out_c, target_h, target_w), dtype=np.float32)
    np.add.at(hist, (channels, y_scaled, x_scaled), 1.0)

    if count_cutoff is not None:
        np.clip(hist, 0, count_cutoff, out=hist)

    image = torch.from_numpy(hist)
    if normalize:
        image = torch.log1p(image)
        image = minmax_normalize_per_channel(image)

    if output_size != target_h or output_size != target_w:
        image = F.interpolate(
            image.unsqueeze(0),
            size=(output_size, output_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    return image.contiguous()


class EventStackedHistogramDataset(Dataset):
    """Reads dv/raw/rgb NPZ and outputs a stacked-histogram tensor.

    Sister of ``EventCountImageDataset`` but yields a
    ``[2*T, output_size, output_size]`` tensor (time + polarity preserved
    as channels) instead of the polarity-stripped, time-pooled count image.

    Constructor knobs other than ``T`` / ``count_cutoff`` mirror the
    timestack class so a single ``--representation`` switch in the trainer
    / evaluator suffices to swap dataset class.
    """

    def __init__(
        self,
        data_dir: str,
        modality: str = "dv",
        T: int = 10,
        target_h: int = 128,
        target_w: int = 128,
        output_size: int = 224,
        count_cutoff: int | None = 10,
        normalize: bool = True,
        augmentation: str = "none",
        aug_seed: int = 2024,
    ):
        super().__init__()
        assert modality in ("dv", "raw", "rgb"), f"unsupported modality: {modality}"
        assert augmentation in AUGMENTATION_MODES, (
            f"unsupported augmentation: {augmentation}; valid: {AUGMENTATION_MODES}"
        )

        self.data_dir = data_dir
        self.modality = modality
        self.T = T
        self.target_h = target_h
        self.target_w = target_w
        self.output_size = output_size
        self.count_cutoff = count_cutoff
        self.normalize = normalize
        self.augmentation = augmentation
        self.aug_seed = aug_seed
        self.epoch = 0

        pattern = os.path.join(data_dir, f"*_filtered_{modality}.npz")
        self.file_list = sorted(
            glob.glob(pattern), key=lambda f: int(os.path.basename(f).split("_")[0])
        )
        if len(self.file_list) == 0:
            raise FileNotFoundError(
                f"No .npz files for modality '{modality}' under '{data_dir}'.\n"
                f"Search pattern: {pattern}"
            )

        self.labels = []
        for npz_path in self.file_list:
            cls_name = parse_class_from_filename(npz_path)
            self.labels.append(CLASS_TO_IDX[cls_name])

        print(
            f"[EventStackedHistogramDataset] modality={modality}, "
            f"samples={len(self.file_list)}, "
            f"representation=stacked_histogram T={T} cutoff={count_cutoff}, "
            f"output=({2 * T}x{output_size}x{output_size}), augmentation={augmentation}"
        )
        print(
            f"  class distribution: "
            f"{ {CIFAR10_CLASSES[i]: self.labels.count(i) for i in range(10) if self.labels.count(i) > 0} }"
        )

    @property
    def in_channels(self) -> int:
        return 2 * self.T

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch so per-epoch augmentation varies reproducibly."""
        self.epoch = epoch

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        npz_path = self.file_list[idx]
        events = np.load(npz_path)["events"]
        events, coord_w, coord_h = resolve_augmentation(
            events,
            self.augmentation,
            self.aug_seed,
            self.epoch,
            idx,
            self.target_w,
            self.target_h,
        )
        image = events_to_stacked_histogram(
            events=events,
            T=self.T,
            target_h=self.target_h,
            target_w=self.target_w,
            output_size=self.output_size,
            count_cutoff=self.count_cutoff,
            normalize=self.normalize,
            coord_w=coord_w,
            coord_h=coord_h,
        )
        label = self.labels[idx]
        return image, label


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test the MobileNetV2 event datasets")
    parser.add_argument("--data_dir", type=str, required=True, help="directory holding the NPZ files")
    parser.add_argument(
        "--modality", type=str, default="dv", choices=["dv", "raw", "rgb"]
    )
    parser.add_argument(
        "--representation",
        type=str,
        default="timestack",
        choices=["timestack", "stacked_histogram"],
        help="Which event representation to smoke-test (default: timestack)",
    )
    parser.add_argument("--T", type=int, default=10, help="time bins (stacked_histogram only)")
    parser.add_argument("--output_size", type=int, default=224, help="CNN input resolution")
    args = parser.parse_args()

    if args.representation == "timestack":
        dataset = EventCountImageDataset(
            data_dir=args.data_dir,
            modality=args.modality,
            output_size=args.output_size,
        )
    else:
        dataset = EventStackedHistogramDataset(
            data_dir=args.data_dir,
            modality=args.modality,
            T=args.T,
            output_size=args.output_size,
        )
    image, label = dataset[0]
    print(f"image shape: {tuple(image.shape)}")
    print(f"label: {label} ({CIFAR10_CLASSES[label]})")
    print(
        f"value range: min={image.min().item():.4f}, "
        f"max={image.max().item():.4f}, mean={image.mean().item():.4f}"
    )
