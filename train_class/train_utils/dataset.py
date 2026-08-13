"""Event dataset for the spiking classifier.

Reads the ``.npz`` files written by ``process_single_batch`` /
``process_all_batches`` and aggregates the sparse events into the multi-step
frame tensor ``[T_steps, 2, target_H, target_W]`` that QKFormer expects; the
size-2 axis is the ON and OFF polarity channels.

Each ``.npz`` holds an ``events`` key of shape ``(N, 4)``, int64, with columns
``[t_us, x, y, p]`` and ``p`` in {0, 1}.

Example:
    python train_class/dataset.py --data_dir ./data/unified80 --modality dv \
        --T 16 --test_sample
"""

import os
import sys
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Filenames are {id}_{classname}_{variant}_..._filtered_{modality}.npz
# ---------------------------------------------------------------------------
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CIFAR10_CLASSES)}


def parse_class_from_filename(filename: str) -> str:
    """
    Extract the class name from a filename.

    e.g. ``18900_cat_1_9006_20251225_165232_filtered_dv.npz`` -> ``cat``.
    """
    basename = os.path.basename(filename)
    # Strip the leading numeric id.
    parts = basename.split("_")
    # parts[0] = id, parts[1] = classname, ...
    # Match against the known class list rather than splitting blindly.
    for cls_name in CIFAR10_CLASSES:
        # Match the _{classname}_ pattern.
        if f"_{cls_name}_" in basename:
            return cls_name
    raise ValueError(
        f"cannot identify a class in '{basename}'; known classes: {CIFAR10_CLASSES}"
    )


def events_to_frames(
    events: np.ndarray,
    T_steps: int,
    original_h: int,
    original_w: int,
    target_h: int = 128,
    target_w: int = 128,
) -> torch.Tensor:
    """
    Aggregate a sparse event stream into multi-step frames.

    Args:
        events: ``(N, 4)`` array with columns ``[t_us, x, y, p]``.
        T_steps: number of time bins.
        original_h, original_w: source coordinate extent.
        target_h, target_w: output frame resolution.

    Returns:
        FloatTensor of shape ``[T_steps, 2, target_h, target_w]``.
    """
    if len(events) == 0:
        return torch.zeros(T_steps, 2, target_h, target_w, dtype=torch.float32)

    timestamps = events[:, 0].astype(np.float64)
    x_coords = events[:, 1]
    y_coords = events[:, 2]
    polarities = events[:, 3]

    # NPZ polarity is either {0,1} or legacy {-1,+1}. Fail fast on anything else
    # so a mis-encoded column is caught here, not silently aggregated below.
    unique_p = np.unique(polarities)
    bad_p = unique_p[~np.isin(unique_p, (-1, 0, 1))]
    if bad_p.size > 0:
        raise ValueError(
            f"events_to_frames: polarity column (events[:, 3]) must be a "
            f"subset of {{-1, 0, 1}}, found unexpected values: {bad_p.tolist()}"
        )

    # Normalize timestamps to a bin index in [0, T_steps).
    t_min, t_max = timestamps.min(), timestamps.max()
    if t_max == t_min:
        t_bins = np.zeros(len(events), dtype=np.int64)
    else:
        t_bins = ((timestamps - t_min) / (t_max - t_min + 1e-6) * T_steps).astype(
            np.int64
        )
        t_bins = np.clip(t_bins, 0, T_steps - 1)

    # Accumulate at the source resolution first.
    frames = np.zeros((T_steps, 2, original_h, original_w), dtype=np.float32)

    # Clip coordinates into range.
    x_safe = np.clip(x_coords, 0, original_w - 1)
    y_safe = np.clip(y_coords, 0, original_h - 1)

    # Map polarity to a channel index via (p > 0) rather than using p
    # directly as an array index: this is bit-identical for the project's
    # {0,1} convention (0 -> ch0, 1 -> ch1) and also correctly maps legacy
    # Map via (p > 0): indexing with p directly would let p = -1 wrap to channel
    # 1 under fancy indexing, merging every OFF event into the ON channel.
    p_idx = (polarities > 0).astype(np.int64)

    # Accumulate event counts.
    np.add.at(frames, (t_bins, p_idx, y_safe, x_safe), 1.0)

    # Resize to the target resolution.
    frames_tensor = torch.from_numpy(frames)  # [T, 2, orig_H, orig_W]

    if original_h != target_h or original_w != target_w:
        # F.interpolate needs a 4D [N, C, H, W] input.
        T, C, H, W = frames_tensor.shape
        frames_flat = frames_tensor.reshape(T * C, 1, H, W)
        frames_resized = F.interpolate(
            frames_flat, size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        frames_tensor = frames_resized.reshape(T, C, target_h, target_w)

    # Compress the residual cross-modality density difference.
    frames_tensor = torch.log1p(frames_tensor)

    return frames_tensor


class EventNpzDataset(Dataset):
    """
    Load every ``.npz`` of one modality and build frame-stream tensors.

    Args:
        data_dir: root directory holding the ``.npz`` files.
        modality: one of ``dv``, ``raw``, ``rgb``.
        T: number of time steps.
        target_h, target_w: output frame resolution.
    """

    def __init__(
        self,
        data_dir: str,
        modality: str = "dv",
        T: int = 16,
        target_h: int = 128,
        target_w: int = 128,
    ):
        super().__init__()
        assert modality in ("dv", "raw", "rgb"), f"unsupported modality: {modality}"

        self.data_dir = data_dir
        self.modality = modality
        self.T = T
        self.target_h = target_h
        self.target_w = target_w

        # Collect the matching NPZ files.
        pattern = os.path.join(data_dir, f"*_filtered_{modality}.npz")
        self.file_list = sorted(
            glob.glob(pattern), key=lambda f: int(os.path.basename(f).split("_")[0])
        )

        if len(self.file_list) == 0:
            raise FileNotFoundError(
                f"no '{modality}' .npz files under '{data_dir}'\n"
                f"search pattern: {pattern}"
            )

        # Parse labels.
        self.labels = []
        for f in self.file_list:
            cls_name = parse_class_from_filename(f)
            self.labels.append(CLASS_TO_IDX[cls_name])

        # Probe the source resolution from the first file.
        sample_events = np.load(self.file_list[0])["events"]
        self.original_w = int(sample_events[:, 1].max()) + 1
        self.original_h = int(sample_events[:, 2].max()) + 1

        print(
            f"[EventNpzDataset] modality={modality}, samples={len(self.file_list)}, "
            f"source=({self.original_h}x{self.original_w}), "
            f"target=({target_h}x{target_w}), T={T}"
        )
        print(
            f"  class distribution: { {CIFAR10_CLASSES[i]: self.labels.count(i) for i in range(10) if self.labels.count(i) > 0} }"
        )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        npz_path = self.file_list[idx]
        events = np.load(npz_path)["events"]

        frames = events_to_frames(
            events,
            self.T,
            self.original_h,
            self.original_w,
            self.target_h,
            self.target_w,
        )

        label = self.labels[idx]
        return frames, label


def split_dataset(dataset: Dataset, train_ratio: float = 0.9, seed: int = 42):
    """
    Split the dataset into train and test by ratio.

    Uses a stratified split so the class distribution is preserved, falling back
    to a plain random split when a class is too small to stratify.

    Args:
        dataset: must expose a ``.labels`` list of int class ids.
        train_ratio: fraction assigned to train.
        seed: RNG seed.

    Returns:
        (train_subset, test_subset)
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    from collections import Counter

    labels = np.array(dataset.labels)
    test_size = 1.0 - train_ratio

    try:
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=seed
        )
        for train_idx, test_idx in splitter.split(np.zeros(len(labels)), labels):
            train_indices = train_idx.tolist()
            test_indices = test_idx.tolist()
    except ValueError as e:
        print(
            f"  [warn] stratified split failed ({e}); falling back to a random split"
        )
        n = len(dataset)
        n_train = int(n * train_ratio)
        n_test = n - n_train
        generator = torch.Generator().manual_seed(seed)
        train_set, test_set = torch.utils.data.random_split(
            dataset, [n_train, n_test], generator=generator
        )
        train_indices = train_set.indices
        test_indices = test_set.indices

    train_set = torch.utils.data.Subset(dataset, train_indices)
    test_set = torch.utils.data.Subset(dataset, test_indices)

    # Per-split class counts.
    train_labels = labels[train_indices]
    test_labels = labels[test_indices]

    train_dist_name = {
        CIFAR10_CLASSES[k]: v for k, v in dict(Counter(train_labels)).items()
    }
    test_dist_name = {
        CIFAR10_CLASSES[k]: v for k, v in dict(Counter(test_labels)).items()
    }

    print(
        f"  Split: train={len(train_indices)}, test={len(test_indices)} "
        f"(ratio={train_ratio}, seed={seed})"
    )
    print(f"    Train classes: {train_dist_name}")
    print(f"    Test  classes: {test_dist_name}")

    return train_set, test_set


def split_dataset_deterministic(
    dataset: Dataset, train_per_class: int = 0, test_per_class: int = 600
):
    """
    Deterministic split with a fixed number of samples per class.

    With ``train_per_class == 0``, test takes the last ``test_per_class``
    samples of each class in file-numeric order and train takes the rest,
    matching the SpikingJelly CIFAR10-DVS convention.

    With ``train_per_class > 0``, both subsets are drawn by evenly spaced
    sampling across the whole class so they span every acquisition batch and
    stay disjoint.

    Args:
        dataset: must expose a ``.labels`` list of int class ids.
        train_per_class: per-class training samples; 0 uses everything left.
        test_per_class: per-class test samples.

    Returns:
        (train_subset, test_subset)
    """
    from collections import Counter

    labels = np.array(dataset.labels)
    num_classes = len(CIFAR10_CLASSES)
    # Group indices by class, preserving sorted file order
    class_indices = [[] for _ in range(num_classes)]
    for i, label in enumerate(labels):
        class_indices[label].append(i)
    train_indices = []
    test_indices = []
    for cls_id in range(num_classes):
        indices = class_indices[cls_id]  # already in numeric file order
        n_total = len(indices)
        cls_name = CIFAR10_CLASSES[cls_id]
        if train_per_class == 0:
            # 60K mode: train = all except last test_per_class, test = last test_per_class
            actual_test = min(test_per_class, n_total)
            actual_train = max(n_total - actual_test, 0)
            train_indices.extend(indices[:actual_train])
            test_indices.extend(
                indices[n_total - actual_test :] if actual_test > 0 else []
            )
        else:
            # 6K mode: uniform stride sampling for both train and test
            need = train_per_class + test_per_class
            if n_total < need:
                print(
                    f"  [warn] class '{cls_name}' has {n_total} samples "
                    f"< requested {train_per_class}+{test_per_class}={need}; "
                    f"scaling down"
                )
            actual_train = min(train_per_class, n_total * train_per_class // need)
            actual_test = min(test_per_class, n_total - actual_train)
            total_pick = actual_train + actual_test
            picked = [
                indices[int(round(i * (n_total - 1) / (total_pick - 1)))]
                for i in range(total_pick)
            ]
            # First actual_train go to train, rest to test
            train_indices.extend(picked[:actual_train])
            test_indices.extend(picked[actual_train:])

    train_set = torch.utils.data.Subset(dataset, train_indices)
    test_set = torch.utils.data.Subset(dataset, test_indices)
    # Print distribution summary
    train_labels = labels[train_indices]
    test_labels = labels[test_indices]
    train_dist = {CIFAR10_CLASSES[k]: int(v) for k, v in Counter(train_labels).items()}
    test_dist = {CIFAR10_CLASSES[k]: int(v) for k, v in Counter(test_labels).items()}
    if train_per_class == 0:
        train_desc = "all remaining"
    else:
        train_desc = f"{train_per_class} evenly spaced"
    print(
        f"  Deterministic split: train={len(train_indices)}, test={len(test_indices)} "
        f"(per class: {train_desc} train, {test_per_class} test)"
    )
    print(f"    Train classes: {train_dist}")
    print(f"    Test  classes: {test_dist}")
    return train_set, test_set


def split_dataset_deterministic_with_val(
    dataset: Dataset,
    val_per_class: int,
    test_per_class: int = 60,
):
    """3-way deterministic split: train / val / test, all disjoint, class-balanced.

    The 2-way `split_dataset_deterministic` lets the trainer select its best
    checkpoint on the test set, which inflates in-domain Top-1 by a few points.
    Here the trainer only ever sees the validation slice.

    Layout for each class (file-numeric-sorted ``indices`` of length ``n``):
        test : indices[n - test_per_class : n]                       (last N)
        val  : indices[n - test_per_class - val_per_class :
                       n - test_per_class]                            (next-to-last N)
        train: indices[: n - test_per_class - val_per_class]          (the rest)

    The test slice is identical to the one ``split_dataset_deterministic``
    returns for the same ``test_per_class``, so evaluation pipelines pointing at
    that test set are unaffected.

    Args:
        dataset: must expose ``.labels`` (parallel list of int class IDs).
        val_per_class: ≥1. The validation set the trainer is allowed to peek
            at for best-ckpt selection. Must be > 0; for the legacy 2-way
            behaviour, call ``split_dataset_deterministic`` instead.
        test_per_class: held-out test, NEVER seen during training. Defaults
            to 60 to match the project's 6K-split protocol.

    Returns:
        ``(train_subset, val_subset, test_subset)`` — three disjoint
        ``torch.utils.data.Subset`` views over ``dataset``.
    """
    from collections import Counter

    if val_per_class < 1:
        raise ValueError(
            "val_per_class must be >= 1; use split_dataset_deterministic for 2-way."
        )

    labels = np.array(dataset.labels)
    num_classes = len(CIFAR10_CLASSES)
    class_indices = [[] for _ in range(num_classes)]
    for i, label in enumerate(labels):
        class_indices[label].append(i)

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    for cls_id in range(num_classes):
        indices = class_indices[cls_id]
        n_total = len(indices)
        cls_name = CIFAR10_CLASSES[cls_id]
        actual_test = min(test_per_class, n_total)
        actual_val = min(val_per_class, max(n_total - actual_test, 0))
        actual_train = n_total - actual_test - actual_val
        if actual_train <= 0:
            print(
                f"  [warn] class '{cls_name}' has {n_total} samples "
                f"< val({val_per_class})+test({test_per_class}); train is empty"
            )
        train_indices.extend(indices[:actual_train])
        val_indices.extend(indices[actual_train : actual_train + actual_val])
        test_indices.extend(indices[n_total - actual_test :])

    train_set = torch.utils.data.Subset(dataset, train_indices)
    val_set = torch.utils.data.Subset(dataset, val_indices)
    test_set = torch.utils.data.Subset(dataset, test_indices)

    train_dist = {CIFAR10_CLASSES[k]: int(v) for k, v in Counter(labels[train_indices]).items()}
    val_dist = {CIFAR10_CLASSES[k]: int(v) for k, v in Counter(labels[val_indices]).items()}
    test_dist = {CIFAR10_CLASSES[k]: int(v) for k, v in Counter(labels[test_indices]).items()}
    print(
        f"  3-way deterministic split: train={len(train_indices)}, "
        f"val={len(val_indices)}, test={len(test_indices)} "
        f"(per class: all remaining train, {val_per_class} val, {test_per_class} test)"
    )
    print(f"    Train classes: {train_dist}")
    print(f"    Val   classes: {val_dist}")
    print(f"    Test  classes: {test_dist}")
    return train_set, val_set, test_set


# ---------------------------------------------------------------------------
# Smoke test: verify loading and frame aggregation.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test EventNpzDataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/unified80",
        help="directory holding the NPZ files",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="dv",
        choices=["dv", "raw", "rgb"],
        help="event modality",
    )
    parser.add_argument("--T", type=int, default=16, help="number of time steps")
    parser.add_argument("--test_sample", action="store_true", help="run the smoke test")
    args = parser.parse_args()

    if args.test_sample:
        ds = EventNpzDataset(args.data_dir, modality=args.modality, T=args.T)
        frames, label = ds[0]
        print("\nSmoke test passed.")
        print(f"  Frame tensor: {frames.shape}  (expected [{args.T}, 2, 128, 128])")
        print(f"  Label: {label} ({CIFAR10_CLASSES[label]})")
        print(
            f"  Range: min={frames.min():.4f}, max={frames.max():.4f}, mean={frames.mean():.4f}"
        )
        assert frames.shape == (args.T, 2, 128, 128), f"shape mismatch: got {frames.shape}"
        print("  Shape check passed.")
