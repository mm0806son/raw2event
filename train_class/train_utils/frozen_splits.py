"""Bind a frozen train/val/test list to what the trainer actually computes.

Freezing a split into JSON does nothing on its own. ``EventNpzDataset`` globs
its directory and ``split_dataset_deterministic_with_val`` recomputes the three
subsets from that glob every run; neither ever reads the frozen file. So a
prefix that failed to generate in one arm, or an extra file left in the
directory, would shift the split silently -- and the two arms would be trained
on different data while still reporting a "paired" contrast.

This module closes that gap: before training starts, recompute the split the
way the trainer will, map the indices back to prefixes, and refuse to start
unless the result equals the frozen lists exactly, in order.

Order matters, not just membership. The frozen digests are order-sensitive, and
a differently ordered train list means a different minibatch sequence under a
fixed seed -- which would break any paired comparison between two runs that
are supposed to differ in one factor only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SPLIT_NAMES = ("train", "val", "test")


class BindError(RuntimeError):
    """Raised when the computed split does not match the frozen one."""


def _sha256(prefixes: list[str]) -> str:
    """Order-sensitive digest: a reordering between runs must not go unnoticed."""
    payload = "\n".join(prefixes).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prefix_of_path(path: str | Path, modality: str | None = None) -> str:
    """Recover a prefix from ``{prefix}_filtered_{modality}.npz``.

    Refuses anything without the suffix rather than falling back to the stem:
    a silent fallback would yield prefixes that look right but carry a trailing
    fragment, producing a mismatch report that points nowhere useful.
    """
    name = Path(path).name
    suffixes = (
        [f"_filtered_{modality}.npz"]
        if modality
        else [f"_filtered_{m}.npz" for m in ("raw", "rgb", "dv")]
    )
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    raise BindError(
        f"{name!r} does not end with an expected suffix {suffixes}; "
        "refusing to guess a prefix"
    )


def computed_splits(
    file_list: list[str],
    train_idx,
    val_idx,
    test_idx,
    modality: str | None = None,
) -> dict[str, list[str]]:
    """Map subset indices back to prefixes, preserving subset order."""
    return {
        "train": [prefix_of_path(file_list[i], modality) for i in train_idx],
        "val": [prefix_of_path(file_list[i], modality) for i in val_idx],
        "test": [prefix_of_path(file_list[i], modality) for i in test_idx],
    }


def load_frozen(path: str | Path) -> dict:
    """Read the frozen split file."""
    return json.loads(Path(path).read_text())


def _first_divergence(frozen: list[str], computed: list[str]) -> str:
    for i, (a, b) in enumerate(zip(frozen, computed)):
        if a != b:
            return f"position {i}: frozen {a!r} vs computed {b!r}"
    return f"lengths differ: frozen {len(frozen)} vs computed {len(computed)}"


def assert_bound(frozen: dict, computed: dict[str, list[str]]) -> None:
    """Refuse to proceed unless the computed split equals the frozen one.

    Checked in this order deliberately: the frozen file's own digest first, so
    a file that was hand-edited to match a bad run is rejected before its
    contents are ever compared to anything.
    """
    recorded = frozen.get("sha256", {})
    for name in SPLIT_NAMES:
        if name not in frozen:
            raise BindError(f"frozen splits have no '{name}' list")
        if _sha256(frozen[name]) != recorded.get(name):
            raise BindError(
                f"frozen '{name}' does not match its recorded SHA-256; the file "
                "was edited after it was frozen"
            )

    for name in SPLIT_NAMES:
        if name not in computed:
            raise BindError(f"computed splits have no '{name}' list")
        want, got = frozen[name], computed[name]
        if want == got:
            continue
        detail = _first_divergence(want, got)
        if sorted(want) == sorted(got):
            raise BindError(
                f"computed '{name}' has the right members in a different order "
                f"({detail}); the frozen digest and the minibatch sequence both "
                "depend on order"
            )
        missing = sorted(set(want) - set(got))
        extra = sorted(set(got) - set(want))
        raise BindError(
            f"computed '{name}' does not match the frozen list "
            f"(frozen {len(want)}, computed {len(got)}; {detail}); "
            f"missing {len(missing)} e.g. {missing[:3]}; "
            f"unexpected {len(extra)} e.g. {extra[:3]}"
        )


def bind_or_die(
    data_dir: str | Path,
    modality: str,
    frozen_path: str | Path,
    val_per_class: int = 60,
    test_per_class: int = 60,
) -> dict[str, list[str]]:
    """Recompute the trainer's split over ``data_dir`` and bind it, or raise.

    Uses the trainer's own dataset and split function rather than a
    reimplementation -- a reimplementation would drift, and drifting is the
    exact failure this exists to catch. Imported lazily because both pull in
    torch, which the project venv does not have.
    """
    from train_class.train_utils.dataset import (
        EventNpzDataset,
        split_dataset_deterministic_with_val,
    )

    dataset = EventNpzDataset(str(data_dir), modality=modality)
    train_set, val_set, test_set = split_dataset_deterministic_with_val(
        dataset, val_per_class=val_per_class, test_per_class=test_per_class
    )
    computed = computed_splits(
        dataset.file_list,
        train_set.indices,
        val_set.indices,
        test_set.indices,
        modality=modality,
    )
    assert_bound(load_frozen(frozen_path), computed)
    return computed


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data_dir", required=True)
    p.add_argument("--modality", default="raw", choices=("raw", "rgb", "dv"))
    p.add_argument(
        "--frozen", required=True, help="Path to the frozen split JSON"
    )
    p.add_argument("--val_per_class", type=int, default=60)
    p.add_argument("--test_per_class", type=int, default=60)
    args = p.parse_args(argv)

    try:
        computed = bind_or_die(
            args.data_dir,
            args.modality,
            args.frozen,
            val_per_class=args.val_per_class,
            test_per_class=args.test_per_class,
        )
    except BindError as exc:
        print(f"SPLIT BINDING FAILED: {exc}")
        return 1
    print(
        "split bound: "
        + "  ".join(f"{name}={len(computed[name])}" for name in SPLIT_NAMES)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
