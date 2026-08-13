"""Extract penultimate (B, 256) embeddings from a QKFormer ckpt over the
canonical 559-prefix list.

Penultimate definition: the input tensor to ``model.head`` (the classification
linear layer). After ``vit_snn`` forward this tensor has shape ``(B, 256)``
because ``embed_dims=256`` and the model already pools over time / space /
tokens before the head. We L2-normalize before saving so cosine similarity is
just an inner product downstream.

Usage (single ckpt, single variant):

    python -m tools.retrieval.extract_embeddings \
        --ckpt        ./train_class/output/qkf_V01_raw_s0/<run_dir>/best.pth \
        --data_dir    ./output/sim_branches/V01 \
        --modality    raw \
        --prefix_list tools/v2e_baseline/canonical_test_v2e_runnable.txt \
        --variant     V01 \
        --seed        0 \
        --output      ./output/retrieval/embeddings/V01_raw_s0.npz

The canonical 559-prefix list is identical to the cross-modal eval split
(cross_modal_eval used the 600-train test_indices, here we go prefix-driven so
ALL 559 must be hittable in the variant directory; missing prefix = fail-fast).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_CLASS = REPO_ROOT / "train_class"
QKFORMER_DIR = TRAIN_CLASS / "QKFormer" / "cifar10-dvs"

# Mirror train_qkformer.py path setup so `import model` finds the upstream
# QKFormer source registered into the timm model registry.
for p in (str(QKFORMER_DIR), str(TRAIN_CLASS), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from spikingjelly.clock_driven import functional  # noqa: E402
from spikingjelly.clock_driven import neuron as sj_neuron  # noqa: E402

from train_utils.dataset import (  # noqa: E402
    CIFAR10_CLASSES,
    CLASS_TO_IDX,
    events_to_frames,
    parse_class_from_filename,
)


def _patch_lif_backend_when_cupy_missing() -> None:
    """Same shim as train_qkformer.py: fall back to torch backend when CuPy
    is not installed (full GPU nodes typically have CuPy; CPU sandboxes rarely do)."""
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


def build_model(num_classes: int = 10) -> nn.Module:
    """Reproduce the exact constructor train_qkformer.py uses (L584-600)."""
    _patch_lif_backend_when_cupy_missing()
    import model as qkformer_model  # noqa: F401  — register to timm registry
    from model import vit_snn

    return vit_snn(
        patch_size=16,
        embed_dims=256,
        num_heads=16,
        mlp_ratios=1,
        in_channels=2,
        num_classes=num_classes,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=4,
        sr_ratios=1,
        drop_rate=0.0,
        drop_path_rate=0.1,
    )


def load_ckpt_into(model: nn.Module, ckpt_path: Path) -> dict:
    """Load `checkpoint["model"]` (training save format) with strict=False to
    tolerate the DDP `module.` prefix if present."""
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = ckpt.get("model", ckpt)
    state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return {"missing": list(missing), "unexpected": list(unexpected)}


def load_prefix_list(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def resolve_npz_for_prefix(data_dir: Path, prefix: str, modality: str) -> Path:
    """Match EventNpzDataset's filename convention: `{prefix}_filtered_{modality}.npz`.

    Some sim variants use the suffix ``rawY`` on disk but the training scripts
    treat them as ``raw`` (see the v2e batch-submit docs). We accept
    either suffix when modality == "raw".
    """
    candidates = [data_dir / f"{prefix}_filtered_{modality}.npz"]
    if modality == "raw":
        candidates.append(data_dir / f"{prefix}_filtered_rawY.npz")
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"No NPZ for prefix='{prefix}' modality='{modality}' under {data_dir}; "
        f"tried: {[str(c) for c in candidates]}"
    )


@torch.no_grad()
def extract(
    model: nn.Module,
    npz_paths: list[Path],
    labels: list[int],
    device: torch.device,
    T: int = 16,
    target_h: int = 128,
    target_w: int = 128,
    batch_size: int = 16,
) -> np.ndarray:
    """Forward-hook on ``model.head`` input → (N, 256) penultimate embedding,
    L2-normalized along D. Resets SpikingJelly state per batch.

    The hook captures the first positional arg (i.e. the head input) and
    stashes it on a closure list for the duration of one batch; we then drain
    it after the batch finishes."""
    model.train(False)  # equivalent to model.eval(); avoids hook false-positive
    captured: list[torch.Tensor] = []

    def _hook(_module, inputs, _output):
        # inputs is a tuple; head is nn.Linear so inputs[0] has shape (B, 256).
        captured.append(inputs[0].detach())

    handle = model.head.register_forward_hook(_hook)
    try:
        out_list: list[np.ndarray] = []
        N = len(npz_paths)
        for start in range(0, N, batch_size):
            batch_paths = npz_paths[start : start + batch_size]
            frames_list = []
            for p in batch_paths:
                events = np.load(p)["events"]
                # Probe per-file resolution — original_h/w varies between
                # Davis346 (260×346) and v2e variants (e.g. 480×640). Mirror
                # EventNpzDataset's convention of "max coordinate + 1".
                if len(events) == 0:
                    orig_h, orig_w = 260, 346
                else:
                    orig_w = int(events[:, 1].max()) + 1
                    orig_h = int(events[:, 2].max()) + 1
                frames = events_to_frames(
                    events,
                    T_steps=T,
                    original_h=orig_h,
                    original_w=orig_w,
                    target_h=target_h,
                    target_w=target_w,
                )
                frames_list.append(frames)
            x = torch.stack(frames_list, dim=0).to(device, non_blocking=True).float()
            captured.clear()
            _ = model(x)
            functional.reset_net(model)
            assert len(captured) == 1, f"hook fired {len(captured)} times, expected 1"
            feat = captured[0]  # (B, 256)
            feat = nn.functional.normalize(feat, p=2, dim=1)
            out_list.append(feat.cpu().numpy().astype(np.float32))
            print(
                f"  [{start + len(batch_paths):4d}/{N}] batch ok · "
                f"feat={tuple(feat.shape)} norm[0]={float(feat[0].norm()):.4f}",
                flush=True,
            )
        embeddings = np.concatenate(out_list, axis=0)
    finally:
        handle.remove()

    assert embeddings.shape == (len(npz_paths), 256), embeddings.shape
    return embeddings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--modality", choices=("dv", "raw", "rgb"), required=True)
    ap.add_argument("--prefix_list", type=Path, required=True)
    ap.add_argument("--variant", required=True, help="V01..V08 or DV (free-form tag for meta)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--target_h", type=int, default=128)
    ap.add_argument("--target_w", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="smoke: cap to first N prefixes (0 = all)")
    args = ap.parse_args()

    if not args.ckpt.exists():
        raise FileNotFoundError(f"ckpt missing: {args.ckpt}")
    if not args.data_dir.is_dir():
        raise NotADirectoryError(f"data_dir missing: {args.data_dir}")
    if not args.prefix_list.exists():
        raise FileNotFoundError(f"prefix_list missing: {args.prefix_list}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    prefixes = load_prefix_list(args.prefix_list)
    if args.limit and args.limit > 0:
        prefixes = prefixes[: args.limit]
    print(f"[extract] {len(prefixes)} prefixes from {args.prefix_list}", flush=True)

    npz_paths: list[Path] = []
    labels: list[int] = []
    for prefix in prefixes:
        path = resolve_npz_for_prefix(args.data_dir, prefix, args.modality)
        npz_paths.append(path)
        cls = parse_class_from_filename(path.name)
        labels.append(CLASS_TO_IDX[cls])
    print(f"[extract] resolved all {len(npz_paths)} NPZ paths under {args.data_dir}", flush=True)

    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    print(f"[extract] device={device}", flush=True)

    model = build_model(num_classes=10).to(device)
    info = load_ckpt_into(model, args.ckpt)
    print(
        f"[extract] loaded {args.ckpt}\n"
        f"          missing={len(info['missing'])} unexpected={len(info['unexpected'])}",
        flush=True,
    )

    embeddings = extract(
        model,
        npz_paths,
        labels,
        device=device,
        T=args.T,
        target_h=args.target_h,
        target_w=args.target_w,
        batch_size=args.batch_size,
    )

    meta = {
        "variant": args.variant,
        "seed": args.seed,
        "modality": args.modality,
        "ckpt_path": str(args.ckpt),
        "data_dir": str(args.data_dir),
        "prefix_list": str(args.prefix_list),
        "T": args.T,
        "target_h": args.target_h,
        "target_w": args.target_w,
        "embed_dim": 256,
        "l2_normalized": True,
        "pool_strategy": "model.head input (post vit_snn temporal/spatial pool)",
        "n_prefixes": len(prefixes),
    }
    max_prefix_len = max(len(p) for p in prefixes)
    np.savez_compressed(
        args.output,
        prefix=np.array(prefixes, dtype=f"<U{max_prefix_len}"),
        embedding=embeddings,
        label=np.array(labels, dtype=np.int8),
    )
    sidecar_meta = args.output.with_suffix(".meta.json")
    sidecar_meta.write_text(json.dumps(meta, indent=2))
    print(f"[extract] wrote {args.output} · embeddings={embeddings.shape}", flush=True)
    print(f"[extract] wrote sidecar meta → {sidecar_meta}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
