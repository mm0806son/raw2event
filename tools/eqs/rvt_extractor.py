"""RVT backbone adapter for EQS. Requires torch, a GPU, and an RVT checkout.

Import-guarded so the pure-NumPy core in ``eqs_score`` never depends on torch.

Adapter invariants:

* Input ``events`` is ``(N, 4)`` ``[t, x, y, p]``, ``t`` in microseconds and
  ``p`` in ``{0, 1}`` — the encoding ``StackedHistogram`` expects.
* Frames are zero-padded, never resized, up to a multiple of ``pad_multiple``:
  resizing would distort event coordinates and the spatial statistics.
* Activations come from stages 1/2/3 (strides /4, /8, /16), not the LSTM states,
  and ``prev_states`` is reset every call so streams score independently.

Backbone construction depends on the checkpoint and its hydra config; verify the
returned stage keys, shapes, and strides against the checkpoint in use.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn.functional as F

    _HAVE_TORCH = True
except ImportError:  # pragma: no cover - exercised only off-cluster
    _HAVE_TORCH = False

T_COL, X_COL, Y_COL, P_COL = 0, 1, 2, 3


def _require_torch() -> None:
    if not _HAVE_TORCH:
        raise ImportError(
            "tools.eqs.rvt_extractor requires torch + the RVT repo on a GPU machine. "
            "The pure-NumPy EQS core lives in tools.eqs.eqs_score (no torch needed)."
        )


def load_rvt_backbone(
    rvt_repo: str,
    checkpoint: str,
    config_name: str = "small",
    dataset: str = "gen1",
    target_hw: tuple[int, int] = (260, 346),
):
    """Load the pretrained RVT detection backbone (Gen1-small by default).

    ``rvt_repo`` is an RVT working tree (e.g. the EQS-vendored ``RVT/``).  The public RVT checkpoints carry only a raw
    ``state_dict`` (no ``hyper_parameters``), and the model config is assembled by
    hydra composition — so we mirror RVT's own ``validation.py``: compose
    ``val.yaml`` with ``dataset=<dataset> +experiment/<dataset>=<config_name>``,
    run ``dynamically_modify_train_config`` (sets ``backbone.in_res_hw`` etc.),
    build the backbone, and load the ``mdl.backbone.*`` weights.  Returns an
    ``nn.Module`` whose ``forward(x, prev_states=None)`` yields the stage-feature
    dict.  Confirmed against rvt-s.ckpt with ``probe_rvt_stages.py``.
    """
    _require_torch()
    rvt_path = str(Path(rvt_repo).resolve())
    if rvt_path not in sys.path:
        sys.path.insert(0, rvt_path)

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from config.modifier import dynamically_modify_train_config
    from models.detection.recurrent_backbone import build_recurrent_backbone

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base="1.2", config_dir=str(Path(rvt_path) / "config")
    ):
        cfg = compose(
            config_name="val",
            overrides=[
                f"dataset={dataset}",
                f"+experiment/{dataset}={config_name}",
                f"checkpoint={checkpoint}",
            ],
        )
    dynamically_modify_train_config(cfg)

    # RVT hardcodes the sensor resolution per dataset (gen1 = 240x304 in
    # spatial.py) and derives the MaxViT partition (window/grid) sizes from it, so
    # the model only accepts input == in_res_hw.  Our streams are DAVIS346
    # (260x346), so we re-run the modifier's own arithmetic for our resolution:
    # round up to ``32 * partition_split_32`` and set partition_size = hw / that.
    # This is the standard RVT mechanism for a new sensor (pad, not resize); the
    # extractor pads every event tensor to exactly this in_res_hw.
    import math

    from omegaconf import open_dict

    bb = cfg.model.backbone
    mult = 32 * int(bb.partition_split_32)
    mdl_hw = tuple(int(math.ceil(x / mult) * mult) for x in target_hw)
    with open_dict(cfg):
        bb.in_res_hw = list(mdl_hw)
        bb.stage.attention.partition_size = [mdl_hw[0] // mult, mdl_hw[1] // mult]
    backbone = build_recurrent_backbone(bb)

    # weights_only=False: this is the user's own official RVT checkpoint (trusted,
    # local), and a Lightning ckpt pickles non-tensor objects under non-state keys.
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    bb_state = {
        k.split("mdl.backbone.", 1)[1]: v
        for k, v in state.items()
        if "mdl.backbone." in k
    }
    if not bb_state:
        raise RuntimeError(
            "no 'mdl.backbone.*' keys found in checkpoint; inspect with the probe script"
        )
    missing, unexpected = backbone.load_state_dict(bb_state, strict=False)
    # Any *missing* key means a backbone parameter stayed randomly initialised —
    # that silently contaminates every EQS number, so fail loudly rather than
    # produce plausible-looking garbage.  Unexpected keys (e.g. the checkpoint's
    # detection head/neck) are fine and only logged.
    if missing:
        raise RuntimeError(
            f"{len(missing)} backbone params not found in checkpoint (would be random-init); "
            f"first few: {list(missing)[:5]}. Re-check checkpoint/config with the probe script."
        )
    print(
        f"[rvt] loaded backbone: {len(bb_state)} tensors, unexpected={len(unexpected)}; "
        f"in_res_hw={mdl_hw} partition_size={list(bb.stage.attention.partition_size)}"
    )
    backbone.eval()
    backbone._eqs_in_res_hw = mdl_hw  # the exact (H, W) the extractor must pad to
    return backbone


class RVTStageExtractor:
    """Turn an event NPZ array into per-stage activation maps (NumPy)."""

    def __init__(
        self,
        backbone,
        bins: int = 10,
        height: int = 260,
        width: int = 346,
        count_cutoff: int = 10,
        stages: tuple[int, ...] = (1, 2, 3),
        pad_multiple: int = 32,
        device: str = "cuda",
    ):
        _require_torch()
        self.backbone = backbone.to(device)
        self.bins = bins
        self.height = height
        self.width = width
        self.count_cutoff = count_cutoff
        self.stages = stages
        self.pad_multiple = pad_multiple
        self.device = device
        # RVT's MaxViT only accepts input == in_res_hw (partition sizes are baked
        # in at build time); load_rvt_backbone stashes the exact target here.
        self.target_hw = getattr(backbone, "_eqs_in_res_hw", None)
        # Official RVT representation builder (must be importable from rvt_repo).
        from data.utils.representations import StackedHistogram

        self._repr = StackedHistogram(
            bins=bins, height=height, width=width, count_cutoff=count_cutoff
        )

    def _to_tensor(self, events: np.ndarray) -> "torch.Tensor":
        ev = np.asarray(events)
        if ev.ndim != 2 or ev.shape[1] != 4:
            raise ValueError(f"events must be (N,4) [t,x,y,p], got {ev.shape}")
        # StackedHistogram.construct bins by [time[0], time[-1]] and asserts the
        # stream is time-sorted. Real/sim streams already are, but a perturbation
        # like time_shuffle deliberately reorders timestamps — so sort by t here
        # (stable). This is a no-op for sorted input and keeps each (x,y,p) paired
        # with its (possibly shuffled) timestamp, preserving the perturbation's
        # intent while satisfying the representation's contract.
        if ev.shape[0] > 1:
            ev = ev[np.argsort(ev[:, T_COL], kind="stable")]
        x = torch.from_numpy(ev[:, X_COL].astype(np.int64))
        y = torch.from_numpy(ev[:, Y_COL].astype(np.int64))
        # Normalise polarity to {0,1}: raw/Pi pipelines encode OFF/ON as {0,1},
        # but v2e variants emit {-1,+1}. StackedHistogram asserts pol >= 0, so map
        # any negative polarity to 0. (pol > 0) is identity for {0,1} input, so it
        # leaves the raw variants unchanged.
        pol = torch.from_numpy((ev[:, P_COL] > 0).astype(np.int64))
        time = torch.from_numpy(ev[:, T_COL].astype(np.int64))
        tensor = self._repr.construct(x, y, pol, time).float()  # (2*bins, H, W)
        return tensor.unsqueeze(0)  # (1, 2*bins, H, W)

    def _pad(self, x: "torch.Tensor") -> "torch.Tensor":
        """Zero-pad (bottom/right) to exactly in_res_hw, or to a 32-multiple if
        the backbone did not stash a target (e.g. a stub in tests)."""
        _, _, h, w = x.shape
        if self.target_hw is not None:
            th, tw = self.target_hw
            if h > th or w > tw:
                raise ValueError(f"input {(h, w)} exceeds RVT in_res_hw {(th, tw)}")
        else:
            m = self.pad_multiple
            th, tw = ((h + m - 1) // m) * m, ((w + m - 1) // m) * m
        ph, pw = th - h, tw - w
        return F.pad(x, (0, pw, 0, ph)) if (ph or pw) else x

    def __call__(self, events: np.ndarray) -> list[np.ndarray]:
        """Extract the requested stage activations for one stream; resets state."""
        with torch.no_grad():
            x = self._pad(self._to_tensor(events).to(self.device))
            feats, _ = self.backbone(x, prev_states=None)
            missing = [s for s in self.stages if s not in feats]
            if missing:
                raise RuntimeError(
                    f"backbone features missing stages {missing}; got keys {list(feats)}. "
                    "RVT API differs from the assumed dict{1..4}; check with the probe script."
                )
            out = []
            for s in self.stages:
                fmap = feats[s].squeeze(0).cpu().numpy()
                if fmap.ndim != 3:
                    raise RuntimeError(
                        f"stage {s} map is {fmap.shape}, expected (C,H,W)"
                    )
                out.append(fmap)
            return out


def eqs_between(
    extractor: RVTStageExtractor, ev_a: np.ndarray, ev_b: np.ndarray, **lfs_kwargs
) -> dict:
    """Convenience: EQS between two event streams via the extractor + LFS block."""
    from tools.eqs.eqs_score import latent_feature_similarity

    return latent_feature_similarity(extractor(ev_a), extractor(ev_b), **lfs_kwargs)
