"""Latent Feature Similarity block: the EQS metric core, in pure NumPy.

Reimplemented from Chanda et al., CVPRW'25 (arXiv:2504.12515, §4/§4.1), whose
official repository does not release this block. Decoupled from the RVT backbone
so it consumes already-extracted activation maps and runs without torch.

Per stage: split each ``(C, H, W)`` map into ``patch``x``patch`` blocks
(zero-padded to divide evenly), average within each block, take the cosine
similarity per block between the two streams, then average over blocks and
stages.

The source paper is internally inconsistent about direction, so both are
returned: ``eqs_similarity`` (higher is closer to real) and ``eqs_distance``
= 1 - similarity (lower is closer, matching the other upstream axes).
"""

from __future__ import annotations

import numpy as np

ZeroNormPolicy = str  # "drop" | "eps" | "ones"


def _as_chw(act: np.ndarray) -> np.ndarray:
    """Coerce an activation map to (C, H, W), squeezing a leading batch dim."""
    a = np.asarray(act, dtype=np.float64)
    if a.ndim == 4:
        if a.shape[0] != 1:
            raise ValueError(f"expected batch size 1, got shape {a.shape}")
        a = a[0]
    if a.ndim != 3:
        raise ValueError(f"activation must be (C,H,W) or (1,C,H,W), got {a.shape}")
    return a


def patch_pool(act: np.ndarray, patch: int) -> np.ndarray:
    """Average a (C, H, W) map over non-overlapping patchxpatch blocks.

    Returns ``(n_patches, C)``.  Spatial dims are zero-padded up to a multiple of
    ``patch``; the per-patch denominator is the full ``patch*patch`` area (padded
    cells contribute zeros), matching the paper's fixed ``1/||p_i||`` averaging.
    """
    if patch < 1:
        raise ValueError(f"patch must be >= 1, got {patch}")
    a = _as_chw(act)
    c, h, w = a.shape
    n_ph = (h + patch - 1) // patch
    n_pw = (w + patch - 1) // patch
    h_pad, w_pad = n_ph * patch, n_pw * patch
    if (h_pad, w_pad) != (h, w):
        padded = np.zeros((c, h_pad, w_pad), dtype=a.dtype)
        padded[:, :h, :w] = a
        a = padded
    # (C, n_ph, patch, n_pw, patch) -> mean over the two patch axes.
    blocks = a.reshape(c, n_ph, patch, n_pw, patch)
    pooled = blocks.sum(axis=(2, 4)) / float(patch * patch)  # (C, n_ph, n_pw)
    return pooled.reshape(c, n_ph * n_pw).T  # (n_patches, C)


def _patch_cosine(
    va: np.ndarray, vb: np.ndarray, zero_norm_policy: ZeroNormPolicy, eps: float
) -> tuple[np.ndarray, int]:
    """Per-patch cosine similarity between two (n_patches, C) vector stacks.

    Returns ``(similarities, n_zero_norm)`` where ``similarities`` holds only the
    *kept* patches (under "drop" the zero-norm patches are removed; under "eps"/
    "ones" they are kept with a defined value).  ``n_zero_norm`` counts patches
    where either stream's patch vector has ~zero norm.
    """
    na = np.linalg.norm(va, axis=1)
    nb = np.linalg.norm(vb, axis=1)
    dot = np.einsum("ij,ij->i", va, vb)
    zero = (na <= eps) | (nb <= eps)
    n_zero = int(zero.sum())
    if zero_norm_policy == "drop":
        keep = ~zero
        denom = na[keep] * nb[keep]
        sims = dot[keep] / denom
        return sims, n_zero
    if zero_norm_policy == "eps":
        sims = dot / (na * nb + eps)
        return sims, n_zero
    if zero_norm_policy == "ones":
        denom = na * nb
        sims = np.ones_like(dot)
        nz = ~zero
        sims[nz] = dot[nz] / denom[nz]
        return sims, n_zero
    raise ValueError(f"unknown zero_norm_policy: {zero_norm_policy!r}")


def latent_feature_similarity(
    acts_a: list[np.ndarray],
    acts_b: list[np.ndarray],
    patch: int = 3,
    zero_norm_policy: ZeroNormPolicy = "drop",
    eps: float = 1e-8,
) -> dict:
    """EQS between two streams from their per-scale activation maps.

    ``acts_a`` / ``acts_b`` are equal-length lists of (C, H, W) (or (1, C, H, W))
    arrays — one per RVT stage, in order.  Returns a dict with per-scale and
    aggregate cosine *similarity* (higher = closer), the derived *distance*
    (1 - sim), and the overall zero-norm patch fraction (a sanity diagnostic:
    a high fraction means the chosen patch/scale is mostly empty background).
    """
    if len(acts_a) != len(acts_b):
        raise ValueError(f"scale count mismatch: {len(acts_a)} vs {len(acts_b)}")
    if not acts_a:
        raise ValueError("need at least one scale of activations")

    per_scale: list[float] = []
    total_patches = 0
    total_zero = 0
    for a, b in zip(acts_a, acts_b):
        va = patch_pool(a, patch)
        vb = patch_pool(b, patch)
        if va.shape != vb.shape:
            raise ValueError(
                f"scale shape mismatch after pooling: {va.shape} vs {vb.shape}"
            )
        sims, n_zero = _patch_cosine(va, vb, zero_norm_policy, eps)
        total_patches += va.shape[0]
        total_zero += n_zero
        per_scale.append(float(np.mean(sims)) if sims.size else float("nan"))

    sim_mean = (
        float(np.nanmean(per_scale)) if np.any(np.isfinite(per_scale)) else float("nan")
    )
    out = {
        "eqs_similarity_per_scale": per_scale,
        "eqs_similarity_mean": sim_mean,
        "eqs_distance_mean": (1.0 - sim_mean)
        if np.isfinite(sim_mean)
        else float("nan"),
        "zero_norm_patch_fraction": (total_zero / total_patches)
        if total_patches
        else float("nan"),
        "n_scales": len(per_scale),
        "patch": patch,
        "zero_norm_policy": zero_norm_policy,
    }
    return out
