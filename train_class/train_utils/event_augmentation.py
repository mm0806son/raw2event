"""Event-domain augmentation primitives (pure NumPy).

All transforms operate on the raw ``(N, 4)`` ``[t_us, x, y, p]`` array before
representation conversion, return a fresh array, and draw randomness only from
a caller-supplied ``np.random.Generator``. The coordinate canvas is passed in
explicitly so a flip mirrors within the sample's own extent.
"""

from __future__ import annotations

import numpy as np

# Valid values for the trainer's ``--augmentation`` flag.
AUGMENTATION_MODES: tuple[str, ...] = (
    "none",
    "eventdrop",
    "flip_h",
    "eventdrop_flip",
    "noise_inject",
    "polarity_rebalance",
)

# Number of independent EventDrop strategies (random / time / area).
_NUM_DROP_STRATEGIES = 3


def flip_horizontal(events: np.ndarray, coord_w: int) -> np.ndarray:
    """Mirror the x coordinate within ``[0, coord_w - 1]``.

    ``t``, ``y`` and ``p`` are untouched. Returns a fresh array; an empty
    input passes through unchanged (as a copy).

    This is a mirror in the raw event coordinate frame. When the
    representation later rescales ``coord_w`` onto a target grid that does
    not divide it evenly, the flipped bin can differ from the exact
    tensor-grid mirror by one cell (sub-pixel rounding) — acceptable for a
    flip augmentation and consistent with the representation's own
    floor-based binning.
    """
    out = events.copy()
    if len(out) == 0:
        return out
    out[:, 1] = (coord_w - 1) - out[:, 1]
    return out


def event_drop(
    events: np.ndarray,
    rng: np.random.Generator,
    coord_w: int,
    coord_h: int,
    drop_ratio: float = 0.5,
) -> np.ndarray:
    """EventDrop: drop a subset of events by one randomly chosen strategy.

    Strategy is drawn uniformly per call:
      0. random — drop each event independently with probability
         ``drop_ratio``.
      1. time   — drop every event inside a random time window covering a
         ``drop_ratio`` fraction of ``[t_min, t_max]``. No-op when all
         events share one timestamp.
      2. area   — drop every event inside a random axis-aligned rectangle
         whose side lengths are ``sqrt(drop_ratio)`` of the canvas (so the
         expected dropped area is ``drop_ratio``).

    Returns the surviving rows in their original order (so a time-sorted
    input stays sorted). If a strategy would remove every event, the
    original array is returned unchanged (copied) — never an empty frame.
    """
    out = events.copy()
    n = len(out)
    if n == 0:
        return out

    strategy = int(rng.integers(_NUM_DROP_STRATEGIES))

    if strategy == 0:
        keep = rng.random(n) >= drop_ratio
    elif strategy == 1:
        t = out[:, 0]
        t_min, t_max = t.min(), t.max()
        if t_max == t_min:
            keep = np.ones(n, dtype=bool)
        else:
            span = t_max - t_min
            window = drop_ratio * span
            start = rng.uniform(t_min, t_max - window) if window < span else t_min
            end = start + window
            keep = ~((t >= start) & (t <= end))
    else:
        x = out[:, 1]
        y = out[:, 2]
        frac = np.sqrt(drop_ratio)
        rect_w = frac * coord_w
        rect_h = frac * coord_h
        x0 = rng.uniform(0, max(coord_w - rect_w, 0.0))
        y0 = rng.uniform(0, max(coord_h - rect_h, 0.0))
        keep = ~((x >= x0) & (x < x0 + rect_w) & (y >= y0) & (y < y0 + rect_h))

    dropped = out[keep]
    if len(dropped) == 0:
        # Degenerate draw removed everything; fall back to the full sample
        # rather than hand an empty frame to the representation.
        return out
    return dropped


def inject_noise(
    events: np.ndarray,
    rng: np.random.Generator,
    coord_w: int,
    coord_h: int,
    noise_ratio: float = 0.10,
    noise_pos_ratio: float = 0.5,
) -> np.ndarray:
    """Inject background-activity and hot-pixel noise events.

    Real sensors emit background activity that the simulator does not. This adds
    ``round(noise_ratio * N)`` events at random space-time positions inside the
    canvas.

    Injected events:
      * ``t`` ~ Uniform[t_min, t_max] (reuses the single timestamp when the
        sample is instantaneous);
      * ``x`` ~ Uniform{0, .., coord_w - 1}, ``y`` ~ Uniform{0, .., coord_h - 1};
      * ``p`` ~ Bernoulli(noise_pos_ratio) (default 0.5 — a balanced prior that
        also nudges the RAW pos_ratio of 0.69 gently toward DV's 0.53).

    The result is stably re-sorted by ``t`` to preserve the non-decreasing time
    order downstream code assumes. Original events are never removed.
    """
    out = events.copy()
    n = len(out)
    n_noise = int(round(noise_ratio * n))
    if n == 0 or n_noise <= 0:
        return out

    t = out[:, 0]
    t_min, t_max = int(t.min()), int(t.max())
    if t_max > t_min:
        # endpoint inclusive so injected noise can land on the last timestamp.
        t_noise = rng.integers(t_min, t_max + 1, size=n_noise)
    else:
        t_noise = np.full(n_noise, t_min, dtype=t.dtype)
    x_noise = rng.integers(0, coord_w, size=n_noise)
    y_noise = rng.integers(0, coord_h, size=n_noise)
    p_noise = (rng.random(n_noise) < noise_pos_ratio).astype(out.dtype)

    noise = np.empty((n_noise, 4), dtype=out.dtype)
    noise[:, 0] = t_noise
    noise[:, 1] = x_noise
    noise[:, 2] = y_noise
    noise[:, 3] = p_noise

    combined = np.concatenate([out, noise], axis=0)
    order = np.argsort(combined[:, 0], kind="stable")
    return combined[order]


def rebalance_polarity(
    events: np.ndarray,
    rng: np.random.Generator,
    flip_ratio: float = 0.10,
) -> np.ndarray:
    """Randomly flip a fraction of event polarities (sim-completion).

    Targets the RAW polarity-conditional spatial coverage deficit (Stage III:
    84.5% of the spatial drift sits in the OFF channel, RAW OFF active-pixel
    ratio 0.31 vs DV 0.89) and the pos_ratio bias (RAW 0.69 vs DV 0.53). Each
    event's polarity is flipped (``p -> 1 - p``) independently with probability
    ``flip_ratio``. The flip is symmetric (pos->neg and neg->pos), so it acts as
    a random signed-polarity dropout that makes the model invariant to the
    simulator's coverage bias rather than hard-coding the test distribution.
    Because RAW is positive-heavy, the net effect also pulls pos_ratio toward
    0.5 and seeds OFF-channel coverage at positions previously held by ON
    events.

    Only the ``p`` column changes; ``t`` / ``x`` / ``y`` are untouched, so the
    time order is preserved and no re-sort is needed. Returns a fresh array; an
    empty input or ``flip_ratio == 0`` passes through unchanged (as a copy).
    """
    out = events.copy()
    n = len(out)
    if n == 0 or flip_ratio <= 0.0:
        return out
    flip_mask = rng.random(n) < flip_ratio
    out[flip_mask, 3] = 1 - out[flip_mask, 3]
    return out


def apply_event_augmentation(
    events: np.ndarray,
    mode: str,
    rng: np.random.Generator,
    coord_w: int,
    coord_h: int,
    drop_ratio: float = 0.5,
    flip_prob: float = 0.5,
    noise_ratio: float = 0.10,
    flip_ratio: float = 0.10,
) -> np.ndarray:
    """Apply the augmentation selected by ``mode`` to a raw event array.

    Args:
        events: ``(N, 4)`` ``[t_us, x, y, p]`` array (typically straight from
            ``np.load``; never mutated).
        mode: one of :data:`AUGMENTATION_MODES`.
        rng: seeded ``np.random.Generator`` driving all randomness.
        coord_w, coord_h: coordinate canvas pinned to the sample's
            pre-augmentation extent.
        drop_ratio: EventDrop drop fraction (IJCAI 2021 default 0.5).
        flip_prob: probability of applying the horizontal mirror.
        noise_ratio: ``noise_inject`` injected-event fraction (default 0.10).
        flip_ratio: ``polarity_rebalance`` per-event flip probability
            (default 0.10).

    Returns:
        The augmented ``(N', 4)`` array. For ``mode == "none"`` the input is
        returned as-is.
    """
    if mode == "none":
        return events
    if mode not in AUGMENTATION_MODES:
        raise ValueError(
            f"unknown augmentation mode: {mode!r}; valid: {AUGMENTATION_MODES}"
        )
    if len(events) == 0:
        return events.copy()

    out = events
    if mode in ("eventdrop", "eventdrop_flip"):
        out = event_drop(out, rng, coord_w, coord_h, drop_ratio=drop_ratio)
    if mode in ("flip_h", "eventdrop_flip"):
        if rng.random() < flip_prob:
            out = flip_horizontal(out, coord_w)
    if mode == "noise_inject":
        out = inject_noise(out, rng, coord_w, coord_h, noise_ratio=noise_ratio)
    if mode == "polarity_rebalance":
        out = rebalance_polarity(out, rng, flip_ratio=flip_ratio)
    return out


def resolve_augmentation(
    events: np.ndarray,
    augmentation: str,
    aug_seed: int,
    epoch: int,
    idx: int,
    fallback_w: int,
    fallback_h: int,
) -> tuple[np.ndarray, int | None, int | None]:
    """Apply augmentation and return the coordinate canvas for the representation.

    For ``augmentation == "none"`` returns ``(events, None, None)`` so the
    representation keeps its legacy per-sample dynamic extent and baseline
    runs stay bit-identical. Otherwise the canvas is pinned to the sample's
    PRE-augmentation extent and returned so the representation rescales
    against it (EventDrop = pure occlusion, flip = clean mirror), making the
    result invariant to the unified80-vs-native-346 coordinate ambiguity.

    The RNG is seeded from ``[aug_seed, epoch, idx]`` so augmentation is
    reproducible, varies per epoch, and is independent of DataLoader worker
    count / fork (each worker reconstructs the same generator for a given
    sample).
    """
    if augmentation == "none":
        return events, None, None
    if len(events) > 0:
        coord_w = int(events[:, 1].max()) + 1
        coord_h = int(events[:, 2].max()) + 1
    else:
        coord_w, coord_h = fallback_w, fallback_h
    rng = np.random.default_rng([aug_seed, epoch, idx])
    events = apply_event_augmentation(events, augmentation, rng, coord_w, coord_h)
    return events, coord_w, coord_h
