"""Structure-destroying perturbations used as EQS validity controls.

Feeding a real recording against a structure-destroyed copy of itself must score
clearly below the identity, otherwise the metric is not responding to event
structure on this data.

All functions are pure, deterministic given ``seed``, and operate on ``(N, 4)``
``[t, x, y, p]``. They permute existing values rather than inventing new ones, so
each column's value multiset is preserved and only the cross-column structure is
destroyed.
"""

from __future__ import annotations

import numpy as np

T, X, Y, P = 0, 1, 2, 3


def _checked(events: np.ndarray) -> np.ndarray:
    ev = np.asarray(events)
    if ev.ndim != 2 or ev.shape[1] != 4:
        raise ValueError(f"events must be (N, 4) [t,x,y,p], got {ev.shape}")
    return ev


def polarity_flip(events: np.ndarray, seed: int = 0) -> np.ndarray:
    """Reflect polarity across its midpoint: {0,1}->{1,0}, {-1,1}->{1,-1}.

    Destroys ON/OFF identity while preserving spatial/temporal structure.
    ``seed`` is unused (deterministic) but kept for a uniform perturbation API.
    """
    ev = _checked(events).copy()
    if ev.shape[0] == 0:
        return ev
    pol = ev[:, P]
    ev[:, P] = pol.max() + pol.min() - pol
    return ev


def time_shuffle(events: np.ndarray, seed: int = 0) -> np.ndarray:
    """Randomly permute the timestamp column, destroying temporal ordering.

    Events keep their (x, y, p) but are reassigned to random time bins, so the
    per-pixel timing structure the RVT temporal bins encode is broken.
    """
    ev = _checked(events).copy()
    if ev.shape[0] < 2:
        return ev
    rng = np.random.default_rng(seed)
    ev[:, T] = ev[rng.permutation(ev.shape[0]), T]
    return ev


def coord_shuffle(events: np.ndarray, seed: int = 0) -> np.ndarray:
    """Jointly permute the (x, y) columns, destroying spatial structure.

    The (x, y) *pairs* are shuffled together across events (keeping the set of
    occupied pixels intact) while t and p stay put, so edges/objects dissolve
    into spatial noise without changing the marginal pixel-occupancy histogram.
    """
    ev = _checked(events).copy()
    if ev.shape[0] < 2:
        return ev
    rng = np.random.default_rng(seed)
    perm = rng.permutation(ev.shape[0])
    ev[:, [X, Y]] = ev[np.ix_(perm, [X, Y])]
    return ev


PERTURBATIONS = {
    "polarity_flip": polarity_flip,
    "time_shuffle": time_shuffle,
    "coord_shuffle": coord_shuffle,
}
