"""Two-level seed x recording paired bootstrap, and max-T simultaneous CIs.

A bootstrap over test recordings alone holds the training seeds fixed, so
between-seed variance is invisible and the interval comes out too narrow. This
resamples seeds with replacement and, within each drawn seed, recordings with
replacement.

Also provides a max-T simultaneous comparison of one reference variant against
many others, controlling the familywise error rate so a "best variant" claim is
not an artifact of many marginal intervals.
"""

from __future__ import annotations

import numpy as np


def _delta_matrix(correct_left: dict, correct_right: dict) -> tuple[np.ndarray, list]:
    """Stack per-seed paired deltas (left - right) into an (S, N) matrix.

    Seeds are the intersection of both dicts (a model must exist on both sides
    of the pair for that seed to contribute). All arrays must share the same
    prefix ordering (guaranteed by the canonical deterministic test split).
    """
    seeds = sorted(set(correct_left) & set(correct_right))
    if not seeds:
        raise ValueError("no shared seeds between left and right runs")
    rows = []
    n = len(np.asarray(correct_left[seeds[0]]))
    for s in seeds:
        left = np.asarray(correct_left[s], dtype=np.float64)
        right = np.asarray(correct_right[s], dtype=np.float64)
        if left.shape != right.shape or left.shape[0] != n:
            raise ValueError(f"seed {s}: shape mismatch ({left.shape} vs {right.shape})")
        rows.append(left - right)
    return np.stack(rows, axis=0), seeds


def _bootstrap_stats(deltas: np.ndarray, n_bootstrap: int, rng: np.random.Generator) -> np.ndarray:
    """Crossed seed x prefix bootstrap statistic distribution (delta units).

    The prefixes are a *crossed* factor (the same N test prefixes are evaluated
    by every seed), not nested under seed. So each draw resamples S seeds AND a
    single *shared* prefix index vector, then averages the seed x prefix
    submatrix. (An earlier version resampled prefixes independently per seed,
    which treats prefixes as nested and slightly mis-estimates the CI; fixed per
    an independent review round.)
    """
    s_count, n = deltas.shape
    stats = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sidx = rng.integers(0, s_count, s_count)
        pidx = rng.integers(0, n, n)
        stats[b] = deltas[np.ix_(sidx, pidx)].mean()
    return stats


def seed_level_paired_t(correct_left: dict, correct_right: dict,
                        conf: float = 0.95) -> dict:
    """Conservative seed-level paired-t CI on the per-seed mean deltas.

    With only ~3 seeds a nonparametric seed bootstrap cannot manufacture
    negative seed effects from all-positive observations, so it understates
    uncertainty. The Student-t interval over the per-seed deltas is the more
    honest primary statistic; the bootstrap is a sensitivity check.
    """
    from statistics import NormalDist  # noqa: F401  (kept for clarity)

    deltas, seeds = _delta_matrix(correct_left, correct_right)
    seed_means = deltas.mean(axis=1) * 100.0
    s_count = len(seed_means)
    mean = float(seed_means.mean())
    if s_count < 2:
        return {"n_seeds": s_count, "mean_pp": mean,
                "ci_low_pp": float("nan"), "ci_high_pp": float("nan"),
                "seed_deltas_pp": [float(x) for x in seed_means]}
    sd = float(seed_means.std(ddof=1))
    se = sd / np.sqrt(s_count)
    # Student-t critical value (two-sided) without scipy: small lookup table.
    t_crit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
              6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}.get(s_count - 1, 1.96)
    half = t_crit * se
    return {
        "n_seeds": int(s_count),
        "seeds": list(seeds),
        "mean_pp": mean,
        "ci_low_pp": mean - half,
        "ci_high_pp": mean + half,
        "sd_pp": sd,
        "seed_deltas_pp": [float(x) for x in seed_means],
    }


def hierarchical_paired_bootstrap_ci(
    correct_left: dict,
    correct_right: dict,
    n_bootstrap: int = 2000,
    rng_seed: int = 0,
) -> dict:
    """Mean +/- 95% CI of a paired delta using a seed x prefix bootstrap.

    Args:
        correct_left / correct_right: dict seed -> per-sample correctness array
            ({0,1}). Pairing is by matching seed; prefixes must be aligned.
    Returns a JSON-friendly dict with point estimate, percentile CI, and the
    per-seed deltas + between-seed SD that drive the extra width.
    """
    deltas, seeds = _delta_matrix(correct_left, correct_right)
    s_count, n = deltas.shape
    seed_means = deltas.mean(axis=1)
    point_pp = float(seed_means.mean() * 100.0)

    rng = np.random.default_rng(rng_seed)
    stats = _bootstrap_stats(deltas, n_bootstrap, rng)
    lo, hi = np.percentile(stats, [2.5, 97.5])

    return {
        "n_seeds": int(s_count),
        "n_test": int(n),
        "seeds": list(seeds),
        "mean_pp": point_pp,
        "ci_low_pp": float(lo * 100.0),
        "ci_high_pp": float(hi * 100.0),
        "seed_deltas_pp": [float(x * 100.0) for x in seed_means],
        "seed_sd_pp": float(seed_means.std(ddof=1) * 100.0) if s_count > 1 else 0.0,
    }


def verdict(mean_pp: float, ci_low_pp: float, ci_high_pp: float,
            gain_threshold: float = 5.0) -> str:
    """Significance rule, mirroring ``cross_modal_eval_with_ci.verdict``."""
    if ci_low_pp > 0 and mean_pp >= gain_threshold:
        return "PASS"
    if ci_low_pp <= 0 <= ci_high_pp:
        return "INCONCLUSIVE (CI crosses zero)"
    if mean_pp < 0:
        return "WITHDRAW"
    return "INCONCLUSIVE (mean below threshold)"


def simultaneous_vs_reference(
    ref: dict,
    others: dict,
    n_bootstrap: int = 2000,
    rng_seed: int = 0,
    gain_threshold: float = 5.0,
) -> dict:
    """max-T simultaneous comparison of ``ref`` vs each variant in ``others``.

    Within each bootstrap draw the SAME resampled seeds/prefixes are applied to
    every comparison, preserving their correlation. The max-T statistic
    ``max_c |theta_c^* - theta_c| / se_c`` yields a family-wise 95% critical
    value ``q``; each comparison's simultaneous band is ``theta_c +/- q*se_c``,
    which is necessarily at least as wide as its marginal CI. A ``ref``-is-best
    claim should survive these simultaneous bands, not just marginal ones.

    Args:
        ref: dict seed -> correctness for the reference variant (e.g. V01).
        others: dict variant_name -> (dict seed -> correctness).
    """
    if not others:
        raise ValueError("need at least one comparison variant")

    data = {name: _delta_matrix(ref, oc)[0] for name, oc in others.items()}
    names = list(data)
    s_count, n = data[names[0]].shape
    point_pp = {name: float(data[name].mean(axis=1).mean() * 100.0) for name in names}

    rng = np.random.default_rng(rng_seed)
    boot = {name: np.empty(n_bootstrap) for name in names}
    for b in range(n_bootstrap):
        # Crossed resample shared across all comparisons within a draw, to
        # preserve their correlation (same seeds + same prefixes everywhere).
        sidx = rng.integers(0, s_count, s_count)
        pidx = rng.integers(0, n, n)
        sel = np.ix_(sidx, pidx)
        for name in names:
            boot[name][b] = data[name][sel].mean()

    se = {name: float(boot[name].std(ddof=1)) for name in names}
    max_t = np.zeros(n_bootstrap)
    for name in names:
        if se[name] > 0:
            t = np.abs((boot[name] - boot[name].mean()) / se[name])
            max_t = np.maximum(max_t, t)
    q = float(np.percentile(max_t, 97.5))

    per = {}
    for name in names:
        lo, hi = np.percentile(boot[name], [2.5, 97.5])
        half = q * se[name]
        per[name] = {
            "mean_pp": point_pp[name],
            "ci_low_pp": float(lo * 100.0),
            "ci_high_pp": float(hi * 100.0),
            "sim_ci_low_pp": float((point_pp[name] / 100.0 - half) * 100.0),
            "sim_ci_high_pp": float((point_pp[name] / 100.0 + half) * 100.0),
            "se_pp": float(se[name] * 100.0),
            "marginal_verdict": verdict(point_pp[name], lo * 100.0, hi * 100.0, gain_threshold),
            "simultaneous_verdict": verdict(
                point_pp[name], (point_pp[name] / 100.0 - half) * 100.0,
                (point_pp[name] / 100.0 + half) * 100.0, gain_threshold),
        }
    return {
        "per_comparison": per,
        "maxT_q": q,
        "n_seeds": int(s_count),
        "n_test": int(n),
        "n_comparisons": len(names),
    }
