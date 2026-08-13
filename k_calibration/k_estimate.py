#!/usr/bin/env python3
"""Three-step regression for the drift parameters k1, k2, k4, k5.

Fits three weighted linear regressions over binned brightness and brightness
change. Accepts either the ``.pt`` tensors written by ``k_preprocess`` or a
pre-computed fit-results CSV.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="K parameter estimation via three-step regression (Eq. 7-9)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # input: either --data_dir (raw) or --input (CSV)
    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument(
        "--data_dir",
        help="Directory containing calib scenes with events_with_luminance_*.pt",
    )
    inp.add_argument(
        "--input", help="Path to fit_results CSV (legacy, from calib_distribution.py)"
    )
    p.add_argument("--output_dir", required=True, help="Directory for outputs")
    p.add_argument("--pair", default=None, help="Device-pair name (e.g. Raw2DVS346)")
    p.add_argument(
        "--source",
        choices=["raw", "rgb"],
        default="raw",
        help="Luminance source type (selects which .pt file to load)",
    )
    p.add_argument(
        "--dt",
        type=float,
        default=16666.7,
        help="Frame interval in microseconds (only used for CSV input)",
    )
    p.add_argument("--min_count", type=int, default=50, help="Min events per bin")
    p.add_argument(
        "--n_lbar_bins",
        type=int,
        default=20,
        help="Number of L-bar bins for 2D binning",
    )
    p.add_argument(
        "--n_dl_bins",
        type=int,
        default=20,
        help="Number of dL bins for 2D binning",
    )
    p.add_argument(
        "--lbar_min",
        type=float,
        default=None,
        help="Minimum mean luminance (L-bar) to keep. Events/bins below this are excluded.",
    )
    p.add_argument(
        "--lbar_max",
        type=float,
        default=None,
        help="Maximum mean luminance (L-bar) to keep. Events/bins above this are excluded.",
    )
    p.add_argument(
        "--plot", action="store_true", help="Generate additional diagnostic plots"
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Weighted least squares
# ---------------------------------------------------------------------------


def wls(
    X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None
) -> tuple[np.ndarray, float, float]:
    """Weighted least squares with intercept.

    Parameters
    ----------
    X : (n, p) feature matrix (no intercept column -- added internally).
    y : (n,) targets.
    w : (n,) positive weights.  None -> uniform.

    Returns
    -------
    beta : (p+1,) coefficients.  beta[:-1] = slopes, beta[-1] = intercept.
    r2   : weighted R-squared.
    rmse : weighted RMSE.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape

    # add intercept column
    X_aug = np.column_stack([X, np.ones(n)])

    if w is None:
        w = np.ones(n)
    w = np.asarray(w, dtype=np.float64)

    W_sqrt = np.sqrt(w)
    Xw = X_aug * W_sqrt[:, None]
    yw = y * W_sqrt

    beta, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)

    y_pred = X_aug @ beta
    ss_res = np.sum(w * (y - y_pred) ** 2)
    y_mean = np.sum(w * y) / np.sum(w)
    ss_tot = np.sum(w * (y - y_mean) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = np.sqrt(ss_res / np.sum(w)) if np.sum(w) > 0 else 0.0

    return beta, r2, rmse


# ---------------------------------------------------------------------------
# Raw data loading: .pt tensors -> intervals -> IG fit -> DataFrame
# ---------------------------------------------------------------------------

# IG distribution threshold (paper: Theta)
_THETA = 1.0
# Maximum inter-event interval in microseconds (1 second)
_MAX_INTERVAL_US = 1_000_000.0


def _ig_mle(tau: np.ndarray, threshold: float = _THETA) -> tuple[float, float]:
    """Inverse Gaussian MLE for drift rate and lambda.

    Parameters
    ----------
    tau : positive inter-event intervals (seconds).
    threshold : Theta.

    Returns
    -------
    mu_hat  : drift rate magnitude = Theta / mean(tau).
    lambda_hat : shape parameter.
    """
    delta_hat = tau.mean()
    n = len(tau)
    denom = np.sum((tau - delta_hat) ** 2 / (tau * delta_hat**2))
    lambda_hat = n / denom if denom > 0 else float(n)
    mu_hat = threshold / delta_hat if delta_hat > 0 else 0.0
    return mu_hat, lambda_hat


def load_pt_data(
    data_dir: str,
    source: str,
    n_lbar_bins: int = 20,
    n_dl_bins: int = 20,
    min_events: int = 50,
    lbar_min: float | None = None,
    lbar_max: float | None = None,
) -> pd.DataFrame:
    """Load .pt luminance tensors, compute intervals, bin, fit IG.

    Each .pt file has shape (N, 7) with columns:
        [timestamp_us, x, y, polarity, prev_lum, next_lum, frame_time_diff_us]

    Pipeline:
        1. Load and concatenate all scene tensors.
        2. Sort events by (x, y, timestamp) within each pixel.
        3. Compute inter-event intervals between consecutive events at same pixel.
        4. Associate each interval with the second event's luminance.
        5. Bin by (L_bar, dL) on a 2D grid.
        6. Fit IG per bin -> mu_hat (drift rate magnitude).
        7. Return DataFrame with columns matching the regression interface.
    """
    import torch

    data_path = Path(data_dir)
    suffix = f"events_with_luminance_{source}.pt"
    pt_files = sorted(data_path.rglob(suffix))
    if not pt_files:
        raise FileNotFoundError(f"No {suffix} files found under {data_dir}")

    # 1. Load and concatenate, tagging each event with its scene ID
    arrays = []
    for scene_id, f in enumerate(pt_files):
        t = (
            torch.load(f, weights_only=True, map_location="cpu")
            .numpy()
            .astype(np.float64)
        )
        # append scene_id column
        sid = np.full((t.shape[0], 1), scene_id, dtype=np.float64)
        arrays.append(np.hstack([t, sid]))
        print(f"  Loaded {f.relative_to(data_path)}: {t.shape[0]:,} events")
    data = np.concatenate(arrays, axis=0)
    n_events = len(data)
    print(f"  Total: {n_events:,} events from {len(pt_files)} scene(s)")

    ts = data[:, 0]  # timestamp in microseconds
    x = data[:, 1].astype(np.int32)
    y = data[:, 2].astype(np.int32)
    polarity = data[:, 3].astype(np.int32)
    prev_lum = data[:, 4]
    next_lum = data[:, 5]
    dt_frame = data[:, 6]  # frame_time_diff in microseconds
    scene_id = data[:, 7].astype(np.int32)

    # 2. Sort by (scene, x, y, timestamp) for per-pixel-per-scene grouping
    order = np.lexsort((ts, y, x, scene_id))
    ts = ts[order]
    x = x[order]
    y = y[order]
    polarity = polarity[order]
    prev_lum = prev_lum[order]
    next_lum = next_lum[order]
    dt_frame = dt_frame[order]
    scene_id = scene_id[order]

    # 3. Identify pixel boundaries and compute intervals
    # Only pair consecutive events from the SAME scene AND same pixel
    same_pixel = (np.diff(x) == 0) & (np.diff(y) == 0)
    same_scene = np.diff(scene_id) == 0
    dt = np.diff(ts)  # in microseconds
    valid = same_pixel & same_scene & (dt > 0) & (dt < _MAX_INTERVAL_US)

    # interval[i] = ts[i+1] - ts[i]; associate with event[i+1]
    # Keep in microseconds to match kdL units (dL / dt_frame_us)
    idx = np.where(valid)[0] + 1
    intervals_us = dt[valid]

    # luminance and polarity of the second event in each pair
    i_lbar = (prev_lum[idx] + next_lum[idx]) / 2.0
    i_dl = next_lum[idx] - prev_lum[idx]
    i_dt_frame = dt_frame[idx]
    i_polarity = polarity[idx]  # 1=ON, 0=OFF

    print(
        f"  {len(intervals_us):,} valid intervals (max {_MAX_INTERVAL_US / 1e6:.0f}s)"
    )

    if len(intervals_us) == 0:
        raise ValueError(
            "No valid inter-event intervals found. "
            "Check that event timestamps are not all identical "
            "(common cause: preprocessed .pt file has degenerate timestamps)."
        )

    # 3b. Apply luminance range filter
    lbar_mask = np.ones(len(i_lbar), dtype=bool)
    if lbar_min is not None:
        lbar_mask &= i_lbar >= lbar_min
    if lbar_max is not None:
        lbar_mask &= i_lbar <= lbar_max
    if not lbar_mask.all():
        n_before = len(i_lbar)
        intervals_us = intervals_us[lbar_mask]
        i_lbar = i_lbar[lbar_mask]
        i_dl = i_dl[lbar_mask]
        i_dt_frame = i_dt_frame[lbar_mask]
        i_polarity = i_polarity[lbar_mask]
        print(
            f"  L-bar filter [{lbar_min}, {lbar_max}]: "
            f"{n_before:,} -> {len(i_lbar):,} intervals "
            f"({n_before - len(i_lbar):,} removed)"
        )

    # 4. Create 2D bins (L_bar x dL)
    lbar_lo, lbar_hi = np.percentile(i_lbar, [1, 99])
    dl_lo, dl_hi = np.percentile(i_dl, [1, 99])
    # guard against degenerate ranges (constant luminance)
    if lbar_hi - lbar_lo < 1e-6:
        lbar_lo -= 0.5
        lbar_hi += 0.5
    if dl_hi - dl_lo < 1e-6:
        dl_lo -= 0.5
        dl_hi += 0.5
    lbar_edges = np.linspace(lbar_lo, lbar_hi, n_lbar_bins + 1)
    dl_edges = np.linspace(dl_lo, dl_hi, n_dl_bins + 1)

    lbar_bin = np.digitize(i_lbar, lbar_edges) - 1
    dl_bin = np.digitize(i_dl, dl_edges) - 1
    # clip to valid range (edges are inclusive at boundaries)
    lbar_bin = np.clip(lbar_bin, 0, n_lbar_bins - 1)
    dl_bin = np.clip(dl_bin, 0, n_dl_bins - 1)

    # 5. Fit IG per 2D bin
    rows = []
    for i in range(n_lbar_bins):
        for j in range(n_dl_bins):
            mask = (lbar_bin == i) & (dl_bin == j)
            count = mask.sum()
            if count < min_events:
                continue

            tau = intervals_us[mask]
            mu_hat, lambda_hat = _ig_mle(tau)

            lbar_center = (lbar_edges[i] + lbar_edges[i + 1]) / 2.0
            dl_center = (dl_edges[j] + dl_edges[j + 1]) / 2.0
            dt_mean = i_dt_frame[mask].mean()  # per-event frame interval

            # kdL = rate of luminance change (per microsecond in raw units)
            kdl = dl_center / dt_mean

            # signed drift rate: use event polarity (ON=+1, OFF=-1)
            pol = i_polarity[mask]
            frac_on = pol.mean()  # fraction of ON events in this bin
            sign = 1.0 if frac_on >= 0.5 else -1.0
            mu_signed = mu_hat * sign

            rows.append(
                {
                    "Lbar": lbar_center,
                    "dLbar": dl_center,
                    "kdL": kdl,
                    "MeanMin": lbar_edges[i],
                    "MeanMax": lbar_edges[i + 1],
                    "DiffMin": dl_edges[j],
                    "DiffMax": dl_edges[j + 1],
                    "Mu_signed": mu_signed,
                    "frac_on": frac_on,
                    "mu_hat": mu_hat,
                    "lambda_hat": lambda_hat,
                    "dt_frame_mean": dt_mean,
                    "Count": count,
                }
            )

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise ValueError(
            "No bins survived IG fitting (check data quality / min_events)"
        )
    print(f"  {len(df)} bins after IG fitting (min_events={min_events})")
    return df


# ---------------------------------------------------------------------------
# CSV data loading (legacy path)
# ---------------------------------------------------------------------------


def load_csv(
    path: str,
    min_count: int,
    dt: float,
    lbar_min: float | None = None,
    lbar_max: float | None = None,
) -> pd.DataFrame:
    """Load fit_results CSV and compute derived columns.

    Two CSV formats exist:

    Old format (has both ``MuHat`` and ``Mu`` columns):
        ``MuHat`` = delta_hat = mean(tau)  (IG mean, large values like 30 000)
        ``Mu``    = signed drift rate with INVERTED sign convention
        Physical drift rate = ``-Mu`` = 1/MuHat (unsigned magnitude)

    New format (``MuHat`` only, no ``Mu`` column):
        ``MuHat`` = mu_hat = Theta/mean(tau)  (drift rate magnitude, small values)
        Physical drift rate = MuHat (unsigned)

    In both cases we store the **signed** physical drift rate as ``Mu_signed``.
    """
    df = pd.read_csv(path)

    if "MuHat" not in df.columns and "Mu" not in df.columns:
        raise ValueError("CSV must contain 'MuHat' and/or 'Mu' column")

    df = df[df["Count"] >= min_count].copy()

    # derived features
    df["Lbar"] = (df["MeanMin"] + df["MeanMax"]) / 2.0
    df["dLbar"] = (df["DiffMin"] + df["DiffMax"]) / 2.0
    df["kdL"] = df["dLbar"] / dt

    # extract physical drift rate depending on CSV format
    if "Mu" in df.columns and "MuHat" in df.columns:
        # old format: Mu is signed drift rate (inverted), MuHat is IG mean
        # physical drift rate = -Mu
        df["_mu_mag"] = df["Mu"].abs()  # = 1/MuHat = drift rate magnitude
        df = df[df["_mu_mag"] > 0].copy()
        # sign: use -Mu (negate the inverted convention)
        df["Mu_signed"] = -df["Mu"]
        print(f"Loaded {len(df)} bins (old format: Mu_signed = -Mu)")
    elif "MuHat" in df.columns:
        # new format: MuHat is drift rate magnitude (unsigned)
        df = df[df["MuHat"].notna() & (df["MuHat"] > 0)].copy()
        # infer sign from polarity or kdL direction
        if "P" in df.columns:
            sign = df["P"].map({1: 1.0, 0: -1.0})
        else:
            sign = np.sign(df["kdL"])
        df["Mu_signed"] = df["MuHat"] * sign
        print(f"Loaded {len(df)} bins (new format: MuHat as drift rate)")
    else:
        # only Mu column
        df = df[df["Mu"].notna() & (df["Mu"] != 0)].copy()
        df["Mu_signed"] = -df["Mu"]
        print(f"Loaded {len(df)} bins (Mu-only format)")

    # Apply luminance range filter
    n_before = len(df)
    if lbar_min is not None:
        df = df[df["Lbar"] >= lbar_min].copy()
    if lbar_max is not None:
        df = df[df["Lbar"] <= lbar_max].copy()
    if len(df) < n_before:
        print(
            f"  L-bar filter [{lbar_min}, {lbar_max}]: "
            f"{n_before} -> {len(df)} bins ({n_before - len(df)} removed)"
        )

    if len(df) < 3:
        raise ValueError(f"Only {len(df)} valid rows after filtering (need >= 3)")

    return df


# ---------------------------------------------------------------------------
# Step 1: per-L-bar regression  mu = a*kdL + b   (paper Eq. 7)
# ---------------------------------------------------------------------------


def step1_per_lbar_regression(
    df: pd.DataFrame,
    n_lbar_bins: int = 10,
    min_bin_n: int = 10,
    min_bin_r2: float = 0.3,
) -> tuple[pd.DataFrame, int]:
    """For each unified L-bar bin, regress Mu_signed on kdL.

    Both polarities are merged into common L-bar bins so that each bin
    spans the full k_dL range (positive and negative), giving the
    regression enough dynamic range.

    Parameters
    ----------
    min_bin_n : minimum points per bin to attempt regression.
    min_bin_r2 : minimum R-squared to accept a bin's slope for Step 2.

    Returns the dataframe with a_n, b_n, lbar_n, r2_n columns added,
    and the count of valid regressions.
    """
    print("\n--- Step 1: mu = a*kdL + b  (per-L-bar bin, Eq. 7) ---")

    # create unified L-bar bins from the full range of Lbar values
    lbar_min, lbar_max = df["Lbar"].min(), df["Lbar"].max()
    bin_edges = np.linspace(lbar_min, lbar_max, n_lbar_bins + 1)
    df["lbar_bin"] = pd.cut(df["Lbar"], bins=bin_edges, include_lowest=True)

    df["a_n"] = np.nan
    df["b_n"] = np.nan
    df["lbar_n"] = np.nan
    df["r2_n"] = np.nan
    valid = 0

    for bin_label, group in df.groupby("lbar_bin", observed=True):
        if len(group) < min_bin_n:
            print(f"  L-bar {bin_label}: skipped (n={len(group)} < {min_bin_n})")
            continue

        X = group["kdL"].values
        y = group["Mu_signed"].values
        w = group["Count"].values.astype(np.float64)

        beta, r2, _ = wls(X, y, w)
        a_n, b_n = beta[0], beta[1]
        lbar = group["Lbar"].mean()

        tag = ""
        if r2 < min_bin_r2:
            tag = "  <- low R2, excluded from Step 2"
        elif a_n <= 0:
            tag = "  <- a<=0, excluded from Step 2"
        else:
            valid += 1

        df.loc[group.index, "a_n"] = a_n
        df.loc[group.index, "b_n"] = b_n
        df.loc[group.index, "lbar_n"] = lbar
        df.loc[group.index, "r2_n"] = r2
        print(
            f"  L-bar~{lbar:.1f} [{bin_label}]: mu = {a_n:.6g}*kdL + {b_n:.6g}"
            f"  (R2={r2:.4f}, n={len(group)}){tag}"
        )

    if valid < 2:
        raise ValueError(f"Only {valid} valid L-bar bins (need >= 2)")
    print(f"  -> {valid} valid L-bar bins (after quality filter)")
    return df, valid


# ---------------------------------------------------------------------------
# Step 2: cross-L-bar regression  1/a = (1/k1)*L-bar + k2/k1   (paper Eq. 8)
# ---------------------------------------------------------------------------


def step2_cross_lbar_regression(
    df: pd.DataFrame,
    min_bin_r2: float = 0.3,
) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Regress 1/a_n on L-bar to extract k1 and k2.

    Only uses bins where a > 0 and R-squared >= min_bin_r2 (quality filter).
    Uses Theil-Sen estimator for robustness to outlier bins.

    Returns (k1, k2, r2, lbar_arr, inv_a_arr, weights).
    """
    print("\n--- Step 2: 1/a = (1/k1)*L-bar + k2/k1  (Eq. 8) ---")

    # aggregate per unified L-bar bin (matching Step 1 binning)
    unique = (
        df.dropna(subset=["a_n", "lbar_bin"])
        .groupby("lbar_bin", observed=True)
        .agg({"a_n": "first", "lbar_n": "first", "r2_n": "first", "Count": "sum"})
        .reset_index()
    )

    # quality filter: positive slope and sufficient R-squared
    unique = unique[(unique["a_n"] > 0) & (unique["r2_n"] >= min_bin_r2)]
    print(f"  Using {len(unique)} bins after quality filter (a>0, R2>={min_bin_r2})")

    lbar = unique["lbar_n"].values
    inv_a = 1.0 / unique["a_n"].values
    w = unique["Count"].values.astype(np.float64)

    if len(lbar) < 2:
        raise ValueError(f"Only {len(lbar)} valid a_n values (need >= 2)")

    # Theil-Sen: median of all pairwise slopes, then median intercept
    n = len(lbar)
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            if lbar[j] != lbar[i]:
                slopes.append((inv_a[j] - inv_a[i]) / (lbar[j] - lbar[i]))
    inv_k1 = np.median(slopes)
    intercepts = inv_a - inv_k1 * lbar
    k2_over_k1 = np.median(intercepts)

    if inv_k1 == 0:
        raise ValueError("1/k1 = 0; regression degenerate")

    k1 = 1.0 / inv_k1
    k2 = k2_over_k1 * k1

    # compute R-squared for reporting (against the Theil-Sen line)
    y_pred = inv_k1 * lbar + k2_over_k1
    ss_res = np.sum(w * (inv_a - y_pred) ** 2)
    y_mean = np.average(inv_a, weights=w)
    ss_tot = np.sum(w * (inv_a - y_mean) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    print(f"  k1 = {k1:.6g}  (Theil-Sen)")
    print(f"  k2 = {k2:.6g}")
    print(f"  R2 = {r2:.4f}")

    return k1, k2, r2, lbar, inv_a, w


# ---------------------------------------------------------------------------
# Global profile-likelihood validation (grid search over k2)
# ---------------------------------------------------------------------------


def validate_global_fit(
    df: pd.DataFrame,
    k2_range: tuple[float, float] = (-100.0, 200.0),
    n_k2: int = 301,
) -> tuple[float, float, float, float, float]:
    """Direct global regression: mu = k1*kdL/(L-bar+k2) + k5*L-bar + k4.

    Grid-search k2, then OLS for (k1, k5, k4).
    Returns (k1, k2, k4, k5, r2).
    """
    print("\n--- Global validation: profile-likelihood over k2 ---")

    valid = df.dropna(subset=["Mu_signed"]).copy()
    lbar = valid["Lbar"].values
    kdl = valid["kdL"].values
    mu = valid["Mu_signed"].values
    w = valid["Count"].values.astype(np.float64)

    best_r2 = -np.inf
    best_k2 = 0.0
    best_beta = None

    for k2_try in np.linspace(k2_range[0], k2_range[1], n_k2):
        denom = lbar + k2_try
        if np.any(denom == 0):
            continue
        c = kdl / denom
        X = np.column_stack([c, lbar])
        beta, r2, _ = wls(X, mu, w)
        if r2 > best_r2:
            best_r2 = r2
            best_k2 = k2_try
            best_beta = beta

    k1_g, k5_g, k4_g = best_beta[0], best_beta[1], best_beta[2]

    print(f"  k1 = {k1_g:.6g}")
    print(f"  k2 = {best_k2:.6g}")
    print(f"  k4 = {k4_g:.6g}")
    print(f"  k5 = {k5_g:.6g}")
    print(f"  R2 = {best_r2:.4f}")

    return k1_g, best_k2, k4_g, k5_g, best_r2


# ---------------------------------------------------------------------------
# Step 3: global regression  mu = k1'*c + k5*L-bar + k4   (paper Eq. 9)
# ---------------------------------------------------------------------------


def step3_global_regression(
    df: pd.DataFrame, k2: float
) -> tuple[float, float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Multivariate regression on c_n = kdL/(L-bar+k2) and L-bar.

    Returns (k1_prime, k4, k5, r2, c_arr, lbar_arr, mu_arr, weights).
    """
    print("\n--- Step 3: mu = k1'*c + k5*L-bar + k4  (Eq. 9) ---")

    # use all individual data points (not aggregated per bin)
    valid = df.dropna(subset=["Lbar", "Mu_signed"]).copy()
    denom = valid["Lbar"].values + k2
    nonzero = denom != 0
    valid = valid[nonzero]
    denom = denom[nonzero]

    c_arr = valid["kdL"].values / denom
    lbar_arr = valid["Lbar"].values
    mu_arr = valid["Mu_signed"].values
    w_arr = valid["Count"].values.astype(np.float64)

    if len(c_arr) < 3:
        raise ValueError(f"Only {len(c_arr)} bins for Step 3 (need >= 3)")

    X = np.column_stack([c_arr, lbar_arr])
    beta, r2, _ = wls(X, mu_arr, w_arr)
    # beta = [k1', k5, k4(intercept)]
    k1_prime = beta[0]
    k5 = beta[1]
    k4 = beta[2]

    print(f"  k1' = {k1_prime:.6g}")
    print(f"  k4  = {k4:.6g}")
    print(f"  k5  = {k5:.6g}")
    print(f"  R2  = {r2:.4f}")

    return k1_prime, k4, k5, r2, c_arr, lbar_arr, mu_arr, w_arr


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_results(
    output_dir: Path,
    pair: str | None,
    source: str,
    k1: float,
    k2: float,
    k4: float,
    k5: float,
    k1_prime: float,
    *,
    k1_global: float | None = None,
    k2_global: float | None = None,
) -> Path:
    """Write stage1_params.json."""
    k1_err = abs(k1_prime - k1) / abs(k1) * 100 if k1 != 0 else float("inf")
    consistency = "pass" if k1_err <= 5.0 else "fail"

    result = {
        "pair": pair,
        "source": source,
        "stage1_params": {
            "k1": k1,
            "k2": k2,
            "k4": k4,
            "k5": k5,
        },
        "k1_prime": k1_prime,
        "k1_error_percent": round(k1_err, 4),
        "regression_consistency": consistency,
        "calibration_date": date.today().isoformat(),
    }
    if k1_global is not None:
        result["global_validation"] = {"k1": k1_global, "k2": k2_global}

    out_path = output_dir / "stage1_params.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)

    print(f"\n{'=' * 50}")
    print(
        f"k1 = {k1:.6g}   k1' = {k1_prime:.6g}   error = {k1_err:.2f}%  [{consistency}]"
    )
    print(f"k2 = {k2:.6g}")
    print(f"k4 = {k4:.6g}")
    print(f"k5 = {k5:.6g}")
    if k1_global is not None:
        g_err = abs(k1_global - k1) / abs(k1) * 100 if k1 != 0 else float("inf")
        print(
            f"k1(global) = {k1_global:.6g}   k2(global) = {k2_global:.6g}"
            f"   dk1 = {g_err:.1f}%"
        )
    print(f"-> {out_path}")
    if consistency == "fail":
        print(
            f"\nWARNING: k1/k1' consistency check failed ({k1_err:.1f}% > 5%)."
            " Stage 1 results may be unreliable for downstream use.",
            file=sys.stderr,
        )
    return out_path


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_step2(
    lbar: np.ndarray,
    inv_a: np.ndarray,
    w: np.ndarray,
    k1: float,
    k2: float,
    output_dir: Path,
) -> None:
    """Step 2 scatter + fit line: 1/a vs L-bar."""
    fig, ax = plt.subplots(figsize=(9, 6))

    sizes = w / w.max() * 200 + 20
    ax.scatter(lbar, inv_a, s=sizes, alpha=0.7, label="Data")

    x_line = np.linspace(lbar.min(), lbar.max(), 100)
    y_line = (1.0 / k1) * x_line + k2 / k1
    ax.plot(
        x_line,
        y_line,
        "r-",
        lw=2,
        label=f"1/a = (1/k1)*L-bar + k2/k1\nk1={k1:.4g}, k2={k2:.4g}",
    )

    ax.set_xlabel("L-bar (mean luminance)")
    ax.set_ylabel("1/a")
    ax.set_title("Step 2: Cross-L-bar regression (Eq. 8)")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / "step2_regression.png", dpi=200)
    plt.close(fig)


def plot_step3(
    c_arr: np.ndarray,
    lbar_arr: np.ndarray,
    mu_arr: np.ndarray,
    w_arr: np.ndarray,
    k1_prime: float,
    k4: float,
    k5: float,
    output_dir: Path,
) -> None:
    """Step 3 scatter: mu vs c (coloured by L-bar)."""
    fig, ax = plt.subplots(figsize=(9, 6))

    sizes = w_arr / w_arr.max() * 200 + 20
    sc = ax.scatter(c_arr, mu_arr, c=lbar_arr, s=sizes, alpha=0.7, cmap="viridis")
    plt.colorbar(sc, ax=ax, label="L-bar")

    c_line = np.linspace(c_arr.min(), c_arr.max(), 100)
    l_mean = np.average(lbar_arr, weights=w_arr)
    y_line = k1_prime * c_line + k5 * l_mean + k4
    ax.plot(
        c_line,
        y_line,
        "r-",
        lw=2,
        label=f"mu = k1'*c + k5*L-bar + k4\nk1'={k1_prime:.4g}, k5={k5:.4g}, k4={k4:.4g}",
    )

    ax.set_xlabel("c = kdL / (L-bar + k2)")
    ax.set_ylabel("mu (drift rate)")
    ax.set_title("Step 3: Global regression (Eq. 9)")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / "step3_regression.png", dpi=200)
    plt.close(fig)


def plot_step1_overlay(df: pd.DataFrame, output_dir: Path) -> None:
    """Step 1 overlay: all unified L-bar bins' mu vs kdL with fit lines."""
    fig, ax = plt.subplots(figsize=(10, 7))

    valid = df.dropna(subset=["a_n", "lbar_bin"])
    bins = valid.groupby("lbar_bin", observed=True).first().reset_index()
    cmap = plt.cm.viridis
    norm = plt.Normalize(bins["lbar_n"].min(), bins["lbar_n"].max())

    for _, row in bins.iterrows():
        bl = row["lbar_bin"]
        sub = valid[valid["lbar_bin"] == bl]

        colour = cmap(norm(row["lbar_n"]))
        ax.scatter(sub["kdL"], sub["Mu_signed"], color=colour, s=20, alpha=0.5)

        x = np.linspace(sub["kdL"].min(), sub["kdL"].max(), 50)
        ax.plot(x, row["a_n"] * x + row["b_n"], color=colour, lw=1.2)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="L-bar")

    ax.set_xlabel("kdL = dL / dt")
    ax.set_ylabel("mu (drift rate)")
    ax.set_title("Step 1: Per-L-bar regressions (Eq. 7)")
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / "step1_all_bins.png", dpi=200)
    plt.close(fig)


def plot_step1_3d(
    df: pd.DataFrame,
    k1: float,
    k2: float,
    k4: float,
    k5: float,
    output_dir: Path,
) -> None:
    """Step 1 3D surface: mu vs (dL, L-bar). DVS-Voltmeter Fig. 2(a) style."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    valid = df.dropna(subset=["Mu_signed"])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # scatter: data points
    ax.scatter(
        valid["dLbar"],
        valid["Lbar"],
        valid["Mu_signed"],
        c="steelblue",
        s=8,
        alpha=0.5,
        depthshade=True,
    )

    # fit surface: mu = k1 * kdL/(L+k2) + k4 + k5*L
    dl_range = np.linspace(valid["dLbar"].min(), valid["dLbar"].max(), 40)
    l_range = np.linspace(valid["Lbar"].min(), valid["Lbar"].max(), 40)
    DL, L = np.meshgrid(dl_range, l_range)
    # use mean dt_frame from data
    dt_mean = (
        valid["dt_frame_mean"].mean() if "dt_frame_mean" in valid.columns else 16667.0
    )
    KDL = DL / dt_mean
    MU = k1 * KDL / (L + k2) + k4 + k5 * L

    ax.plot_surface(DL, L, MU, alpha=0.25, color="salmon", edgecolor="none")

    ax.set_xlabel(r"$\Delta L$", fontsize=12)
    ax.set_ylabel(r"$\bar{L}$", fontsize=12)
    ax.set_zlabel(r"$\mu$", fontsize=12)
    ax.set_title(r"(a) $\mu = a(\bar{L})\Delta L + b(\bar{L})$", fontsize=13)
    ax.view_init(elev=25, azim=-50)
    fig.tight_layout()
    fig.savefig(output_dir / "fig2a_step1_3d.png", dpi=200)
    plt.close(fig)


def plot_step3_3d(
    c_arr: np.ndarray,
    lbar_arr: np.ndarray,
    mu_arr: np.ndarray,
    k1_prime: float,
    k4: float,
    k5: float,
    output_dir: Path,
) -> None:
    """Step 3 3D surface: mu vs (L-bar, c). DVS-Voltmeter Fig. 2(c) style."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # scatter: data points
    ax.scatter(
        lbar_arr,
        c_arr,
        mu_arr,
        c="steelblue",
        s=8,
        alpha=0.5,
        depthshade=True,
    )

    # fit surface: mu = k1'*c + k5*L + k4
    l_range = np.linspace(lbar_arr.min(), lbar_arr.max(), 40)
    c_range = np.linspace(c_arr.min(), c_arr.max(), 40)
    L, C = np.meshgrid(l_range, c_range)
    MU = k1_prime * C + k5 * L + k4

    ax.plot_surface(L, C, MU, alpha=0.25, color="salmon", edgecolor="none")

    ax.set_xlabel(r"$\bar{L}$", fontsize=12)
    ax.set_ylabel(r"$c$", fontsize=12)
    ax.set_zlabel(r"$\mu$", fontsize=12)
    ax.set_title(r"(c) $\mu = k_1 c + k_5 \bar{L} + k_4$", fontsize=13)
    ax.view_init(elev=25, azim=-50)
    fig.tight_layout()
    fig.savefig(output_dir / "fig2c_step3_3d.png", dpi=200)
    plt.close(fig)


def plot_summary(
    k1: float,
    k2: float,
    k4: float,
    k5: float,
    k1_prime: float,
    pair: str | None,
    output_dir: Path,
) -> None:
    """One-page text summary figure."""
    k1_err = abs(k1_prime - k1) / abs(k1) * 100 if k1 != 0 else float("inf")
    consistency = "PASS" if k1_err <= 5.0 else "FAIL"

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    lines = [
        "K Parameter Summary" + (f"  ({pair})" if pair else ""),
        "",
        f"k1  = {k1:.6g}",
        f"k2  = {k2:.6g}",
        f"k4  = {k4:.6g}",
        f"k5  = {k5:.6g}",
        "",
        f"k1' = {k1_prime:.6g}  (Step 3 verification)",
        f"|k1' - k1|/|k1| = {k1_err:.2f}%  [{consistency}]",
    ]
    ax.text(
        0.05,
        0.95,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        fontsize=12,
        fontfamily="monospace",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "K_parameters_summary.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load data
    if args.data_dir:
        source = args.source
        print(f"Loading raw .pt data from {args.data_dir} (source={source})")
        df = load_pt_data(
            args.data_dir,
            source,
            n_lbar_bins=args.n_lbar_bins,
            n_dl_bins=args.n_dl_bins,
            min_events=args.min_count,
            lbar_min=args.lbar_min,
            lbar_max=args.lbar_max,
        )
        # save intermediate fit results for inspection
        csv_path = output_dir / f"{source}_fit_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved fit results to {csv_path}")
    else:
        input_name = Path(args.input).stem.upper()
        source = "raw" if "RAW" in input_name else "rgb"
        df = load_csv(
            args.input, args.min_count, args.dt,
            lbar_min=args.lbar_min, lbar_max=args.lbar_max,
        )

    # step 1
    df, n_valid = step1_per_lbar_regression(df)

    # step 2
    k1, k2, r2_step2, lbar_s2, inv_a_s2, w_s2 = step2_cross_lbar_regression(df)

    # step 3
    k1_prime, k4, k5, r2_step3, c_s3, lbar_s3, mu_s3, w_s3 = step3_global_regression(
        df, k2
    )

    # global validation
    k1_g, k2_g, k4_g, k5_g, r2_g = validate_global_fit(df)

    # save
    save_results(
        output_dir,
        args.pair,
        source,
        k1,
        k2,
        k4,
        k5,
        k1_prime,
        k1_global=k1_g,
        k2_global=k2_g,
    )

    # plots (always: step2, step3, summary; optional: step1 overlay)
    plot_step2(lbar_s2, inv_a_s2, w_s2, k1, k2, output_dir)
    plot_step3(c_s3, lbar_s3, mu_s3, w_s3, k1_prime, k4, k5, output_dir)
    plot_summary(k1, k2, k4, k5, k1_prime, args.pair, output_dir)

    if args.plot:
        plot_step1_overlay(df, output_dir)
        plot_step1_3d(df, k1, k2, k4, k5, output_dir)
        plot_step3_3d(c_s3, lbar_s3, mu_s3, k1_prime, k4, k5, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
