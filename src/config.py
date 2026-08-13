#! python3
# -*- encoding: utf-8 -*-

import json

from easydict import EasyDict as edict

__C = edict()
cfg = __C

# SENSOR
__C.SENSOR = edict()

# K parameters for each camera type.
# Raw2DVS346 / RGB2DVS346: Stage 1 physical regression (k_calibration/k_estimate.py)
# with DVS-Voltmeter DVS346 priors k3=1e-4, k6=1e-5. Stage 2 Optuna refinement
# is bypassed because of a known identifiability degeneracy under
# uniform-screen calibration scenes.
# The Raw2DVS346 k4 entry is further adjusted to 1e-6 via the global k4
# polarity probe (tools/k_diagnostics/global_k4_polarity_probe*.py): this
# lifts polarity_delta from 0.176 -> 0.109 and active_pixel_ratio from
# 1.55 -> 1.01 across 3 CIFAR samples (4/5 fidelity dimensions pass).
# Override a specific pair at runtime via ``load_K_from_file("<pair>_K.json")``
# or ``set_camera_type("<pair>", K_file=...)`` when a newer calibration is available.
K_MAP = {
    "DVS346": [0.00018 * 29250, 20, 0.0001, 1e-7, 5e-9, 0.00001],
    "DVS240": [0.000094 * 47065, 23, 0.0002, 1e-7, 5e-8, 0.00001],
    "RGB2DVS346": [
        7.612017342884227,
        204.46017142991028,
        1e-4,
        7.354632959846225e-06,
        -2.5565783709920732e-08,
        1e-5,
    ],
    "Raw2DVS346": [
        1.6612259044483269,
        -35.55831455106693,
        1e-4,
        1e-6,
        -2.0353994379776813e-09,
        1e-5,
    ],
}


def load_K_from_file(json_path):
    """Load device-pair K parameters from a calibration JSON file and add to K_MAP.

    The JSON must contain at least:
        {"pair": "<camera_type>", "K": [k1, k2, k3, k4, k5, k6]}

    K may also be 8-dim [k1..k6, k_on, k_off]; a 6-dim K is padded with
    k_on = k_off = 1.0. An optional ``"version"`` field is read when present but
    not enforced — the length is the source of truth.

    Returns:
        pair (str): the camera-type key that was added/updated in K_MAP.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    pair = data.get("pair")
    K = data.get("K")
    if pair is None or K is None:
        raise ValueError(f"JSON file '{json_path}' must contain 'pair' and 'K' fields.")
    if len(K) not in (6, 8):
        raise ValueError(
            f"K must have 6 (legacy) or 8 (polarity-aware) elements, got {len(K)}."
        )
    # Pad 6-dim K with default unit thresholds so cfg.SENSOR.K is always 8-dim.
    if len(K) == 6:
        K = list(K) + [1.0, 1.0]

    K_MAP[pair] = K
    return pair


def get_K(pair):
    """Return the K parameters registered for ``pair`` in ``K_MAP``.

    Raises ``RuntimeError`` with actionable guidance when the pair is unknown.
    """
    if pair not in K_MAP:
        raise RuntimeError(
            f"No K parameters registered for '{pair}'. Run the k_calibration "
            "pipeline (k_preprocess.py --source raw → k_estimate.py → "
            f"k_optimize.py) and either paste the result into src/config.py "
            f"K_MAP['{pair}'] or call src.config.load_K_from_file('{pair}_K.json')."
        )
    return K_MAP[pair]


def set_camera_type(camera_type, K_file=None):
    """Dynamically update the sensor camera type and its K parameters.

    Args:
        camera_type: Key in K_MAP (e.g. 'Raw2DVS346').
        K_file: Optional path to a calibration JSON file.  If provided, the K
                values from the file are loaded into K_MAP before setting the
                camera type, allowing calibrated parameters to override defaults.
    """
    if K_file is not None:
        loaded_pair = load_K_from_file(K_file)
        if loaded_pair != camera_type:
            raise ValueError(
                f"K_file pair '{loaded_pair}' does not match camera_type '{camera_type}'."
            )
    if camera_type not in K_MAP:
        raise ValueError(
            f"Unknown camera type: {camera_type}. Valid types: {list(K_MAP.keys())}"
        )
    cfg.SENSOR.CAMERA_TYPE = camera_type
    cfg.SENSOR.K = K_MAP[camera_type]


# DVS346 is the import-time default so ``cfg.SENSOR.K`` is always populated.
# Real simulation call sites should pass ``k_values=get_K("Raw2DVS346")``
# (or the appropriate pair key) explicitly rather than relying on this default.
set_camera_type("DVS346")


# Directories
__C.DIR = edict()
__C.DIR.IN_PATH = "data_samples/interp/"
__C.DIR.OUT_PATH = "data_samples/output/"


# Visualize
__C.Visual = edict()
__C.Visual.FRAME_STEP = 5
