# Raw2Event

Code for the Raw2Event benchmark: paired pre-ISP raw-Bayer frames, post-ISP RGB
frames, and real DAVIS346 events captured from the same scenes, together with the
tooling used to turn frames into events and to compare simulated streams against
the matched real recordings.

The package covers five things:

- device-pair `K` calibration for the DVS-Voltmeter event model,
- frame-to-event generation and a twelve-variant reference simulator suite
  (DVS-Voltmeter and v2e, on raw and RGB inputs, with and without frame
  interpolation),
- upstream comparison of simulated streams against matched real events,
- downstream CIFAR-10 classification and retrieval on simulated and real events,
- the acquisition software that produced the dataset.

The dataset is hosted separately:
<https://huggingface.co/datasets/raw2event/raw2event>.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The v2e baseline is an external project. `external/v2e` ships as a placeholder:

```bash
# from a Git checkout
git submodule update --init --recursive external/v2e
# or, from a plain archive
git clone https://github.com/SensorsINI/v2e.git external/v2e

cd external/v2e && git checkout cac638444557a224ba6ee5c29260916f152c00bd && cd -
```

Retraining the spiking classifier additionally needs the QKFormer backbone.
It ships as a submodule at `train_class/QKFormer`, which is where
`train_class/train_qkformer.py` expects to find `model.py`, `utils.py`, and
`autoaugment.py` under `cifar10-dvs/`:

```bash
# from a Git checkout
git submodule update --init --recursive train_class/QKFormer
# or, from a plain archive
git clone https://github.com/linenmin/QKFormer.git train_class/QKFormer

pip install -r train_class/requirements.txt
```

The regression tests run from the repository root:

```bash
pytest tests/ -q
```

---

## Data

Each recording provides three synchronized streams of one scene: pre-ISP raw
Bayer frames, post-ISP RGB frames, and real DAVIS346 events. After per-modality
cropping and rescaling, all three share a unified `80x80` canvas (`unified80`)
derived from `346x260` sensor coordinates.

Download `unified80/` into `./data/unified80/`, or point `--data_dir` elsewhere.
The recording lists used by the commands below ship under
`tools/v2e_baseline/`:

| File | Contents |
|---|---|
| `tools/v2e_baseline/canonical_test_600_prefixes.txt` | 600-recording evaluation set |
| `tools/v2e_baseline/canonical_test_559_prefixes.txt` | 559 recordings runnable through v2e |
| `tools/v2e_baseline/canonical_test_v2e_runnable.txt` | v2e runnability record for the above |
| `tools/v2e_baseline/canonical_test_dropped_no_aedat.txt` | 41 recordings without an AEDAT4 file |

NPZ convention: `events` has shape `(N, 4)` with columns `[t_us, x, y, p]`.

---

## Pipeline

Every step below is runnable as shown; add `--help` to any command for the full
argument list.

### 1. Device-pair `K` calibration

Associate real events with frame brightness, then fit the physical parameters by
three-step regression, once per input branch. The released values live in
`src/config.py:K_MAP`.

```bash
python k_calibration/k_preprocess.py --data_dir <calibration_data> --source both

python k_calibration/k_estimate.py \
    --data_dir <calibration_data> --output_dir <raw_fit> \
    --source raw --pair Raw2DVS346

python k_calibration/k_estimate.py \
    --data_dir <calibration_data> --output_dir <rgb_fit> \
    --source rgb --pair RGB2DVS346
```

`k_calibration/k_optimize.py` provides an optional Optuna refinement over the
full parameter vector. The `k_4` value used for the raw branch is selected by a
polarity scan:

```bash
python -m tools.k_diagnostics.global_k4_polarity_probe --help
python -m tools.k_diagnostics.global_k4_polarity_probe_refine --help
```

### 2. Event generation

Filename-based modality detection picks the raw or RGB device-pair parameters;
`--is_rgb` forces the same choice for direct frame-file input.

```bash
python generate_event.py \
    --video <recording.mkv> --method dvs --sim_backend auto \
    --output <output_stem> --save_aedat4
```

Batch mode over a directory:

```bash
python generate_event.py \
    --batch_dir <input_dir> --output_dir <output_dir> \
    --method dvs --workers 8
```

### 3. Reference simulator variants

The twelve reference variants combine two simulator families, two input
branches, default and tuned parameters, and optional SuperSloMo interpolation.

```bash
bash tools/v2e_baseline/download_slomo_ckpt.sh

# one v2e variant over a recording list
python -m tools.v2e_baseline.run_v2e_batch \
    --variant V05 --input_dir ./data --output_dir ./output/sim/V05 \
    --prefix_list tools/v2e_baseline/canonical_test_559_prefixes.txt

# DVS-Voltmeter with the literature-default K
python -m tools.v2e_baseline.raw_dvsv_default_k_gen --help

# contrast-threshold sweep behind the tuned v2e settings
python -m tools.v2e_baseline.threshold_sweep --help
```

`tools/v2e_baseline/smoke_one_prefix.py` runs a single recording end to end as a
sanity check before launching a batch.

### 4. Data preparation for training

```bash
python train_class/process_all_batches.py --help        # offline K
python train_class/process_all_batches_runtime_k.py --help  # runtime-loaded K
python train_class/rescale_npz_to_unified.py --help     # to the unified canvas
```

### 5. Upstream comparison against real events

Scores each simulated stream against its matched DAVIS346 recording along seven
per-recording dimensions: event-count ratio, polarity deviation, active-pixel
ratio, spatial entropy ratio, per-pixel spatial EMD, inter-event-interval total
variation, and interval Sinkhorn EMD.

```bash
python tools/v2e_baseline/k_health_v2e_compare.py \
    --config <variant_config.json> \
    --dv_npz_dir ./data/unified80 \
    --prefix_list tools/v2e_baseline/canonical_test_559_prefixes.txt \
    --output ./output/upstream/k_health_table.json
```

Writes `k_health_table.json` (aggregates plus per-recording rows) and a
`k_health_table.md` rendering of the aggregate table. Start from
`tools/v2e_baseline/k_health_config_template.json`; its companion README
documents the variant-to-directory mapping. `tools/k_diagnostics/k_health_check.py`
is the per-recording implementation behind it.

### 6. Learned realism metric (EQS)

EQS compares two event streams in the latent space of a pretrained RVT detection
backbone. It needs torch, a GPU, and an RVT checkout. Run the perturbation
controls first: if structure-destroyed copies of real recordings do not score
clearly below the identity score, the adapter is not discriminative on this data.

```bash
python -m tools.eqs.eqs_compare \
    --config <variant_config.json> --dv_npz_dir ./data/unified80 \
    --prefix_list tools/v2e_baseline/canonical_test_559_prefixes.txt \
    --rvt_repo ./external/RVT --checkpoint ./external/rvt_gen1_small.ckpt \
    --controls --output ./output/eqs/eqs_summary.json

python -m tools.eqs.eqs_downstream_correlation \
    --upstream_json ./output/eqs/eqs_summary.json \
    --downstream_json ./output/cross_modal_eval/summary.json \
    --downstream_key real_dv_acc --output ./output/eqs/eqs_vs_transfer.json
```

`tools/eqs/eqs_score.py` is a pure-NumPy implementation of the metric core and
runs without torch.

### 7. Classifier training

Two architectures, three seeds each. Both pin the PyTorch, NumPy, Python and
DataLoader-worker RNGs. QKFormer selects its best epoch on same-modality test
accuracy; MobileNetV2 selects on a disjoint 60-per-class validation split.

```bash
python train_class/train_qkformer.py \
    --data_dir ./data/unified80 --modality raw \
    --epochs 96 --T 16 --batch_size 16 --seed 0

python train_class/train_mobileNetV2.py \
    --data_dir ./data/unified80 --modality raw \
    --epochs 100 --batch_size 16 \
    --representation stacked_histogram --rep_T 10 --rep_count_cutoff 10 \
    --seed 0
```

Both trainers log to Weights & Biases by default; edit `train_class/wandb_env.yaml`
(or copy it to `wandb_env.local.yaml`) to point at your own project, or pass
`--no_wandb` to turn logging off.

`--representation timestack` selects a polarity-agnostic count image instead of
the stacked histogram. `--frozen_splits <file.json>` refuses to start unless the
computed split matches a frozen list exactly, in order — use it whenever two runs
must be compared as a matched pair.

### 8. Cross-modal evaluation and confidence intervals

```bash
python train_class/evaluate_cross_modality.py \
    --model_family qkformer --run raw=<run_dir> \
    --eval_modalities dv --split_source <dv_run_dir> \
    --data_dir ./data/unified80 --checkpoint_tag best

python -m tools.v2e_baseline.build_cross_modal_manifest --help
python -m tools.v2e_baseline.cross_modal_eval_with_ci --help
```

Recording-level bootstrap alone understates the spread when training seeds also
vary. Dump the per-sample correctness, then resample seeds and recordings jointly
and compute simultaneous intervals across a family of comparisons:

```bash
python -m tools.v2e_baseline.dump_eval_correctness \
    --manifest ./output/cross_modal_eval/manifest.json \
    --test_data_dir ./data/unified80 \
    --output ./output/correctness/correctness.npz

python -m tools.v2e_baseline.analyze_correctness \
    --npz ./output/correctness/correctness.npz --out ./output/correctness/
```

### 9. Per-recording fidelity-transfer correlation

Relates each recording's upstream distance to whether that recording is
classified correctly after transfer.

```bash
python -m tools.v2e_baseline.dump_per_prefix_correctness --help
python -m tools.v2e_baseline.within_prefix_correlation --help
```

### 10. Frozen-feature cross-modal retrieval

Freezes the trained encoder, embeds every recording, and retrieves the nearest
simulated recording for each real one.

```bash
python -m tools.retrieval.run_pipeline \
    --ckpt_root ./train_class/output --real_dv_dir ./data/unified80 \
    --sim_dir_map ./output/sim_dir_map.json \
    --prefix_list tools/v2e_baseline/canonical_test_559_prefixes.txt \
    --output_root ./output/retrieval/
```

`--sim_dir_map` is a JSON mapping each variant code to its simulated-events
directory; see the script docstring for the schema.

### 11. Acquisition and spatial-processing audits

These test whether the observed simulator rankings are artifacts of the capture
rig rather than properties of the simulators. Each writes a per-recording CSV
next to its summary JSON.

```bash
# measured Pi frame cadence, from the released metadata sidecars
python -m tools.audits.cadence_audit \
    --metadata-dir ./data/corpus/metadata --output-dir ./output/audits/cadence/

# event-rate power spectra for matched real / raw-sim / RGB-sim streams
python -m tools.audits.rate_spectrum_audit \
    --data-dir ./data/unified80 --modalities dv raw rgb \
    --output-dir ./output/audits/rate_spectrum/

# remove the frame cadence and its harmonics, recompute the temporal rankings
# --variant takes ID=directory:suffix and may be repeated
python -m tools.audits.notch_ranking_robustness \
    --reference-dir ./data/unified80 --variant V01=./output/sim/V01:raw \
    --prefix-list tools/v2e_baseline/canonical_test_559_prefixes.txt \
    --output-dir ./output/audits/notch/

# what the AprilTag-driven ROI extraction and normalization preserve
python -m tools.audits.roi_retention_audit \
    --native-dir ./data/native_dv --unified-dir ./data/unified80 \
    --output-dir ./output/audits/roi/

# whether the spatial rankings survive resampling to lower grids
python -m tools.audits.resolution_sensitivity_audit \
    --dv-dir ./data/unified80 --variant V01=./output/sim/V01:raw \
    --prefix-list tools/v2e_baseline/canonical_test_559_prefixes.txt \
    --resolutions 64 40 20 --output-dir ./output/audits/resolution/
```

---

## Repository Layout

```
src/                     event simulator: config (K_MAP), stochastic core,
                         and process_data/ readers, filters, AprilTag detection
generate_event.py        frame-to-event CLI, single file or batch
k_calibration/           device-pair K calibration: preprocessing, three-step
                         regression, optional Optuna refinement
train_class/             data preparation, QKFormer and MobileNetV2 trainers,
                         cross-modal evaluation; train_utils/ holds the datasets,
                         event representations, augmentations, split binding
tools/v2e_baseline/      twelve-variant suite, upstream comparison, cross-modal
                         evaluation, bootstrap and simultaneous CIs, recording lists
tools/k_diagnostics/     per-recording upstream implementation, k4 polarity scan
tools/eqs/               learned realism metric and its perturbation controls
tools/audits/            acquisition cadence, spectra, ROI and resolution audits
tools/retrieval/         frozen-feature cross-modal retrieval
tools/visualization/     demo notebook for src/process_data/
data_collection/         acquisition software (Pi + DAVIS346 + Dobot Magician)
tests/                   calibration, simulator, metric, and evaluation checks
external/v2e/            v2e baseline (submodule)
train_class/QKFormer/    QKFormer backbone (submodule)
```

---

## Data Collection

`data_collection/` contains the software that produced the dataset: a Raspberry
Pi 5 with a Pi Camera Module 3 (raw Bayer plus on-device ISP-RGB) co-located with
a DAVIS346, and a Dobot Magician arm executing a deterministic trajectory over
the displayed stimulus. The workstation drives the DAVIS346 and the robot, the Pi
captures the synchronized frame pair, and the two hosts are synchronized over
MQTT. See `data_collection/README.md`.

Reproducing the acquisition needs the physical rig. None of the steps above
depend on it.

---

## License

Code: Apache-2.0 (see `LICENSE`). Dataset: CC-BY-4.0, declared at the dataset
entry. The Dobot SDK under `data_collection/dobot/` is vendor-provided under its
own license; see `data_collection/dobot/LICENSE`.

---

## Acknowledgements

The simulator builds on **DVS-Voltmeter** (Lin et al., ECCV 2022); the baselines
use **v2e** (Hu et al., CVPRW 2021); the spiking classifier is **QKFormer**
(Zhou et al., NeurIPS 2024).

```bibtex
@inproceedings{lin2022dvsvoltmeter,
  title     = {DVS-Voltmeter: Stochastic Process-based Event Simulator for Dynamic Vision Sensors},
  author    = {Lin, Songnan and Ma, Ye and Guo, Zhenhua and Wen, Bihan},
  booktitle = {ECCV},
  year      = {2022}
}

@inproceedings{hu2021v2e,
  title     = {v2e: From Video Frames to Realistic DVS Events},
  author    = {Hu, Yuhuang and Liu, Shih-Chii and Delbruck, Tobi},
  booktitle = {CVPRW},
  year      = {2021}
}

@inproceedings{zhou2024qkformer,
  title     = {QKFormer: Hierarchical Spiking Transformer using Q-K Attention},
  author    = {Zhou, Chenlin and others},
  booktitle = {NeurIPS},
  year      = {2024}
}
```
