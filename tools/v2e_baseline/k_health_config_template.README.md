# `k_health_config_template.json` — usage notes

This template feeds `tools/v2e_baseline/k_health_v2e_compare.py --config <path>`. The
script reads the top-level JSON object as a `dict[variant_id -> spec]`; each spec is
consumed as:

```python
sim_dir    = Path(spec["npz_dir"])              # required
npz_suffix = spec.get("npz_suffix", "rgb")      # filename pattern: {prefix}_filtered_{npz_suffix}.npz
label      = spec.get("label", variant)         # printed in stdout + markdown summary
```

Any other field is ignored (the script never reads them). `--dv_npz_dir` is passed
on the CLI separately and the DV reference filename is hard-coded to
`{prefix}_filtered_dv.npz`.

## Required environment variables

The JSON template contains exactly two `envsubst` placeholders (always braced —
`${VAR}`, never bare `$VAR`):

- `${LOCAL_DST}`
- `${UNIFIED80_DIR}`

A third variable, `DV_NPZ_DIR`, is **not substituted into the JSON**. It is consumed
only via the `--dv_npz_dir` CLI flag of `k_health_v2e_compare.py`. The template does
not reference `${DV_NPZ_DIR}` anywhere — running `envsubst` on the template will
leave `DV_NPZ_DIR` untouched (because it is not present), and the variable must
still be exported in the calling shell so the `--dv_npz_dir "$DV_NPZ_DIR"` flag
resolves correctly.

| Var              | Where it is consumed                                                 | Example (this dev machine)                                                |
| ---------------- | -------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| `LOCAL_DST`      | `envsubst` placeholder in the JSON; root of `v2e_compare` tree       | `./output/v2e_compare`                |
| `UNIFIED80_DIR`  | `envsubst` placeholder in the JSON; Raw2Event unified80 NPZ root for V01/V02 | `./data/unified80`   |
| `DV_NPZ_DIR`     | CLI flag `--dv_npz_dir` only (not in JSON)                            | `./data/unified80`           |

## Usage

`envsubst` silently substitutes empty strings for unset variables, which would
produce broken paths like `/V03_V03_dvsv_default_k`. Guard against this by
asserting all three variables are set before invoking `envsubst`.

`envsubst` itself does **not** support `${VAR:?msg}` expansion (that is a bash
parameter-expansion feature, not an `envsubst` feature), so the guard must live in
the surrounding shell script and the JSON template stays as plain `${VAR}`:

```bash
# Fail fast if any required variable is unset/empty.
: "${LOCAL_DST:?set LOCAL_DST}" "${UNIFIED80_DIR:?set UNIFIED80_DIR}" "${DV_NPZ_DIR:?set DV_NPZ_DIR}"

envsubst < tools/v2e_baseline/k_health_config_template.json \
  > /tmp/k_health_config.json

python tools/v2e_baseline/k_health_v2e_compare.py \
  --config /tmp/k_health_config.json \
  --dv_npz_dir "$DV_NPZ_DIR" \
  --prefix_list tools/v2e_baseline/canonical_test_v2e_runnable.txt \
  --output output/diagnostics_20260501_v2e_compare/k_health_table.json
```

A standard batch invocation, with all three variables set inline:

```bash
LOCAL_DST=./output/v2e_compare \
DV_NPZ_DIR=./data/unified80 \
UNIFIED80_DIR=./data/unified80 \
bash -c '
  : "${LOCAL_DST:?}" "${UNIFIED80_DIR:?}" "${DV_NPZ_DIR:?}"
  envsubst < tools/v2e_baseline/k_health_config_template.json > /tmp/k_health_config.json
  python tools/v2e_baseline/k_health_v2e_compare.py \
    --config /tmp/k_health_config.json \
    --dv_npz_dir "$DV_NPZ_DIR" \
    --prefix_list tools/v2e_baseline/canonical_test_v2e_runnable.txt \
    --output output/diagnostics_20260501_v2e_compare/k_health_table.json
'
```

## Variant -> directory / suffix mapping

| ID  | npz_dir (after envsubst)                                                | npz_suffix | Source / status (2026-05-04)                           |
| --- | ------------------------------------------------------------------------ | ---------- | ------------------------------------------------------- |
| V01 | `${UNIFIED80_DIR}`                                                       | `raw`      | produced on the GPU cluster (not on local dev)                        |
| V02 | `${UNIFIED80_DIR}`                                                       | `rgb`      | produced on the GPU cluster (not on local dev)         |
| V03 | `${LOCAL_DST}/V03_V03_dvsv_default_k`                                    | `rawY`     | cluster only (5917 NPZ verified 2026-05-04); double-prefix is the real disk name |
| V04 | `${LOCAL_DST}/V04_V04_dvsv_default_k`                                    | `rgb`      | cluster only (5917 NPZ verified 2026-05-04); double-prefix is the real disk name |
| V05 | `${LOCAL_DST}/V05_v2e_rgb_native50_default`                              | `rgb`      | cluster only; dir name from Stage 3 submit example      |
| V06 | `${LOCAL_DST}/V06_v2e_rgb_native50_tuned`                                | `rgb`      | produced on the GPU cluster; symmetric                                  |
| V07 | `${LOCAL_DST}/V07_v2e_rawY_native50_default`                             | `rawY`     | produced on the GPU cluster; symmetric                                  |
| V08 | `${LOCAL_DST}/V08_v2e_rawY_native50_tuned`                               | `rawY`     | produced on the GPU cluster; symmetric                                  |
| V09 | `${LOCAL_DST}/V09_v2e_rgb_slomo_default`                                 | `rgb`      | **on this dev machine, 559 NPZ verified**               |
| V10 | `${LOCAL_DST}/V10_v2e_rgb_slomo_tuned`                                   | `rgb`      | **on this dev machine, 559 NPZ verified**               |
| V11 | `${LOCAL_DST}/V11_v2e_rawY_slomo_default`                                | `rawY`     | **on this dev machine, 559 NPZ verified**               |
| V12 | `${LOCAL_DST}/V12_v2e_rawY_slomo_tuned`                                  | `rawY`     | **on this dev machine, 559 NPZ verified**               |

### V03/V04 directory-name note

`tools/v2e_baseline/raw_dvsv_default_k_gen.py` does **not** fix the V03/V04 output
directory name in code — its `VARIANT_SPEC` only declares `input_modality`,
`k_label`, and `npz_suffix`; the output path is purely user-controlled via
`--output_dir`. The actual on-disk dirnames are determined by the batch-script
default `OUTPUT_NAME=${VARIANT}_${VARIANT}_dvsv_default_k`, which produces the
double-prefix names used throughout:

```text
${LOCAL_DST}/V03_V03_dvsv_default_k   # 5917 NPZ
${LOCAL_DST}/V04_V04_dvsv_default_k   # 5917 NPZ
```

These are the canonical paths used by this template. Earlier 2026-05-03 docs
referenced single-prefix names (`V03_raw_dvsv_default_k` / `V04_rgb_dvsv_default_k`)
which exist on disk only as empty residuals (3 inodes each, no NPZ). Do not point
at the single-prefix dirs.

Future re-runs with a different `OUTPUT_NAME` must update this template
in lockstep with the corresponding data-prep schema.

## Missing-variant behavior

`k_health_v2e_compare.py` skips any `(variant, prefix)` pair where the simulated
NPZ does not exist (the inner loop has `if not (sim_path.exists() and dv_path.exists()): continue`).
If a variant directory is entirely missing or empty, every prefix is skipped, so
the variant produces **zero per-prefix rows**.

The aggregate markdown table is then built by the script around line 143-146:

```python
rows = [r for r in all_rows if r["variant"] == variant]
if not rows:
    continue                       # variant is skipped from the markdown summary
```

So a fully-missing variant appears in the raw `results.json` as an empty list (the
key is present, value is `[]`), but is **omitted entirely** from the aggregate
markdown summary — it does **not** show up as `n=0`. To diagnose a missing
variant, inspect the raw JSON, not the markdown table.

## Subset runs

The script offers no `--variants` flag (confirmed by re-reading
`k_health_v2e_compare.py` — see "Schema notes" below). To run on a subset, pre-edit
the JSON (drop unwanted top-level keys) before passing it. Example for V09-V12 only
(the variants currently on disk):

```bash
: "${LOCAL_DST:?}" "${UNIFIED80_DIR:?}"
envsubst < tools/v2e_baseline/k_health_config_template.json \
  | jq '{V09, V10, V11, V12}' \
  > /tmp/k_health_v09_v12.json
```

## Schema notes / observations (no code changes made)

- `k_health_v2e_compare.py` accepts only four CLI flags: `--config`, `--dv_npz_dir`,
  `--prefix_list`, `--output`, `--sinkhorn_reg`. There is **no** `--variants`
  selector; subsetting must happen in the JSON.
- The script iterates over **every** top-level key in the JSON, so do not add
  `_doc` / `_env_vars` / `_meta` siblings — they would be parsed as variant IDs
  and immediately fail the `Path(spec["npz_dir"])` lookup. Keep documentation in
  this README, not in the JSON.
- `npz_suffix` defaults to `"rgb"` if omitted. The template always sets it
  explicitly to avoid silent mis-matches on rawY variants.
- Sinkhorn EMD requires the optional `ot` (POT) package; if it is not installed,
  `sinkhorn_dt_emd` returns `NaN` and the row is still produced (`np.nanmean`
  ignores it in the summary). Any `pip install POT` failure inside the cluster
  container is unrelated to this template.
