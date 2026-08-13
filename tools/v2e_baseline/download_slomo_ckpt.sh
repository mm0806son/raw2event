#!/bin/bash
# Download SuperSloMo checkpoint required by v2e slomo protocol (V09..V12).
# v2e ships pre-trained weights via Google Drive; the SensorsINI/v2e README
# points to multiple mirror options. This script tries gdown first; if that
# fails, falls back to manual instructions.

set -euo pipefail

# Default destination = ./external/, matching the SLOMO_MODEL default in
# run_v2e_batch.py. Override via DEST=... before sourcing this script.
DEST="${DEST:-./external/SuperSloMo39.ckpt}"
mkdir -p "$(dirname "$DEST")"

if [ -f "$DEST" ] && [ -s "$DEST" ]; then
  echo "SuperSloMo39.ckpt already present at $DEST"
  exit 0
fi

# v2e README references this Google Drive ID
GDRIVE_ID="${GDRIVE_ID:-1uzG6LP6ARrOu58yYWUKxsCJp7Nfnzfwz}"

if command -v gdown >/dev/null 2>&1; then
  echo "Trying gdown ${GDRIVE_ID} -> $DEST"
  gdown --id "${GDRIVE_ID}" -O "$DEST" || {
    echo "gdown failed; falling through to manual instructions"
    GDOWN_FAIL=1
  }
else
  echo "gdown not installed; pip install gdown to use this fallback"
  GDOWN_FAIL=1
fi

if [ "${GDOWN_FAIL:-0}" -eq 1 ] || ! [ -s "$DEST" ]; then
  cat <<'EOF'

Manual download instructions:
  1. Visit https://github.com/SensorsINI/v2e/blob/master/README.md
  2. Follow the SuperSloMo download link (Drive)
  3. Place the file at: ./external/SuperSloMo39.ckpt
     (or override via DEST=... before running this script)

Or, if you have a copy elsewhere:
  cp /path/to/SuperSloMo39.ckpt ./external/SuperSloMo39.ckpt
EOF
  exit 1
fi

echo "Downloaded: $DEST"
ls -lh "$DEST"
