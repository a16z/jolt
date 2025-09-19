#!/bin/bash
set -euo pipefail

echo "Generating fixtures files..."
cargo run --release -p fibonacci -- --save

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIXTURE_DIR=$SCRIPT_DIR/fixtures

FILES=(
  "jolt_verifier_preprocessing.dat"
  "fib_proof.bin"
  "fib_io_device.bin"
)

for entry in "${FILES[@]}"; do
  read -r SRC <<<"$entry"

  SRC_TMP="/tmp/$SRC"
  RAW_PATH="$FIXTURE_DIR/$SRC"

  if [[ ! -f "$SRC_TMP" ]]; then
    echo "WARN: $SRC_TMP not found; skipping"
    continue
  fi

  echo "Copying $SRC_TMP -> $RAW_PATH"
  cp "$SRC_TMP" "$RAW_PATH"
done

echo "Done generating test fixtures."
