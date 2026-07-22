#!/bin/bash
set -euo pipefail

echo "Generating fixtures files..."
cargo run --profile build-fast -p fibonacci -- --save

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIXTURE_DIR=$SCRIPT_DIR/fixtures

FILES=(
  "jolt_verifier_preprocessing.dat"
  "fib_proof.bin"
  "fib_io_device.bin"
)

mkdir -p "$FIXTURE_DIR"

for entry in "${FILES[@]}"; do
  read -r SRC <<<"$entry"

  SRC_TMP="/tmp/$SRC"
  RAW_PATH="$FIXTURE_DIR/$SRC"

  if [[ ! -f "$SRC_TMP" ]]; then
    echo "ERROR: expected fixture $SRC_TMP was not produced by 'fibonacci --save'" >&2
    echo "       (the --save output filename contract may have drifted)" >&2
    exit 1
  fi

  echo "Copying $SRC_TMP -> $RAW_PATH"
  cp "$SRC_TMP" "$RAW_PATH"
done

echo "Done generating test fixtures."
