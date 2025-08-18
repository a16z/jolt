#!/bin/bash
set -euo pipefail

echo "Generating fixtures files..."
cargo run --release -p fibonacci -- --save

FIXTURE_DIR=./tests/fixtures
BYTE_RS_DIR=./tests

# input/output name pairs: ("input_filename" "output_filename.rs")
FILES=(
  "jolt_verifier_preprocessing.dat jolt_verifier_preprocessing_bytes.rs"
  "fib_proof.bin fib_proof_bytes.rs"
  "fib_io_device.bin fib_io_device_bytes.rs"
)

for entry in "${FILES[@]}"; do
  read -r SRC DEST <<<"$entry"

  SRC_TMP="/tmp/$SRC"
  RAW_PATH="$FIXTURE_DIR/$SRC"
  RS_PATH="$BYTE_RS_DIR/$DEST"

  if [[ ! -f "$SRC_TMP" ]]; then
    echo "WARN: $SRC_TMP not found; skipping"
    continue
  fi

  echo "Copying $SRC_TMP -> $RAW_PATH"
  cp "$SRC_TMP" "$RAW_PATH"

  NAME_UPPER=$(basename "$DEST" .rs | tr '-' '_' | tr '[:lower:]' '[:upper:]')

  echo "Writing $RS_PATH"
  {
    echo "pub static ${NAME_UPPER}: &[u8] = &["
    xxd -p -c 16 "$RAW_PATH" | awk '{
      line = $0
      out = "    "
      for (i = 1; i <= length(line); i += 2) {
        byte = substr(line, i, 2)
        if (i + 1 < length($0)) {
          out = out "0x" byte ", "
        } else {
          out = out "0x" byte ","
        }
      }
      print out
    }'
    echo "];"
  } > "$RS_PATH"
done

echo "Done generating test fixtures."
