#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
: "${HOL_LIGHT_DIR:?set HOL_LIGHT_DIR to a module-built HOL Light checkout}"
: "${S2N_BIGNUM_DIR:?set S2N_BIGNUM_DIR to an s2n-bignum checkout}"

PROOF_TARGET=$(mktemp -d)
cleanup() {
  rm -rf "$PROOF_TARGET"
}
trap cleanup EXIT

CARGO_TARGET_DIR="$PROOF_TARGET/cargo" \
  cargo build --locked -p jolt-field --release --no-default-features \
    --features solinas,fp128-proof-linkage \
    --example fp128_production_witness

find_one_object() {
  local name=$1
  local -a matches=()
  while IFS= read -r path; do
    matches+=("$path")
  done < <(find "$PROOF_TARGET/cargo/release/build" -path "*/out/$name" -print)
  if [[ ${#matches[@]} -ne 1 ]]; then
    echo "expected one fresh $name object, found ${#matches[@]}" >&2
    return 1
  fi
  printf '%s\n' "${matches[0]}"
}

ADD_OBJECT=$(find_one_object fp128_add.o)
SUB_OBJECT=$(find_one_object fp128_sub.o)
PRODUCTION_WITNESS="$PROOF_TARGET/cargo/release/examples/fp128_production_witness"

python3 "$REPO_ROOT/scripts/check_fp128_proof_artifacts.py" \
  --add-object "$ADD_OBJECT" \
  --sub-object "$SUB_OBJECT" \
  --production-witness "$PRODUCTION_WITNESS"

relative_to_s2n_arm() {
  python3 - "$S2N_BIGNUM_DIR/arm" "$1" <<'PY'
import os
import sys

print(os.path.relpath(sys.argv[2], sys.argv[1]))
PY
}

build_proof() {
  local proof_source=$1
  local native_output=$2
  local proof_relative
  local output_relative
  proof_relative=$(relative_to_s2n_arm "$proof_source")
  output_relative=$(relative_to_s2n_arm "$native_output")
  (
    cd "$S2N_BIGNUM_DIR/arm"
    if [[ -d "$HOL_LIGHT_DIR/_opam" ]]; then
      opam exec --switch "$HOL_LIGHT_DIR" -- \
        ../tools/build-proof.sh \
        "$proof_relative" \
        "$HOL_LIGHT_DIR/hol.sh" \
        "$output_relative"
    else
      opam exec -- \
        ../tools/build-proof.sh \
        "$proof_relative" \
        "$HOL_LIGHT_DIR/hol.sh" \
        "$output_relative"
    fi
  )
}

run_proof() {
  local object_variable=$1
  local object_path=$2
  local native_path=$3
  local theorem_name=$4
  local log_path=$5
  if ! env "$object_variable=$object_path" "$native_path" >"$log_path" 2>&1; then
    tail -200 "$log_path" >&2
    return 1
  fi
  grep -F "val $theorem_name : thm" "$log_path"
  grep -F "Running time:" "$log_path"
}

ADD_NATIVE="$PROOF_TARGET/fp128_add_correct.native"
SUB_NATIVE="$PROOF_TARGET/fp128_sub_correct.native"
build_proof "$REPO_ROOT/proofs/hol-light/fp128_add_correct.ml" "$ADD_NATIVE"
build_proof "$REPO_ROOT/proofs/hol-light/fp128_sub_correct.ml" "$SUB_NATIVE"

run_proof \
  JOLT_FP128_ADD_OBJECT "$ADD_OBJECT" "$ADD_NATIVE" \
  JOLT_FP128_ADD_SUBROUTINE_CORRECT "$PROOF_TARGET/add.log"
run_proof \
  JOLT_FP128_SUB_OBJECT "$SUB_OBJECT" "$SUB_NATIVE" \
  JOLT_FP128_SUB_SUBROUTINE_CORRECT "$PROOF_TARGET/sub.log"

echo "Fp128 public-witness addition and subtraction proofs passed."
