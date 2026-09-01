#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PROOF_DIR="$REPO_ROOT/proofs/hol-light"
: "${HOL_LIGHT_DIR:?set HOL_LIGHT_DIR to a HOL Light checkout}"
: "${S2N_BIGNUM_DIR:?set S2N_BIGNUM_DIR to an s2n-bignum checkout}"

if [[ $# -ne 2 ]]; then
  echo "usage: $0 x86_64 add|sub|mul|mul_bmi2_adx | aarch64 mul" >&2
  exit 2
fi
ARCHITECTURE=$1
OPERATION=$2
if [[ "$ARCHITECTURE" == x86_64 && "$OPERATION" =~ ^(add|sub|mul|mul_bmi2_adx)$ ]]; then
  DEV_ENTRY=fp128_x86_64_dev.ml
elif [[ "$ARCHITECTURE" == aarch64 && "$OPERATION" == mul ]]; then
  DEV_ENTRY=fp128_aarch64_dev.ml
else
  echo "usage: $0 x86_64 add|sub|mul|mul_bmi2_adx | aarch64 mul" >&2
  exit 2
fi

"$PROOF_DIR/check.sh" bytes "$ARCHITECTURE"

TARGET_DIR="$REPO_ROOT/target/fp128-formal-verification/$ARCHITECTURE"
if [[ "$ARCHITECTURE" == x86_64 && "$(uname -m)" == arm64 ]]; then
  PROFILE_ROOT="$TARGET_DIR/x86_64-apple-darwin/release"
else
  PROFILE_ROOT="$TARGET_DIR/release"
fi

find_newest_object() {
  local name=$1
  local newest=""
  while IFS= read -r path; do
    if [[ -z "$newest" || "$path" -nt "$newest" ]]; then
      newest=$path
    fi
  done < <(find "$PROFILE_ROOT/build" -path "*/out/$name" -print)
  if [[ -z "$newest" ]]; then
    echo "no $name object was produced" >&2
    return 1
  fi
  printf '%s\n' "$newest"
}

ADD_OBJECT=$(find_newest_object fp128_add.o)
SUB_OBJECT=$(find_newest_object fp128_sub.o)
MUL_OBJECT=$(find_newest_object fp128_mul.o)
BMI2_ADX_MUL_OBJECT=""
if [[ "$ARCHITECTURE" == x86_64 ]]; then
  BMI2_ADX_MUL_OBJECT=$(find_newest_object fp128_mul_bmi2_adx.o)
fi
if [[ "$ARCHITECTURE" == x86_64 && "$(uname -s)" == Darwin ]]; then
  ELF_OBJECT_DIR="$TARGET_DIR/x86_64-elf"
  mkdir -p "$ELF_OBJECT_DIR"
  clang --target=x86_64-unknown-linux-gnu \
    -c -I "$REPO_ROOT/crates/jolt-field/asm/x86_64" \
    "$REPO_ROOT/crates/jolt-field/asm/x86_64/fp128_add.S" \
    -o "$ELF_OBJECT_DIR/fp128_add.o"
  clang --target=x86_64-unknown-linux-gnu \
    -c -I "$REPO_ROOT/crates/jolt-field/asm/x86_64" \
    "$REPO_ROOT/crates/jolt-field/asm/x86_64/fp128_sub.S" \
    -o "$ELF_OBJECT_DIR/fp128_sub.o"
  clang --target=x86_64-unknown-linux-gnu \
    -c -I "$REPO_ROOT/crates/jolt-field/asm/x86_64" \
    "$REPO_ROOT/crates/jolt-field/asm/x86_64/fp128_mul.S" \
    -o "$ELF_OBJECT_DIR/fp128_mul.o"
  clang --target=x86_64-unknown-linux-gnu \
    -c -I "$REPO_ROOT/crates/jolt-field/asm/x86_64" \
    "$REPO_ROOT/crates/jolt-field/asm/x86_64/fp128_mul_bmi2_adx.S" \
    -o "$ELF_OBJECT_DIR/fp128_mul_bmi2_adx.o"
  ADD_OBJECT="$ELF_OBJECT_DIR/fp128_add.o"
  SUB_OBJECT="$ELF_OBJECT_DIR/fp128_sub.o"
  MUL_OBJECT="$ELF_OBJECT_DIR/fp128_mul.o"
  BMI2_ADX_MUL_OBJECT="$ELF_OBJECT_DIR/fp128_mul_bmi2_adx.o"
fi
DEV_INIT=$(mktemp)
cleanup() {
  rm -f "$DEV_INIT"
}
trap cleanup EXIT

printf '#use "%s/hol.ml";;\nloadt "%s/fp128_x86_64_dev.ml";;\n' \
  "$HOL_LIGHT_DIR" "$PROOF_DIR" >"$DEV_INIT"
if [[ "$DEV_ENTRY" != fp128_x86_64_dev.ml ]]; then
  printf '#use "%s/hol.ml";;\nloadt "%s/%s";;\n' \
    "$HOL_LIGHT_DIR" "$PROOF_DIR" "$DEV_ENTRY" >"$DEV_INIT"
fi

if [[ -z "${LINE_EDITOR:-}" ]]; then
  if command -v rlwrap >/dev/null 2>&1; then
    LINE_EDITOR=rlwrap
  elif command -v ledit >/dev/null 2>&1; then
    LINE_EDITOR=ledit
  else
    LINE_EDITOR=env
  fi
fi

echo "Starting one persistent HOL Light session for $ARCHITECTURE $OPERATION."
echo "The first load is slow. Later theorem reloads reuse the loaded processor model."

cd "$S2N_BIGNUM_DIR"
export JOLT_FP128_PROOF_DIR="$PROOF_DIR"
export JOLT_FP128_DEV_OPERATION="$OPERATION"
export JOLT_FP128_ADD_OBJECT="$ADD_OBJECT"
export JOLT_FP128_SUB_OBJECT="$SUB_OBJECT"
export JOLT_FP128_MUL_OBJECT="$MUL_OBJECT"
export JOLT_FP128_MUL_BMI2_ADX_OBJECT="$BMI2_ADX_MUL_OBJECT"
export HOL_ML_PATH="$DEV_INIT"
export LINE_EDITOR
if [[ -d "$HOL_LIGHT_DIR/_opam" ]]; then
  opam exec --switch "$HOL_LIGHT_DIR" -- "$HOL_LIGHT_DIR/hol.sh"
else
  opam exec -- "$HOL_LIGHT_DIR/hol.sh"
fi
