#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PROOF_DIR="$REPO_ROOT/proofs/hol-light"
: "${HOL_LIGHT_DIR:?set HOL_LIGHT_DIR to a HOL Light checkout}"
: "${S2N_BIGNUM_DIR:?set S2N_BIGNUM_DIR to an s2n-bignum checkout}"

usage() {
  echo "usage: $0 x86_64 add|sub|mul|mul_bmi2 | aarch64 add|sub|mul" >&2
}

if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi
ARCHITECTURE=$1
OPERATION=$2
if [[ "$ARCHITECTURE" == x86_64 && "$OPERATION" =~ ^(add|sub|mul|mul_bmi2)$ ]]; then
  DEV_ENTRY=fp64_x86_64_dev.ml
elif [[ "$ARCHITECTURE" == aarch64 && "$OPERATION" =~ ^(add|sub|mul)$ ]]; then
  DEV_ENTRY=fp64_aarch64_dev.ml
else
  usage
  exit 2
fi

"$PROOF_DIR/check-fp64.sh" bytes "$ARCHITECTURE"

TARGET_DIR="$REPO_ROOT/target/fp64-formal-verification/$ARCHITECTURE"
if [[ "$ARCHITECTURE" == x86_64 && "$(uname -s)" == Darwin ]]; then
  PROFILE_ROOT="$TARGET_DIR/x86_64-apple-darwin/release"
else
  PROFILE_ROOT="$TARGET_DIR/release"
fi

find_newest_object() {
  local name=$1
  local profile_root=${2:-$PROFILE_ROOT}
  local newest=""
  while IFS= read -r path; do
    if [[ -z "$newest" || "$path" -nt "$newest" ]]; then
      newest=$path
    fi
  done < <(find "$profile_root/build" -path "*/out/$name" -print)
  if [[ -z "$newest" ]]; then
    echo "no $name object was produced" >&2
    return 1
  fi
  printf '%s\n' "$newest"
}

ADD_OBJECT=$(find_newest_object fp64_add.o)
SUB_OBJECT=$(find_newest_object fp64_sub.o)
MUL_OBJECT=$(find_newest_object fp64_mul.o)
BMI2_MUL_OBJECT=""
if [[ "$ARCHITECTURE" == x86_64 ]]; then
  BMI2_PROFILE_ROOT="$TARGET_DIR/bmi2"
  if [[ "$(uname -s)" == Darwin ]]; then
    BMI2_PROFILE_ROOT="$BMI2_PROFILE_ROOT/x86_64-apple-darwin/release"
  else
    BMI2_PROFILE_ROOT="$BMI2_PROFILE_ROOT/release"
  fi
  BMI2_MUL_OBJECT=$(find_newest_object fp64_mul_bmi2.o "$BMI2_PROFILE_ROOT")
fi

if [[ "$ARCHITECTURE" == x86_64 && "$(uname -s)" == Darwin ]]; then
  ELF_OBJECT_DIR="$TARGET_DIR/x86_64-elf"
  mkdir -p "$ELF_OBJECT_DIR"
  for operation in add sub mul mul_bmi2; do
    clang --target=x86_64-unknown-linux-gnu \
      -c -I "$REPO_ROOT/crates/jolt-field/asm/x86_64" \
      "$REPO_ROOT/crates/jolt-field/asm/x86_64/fp64_$operation.S" \
      -o "$ELF_OBJECT_DIR/fp64_$operation.o"
  done
  ADD_OBJECT="$ELF_OBJECT_DIR/fp64_add.o"
  SUB_OBJECT="$ELF_OBJECT_DIR/fp64_sub.o"
  MUL_OBJECT="$ELF_OBJECT_DIR/fp64_mul.o"
  BMI2_MUL_OBJECT="$ELF_OBJECT_DIR/fp64_mul_bmi2.o"
fi

DEV_INIT=$(mktemp)
cleanup() {
  rm -f "$DEV_INIT"
}
trap cleanup EXIT

printf '#use "%s/hol.ml";;\nloadt "%s/%s";;\n' \
  "$HOL_LIGHT_DIR" "$PROOF_DIR" "$DEV_ENTRY" >"$DEV_INIT"

if [[ -z "${LINE_EDITOR:-}" ]]; then
  if command -v rlwrap >/dev/null 2>&1; then
    LINE_EDITOR=rlwrap
  elif command -v ledit >/dev/null 2>&1; then
    LINE_EDITOR=ledit
  else
    LINE_EDITOR="env"
  fi
fi

echo "Starting one persistent HOL Light session for $ARCHITECTURE Fp64 $OPERATION."
echo "The first load imports the processor model. Later theorem reloads reuse it."

cd "$S2N_BIGNUM_DIR"
export JOLT_FP64_PROOF_DIR="$PROOF_DIR"
export JOLT_FP64_DEV_OPERATION="$OPERATION"
export JOLT_FP64_ADD_OBJECT="$ADD_OBJECT"
export JOLT_FP64_SUB_OBJECT="$SUB_OBJECT"
export JOLT_FP64_MUL_OBJECT="$MUL_OBJECT"
export JOLT_FP64_MUL_BMI2_OBJECT="$BMI2_MUL_OBJECT"
export HOL_ML_PATH="$DEV_INIT"
export LINE_EDITOR
if [[ -d "$HOL_LIGHT_DIR/_opam" ]]; then
  opam exec --switch "$HOL_LIGHT_DIR" -- "$HOL_LIGHT_DIR/hol.sh"
else
  opam exec -- "$HOL_LIGHT_DIR/hol.sh"
fi
