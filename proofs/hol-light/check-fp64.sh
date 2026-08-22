#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PROOF_DIR="$REPO_ROOT/proofs/hol-light"

usage() {
  echo "usage: $0 [all|bytes] [aarch64|x86_64] [--clean]" >&2
}

normalize_architecture() {
  case "$1" in
    arm64 | aarch64) printf '%s\n' aarch64 ;;
    x86_64 | amd64) printf '%s\n' x86_64 ;;
    *) return 1 ;;
  esac
}

MODE=all
ARCHITECTURE=""
CLEAN=false
for argument in "$@"; do
  case "$argument" in
    all | bytes) MODE=$argument ;;
    aarch64 | arm64 | x86_64 | amd64)
      ARCHITECTURE=$(normalize_architecture "$argument")
      ;;
    --clean) CLEAN=true ;;
    -h | --help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done

if [[ -z "$ARCHITECTURE" ]]; then
  ARCHITECTURE=$(normalize_architecture "$(uname -m)") || {
    echo "unsupported host architecture: $(uname -m)" >&2
    exit 2
  }
fi

HOST_ARCHITECTURE=$(normalize_architecture "$(uname -m)") || true
CARGO_BUILD_ARGS=(build --locked -p jolt-field --release)
TARGET_TRIPLE=""
if [[ "$ARCHITECTURE" != "$HOST_ARCHITECTURE" ]]; then
  if [[ "$(uname -s)" == Darwin && "$ARCHITECTURE" == x86_64 ]]; then
    TARGET_TRIPLE=x86_64-apple-darwin
    CARGO_BUILD_ARGS+=(--target "$TARGET_TRIPLE")
  else
    echo "cross checking $ARCHITECTURE is not configured on this host" >&2
    exit 2
  fi
fi

if $CLEAN; then
  PROOF_TARGET=$(mktemp -d)
else
  PROOF_TARGET="$REPO_ROOT/target/fp64-formal-verification/$ARCHITECTURE"
  mkdir -p "$PROOF_TARGET"
fi

cleanup() {
  local status=$?
  if $CLEAN; then
    if [[ $status -eq 0 ]]; then
      rm -rf "$PROOF_TARGET"
    else
      echo "failed proof workspace preserved at $PROOF_TARGET" >&2
    fi
  fi
}
trap cleanup EXIT

if [[ -n "$TARGET_TRIPLE" ]]; then
  PROFILE_ROOT="$PROOF_TARGET/$TARGET_TRIPLE/release"
else
  PROFILE_ROOT="$PROOF_TARGET/release"
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

if [[ "$MODE" == bytes ]]; then
  echo "[1/2] Building the $ARCHITECTURE Fp64 inspection witnesses"
else
  echo "[1/5] Building the $ARCHITECTURE Fp64 inspection witnesses"
fi
CARGO_TARGET_DIR="$PROOF_TARGET" \
  cargo "${CARGO_BUILD_ARGS[@]}" \
    --no-default-features \
    --features solinas,fp64-proof-linkage \
    --example fp64_production_witness

ADD_OBJECT=$(find_newest_object fp64_add.o)
SUB_OBJECT=$(find_newest_object fp64_sub.o)
MUL_OBJECT=$(find_newest_object fp64_mul.o)
PRODUCTION_WITNESS="$PROFILE_ROOT/examples/fp64_production_witness"

BMI2_MUL_OBJECT=""
BMI2_PRODUCTION_WITNESS=""
if [[ "$ARCHITECTURE" == x86_64 ]]; then
  BMI2_TARGET="$PROOF_TARGET/bmi2"
  if [[ -n "$TARGET_TRIPLE" ]]; then
    BMI2_PROFILE_ROOT="$BMI2_TARGET/$TARGET_TRIPLE/release"
  else
    BMI2_PROFILE_ROOT="$BMI2_TARGET/release"
  fi
  RUSTFLAGS="${RUSTFLAGS:-} -C target-feature=+bmi2" \
  CARGO_TARGET_DIR="$BMI2_TARGET" \
    cargo "${CARGO_BUILD_ARGS[@]}" \
      --no-default-features \
      --features solinas,fp64-proof-linkage \
      --example fp64_production_witness
  BMI2_MUL_OBJECT=$(find_newest_object fp64_mul_bmi2.o "$BMI2_PROFILE_ROOT")
  BMI2_PRODUCTION_WITNESS="$BMI2_PROFILE_ROOT/examples/fp64_production_witness"
fi

if [[ "$MODE" == bytes ]]; then
  echo "[2/2] Checking exact Fp64 object and inspection witness bytes"
else
  echo "[2/5] Checking exact Fp64 object and inspection witness bytes"
fi
CHECKER_ARGS=(
  --architecture "$ARCHITECTURE"
  --add-object "$ADD_OBJECT"
  --sub-object "$SUB_OBJECT"
  --mul-object "$MUL_OBJECT"
  --production-witness "$PRODUCTION_WITNESS"
)
if [[ "$ARCHITECTURE" == x86_64 ]]; then
  CHECKER_ARGS+=(
    --mul-bmi2-object "$BMI2_MUL_OBJECT"
    --bmi2-production-witness "$BMI2_PRODUCTION_WITNESS"
  )
fi
python3 "$REPO_ROOT/scripts/check_fp64_proof_artifacts.py" "${CHECKER_ARGS[@]}"

if [[ "$MODE" == bytes ]]; then
  exit 0
fi

: "${HOL_LIGHT_DIR:?set HOL_LIGHT_DIR to a module-built HOL Light checkout}"
: "${S2N_BIGNUM_DIR:?set S2N_BIGNUM_DIR to an s2n-bignum checkout}"

ADD_PROOF_OBJECT=$ADD_OBJECT
SUB_PROOF_OBJECT=$SUB_OBJECT
MUL_PROOF_OBJECT=$MUL_OBJECT
BMI2_MUL_PROOF_OBJECT=$BMI2_MUL_OBJECT
if [[ "$ARCHITECTURE" == x86_64 && "$(uname -s)" == Darwin ]]; then
  ELF_OBJECT_DIR="$PROOF_TARGET/x86_64-elf"
  mkdir -p "$ELF_OBJECT_DIR"
  for operation in add sub mul mul_bmi2; do
    clang --target=x86_64-unknown-linux-gnu \
      -c -I "$REPO_ROOT/crates/jolt-field/asm/x86_64" \
      "$REPO_ROOT/crates/jolt-field/asm/x86_64/fp64_$operation.S" \
      -o "$ELF_OBJECT_DIR/fp64_$operation.o"
  done
  ADD_PROOF_OBJECT="$ELF_OBJECT_DIR/fp64_add.o"
  SUB_PROOF_OBJECT="$ELF_OBJECT_DIR/fp64_sub.o"
  MUL_PROOF_OBJECT="$ELF_OBJECT_DIR/fp64_mul.o"
  BMI2_MUL_PROOF_OBJECT="$ELF_OBJECT_DIR/fp64_mul_bmi2.o"
fi

if [[ "$ARCHITECTURE" == aarch64 ]]; then
  S2N_ARCH_DIR=arm
else
  S2N_ARCH_DIR=x86
fi

relative_to_s2n_arch() {
  python3 - "$S2N_BIGNUM_DIR/$S2N_ARCH_DIR" "$1" <<'PY'
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
  proof_relative=$(relative_to_s2n_arch "$proof_source")
  output_relative=$(relative_to_s2n_arch "$native_output")
  (
    cd "$S2N_BIGNUM_DIR/$S2N_ARCH_DIR"
    if [[ -d "$HOL_LIGHT_DIR/_opam" ]]; then
      JOLT_FP64_PROOF_DIR="$PROOF_DIR" \
        opam exec --switch "$HOL_LIGHT_DIR" -- \
          ../tools/build-proof.sh "$proof_relative" \
          "$HOL_LIGHT_DIR/hol.sh" "$output_relative"
    else
      JOLT_FP64_PROOF_DIR="$PROOF_DIR" \
        opam exec -- \
          ../tools/build-proof.sh "$proof_relative" \
          "$HOL_LIGHT_DIR/hol.sh" "$output_relative"
    fi
  )
}

LOG_DIR="$PROOF_TARGET/proof-logs"
mkdir -p "$LOG_DIR"
COMBINED_SOURCE="$PROOF_TARGET/fp64_${ARCHITECTURE}_all.ml"
NATIVE_PROOF="$PROOF_TARGET/fp64_${ARCHITECTURE}_all.native"
LOG_PATH="$LOG_DIR/$ARCHITECTURE-all.log"
CACHE_KEY_PATH="$PROOF_TARGET/fp64_${ARCHITECTURE}_all.cache-key"

if [[ "$ARCHITECTURE" == x86_64 ]]; then
  ARCHITECTURE_LABEL=x86-64
  ADD_THEOREM=JOLT_FP64_ADD_X86_64_SUBROUTINE_CORRECT
  SUB_THEOREM=JOLT_FP64_SUB_X86_64_SUBROUTINE_CORRECT
  MUL_THEOREM=JOLT_FP64_MUL_X86_64_SUBROUTINE_CORRECT
  BMI2_MUL_THEOREM=JOLT_FP64_MUL_X86_64_BMI2_SUBROUTINE_CORRECT
  {
    printf 'loadt "x86/proofs/base.ml";;\n'
    printf 'loadt "%s/fp64_common.ml";;\n' "$PROOF_DIR"
    printf 'loadt "%s/fp64_x86_64_common.ml";;\n' "$PROOF_DIR"
    printf 'print_endline "[HOL 1/5] Proving x86-64 Fp64 addition";;\n'
    printf 'loadt "%s/fp64_add_x86_64_object.ml";;\n' "$PROOF_DIR"
    printf 'loadt "%s/fp64_add_x86_64_correct.ml";;\n' "$PROOF_DIR"
    printf 'print_endline "[HOL 2/5] Proving x86-64 Fp64 subtraction";;\n'
    printf 'loadt "%s/fp64_sub_x86_64_object.ml";;\n' "$PROOF_DIR"
    printf 'loadt "%s/fp64_sub_x86_64_correct.ml";;\n' "$PROOF_DIR"
    printf 'print_endline "[HOL 3/5] Proving baseline x86-64 Fp64 multiplication";;\n'
    printf 'loadt "%s/fp64_mul_x86_64_object.ml";;\n' "$PROOF_DIR"
    printf 'loadt "%s/fp64_mul_x86_64_correct.ml";;\n' "$PROOF_DIR"
    printf 'print_endline "[HOL 4/5] Proving BMI2 x86-64 Fp64 multiplication";;\n'
    printf 'loadt "%s/fp64_mul_x86_64_bmi2_object.ml";;\n' "$PROOF_DIR"
    printf 'loadt "%s/fp64_mul_x86_64_bmi2_correct.ml";;\n' "$PROOF_DIR"
    printf 'print_endline "[HOL 5/5] Checking the Fp64 modulus primality certificate";;\n'
    printf 'loadt "%s/fp64_prime.ml";;\n' "$PROOF_DIR"
  } >"$COMBINED_SOURCE"
  PROOF_SOURCES=("$COMBINED_SOURCE" "$PROOF_DIR"/fp64_*.ml)
else
  ARCHITECTURE_LABEL=AArch64
  ADD_THEOREM=JOLT_FP64_ADD_SUBROUTINE_CORRECT
  SUB_THEOREM=JOLT_FP64_SUB_SUBROUTINE_CORRECT
  MUL_THEOREM=JOLT_FP64_MUL_SUBROUTINE_CORRECT
  BMI2_MUL_THEOREM=""
  {
    printf 'loadt "arm/proofs/base.ml";;\n'
    printf 'loadt "%s/fp64_common.ml";;\n' "$PROOF_DIR"
    printf 'print_endline "[HOL 1/4] Proving AArch64 Fp64 addition";;\n'
    printf 'loadt "%s/fp64_add_object.ml";;\n' "$PROOF_DIR"
    printf 'loadt "%s/fp64_add_correct.ml";;\n' "$PROOF_DIR"
    printf 'print_endline "[HOL 2/4] Proving AArch64 Fp64 subtraction";;\n'
    printf 'loadt "%s/fp64_sub_object.ml";;\n' "$PROOF_DIR"
    printf 'loadt "%s/fp64_sub_correct.ml";;\n' "$PROOF_DIR"
    printf 'print_endline "[HOL 3/4] Proving AArch64 Fp64 multiplication";;\n'
    printf 'loadt "%s/fp64_mul_object.ml";;\n' "$PROOF_DIR"
    printf 'loadt "%s/fp64_mul_correct.ml";;\n' "$PROOF_DIR"
    printf 'print_endline "[HOL 4/4] Checking the Fp64 modulus primality certificate";;\n'
    printf 'loadt "%s/fp64_prime.ml";;\n' "$PROOF_DIR"
  } >"$COMBINED_SOURCE"
  PROOF_SOURCES=("$COMBINED_SOURCE" "$PROOF_DIR"/fp64_*.ml)
fi

for theorem in "$ADD_THEOREM" "$SUB_THEOREM" "$MUL_THEOREM"; do
  printf 'Printf.printf "val %s : thm = %%s\\n" (string_of_thm %s);;\n' \
    "$theorem" "$theorem" >>"$COMBINED_SOURCE"
done
if [[ -n "$BMI2_MUL_THEOREM" ]]; then
  printf 'Printf.printf "val %s : thm = %%s\\n" (string_of_thm %s);;\n' \
    "$BMI2_MUL_THEOREM" "$BMI2_MUL_THEOREM" >>"$COMBINED_SOURCE"
fi
printf 'Printf.printf "val JOLT_FP64_PRIME : thm = %%s\\n" (string_of_thm JOLT_FP64_PRIME);;\n' \
  >>"$COMBINED_SOURCE"

CACHE_KEY=$(
  {
    git -C "$HOL_LIGHT_DIR" rev-parse HEAD
    git -C "$S2N_BIGNUM_DIR" rev-parse HEAD
    git hash-object "${PROOF_SOURCES[@]}"
  } | git hash-object --stdin
)
if ! $CLEAN && [[ -x "$NATIVE_PROOF" ]] &&
  [[ -f "$CACHE_KEY_PATH" ]] && [[ "$(<"$CACHE_KEY_PATH")" == "$CACHE_KEY" ]]; then
  echo "[3/5] Reusing the cached combined $ARCHITECTURE_LABEL Fp64 proof"
else
  echo "[3/5] Building one combined $ARCHITECTURE_LABEL Fp64 proof"
  build_proof "$COMBINED_SOURCE" "$NATIVE_PROOF"
  printf '%s\n' "$CACHE_KEY" >"$CACHE_KEY_PATH"
fi

echo "[4/5] Loading the $ARCHITECTURE_LABEL model and proving Fp64 arithmetic"
JOLT_FP64_PROOF_DIR="$PROOF_DIR" \
JOLT_FP64_ADD_OBJECT="$ADD_PROOF_OBJECT" \
JOLT_FP64_SUB_OBJECT="$SUB_PROOF_OBJECT" \
JOLT_FP64_MUL_OBJECT="$MUL_PROOF_OBJECT" \
JOLT_FP64_MUL_BMI2_OBJECT="$BMI2_MUL_PROOF_OBJECT" \
  "$NATIVE_PROOF" 2>&1 | tee "$LOG_PATH"

echo "[5/5] Confirming the $ARCHITECTURE_LABEL Fp64 theorems"
for theorem in "$ADD_THEOREM" "$SUB_THEOREM" "$MUL_THEOREM"; do
  grep -F "val $theorem : thm" "$LOG_PATH"
done
if [[ -n "$BMI2_MUL_THEOREM" ]]; then
  grep -F "val $BMI2_MUL_THEOREM : thm" "$LOG_PATH"
fi
grep -F "val JOLT_FP64_PRIME : thm" "$LOG_PATH"
echo "Fp64 $ARCHITECTURE_LABEL inspection witness proofs passed."
