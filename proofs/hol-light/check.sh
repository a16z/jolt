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
  PROOF_TARGET="$REPO_ROOT/target/fp128-formal-verification/$ARCHITECTURE"
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

if [[ "$MODE" == bytes ]]; then
  echo "[1/2] Building the $ARCHITECTURE production witness"
else
  echo "[1/5] Building the $ARCHITECTURE production witness"
fi
CARGO_TARGET_DIR="$PROOF_TARGET" \
  cargo "${CARGO_BUILD_ARGS[@]}" \
    --no-default-features \
    --features solinas,fp128-proof-linkage \
    --example fp128_production_witness

ADD_OBJECT=$(find_newest_object fp128_add.o)
SUB_OBJECT=$(find_newest_object fp128_sub.o)
MUL_OBJECT=""
if [[ "$ARCHITECTURE" == aarch64 ]]; then
  MUL_OBJECT=$(find_newest_object fp128_mul.o)
fi
PRODUCTION_WITNESS="$PROFILE_ROOT/examples/fp128_production_witness"

if [[ "$MODE" == bytes ]]; then
  echo "[2/2] Checking exact object and public witness bytes"
else
  echo "[2/5] Checking exact object and public witness bytes"
fi
CHECKER_ARGS=(
  --architecture "$ARCHITECTURE"
  --add-object "$ADD_OBJECT"
  --sub-object "$SUB_OBJECT"
  --production-witness "$PRODUCTION_WITNESS"
)
if [[ -n "$MUL_OBJECT" ]]; then
  CHECKER_ARGS+=(--mul-object "$MUL_OBJECT")
fi
python3 "$REPO_ROOT/scripts/check_fp128_proof_artifacts.py" \
  "${CHECKER_ARGS[@]}"

if [[ "$MODE" == bytes ]]; then
  exit 0
fi

: "${HOL_LIGHT_DIR:?set HOL_LIGHT_DIR to a module-built HOL Light checkout}"
: "${S2N_BIGNUM_DIR:?set S2N_BIGNUM_DIR to an s2n-bignum checkout}"

ADD_PROOF_OBJECT=$ADD_OBJECT
SUB_PROOF_OBJECT=$SUB_OBJECT
if [[ "$ARCHITECTURE" == x86_64 && "$(uname -s)" == Darwin ]]; then
  ELF_OBJECT_DIR="$PROOF_TARGET/x86_64-elf"
  mkdir -p "$ELF_OBJECT_DIR"
  clang --target=x86_64-unknown-linux-gnu \
    -c -I "$REPO_ROOT/crates/jolt-field/asm/x86_64" \
    "$REPO_ROOT/crates/jolt-field/asm/x86_64/fp128_add.S" \
    -o "$ELF_OBJECT_DIR/fp128_add.o"
  clang --target=x86_64-unknown-linux-gnu \
    -c -I "$REPO_ROOT/crates/jolt-field/asm/x86_64" \
    "$REPO_ROOT/crates/jolt-field/asm/x86_64/fp128_sub.S" \
    -o "$ELF_OBJECT_DIR/fp128_sub.o"
  ADD_PROOF_OBJECT="$ELF_OBJECT_DIR/fp128_add.o"
  SUB_PROOF_OBJECT="$ELF_OBJECT_DIR/fp128_sub.o"
fi

S2N_ARCH_DIR=$ARCHITECTURE
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
      JOLT_FP128_PROOF_DIR="$PROOF_DIR" \
        opam exec --switch "$HOL_LIGHT_DIR" -- \
          ../tools/build-proof.sh \
          "$proof_relative" \
          "$HOL_LIGHT_DIR/hol.sh" \
          "$output_relative"
    else
      JOLT_FP128_PROOF_DIR="$PROOF_DIR" \
        opam exec -- \
          ../tools/build-proof.sh \
          "$proof_relative" \
          "$HOL_LIGHT_DIR/hol.sh" \
          "$output_relative"
    fi
  )
}

LOG_DIR="$PROOF_TARGET/proof-logs"
mkdir -p "$LOG_DIR"

COMBINED_SOURCE="$PROOF_TARGET/fp128_${ARCHITECTURE}_all.ml"
NATIVE_PROOF="$PROOF_TARGET/fp128_${ARCHITECTURE}_all.native"
LOG_PATH="$LOG_DIR/$ARCHITECTURE-all.log"
CACHE_KEY_PATH="$PROOF_TARGET/fp128_${ARCHITECTURE}_all.cache-key"

if [[ "$ARCHITECTURE" == x86_64 ]]; then
  ARCHITECTURE_LABEL=x86-64
  ADD_THEOREM=JOLT_FP128_ADD_X86_64_SUBROUTINE_CORRECT
  SUB_THEOREM=JOLT_FP128_SUB_X86_64_SUBROUTINE_CORRECT
  MUL_THEOREM=""
  printf 'print_endline "[HOL 1/3] Proving x86-64 addition";;\n' >"$COMBINED_SOURCE"
  printf 'loadt "%s/fp128_common.ml";;\n' "$PROOF_DIR" >>"$COMBINED_SOURCE"
  printf 'loadt "%s/fp128_x86_64_common.ml";;\n' "$PROOF_DIR" >>"$COMBINED_SOURCE"
  printf 'loadt "%s/fp128_add_x86_64_object.ml";;\n' "$PROOF_DIR" >>"$COMBINED_SOURCE"
  printf 'loadt "%s/fp128_add_x86_64_correct.ml";;\n' "$PROOF_DIR" >>"$COMBINED_SOURCE"
  printf 'print_endline "[HOL 2/3] Proving x86-64 subtraction";;\n' >>"$COMBINED_SOURCE"
  printf 'loadt "%s/fp128_sub_x86_64_object.ml";;\n' "$PROOF_DIR" >>"$COMBINED_SOURCE"
  printf 'loadt "%s/fp128_sub_x86_64_correct.ml";;\n' "$PROOF_DIR" >>"$COMBINED_SOURCE"
  printf 'print_endline "[HOL 3/3] Certifying the A7F7 modulus";;\n' >>"$COMBINED_SOURCE"
  printf 'loadt "%s/fp128_prime.ml";;\n' "$PROOF_DIR" >>"$COMBINED_SOURCE"
  PROOF_SOURCES=(
    "$COMBINED_SOURCE"
    "$PROOF_DIR/fp128_common.ml"
    "$PROOF_DIR/fp128_x86_64_common.ml"
    "$PROOF_DIR/fp128_add_x86_64_object.ml"
    "$PROOF_DIR/fp128_add_x86_64_correct.ml"
    "$PROOF_DIR/fp128_sub_x86_64_object.ml"
    "$PROOF_DIR/fp128_sub_x86_64_correct.ml"
    "$PROOF_DIR/fp128_prime.ml"
  )
else
  ARCHITECTURE_LABEL=AArch64
  ADD_THEOREM=JOLT_FP128_ADD_SUBROUTINE_CORRECT
  SUB_THEOREM=JOLT_FP128_SUB_SUBROUTINE_CORRECT
  MUL_THEOREM=JOLT_FP128_MUL_SUBROUTINE_CORRECT
  printf 'print_endline "[HOL 1/4] Proving AArch64 addition";;\n' >"$COMBINED_SOURCE"
  printf 'loadt "%s/fp128_common.ml";;\n' "$PROOF_DIR" >>"$COMBINED_SOURCE"
  printf 'loadt "%s/fp128_add_correct.ml";;\n' "$PROOF_DIR" >>"$COMBINED_SOURCE"
  printf 'print_endline "[HOL 2/4] Proving AArch64 subtraction";;\n' >>"$COMBINED_SOURCE"
  printf 'loadt "%s/fp128_sub_correct.ml";;\n' "$PROOF_DIR" >>"$COMBINED_SOURCE"
  printf 'print_endline "[HOL 3/4] Proving AArch64 multiplication";;\n' >>"$COMBINED_SOURCE"
  printf 'loadt "%s/fp128_mul_object.ml";;\n' "$PROOF_DIR" >>"$COMBINED_SOURCE"
  printf 'loadt "%s/fp128_mul_correct.ml";;\n' "$PROOF_DIR" >>"$COMBINED_SOURCE"
  printf 'print_endline "[HOL 4/4] Certifying the A7F7 modulus";;\n' >>"$COMBINED_SOURCE"
  printf 'loadt "%s/fp128_prime.ml";;\n' "$PROOF_DIR" >>"$COMBINED_SOURCE"
  PROOF_SOURCES=(
    "$COMBINED_SOURCE"
    "$PROOF_DIR/fp128_common.ml"
    "$PROOF_DIR/fp128_add_correct.ml"
    "$PROOF_DIR/fp128_sub_correct.ml"
    "$PROOF_DIR/fp128_mul_object.ml"
    "$PROOF_DIR/fp128_mul_correct.ml"
    "$PROOF_DIR/fp128_prime.ml"
  )
fi

printf 'Printf.printf "val %s : thm = %%s\\n" (string_of_thm %s);;\n' \
  "$ADD_THEOREM" "$ADD_THEOREM" >>"$COMBINED_SOURCE"
printf 'Printf.printf "val %s : thm = %%s\\n" (string_of_thm %s);;\n' \
  "$SUB_THEOREM" "$SUB_THEOREM" >>"$COMBINED_SOURCE"
if [[ -n "$MUL_THEOREM" ]]; then
  printf 'Printf.printf "val %s : thm = %%s\\n" (string_of_thm %s);;\n' \
    "$MUL_THEOREM" "$MUL_THEOREM" >>"$COMBINED_SOURCE"
fi
printf 'Printf.printf "val JOLT_FP128_A7F7_PRIME : thm = %%s\\n" (string_of_thm JOLT_FP128_A7F7_PRIME);;\n' \
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
  echo "[3/5] Reusing the cached combined $ARCHITECTURE_LABEL proof program"
else
  echo "[3/5] Building one combined $ARCHITECTURE_LABEL proof program"
  build_proof "$COMBINED_SOURCE" "$NATIVE_PROOF"
  printf '%s\n' "$CACHE_KEY" >"$CACHE_KEY_PATH"
fi

echo "[4/5] Loading the $ARCHITECTURE_LABEL model and proving both operations"
JOLT_FP128_PROOF_DIR="$PROOF_DIR" \
JOLT_FP128_ADD_OBJECT="$ADD_PROOF_OBJECT" \
JOLT_FP128_SUB_OBJECT="$SUB_PROOF_OBJECT" \
JOLT_FP128_MUL_OBJECT="$MUL_OBJECT" \
  "$NATIVE_PROOF" 2>&1 | tee "$LOG_PATH"
echo "[5/5] Confirming the $ARCHITECTURE_LABEL theorems"
grep -F "val $ADD_THEOREM : thm" "$LOG_PATH"
grep -F "val $SUB_THEOREM : thm" "$LOG_PATH"
if [[ -n "$MUL_THEOREM" ]]; then
  grep -F "val $MUL_THEOREM : thm" "$LOG_PATH"
fi
grep -F "val JOLT_FP128_A7F7_PRIME : thm" "$LOG_PATH"

echo "Fp128 $ARCHITECTURE public witness proofs passed."
