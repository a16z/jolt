#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PROOF_DIR="$REPO_ROOT/proofs/hol-light"
MATRIX_PATH="$PROOF_DIR/fp64-certified-builds.json"
MATRIX_TOOL="$REPO_ROOT/scripts/fp64_certified_matrix.py"

usage() {
  echo "usage: $0 [all|bytes] [aarch64|x86_64] [--matrix-entry ID] [--evidence-out PATH] [--clean]" >&2
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
MATRIX_ENTRY_ID=""
EVIDENCE_OUT=""
CLEAN=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    all | bytes) MODE=$1 ;;
    aarch64 | arm64 | x86_64 | amd64)
      ARCHITECTURE=$(normalize_architecture "$1")
      ;;
    --matrix-entry)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      MATRIX_ENTRY_ID=$2
      shift
      ;;
    --evidence-out)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      EVIDENCE_OUT=$2
      shift
      ;;
    --clean) CLEAN=true ;;
    -h | --help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
  shift
done

python3 "$MATRIX_TOOL" --matrix "$MATRIX_PATH" validate
HOST_TRIPLE=$(rustc -Vv | sed -n 's/^host: //p')
HOST_ARCHITECTURE=$(normalize_architecture "${HOST_TRIPLE%%-*}") || {
  echo "unsupported Rust host architecture: $HOST_TRIPLE" >&2
  exit 2
}

if [[ -n "$MATRIX_ENTRY_ID" ]]; then
  TARGET_TRIPLE=$(python3 "$MATRIX_TOOL" --matrix "$MATRIX_PATH" get \
    --target-id "$MATRIX_ENTRY_ID" --field target_triple)
  MATRIX_ARCHITECTURE=$(python3 "$MATRIX_TOOL" --matrix "$MATRIX_PATH" get \
    --target-id "$MATRIX_ENTRY_ID" --field architecture)
  if [[ -n "$ARCHITECTURE" && "$ARCHITECTURE" != "$MATRIX_ARCHITECTURE" ]]; then
    echo "matrix entry $MATRIX_ENTRY_ID is for $MATRIX_ARCHITECTURE, not $ARCHITECTURE" >&2
    exit 2
  fi
  ARCHITECTURE=$MATRIX_ARCHITECTURE
else
  if [[ -z "$ARCHITECTURE" ]]; then
    ARCHITECTURE=$HOST_ARCHITECTURE
  fi
  if [[ "$ARCHITECTURE" == "$HOST_ARCHITECTURE" ]]; then
    TARGET_TRIPLE=$HOST_TRIPLE
  elif [[ "$HOST_TRIPLE" == aarch64-apple-darwin && "$ARCHITECTURE" == x86_64 ]]; then
    TARGET_TRIPLE=x86_64-apple-darwin
  else
    echo "cross checking $ARCHITECTURE from $HOST_TRIPLE is not configured" >&2
    exit 2
  fi
  MATRIX_ENTRY_ID=$(python3 "$MATRIX_TOOL" --matrix "$MATRIX_PATH" resolve \
    --target-triple "$TARGET_TRIPLE")
fi

CERTIFICATION_SCOPE=$(python3 "$MATRIX_TOOL" --matrix "$MATRIX_PATH" get \
  --target-id "$MATRIX_ENTRY_ID" --field certification_scope)
python3 "$MATRIX_TOOL" --matrix "$MATRIX_PATH" check-environment \
  --target-id "$MATRIX_ENTRY_ID"
MATRIX_CONTRACT_ID=$(python3 "$MATRIX_TOOL" --matrix "$MATRIX_PATH" contract-id)
build_field() {
  python3 "$MATRIX_TOOL" --matrix "$MATRIX_PATH" build-field --field "$1"
}
RELEASE_OPT_LEVEL=$(build_field release_profile.opt_level)
RELEASE_LTO=$(build_field release_profile.lto)
RELEASE_CODEGEN_UNITS=$(build_field release_profile.codegen_units)
RELEASE_DEBUG=$(build_field release_profile.debug)
RELEASE_DEBUG_ASSERTIONS=$(build_field release_profile.debug_assertions)
RELEASE_OVERFLOW_CHECKS=$(build_field release_profile.overflow_checks)
RELEASE_SPLIT_DEBUGINFO=$(build_field release_profile.split_debuginfo)
RELEASE_STRIP=$(build_field release_profile.strip)
RELEASE_RPATH=$(build_field release_profile.rpath)
RELEASE_INCREMENTAL=$(build_field release_profile.incremental)
RELEASE_PANIC=$(build_field release_profile.panic)
if [[ -z "$EVIDENCE_OUT" ]]; then
  EVIDENCE_OUT="$REPO_ROOT/target/fp64-formal-verification/evidence/$MATRIX_ENTRY_ID.json"
fi
CARGO_BUILD_ARGS=(
  --config "profile.release.opt-level=$RELEASE_OPT_LEVEL"
  --config "profile.release.lto=\"$RELEASE_LTO\""
  --config "profile.release.codegen-units=$RELEASE_CODEGEN_UNITS"
  --config "profile.release.debug=$RELEASE_DEBUG"
  --config "profile.release.debug-assertions=$RELEASE_DEBUG_ASSERTIONS"
  --config "profile.release.overflow-checks=$RELEASE_OVERFLOW_CHECKS"
  --config "profile.release.split-debuginfo=\"$RELEASE_SPLIT_DEBUGINFO\""
  --config "profile.release.strip=\"$RELEASE_STRIP\""
  --config "profile.release.rpath=$RELEASE_RPATH"
  --config "profile.release.incremental=$RELEASE_INCREMENTAL"
  --config "profile.release.panic=\"$RELEASE_PANIC\""
  build --locked -p jolt-field --release --target "$TARGET_TRIPLE"
)

if $CLEAN; then
  PROOF_TARGET=$(mktemp -d)
else
  PROOF_TARGET="$REPO_ROOT/target/fp64-formal-verification/$MATRIX_ENTRY_ID"
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

PROFILE_ROOT="$PROOF_TARGET/$TARGET_TRIPLE/release"

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
JOLT_FP64_PROOF_MATRIX_CONTRACT="$MATRIX_CONTRACT_ID" \
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
  BMI2_PROFILE_ROOT="$BMI2_TARGET/$TARGET_TRIPLE/release"
  JOLT_FP64_PROOF_MATRIX_CONTRACT="$MATRIX_CONTRACT_ID" \
  RUSTFLAGS="-C target-feature=+bmi2" \
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
  --matrix "$MATRIX_PATH"
  --target-id "$MATRIX_ENTRY_ID"
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

operation_value() {
  local profile=$1
  local operation=$2
  local field=$3
  python3 "$MATRIX_TOOL" --matrix "$MATRIX_PATH" operation \
    --target-id "$MATRIX_ENTRY_ID" --profile "$profile" \
    --operation "$operation" --field "$field"
}

ADD_OBJECT_SOURCE=$(operation_value baseline add object_source)
ADD_CORRECT_SOURCE=$(operation_value baseline add correctness_source)
ADD_THEOREM=$(operation_value baseline add subroutine_theorem)
SUB_OBJECT_SOURCE=$(operation_value baseline sub object_source)
SUB_CORRECT_SOURCE=$(operation_value baseline sub correctness_source)
SUB_THEOREM=$(operation_value baseline sub subroutine_theorem)
MUL_OBJECT_SOURCE=$(operation_value baseline mul object_source)
MUL_CORRECT_SOURCE=$(operation_value baseline mul correctness_source)
MUL_THEOREM=$(operation_value baseline mul subroutine_theorem)

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
  BMI2_MUL_OBJECT_SOURCE=$(operation_value bmi2-mul mul object_source)
  BMI2_MUL_CORRECT_SOURCE=$(operation_value bmi2-mul mul correctness_source)
  BMI2_MUL_THEOREM=$(operation_value bmi2-mul mul subroutine_theorem)
  {
    printf 'loadt "x86/proofs/base.ml";;\n'
    printf 'loadt "%s/fp64_common.ml";;\n' "$PROOF_DIR"
    printf 'loadt "%s/fp64_x86_64_common.ml";;\n' "$PROOF_DIR"
    printf 'print_endline "[HOL 1/5] Proving x86-64 Fp64 addition";;\n'
    printf 'loadt "%s/%s";;\n' "$PROOF_DIR" "$ADD_OBJECT_SOURCE"
    printf 'loadt "%s/%s";;\n' "$PROOF_DIR" "$ADD_CORRECT_SOURCE"
    printf 'print_endline "[HOL 2/5] Proving x86-64 Fp64 subtraction";;\n'
    printf 'loadt "%s/%s";;\n' "$PROOF_DIR" "$SUB_OBJECT_SOURCE"
    printf 'loadt "%s/%s";;\n' "$PROOF_DIR" "$SUB_CORRECT_SOURCE"
    printf 'print_endline "[HOL 3/5] Proving baseline x86-64 Fp64 multiplication";;\n'
    printf 'loadt "%s/%s";;\n' "$PROOF_DIR" "$MUL_OBJECT_SOURCE"
    printf 'loadt "%s/%s";;\n' "$PROOF_DIR" "$MUL_CORRECT_SOURCE"
    printf 'print_endline "[HOL 4/5] Proving BMI2 x86-64 Fp64 multiplication";;\n'
    printf 'loadt "%s/%s";;\n' "$PROOF_DIR" "$BMI2_MUL_OBJECT_SOURCE"
    printf 'loadt "%s/%s";;\n' "$PROOF_DIR" "$BMI2_MUL_CORRECT_SOURCE"
    printf 'print_endline "[HOL 5/5] Checking the Fp64 modulus primality certificate";;\n'
    printf 'loadt "%s/fp64_prime.ml";;\n' "$PROOF_DIR"
  } >"$COMBINED_SOURCE"
  PROOF_SOURCES=("$COMBINED_SOURCE" "$PROOF_DIR"/fp64_*.ml)
else
  ARCHITECTURE_LABEL=AArch64
  BMI2_MUL_THEOREM=""
  {
    printf 'loadt "arm/proofs/base.ml";;\n'
    printf 'loadt "%s/fp64_common.ml";;\n' "$PROOF_DIR"
    printf 'print_endline "[HOL 1/4] Proving AArch64 Fp64 addition";;\n'
    printf 'loadt "%s/%s";;\n' "$PROOF_DIR" "$ADD_OBJECT_SOURCE"
    printf 'loadt "%s/%s";;\n' "$PROOF_DIR" "$ADD_CORRECT_SOURCE"
    printf 'print_endline "[HOL 2/4] Proving AArch64 Fp64 subtraction";;\n'
    printf 'loadt "%s/%s";;\n' "$PROOF_DIR" "$SUB_OBJECT_SOURCE"
    printf 'loadt "%s/%s";;\n' "$PROOF_DIR" "$SUB_CORRECT_SOURCE"
    printf 'print_endline "[HOL 3/4] Proving AArch64 Fp64 multiplication";;\n'
    printf 'loadt "%s/%s";;\n' "$PROOF_DIR" "$MUL_OBJECT_SOURCE"
    printf 'loadt "%s/%s";;\n' "$PROOF_DIR" "$MUL_CORRECT_SOURCE"
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

EVIDENCE_ARGS=(
  --matrix "$MATRIX_PATH"
  evidence
  --target-id "$MATRIX_ENTRY_ID"
  --repo-root "$REPO_ROOT"
  --hol-light-dir "$HOL_LIGHT_DIR"
  --s2n-bignum-dir "$S2N_BIGNUM_DIR"
  --output "$EVIDENCE_OUT"
  --artifact "add_object=$ADD_OBJECT"
  --artifact "add_proof_object=$ADD_PROOF_OBJECT"
  --artifact "sub_object=$SUB_OBJECT"
  --artifact "sub_proof_object=$SUB_PROOF_OBJECT"
  --artifact "mul_object=$MUL_OBJECT"
  --artifact "mul_proof_object=$MUL_PROOF_OBJECT"
  --artifact "production_witness=$PRODUCTION_WITNESS"
  --artifact "proof_log=$LOG_PATH"
)
while IFS= read -r profile; do
  EVIDENCE_ARGS+=(--profile "$profile")
done < <(python3 "$MATRIX_TOOL" --matrix "$MATRIX_PATH" profile-ids \
  --target-id "$MATRIX_ENTRY_ID")
if [[ "$ARCHITECTURE" == x86_64 ]]; then
  EVIDENCE_ARGS+=(
    --artifact "mul_bmi2_object=$BMI2_MUL_OBJECT"
    --artifact "mul_bmi2_proof_object=$BMI2_MUL_PROOF_OBJECT"
    --artifact "bmi2_production_witness=$BMI2_PRODUCTION_WITNESS"
  )
fi
if $CLEAN; then
  EVIDENCE_ARGS+=(--clean)
fi
python3 "$MATRIX_TOOL" "${EVIDENCE_ARGS[@]}"
echo "Fp64 matrix entry $MATRIX_ENTRY_ID passed ($CERTIFICATION_SCOPE)."
