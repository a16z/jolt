#!/usr/bin/env bash
# ACT4 runner for Jolt.
#
# Iterates over ACT4-generated self-checking ELFs, skips any whose basename
# appears in a skip file, runs each remaining ELF through `jolt-emu`, and
# reports pass/fail based on the emulator's process exit status (propagated
# from the HTIF endcode; see tracer/src/main.rs and tracer/src/emulator/mod.rs
# ::run_test).
#
# Usage:
#   run.sh --emulator <path> --work-dir <dir> [--skip-file <path>] [--timeout-secs N]
#
# --work-dir is the ACT4 build output directory containing <test>/elfs/*.elf.
# Exits 0 on full success; non-zero if any executed test fails or times out.

set -euo pipefail

EMULATOR=""
WORK_DIR=""
SKIP_FILE=""
TIMEOUT_SECS="60"

usage() {
    cat >&2 <<EOF
usage: $0 --emulator <path> --work-dir <dir> [--skip-file <path>] [--timeout-secs N]
EOF
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --emulator)     EMULATOR="${2:-}"; shift 2 ;;
        --work-dir)     WORK_DIR="${2:-}"; shift 2 ;;
        --skip-file)    SKIP_FILE="${2:-}"; shift 2 ;;
        --timeout-secs) TIMEOUT_SECS="${2:-}"; shift 2 ;;
        -h|--help)      usage ;;
        *)              echo "unknown arg: $1" >&2; usage ;;
    esac
done

[[ -n "$EMULATOR" ]] || { echo "--emulator is required" >&2; usage; }
[[ -n "$WORK_DIR" ]] || { echo "--work-dir is required" >&2; usage; }

if [[ ! -x "$EMULATOR" ]]; then
    echo "emulator not found or not executable: $EMULATOR" >&2
    exit 2
fi
if [[ ! -d "$WORK_DIR" ]]; then
    echo "work dir does not exist: $WORK_DIR" >&2
    echo "did the ACT4 generate step run successfully?" >&2
    exit 2
fi

# Build the skip set as a newline-delimited string. Comments (#) and blank
# lines are ignored. We deliberately avoid bash-4 associative arrays so this
# script runs on macOS's default bash 3.2 as well as CI's bash 5.
SKIP_ENTRIES=""
SKIP_COUNT=0
if [[ -n "$SKIP_FILE" ]]; then
    if [[ ! -f "$SKIP_FILE" ]]; then
        echo "skip file does not exist: $SKIP_FILE" >&2
        exit 2
    fi
    while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
        # Strip comment
        line="${raw_line%%#*}"
        # Strip leading/trailing whitespace
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        [[ -z "$line" ]] && continue
        SKIP_ENTRIES="${SKIP_ENTRIES}${line}"$'\n'
        SKIP_COUNT=$((SKIP_COUNT + 1))
    done < "$SKIP_FILE"
fi

is_skipped() {
    [[ -z "$SKIP_ENTRIES" ]] && return 1
    printf '%s' "$SKIP_ENTRIES" | grep -Fxq -- "$1"
}

# Prefer GNU `timeout` if present. On macOS this is commonly `gtimeout`. If
# neither is available, fall back to running without a timeout.
TIMEOUT_CMD=""
if command -v timeout >/dev/null 2>&1; then
    TIMEOUT_CMD="timeout"
elif command -v gtimeout >/dev/null 2>&1; then
    TIMEOUT_CMD="gtimeout"
fi

# Collect ELFs. ACT4 produces two variants per test:
#   <work>/<name>/elfs/**/foo.elf      — the DUT build, with Sail-computed
#                                        signatures baked in (what we run).
#   <work>/<name>/build/**/foo.sig.elf — signature-generation intermediate
#                                        used only by Sail at build time.
# We explicitly exclude `*.sig.elf` so the runner doesn't execute the
# intermediates through jolt-emu. Using `find` (portable across bash
# versions) in case ACT4's internal layout evolves between releases.
ELFS_LIST="$(find "$WORK_DIR" -type f -name '*.elf' ! -name '*.sig.elf' 2>/dev/null | sort -u)"
if [[ -z "$ELFS_LIST" ]]; then
    echo "no ELFs found under $WORK_DIR" >&2
    exit 2
fi

total=$(printf '%s\n' "$ELFS_LIST" | wc -l | tr -d ' ')
passed=0
failed=0
skipped=0
declare -a FAILURES=()

echo "ACT4: running $total ELF(s) through $EMULATOR"
if [[ -n "$SKIP_FILE" ]]; then
    echo "ACT4: skip list: $SKIP_FILE ($SKIP_COUNT entries)"
fi

while IFS= read -r elf; do
    [[ -z "$elf" ]] && continue
    name="$(basename "$elf" .elf)"
    if is_skipped "$name"; then
        skipped=$((skipped + 1))
        echo "SKIP  $name"
        continue
    fi

    if [[ -n "$TIMEOUT_CMD" ]]; then
        if "$TIMEOUT_CMD" "${TIMEOUT_SECS}s" "$EMULATOR" "$elf" >/dev/null 2>&1; then
            passed=$((passed + 1))
            echo "PASS  $name"
        else
            rc=$?
            failed=$((failed + 1))
            FAILURES+=("$name (exit $rc)")
            echo "FAIL  $name (exit $rc)"
        fi
    else
        if "$EMULATOR" "$elf" >/dev/null 2>&1; then
            passed=$((passed + 1))
            echo "PASS  $name"
        else
            rc=$?
            failed=$((failed + 1))
            FAILURES+=("$name (exit $rc)")
            echo "FAIL  $name (exit $rc)"
        fi
    fi
done <<< "$ELFS_LIST"

echo
echo "ACT4 summary: $passed passed, $failed failed, $skipped skipped (of $total)"

if [[ $failed -gt 0 ]]; then
    echo
    echo "failures:"
    for f in "${FAILURES[@]}"; do
        echo "  - $f"
    done
    exit 1
fi

exit 0
