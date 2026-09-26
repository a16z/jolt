#!/usr/bin/env bash
set -euo pipefail

# Runs the jolt-metal GPU tests on this Mac and prints a Markdown report to
# paste into the PR. Hosted CI runners cannot run the GPU tests, so this
# report is the record of their results (crates/jolt-metal/README.md).
#
# The suite runs twice: normally, and under Metal's API and shader validation
# layers. Shader validation catches out-of-bounds device accesses that happen
# not to corrupt the result, but by default it only logs them and the command
# buffer still completes; ABORT_ON_FAULT turns each one into a failed test.
# Exits non-zero if either run fails.
#
# With --bench, it then runs crates/jolt-metal/benches/field.rs and appends
# the GPU and CPU throughput table (scripts/metal-bench-table.py), then runs
# crates/jolt-metal/benches/limits.rs and appends the machine limits and the
# paired fractions of them. Run it on AC power, not in low power mode. Other
# processes slow the CPU baseline and share the chip's power budget with the
# GPU, so the report records the power source, the energy mode, and the load
# average before and after each benchmark, and absolute rates are read with
# the load they were measured under (specs/jolt-metal-field.md, Measurement
# hygiene).

if [[ "$(uname -s)" != Darwin ]]; then
  echo "error: the Metal report runs on macOS only" >&2
  exit 1
fi
export CARGO_TERM_COLOR=never

bench=false
case "${1:-}" in
  "") ;;
  --bench) bench=true ;;
  *)
    echo "usage: $0 [--bench]" >&2
    exit 2
    ;;
esac

cd "$(git rev-parse --show-toplevel)"
dirty=""
if ! git diff --quiet HEAD -- crates/jolt-metal Cargo.toml Cargo.lock ||
  [[ -n "$(git ls-files --others --exclude-standard crates/jolt-metal)" ]]; then
  dirty=" (uncommitted changes in crates/jolt-metal or the manifests)"
fi

run() {
  local log status
  log="$(mktemp)"
  set +e
  env "$@" cargo nextest run -p jolt-metal --cargo-quiet --no-fail-fast >"$log" 2>&1
  status=$?
  set -e
  echo '```'
  grep -E '^ +(PASS|FAIL|SIGABRT|SIGSEGV|TIMEOUT|Summary) |Invalid Metal usage' "$log" || cat "$log"
  echo '```'
  rm -f "$log"
  return "$status"
}

echo "### jolt-metal local report"
echo
echo "- commit: \`$(git rev-parse HEAD)\`$dirty"
echo "- machine: $(sysctl -n machdep.cpu.brand_string), $(($(sysctl -n hw.memsize) / 1073741824)) GiB"
echo "- macOS: $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
echo "- rustc: $(rustc --version)"
echo
echo '```'
cargo run --quiet -p jolt-metal --example metal_probe 2>/dev/null
echo '```'
echo
echo "#### Tests"
echo
status=0
run || status=1
echo
echo "#### Tests under \`MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1\`"
echo
run MTL_DEBUG_LAYER=1 MTL_DEBUG_LAYER_ERROR_MODE=assert MTL_SHADER_VALIDATION=1 \
  MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 MTL_SHADER_VALIDATION_ABORT_ON_FAULT=1 || status=1

if $bench; then
  echo
  echo "#### Benchmarks"
  echo
  # pmset powermode: 0 automatic, 1 low power, 2 high power.
  echo "- power: $(pmset -g batt | head -1 | sed -E "s/.*'(.*)'.*/\1/"), energy mode $(pmset -g | awk '/ powermode / {print ($2 == 0 ? "automatic" : $2 == 1 ? "low power" : $2 == 2 ? "high power" : $2)}')"
  echo "- GPU: GPU execution time; CPU: wall time on all cores, \`jolt_field\` with \`asm\`"
  rm -rf target/criterion/metal_*
  log="$(mktemp)"
  err="$(mktemp)"
  # Build first, so the load average before the run excludes the compiler.
  bench_status=0
  cargo bench -p jolt-metal --bench field --bench limits --no-run >"$log" 2>&1 || bench_status=$?
  if [[ $bench_status -eq 0 ]]; then
    echo "- load average (1, 5, 15 min) on $(sysctl -n hw.ncpu) cores, before: $(sysctl -n vm.loadavg | tr -d '{}' | xargs)"
    cargo bench -p jolt-metal --bench field -- --noplot >"$log" 2>&1 || bench_status=$?
  fi
  echo "- load average after: $(sysctl -n vm.loadavg | tr -d '{}' | xargs)"
  echo
  if [[ $bench_status -eq 0 ]]; then
    scripts/metal-bench-table.py target/criterion
    echo
    # Its stdout is the Markdown; its stderr joins the log only on failure.
    cargo bench -p jolt-metal --bench limits >"$log" 2>"$err" || {
      bench_status=$?
      cat "$err" >>"$log"
    }
  fi
  if [[ $bench_status -eq 0 ]]; then
    sed 's/^### /#### /' "$log"
  else
    echo '```'
    tail -n 40 "$log"
    echo '```'
    status=1
  fi
  rm -f "$log" "$err"
fi
exit "$status"
