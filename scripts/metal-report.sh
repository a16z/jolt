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

if [[ "$(uname -s)" != Darwin ]]; then
  echo "error: the Metal report runs on macOS only" >&2
  exit 1
fi
export CARGO_TERM_COLOR=never

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
exit "$status"
