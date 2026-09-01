#!/usr/bin/env bash
# One-shot AC1 baseline run for a fresh linux-x86_64 box.
#
# Produces the markdown table that belongs in the x86-tracer slice-0 PR
# description, with machine and revision provenance attached.
#
# Usage, from a clean checkout of the branch under measurement:
#   ./jolt-eval/bin/run_baseline_x86.sh          # full run
#   RUNS=9 ./jolt-eval/bin/run_baseline_x86.sh   # more repetitions
#
# Budget roughly an hour of machine time: the measurement itself is minutes,
# but the first build of jolt-prover-legacy with `host` dominates. Rent cores.
set -euo pipefail

RUNS="${RUNS:-5}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

log() { printf '\n=== %s\n' "$*" >&2; }

# --- preconditions -----------------------------------------------------------
arch="$(uname -m)"
if [ "$arch" != "x86_64" ]; then
  echo "This harness records the linux-x86_64 gate baseline; this box is $arch." >&2
  echo "Numbers from another architecture are not comparable and must not be pasted as AC1." >&2
  exit 1
fi
if [ "$(uname -s)" != "Linux" ]; then
  echo "Expected Linux (peak RSS is read from /proc/self/status); got $(uname -s)." >&2
  exit 1
fi

# Timing hygiene. A shared vCPU with noisy neighbours produces numbers that
# cannot be reproduced later, which defeats the point of recording a baseline.
log "machine"
{
  echo "CPU:    $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')"
  echo "cores:  $(nproc)"
  echo "memory: $(awk '/MemTotal/ {printf "%.1f GiB", $2/1048576}' /proc/meminfo)"
  echo "commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
} >&2
if [ -r /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
  gov="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
  echo "governor: $gov" >&2
  [ "$gov" != "performance" ] && echo "  (note: 'performance' gives steadier numbers)" >&2
fi
if systemd-detect-virt --quiet 2>/dev/null; then
  echo "WARNING: virtualized ($(systemd-detect-virt)). Prefer bare metal for a" >&2
  echo "         baseline that will be quoted months from now." >&2
fi

# --- toolchain ---------------------------------------------------------------
log "toolchain"
if ! command -v cargo >/dev/null; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none
  # shellcheck disable=SC1091
  . "$HOME/.cargo/env"
fi
# rust-toolchain.toml pins the version; this materializes it plus rust-src.
rustup show active-toolchain >/dev/null
rustup component add rust-src >/dev/null 2>&1 || true

# Guest ELFs link ZeroOS, whose musl toolchain the guest build shells out to.
if [ ! -d "$HOME/.zeroos/musl" ]; then
  log "ZeroOS musl toolchain"
  curl -fsSL https://github.com/LayerZero-Labs/ZeroOS/releases/download/musl-toolchain-musl-1.2.3-gcc-9.4.0/zeroos-musl-toolchain-musl-1.2.3-gcc-9.4.0-Linux-x86_64.tar.gz \
    -o /tmp/zeroos-musl.tar.gz
  mkdir -p "$HOME/.zeroos"
  tar -xzf /tmp/zeroos-musl.tar.gz -C "$HOME/.zeroos"
fi

# Guests are built through the CLI, so it must match this checkout.
log "jolt CLI (from this checkout)"
cargo install --path . --locked --force --profile ci --target-dir target

# --- measurement -------------------------------------------------------------
log "baseline (runs=$RUNS, this is the long part)"
# TRACER_PARALLEL unset: the baseline is the serial reference by definition.
unset TRACER_PARALLEL JOLT_TRACER_CHUNK_ROWS
cargo run --release -p jolt-eval --bin trace-gen-baseline -- --runs "$RUNS" | tee /tmp/ac1-baseline.md

log "written to /tmp/ac1-baseline.md — paste into the PR description"
