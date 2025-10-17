#!/bin/bash
# Optimize machine for Jolt benchmarks
#
# Fast system-level tuning for zkVM benchmarks on Ubuntu 24 + AMD Threadripper.
# It sets performance governor, enables THP, reduces I/O jitter, and tames background services.

set -euo pipefail

need_sudo() {
  if [[ "$EUID" -ne 0 ]]; then
    echo "Re-running with sudo..."
    exec sudo --preserve-env=PATH "$0" "$@"
  fi
}
need_sudo "$@"

echo "==> Setting CPU to performance mode"
if command -v powerprofilesctl >/dev/null 2>&1; then
  powerprofilesctl set performance || true
fi

# Try amd-pstate first; fall back to per-core governor
if [[ -d /sys/devices/system/cpu/amd_pstate ]]; then
  echo 1 > /sys/devices/system/cpu/amd_pstate/status 2>/dev/null || true
fi
if [[ -d /sys/devices/system/cpu/cpu0/cpufreq ]]; then
  for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    [[ -f "$g" ]] && echo performance > "$g" || true
  done
fi

echo "==> Enabling Transparent Huge Pages (THP) and defrag"
for f in /sys/kernel/mm/transparent_hugepage/enabled /sys/kernel/mm/transparent_hugepage/defrag; do
  [[ -f "$f" ]] && echo always > "$f" || true
done


echo "==> Reducing I/O writeback jitter"
sysctl -q vm.dirty_ratio=10
sysctl -q vm.dirty_background_ratio=5

echo "==> Turning swap off (reversible with 'swapon -a')"
swapoff -a || true

echo "==> Configuring memory overcommit for large allocations"
sysctl -q vm.overcommit_memory=1

echo "==> Raising vm.max_map_count for mapping-heavy workloads"
sysctl -q vm.max_map_count=1048576

echo "==> Stopping irqbalance to reduce cross-core interrupts (optional)"
if systemctl list-unit-files | grep -q '^irqbalance\.service'; then
  systemctl stop irqbalance || true
fi

echo "==> Increasing file descriptor & memlock limits for current shell"
ulimit -n 1048576 || echo "Failed to increase ulimit -n to 1048576"
ulimit -l unlimited 2>/dev/null || echo "Failed to increase ulimit -l to unlimited"
ulimit -s unlimited 2>/dev/null || echo "Failed to increase ulimit -s to unlimited"

echo "==> Dropping page cache (optional, for clean benchmark runs)"
# Comment out if you prefer warm cache runs.
sync
echo 3 > /proc/sys/vm/drop_caches
echo
echo "==> System Info:"
echo "Max map count: $(cat /proc/sys/vm/max_map_count 2>/dev/null || echo 'N/A')"
echo "Overcommit memory: $(cat /proc/sys/vm/overcommit_memory 2>/dev/null || echo 'N/A')"

echo
echo "==> Please run to set the RUST_MIN_STACK for larger trace sizes:"
echo "  export RUST_MIN_STACK=134217728"

echo "==> Ready-to-run Jolt profiling command:"
echo
echo 'RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C codegen-units=1 -C embed-bitcode=yes" RUST_LOG=info chrt -f 80 cargo run --release -p jolt-core profile --name sha2-chain --format chrome'

echo
echo "All set. Run the command above to benchmark."
