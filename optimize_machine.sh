# Fast system-level tuning for zkVM benchmarks on Ubuntu 24 + AMD Threadripper.
# Run this whole script (ideally with sudo). It sets performance governor,
# enables THP, reduces I/O jitter, tames background services, and shows how to
# pin your run to a NUMA node and specific cores.

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

echo "==> Disabling automatic NUMA balancing (we'll pin manually)"
if [[ -f /proc/sys/kernel/numa_balancing ]]; then
  echo 0 > /proc/sys/kernel/numa_balancing
fi

echo "==> Reducing I/O writeback jitter"
sysctl -q vm.dirty_ratio=10
sysctl -q vm.dirty_background_ratio=5

echo "==> Turning swap off (reversible with 'swapon -a')"
swapoff -a || true

echo "==> Stopping irqbalance to reduce cross-core interrupts (optional)"
if systemctl list-unit-files | grep -q '^irqbalance\.service'; then
  systemctl stop irqbalance || true
fi

echo "==> Increasing file descriptor & memlock limits for current shell"
ulimit -n 1048576 || true
ulimit -l unlimited 2>/dev/null || true

echo "==> Dropping page cache (optional, for clean benchmark runs)"
# Comment out if you prefer warm cache runs.
sync
echo 3 > /proc/sys/vm/drop_caches

echo "==> Detecting NUMA topology and preparing core list on node 0"
if ! command -v lscpu >/dev/null 2>&1; then
  apt-get update && apt-get install -y util-linux
fi

# Build a comma-separated list of CPU IDs that belong to NUMA node 0.
CORES_NODE0=$(lscpu -e=CPU,NODE | awk '$2 == 0 {print $1}' | paste -sd, -)
if [[ -z "${CORES_NODE0}" ]]; then
  echo "Could not determine cores for NUMA node 0; using all cores."
  CORES_NODE0=$(lscpu -e=CPU | awk 'NR>1{print $1}' | paste -sd, -)
fi
echo "Cores on node 0: ${CORES_NODE0}"

echo "==> Ready-to-run Jolt profiling command:"
echo
echo "numactl --cpunodebind=0 --membind=0 taskset -c ${CORES_NODE0} chrt -f 80 RUST_LOG=info cargo run --release -p jolt-core profile --name sha2-chain --format chrome"

echo
echo "All set. Run the command above to benchmark."
