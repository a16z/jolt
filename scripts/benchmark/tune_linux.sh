#!/usr/bin/env bash
# Fast system-level tuning for zkVM benchmarks on Ubuntu 24.
# Run this whole script (ideally with sudo). It sets performance governor,
# enables THP, reduces I/O jitter, tames background services, and shows how to
# pin your run to a NUMA node and specific cores.
#
# Usage:
#   ./tune_linux.sh           # Full system tuning
#   ./tune_linux.sh --cache-only  # Only drop page cache

set -euo pipefail

# Parse command line arguments
CACHE_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cache-only)
      CACHE_ONLY=1
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--cache-only]"
      exit 1
      ;;
  esac
done

need_sudo() {
  if [[ "$EUID" -ne 0 ]]; then
    echo "Re-running with sudo..."
    exec sudo --preserve-env=PATH "$0" "$@"
  fi
}

# Handle cache-only mode
if (( CACHE_ONLY )); then
  need_sudo "$@"
  echo "==> Dropping page cache (cache-only mode)"
  sync
  echo 3 > /proc/sys/vm/drop_caches
  echo "Page cache dropped."
  exit 0
fi

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
sysctl -q vm.max_map_count=1048576

echo "==> Turning swap off (reversible with 'swapon -a')"
swapoff -a || true

echo "==> Configuring memory overcommit for large allocations"
if [[ -f /proc/sys/vm/overcommit_memory ]]; then
  echo 1 > /proc/sys/vm/overcommit_memory  # Allow overcommit
fi
if [[ -f /proc/sys/vm/overcommit_ratio ]]; then
  echo 80 > /proc/sys/vm/overcommit_ratio  # Allow overcommit up to 80% of RAM
fi

echo "==> Relying on Transparent Huge Pages (THP) for dynamic huge page management"
# THP will automatically allocate huge pages as needed, no manual pre-allocation required
echo "THP is enabled and will manage huge pages automatically based on demand"

echo "==> Optimizing memory watermarks for 2TB system"
if [[ -f /proc/sys/vm/min_free_kbytes ]]; then
  echo 262144 > /proc/sys/vm/min_free_kbytes  # 256MB minimum free memory
fi
if [[ -f /proc/sys/vm/lowmem_reserve_ratio ]]; then
  echo 32 > /proc/sys/vm/lowmem_reserve_ratio  # Optimize reserve ratios
fi

echo "==> Configuring aggressive settings for >200GB allocations"
# More aggressive overcommit for very large allocations
if [[ -f /proc/sys/vm/overcommit_ratio ]]; then
  echo 95 > /proc/sys/vm/overcommit_ratio  # Allow overcommit up to 95% of RAM
fi

# Optimize memory compaction for large contiguous allocations
if [[ -f /proc/sys/vm/compaction_proactiveness ]]; then
  echo 20 > /proc/sys/vm/compaction_proactiveness  # Proactive compaction
fi
if [[ -f /proc/sys/vm/compact_memory ]]; then
  echo 1 > /proc/sys/vm/compact_memory  # Trigger immediate compaction
fi

# Delay OOM killer to allow more memory reclaim attempts
if [[ -f /proc/sys/vm/oom_kill_allocating_task ]]; then
  echo 1 > /proc/sys/vm/oom_kill_allocating_task  # Kill allocating task first
fi

# Set more aggressive memory reclaim settings
echo 0 > /proc/sys/vm/swappiness  # Strongly avoid swap
echo 1 > /proc/sys/vm/zone_reclaim_mode  # Reclaim from local zone first
echo 50 > /proc/sys/vm/vfs_cache_pressure  # Less aggressive cache pressure

# Increase memory map limits for very large allocations and high thread counts
echo 4194304 > /proc/sys/vm/max_map_count  # Allow 4M memory maps for 384+ threads (default is usually 65530)

echo "==> Stopping irqbalance to reduce cross-core interrupts (optional)"
if systemctl list-unit-files | grep -q '^irqbalance\.service'; then
  systemctl stop irqbalance || true
fi

echo "==> Increasing file descriptor & memlock limits for current shell"
ulimit -n 1048576 || true
ulimit -l unlimited 2>/dev/null || true
# Set stack size to unlimited (critical for thread stack guard pages)
ulimit -s unlimited 2>/dev/null || true
# Set virtual memory limit to unlimited for large allocations
ulimit -v unlimited 2>/dev/null || true
# Set address space limit to unlimited
ulimit -m unlimited 2>/dev/null || true
# Set data segment size to unlimited
ulimit -d unlimited 2>/dev/null || true

echo "==> System tuning complete (Rayon/Rust env vars handled by benchmark scripts)"

echo "==> Dropping page cache (optional, for clean benchmark runs)"
# Comment out if you prefer warm cache runs.
sync
echo 3 > /proc/sys/vm/drop_caches

if ! command -v lscpu >/dev/null 2>&1; then
  apt-get update && apt-get install -y util-linux
fi
if ! command -v bc >/dev/null 2>&1; then
  apt-get update && apt-get install -y bc
fi

echo
echo "==> System tuning diagnostics:"
echo "Available memory: $(free -h | awk '/^Mem:/ {print $2}')"
echo "Huge pages (2MB): $(cat /proc/sys/vm/nr_hugepages 2>/dev/null || echo 'N/A')"
echo "Huge pages (1GB): $(cat /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages 2>/dev/null || echo 'N/A')"
echo "Overcommit mode: $(cat /proc/sys/vm/overcommit_memory 2>/dev/null || echo 'N/A')"
echo "Overcommit ratio: $(cat /proc/sys/vm/overcommit_ratio 2>/dev/null || echo 'N/A')%"
echo "Max map count: $(cat /proc/sys/vm/max_map_count 2>/dev/null || echo 'N/A')"
echo "Stack size limit: $(ulimit -s 2>/dev/null || echo 'N/A')"
echo
echo "==> Recommended environment variables for benchmarks:"
echo "If running benchmarks directly (not via jolt_runner.sh), set these:"
echo "  export RAYON_NUM_THREADS=$(nproc)                    # Use all CPU cores"
echo "  export RAYON_THREAD_STACK_SIZE=67108864              # 64MB per thread"
echo "  export RUST_MIN_STACK=134217728                      # 128MB stack size"
echo "  export RUST_STACK_GUARD_SIZE=0                       # Disable guard pages"
echo
echo "Monitor memory during run with:"
echo "  watch -n 1 'free -h && echo && cat /proc/meminfo | grep -E \"(MemFree|MemAvailable|Hugepages|Committed)\"'"
echo
echo "If you're still hitting memory allocation issues, try:"
echo "1. Increase huge pages: echo 20480 > /proc/sys/vm/nr_hugepages  # 40GB of 2MB pages"
echo "2. Reduce Rayon threads: export RAYON_NUM_THREADS=\$((\$(nproc)/2))"
echo "3. Disable stack guard pages: export RUST_STACK_GUARD_SIZE=0"
echo "4. Check /var/log/kern.log for OOM killer messages"
echo "5. Monitor with: dmesg -w | grep -i 'killed process'"
echo "6. For 'alternative stack guard page' errors, ensure ulimit -s is unlimited"

echo
echo "All set. Run the command above to benchmark."