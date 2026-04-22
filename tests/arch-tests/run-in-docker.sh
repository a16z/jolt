#!/usr/bin/env bash
#
# Run the ACT4 architectural-test suite inside the same ubuntu:24.04
# container image CI uses. Handy on macOS (where scripts/bootstrap's
# apt-based install won't work) and for reproducing CI failures locally
# without touching the host.
#
# Usage:
#   tests/arch-tests/run-in-docker.sh                      # default: bootstrap + arch-tests-64imac + arch-tests-smoke
#   tests/arch-tests/run-in-docker.sh shell                # drop into an interactive shell with everything ready
#   tests/arch-tests/run-in-docker.sh -- <make-target...>  # run a custom target (e.g. `arch-tests-generate`)
#
# Named volumes persist between runs so Cargo builds, the xpack toolchain,
# and Sail don't get re-downloaded each time:
#   jolt-arch-tests-target → /jolt/target       (Cargo cache + ACT4 workdir)
#   jolt-arch-tests-opt    → /opt/riscv         (toolchain + sail_riscv_sim)
#   jolt-arch-tests-home   → /root              (mise + cargo caches + ssh state)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="ubuntu:24.04"
TARGET_VOL="jolt-arch-tests-target"
OPT_VOL="jolt-arch-tests-opt"
HOME_VOL="jolt-arch-tests-home"

mode="run"
if [[ $# -ge 1 && "$1" == "shell" ]]; then
    mode="shell"
    shift
fi

if [[ $# -ge 1 && "$1" == "--" ]]; then
    shift
fi

# Default Make targets to execute. Caller can override via positional args.
if [[ $# -eq 0 ]]; then
    make_cmds=("make bootstrap" "make arch-tests-64imac" "make arch-tests-smoke")
else
    make_cmds=("make $*")
fi

# The bootstrap snippet runs before the user's make command(s). `set -e`
# propagates to the subsequent commands in the same `bash -lc` invocation,
# so we don't need an explicit `&&` between them — which also avoids a
# multi-line-string gotcha where `&&` ends up on its own line.
bootstrap_script='set -e
export DEBIAN_FRONTEND=noninteractive
if ! command -v make >/dev/null 2>&1; then
    apt-get update -qq
    apt-get install -y --no-install-recommends ca-certificates curl git make
fi
if ! command -v cargo >/dev/null 2>&1; then
    curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
fi
. "$HOME/.cargo/env"
'

if [[ "$mode" == "shell" ]]; then
    exec docker run --rm -it \
        -v "$REPO_ROOT":/jolt \
        -v "$TARGET_VOL":/jolt/target \
        -v "$OPT_VOL":/opt/riscv \
        -v "$HOME_VOL":/root \
        -w /jolt \
        "$IMAGE" bash -lc "${bootstrap_script}exec bash"
fi

# Join with newlines; `set -e` in the prelude aborts on any failure.
joined_cmds="$(printf '%s\n' "${make_cmds[@]}")"

exec docker run --rm -it \
    -v "$REPO_ROOT":/jolt \
    -v "$TARGET_VOL":/jolt/target \
    -v "$OPT_VOL":/opt/riscv \
    -v "$HOME_VOL":/root \
    -w /jolt \
    "$IMAGE" bash -lc "${bootstrap_script}${joined_cmds}"
