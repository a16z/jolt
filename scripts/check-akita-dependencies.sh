#!/usr/bin/env bash
set -euo pipefail

# Include dev dependencies: Akita's tests must also remain curve-free.
graph=$(cargo tree -p jolt-akita --all-features --locked --edges normal,build,dev --prefix none)
if forbidden=$(rg '^(ark-bn254|ark-ec|ark-ff|jolt-dory|light-poseidon) ' <<< "$graph"); then
  echo "Unexpected curve dependency in jolt-akita:" >&2
  echo "$forbidden" >&2
  exit 1
fi
