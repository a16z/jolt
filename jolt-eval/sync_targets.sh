#!/usr/bin/env bash
#
# Synchronizes fuzz targets and Criterion benchmarks with the invariant
# and objective definitions in jolt-eval source code.
#
# Run from the repo root:
#   ./jolt-eval/sync_targets.sh
#
# Idempotent: running twice produces no changes.

set -euo pipefail

EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
FUZZ_DIR="$EVAL_DIR/fuzz"
BENCH_DIR="$EVAL_DIR/benches"

# ── Helpers ──────────────────────────────────────────────────────────

# Convert CamelCase to snake_case, stripping Invariant/Objective suffix
to_snake() {
    echo "$1" \
        | sed 's/Invariant$//' \
        | sed 's/Objective$//' \
        | sed 's/\([A-Z]\)/_\1/g' \
        | sed 's/^_//' \
        | tr '[:upper:]' '[:lower:]'
}

# ── Fuzz targets ─────────────────────────────────────────────────────

echo "=== Syncing fuzz targets ==="

mkdir -p "$FUZZ_DIR/fuzz_targets"

# Find (snake_name, module_path, struct_name) for each fuzzable invariant
fuzz_entries=""
for file in "$EVAL_DIR"/src/invariant/*.rs; do
    [ -f "$file" ] || continue
    basename_rs=$(basename "$file" .rs)
    [ "$basename_rs" = "mod" ] && continue

    # Look for #[invariant(...Fuzz...)] annotations
    { grep -n 'invariant.*Fuzz' "$file" 2>/dev/null || true; } | while IFS=: read -r line _; do
        struct=$(sed -n "$((line+1)),$((line+5))p" "$file" \
            | grep -o 'pub struct [A-Za-z_]*' | head -1 | awk '{print $3}')
        [ -z "$struct" ] && continue
        snake=$(to_snake "$struct")
        echo "$snake invariant::${basename_rs}::${struct}"
    done
done | sort -u > /tmp/jolt_fuzz_entries

# Generate missing fuzz target files
while read -r snake mod_struct; do
    [ -z "$snake" ] && continue
    struct="${mod_struct##*::}"
    target_file="$FUZZ_DIR/fuzz_targets/${snake}.rs"
    if [ ! -f "$target_file" ]; then
        echo "  Creating fuzz target: $snake"
        cat > "$target_file" <<EOF
#![no_main]
use jolt_eval::${mod_struct};
jolt_eval::fuzz_invariant!(${struct}::default());
EOF
    fi
done < /tmp/jolt_fuzz_entries

# Remove stale fuzz target files
for f in "$FUZZ_DIR"/fuzz_targets/*.rs; do
    [ -f "$f" ] || continue
    base=$(basename "$f" .rs)
    if ! grep -q "^$base " /tmp/jolt_fuzz_entries 2>/dev/null; then
        echo "  Removing stale fuzz target: $base"
        rm "$f"
    fi
done

# Regenerate fuzz/Cargo.toml [[bin]] entries
{
    sed '/^\[\[bin\]\]/,$d' "$FUZZ_DIR/Cargo.toml"
    while read -r snake _; do
        [ -z "$snake" ] && continue
        cat <<EOF
[[bin]]
name = "$snake"
path = "fuzz_targets/${snake}.rs"
test = false
doc = false
bench = false

EOF
    done < /tmp/jolt_fuzz_entries
} > "$FUZZ_DIR/Cargo.toml.tmp"
mv "$FUZZ_DIR/Cargo.toml.tmp" "$FUZZ_DIR/Cargo.toml"

# ── Criterion benchmarks ─────────────────────────────────────────────

echo "=== Syncing Criterion benchmarks ==="

mkdir -p "$BENCH_DIR"

# Find (bench_name, module_path, struct_name) for each PerfObjective
bench_entries=""
for file in "$EVAL_DIR"/src/objective/*.rs; do
    [ -f "$file" ] || continue
    basename_rs=$(basename "$file" .rs)
    [ "$basename_rs" = "mod" ] && continue

    { grep -n 'impl PerfObjective for' "$file" 2>/dev/null || true; } | while IFS=: read -r _ rest; do
        struct=$(echo "$rest" | grep -o 'for [A-Za-z_]*' | awk '{print $2}')
        [ -z "$struct" ] && continue
        # Try to find the NAME const
        bench_name=$(grep -A5 "impl $struct" "$file" \
            | grep 'const NAME' | head -1 \
            | grep -o '"[^"]*"' | tr -d '"') || true
        [ -z "$bench_name" ] && bench_name=$(to_snake "$struct" | sed 's/_objective$//')
        echo "$bench_name objective::${basename_rs}::${struct}"
    done
done | sort -u > /tmp/jolt_bench_entries

# Generate missing bench files
while read -r name mod_struct; do
    [ -z "$name" ] && continue
    struct="${mod_struct##*::}"
    bench_file="$BENCH_DIR/${name}.rs"
    if [ ! -f "$bench_file" ]; then
        echo "  Creating benchmark: $name"
        cat > "$bench_file" <<EOF
use jolt_eval::${mod_struct};
jolt_eval::bench_objective!(${struct});
EOF
    fi
done < /tmp/jolt_bench_entries

# Remove stale bench files
for f in "$BENCH_DIR"/*.rs; do
    [ -f "$f" ] || continue
    base=$(basename "$f" .rs)
    if ! grep -q "^$base " /tmp/jolt_bench_entries 2>/dev/null; then
        echo "  Removing stale benchmark: $base"
        rm "$f"
    fi
done

# Update Cargo.toml [[bench]] entries
CARGO_TOML="$EVAL_DIR/Cargo.toml"
tmpfile=$(mktemp)

# Remove existing [[bench]] blocks
awk '
    /^\[\[bench\]\]/ { skip=1; next }
    skip && /^$/ { skip=0; next }
    skip && /^\[/ { skip=0 }
    !skip { print }
' "$CARGO_TOML" > "$tmpfile"

# Insert new [[bench]] entries before the first [[bin]]
{
    sed '/^\[\[bin\]\]/,$d' "$tmpfile"
    while read -r name _; do
        [ -z "$name" ] && continue
        cat <<EOF
[[bench]]
name = "$name"
harness = false

EOF
    done < /tmp/jolt_bench_entries
    sed -n '/^\[\[bin\]\]/,$p' "$tmpfile"
} > "$CARGO_TOML"
rm -f "$tmpfile" /tmp/jolt_fuzz_entries /tmp/jolt_bench_entries

echo "=== Done ==="
