#!/usr/bin/env bash
# sync-stack.sh — Create or sync the Graphite PR stack for jolt-v2 modular crates.
#
# Usage:
#   ./scripts/sync-stack.sh create   # One-time: build stack from main
#   ./scripts/sync-stack.sh sync     # Update stack branches from refactor/crates
#   ./scripts/sync-stack.sh submit   # Push stack and open/update PRs
#
# Prerequisites:
#   - Graphite CLI (`gt`) installed and authenticated
#   - Working tree clean (stash or commit first)
#   - For 'create': no existing jolt-v2/* branches

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

WIP_BRANCH="refactor/crates"
TRUNK="main"

# ── Stack definition (order matters) ─────────────────────────────────────────
# Format: branch_name|crate_dir (empty for scaffolding)|commit message
STACK=(
  "jolt-v2/scaffolding||chore: workspace scaffolding for modular crates"
  "jolt-v2/jolt-field|crates/jolt-field|feat: add jolt-field crate"
  "jolt-v2/jolt-profiling|crates/jolt-profiling|feat: add jolt-profiling crate"
  "jolt-v2/jolt-transcript|crates/jolt-transcript|feat: add jolt-transcript crate"
  "jolt-v2/jolt-instructions|crates/jolt-instructions|feat: add jolt-instructions crate"
  "jolt-v2/jolt-poly|crates/jolt-poly|feat: add jolt-poly crate"
  "jolt-v2/jolt-crypto|crates/jolt-crypto|feat: add jolt-crypto crate"
  "jolt-v2/jolt-host|crates/jolt-host|feat: add jolt-host crate"
)

# ── Helpers ──────────────────────────────────────────────────────────────────

parse_entry() {
  IFS='|' read -r BRANCH CRATE_DIR COMMIT_MSG <<< "$1"
}

ensure_clean() {
  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: Working tree is dirty. Commit or stash first." >&2
    exit 1
  fi
}

add_workspace_member() {
  local member="$1"
  if grep -qF "\"$member\"" Cargo.toml; then
    return 0
  fi
  # Insert new member right after 'members = ['
  sed -i '' "/^members = \[/a\\
\\  \"$member\"," Cargo.toml
}

# ── Scaffolding: lints, CI, workspace deps ───────────────────────────────────

apply_scaffolding() {
  # 1. Workspace lints (append before [workspace.dependencies] if not present)
  if ! grep -q '\[workspace\.lints\.clippy\]' Cargo.toml; then
    # Find the line number of [workspace.dependencies] and insert lints before it
    local deps_line
    deps_line=$(grep -n '^\[workspace\.dependencies\]' Cargo.toml | head -1 | cut -d: -f1)

    local lints_block
    lints_block=$(cat <<'EOF'

[workspace.lints.clippy]
pedantic = { level = "warn", priority = -1 }

# Pedantic overrides — suppressed because they're too noisy for math-heavy ZK code
missing_errors_doc = "allow"
missing_panics_doc = "allow"
must_use_candidate = "allow"
doc_markdown = "allow"
similar_names = "allow"
too_many_lines = "allow"
module_name_repetitions = "allow"
struct_excessive_bools = "allow"
fn_params_excessive_bools = "allow"
items_after_statements = "allow"
uninlined_format_args = "allow"
return_self_not_must_use = "allow"
default_trait_access = "allow"
match_same_arms = "allow"
manual_let_else = "allow"
used_underscore_binding = "allow"
no_effect_underscore_binding = "allow"
needless_pass_by_value = "allow"
trivially_copy_pass_by_ref = "allow"
redundant_closure_for_method_calls = "allow"
unnecessary_wraps = "allow"
if_not_else = "allow"

# Numeric/math code — ZK cryptography uses these patterns pervasively
float_cmp = "allow"
many_single_char_names = "allow"
wildcard_imports = "allow"
inline_always = "allow"
checked_conversions = "allow"

# Cast lints — field arithmetic requires intentional casting
cast_possible_truncation = "allow"
cast_sign_loss = "allow"
cast_precision_loss = "allow"
cast_possible_wrap = "allow"
cast_lossless = "allow"

# Code quality — hard denies to catch AI-generated slop
dbg_macro = "deny"
todo = "deny"
unimplemented = "deny"
print_stdout = "deny"
print_stderr = "deny"
undocumented_unsafe_blocks = "deny"

[workspace.lints.rust]
unused_results = "warn"

EOF
)
    # Insert lints block before [workspace.dependencies]
    local tmp
    tmp=$(mktemp)
    head -n "$((deps_line - 1))" Cargo.toml > "$tmp"
    echo "$lints_block" >> "$tmp"
    tail -n "+$deps_line" Cargo.toml >> "$tmp"
    mv "$tmp" Cargo.toml
  fi

  # 2. Add workspace deps not present on main: light-poseidon, digest
  if ! grep -q 'light-poseidon' Cargo.toml; then
    sed -i '' '/^blake3 = /a\
light-poseidon = "0.4"\
digest = "0.10"' Cargo.toml
  fi

  # 3. Add tracing "attributes" feature (idempotent — only matches the exact old string)
  sed -i '' 's/tracing = { version = "0.1.37", default-features = false }/tracing = { version = "0.1.37", default-features = false, features = ["attributes"] }/' Cargo.toml

  # 4. Copy CI files from WIP branch
  mkdir -p .github/workflows
  git checkout "$WIP_BRANCH" -- .github/pull_request_template.md 2>/dev/null || true
  git checkout "$WIP_BRANCH" -- .github/workflows/bench-crates.yml 2>/dev/null || true
}

# ── Per-crate application ────────────────────────────────────────────────────

apply_crate() {
  local crate_dir="$1"
  local branch="$2"

  # Copy crate directory from WIP branch
  git checkout "$WIP_BRANCH" -- "$crate_dir"

  # Add to workspace members
  add_workspace_member "$crate_dir"

  # Per-crate fixups (workspace dep conflicts with main)
  case "$branch" in
    jolt-v2/jolt-host)
      # main has bincode 1.x in workspace; jolt-host needs bincode 2
      sed -i '' 's/bincode\.workspace = true/bincode = { version = "2", features = ["serde"] }/' \
        "$crate_dir/Cargo.toml"
      ;;
  esac
}

# ── Commands ─────────────────────────────────────────────────────────────────

stage_scaffolding() {
  git add Cargo.toml
  git add .github/pull_request_template.md .github/workflows/bench-crates.yml 2>/dev/null || true
}

stage_crate() {
  local crate_dir="$1"
  git add "$crate_dir"
  # Remove any accidentally staged build artifacts
  git reset HEAD -- "$crate_dir/fuzz/target" 2>/dev/null || true
  git reset HEAD -- "$crate_dir/target" 2>/dev/null || true
  git add Cargo.toml
}

cmd_create() {
  ensure_clean
  echo "Creating stack from $TRUNK..."

  # Use up-to-date main
  git fetch origin "$TRUNK"
  git checkout "$TRUNK"
  git merge --ff-only "origin/$TRUNK" 2>/dev/null || true

  for entry in "${STACK[@]}"; do
    parse_entry "$entry"
    echo "  → $BRANCH"

    gt create "$BRANCH" --no-interactive 2>/dev/null || gt create "$BRANCH" 2>/dev/null

    if [ -z "$CRATE_DIR" ]; then
      apply_scaffolding
      stage_scaffolding
    else
      apply_crate "$CRATE_DIR" "$BRANCH"
      stage_crate "$CRATE_DIR"
    fi

    git commit -m "$COMMIT_MSG"
  done

  echo ""
  echo "Stack created. Run 'gt log short' to inspect."
  echo "Run './scripts/sync-stack.sh submit' to push and open PRs."
}

cmd_sync() {
  echo "Syncing stack from $WIP_BRANCH..."

  for entry in "${STACK[@]}"; do
    parse_entry "$entry"
    echo "  → $BRANCH"

    gt checkout "$BRANCH" 2>/dev/null || {
      echo "  SKIP: branch $BRANCH not found (run 'create' first)"
      continue
    }

    if [ -z "$CRATE_DIR" ]; then
      apply_scaffolding
      stage_scaffolding
    else
      apply_crate "$CRATE_DIR" "$BRANCH"
      stage_crate "$CRATE_DIR"
    fi

    # Only amend if there are staged changes
    if ! git diff --cached --quiet; then
      git commit --amend --no-edit
    else
      echo "  (no changes)"
    fi
  done

  echo ""
  echo "Restacking..."
  gt restack 2>/dev/null || echo "  (restack: nothing to do)"
  echo "Done. Run 'gt log short' to inspect."
}

cmd_submit() {
  echo "Submitting stack..."
  gt submit --no-interactive 2>/dev/null || gt submit
  echo "Done."
}

# ── Main ─────────────────────────────────────────────────────────────────────

case "${1:-help}" in
  create)  cmd_create ;;
  sync)    cmd_sync ;;
  submit)  cmd_submit ;;
  *)
    echo "Usage: $0 {create|sync|submit}"
    echo ""
    echo "  create  — Build the Graphite stack from $TRUNK (one-time)"
    echo "  sync    — Update stack branches from $WIP_BRANCH"
    echo "  submit  — Push branches and open/update PRs"
    exit 1
    ;;
esac
