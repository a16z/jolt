#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: stack/update-stack.sh [--apply] [--rebuild] [--commit] [--push] [--cargo-metadata] [--check-coverage] [--from REF] [--base REF] [--start-at NN] [--only NN]

Materializes the draft PR stack described by stack/branches.tsv.

Default mode is a dry run. With --apply, the script creates or updates stack
branches by restoring branch-owned paths from the source ref. It applies
incremental root manifest changes for new workspace crates.

Options:
  --apply       perform git operations; without this, print the planned actions
  --rebuild     reset existing stack branches to their configured base first
  --commit      commit restored changes using the title from stack/branches.tsv
  --push        push updated stack branches to origin with --force-with-lease
  --cargo-metadata
                run cargo metadata after manifest changes to refresh Cargo.lock
  --check-coverage
                fail if a source-ref diff path is not assigned to a stack slice
  --from REF    source ref to slice from (default: refactor/audit-prep)
  --base REF    base ref for the first stack branch (default: origin/main)
  --start-at NN first stack item to sync (default: 08, or STACK_START_AT)
  --only NN     update only one stack item, e.g. 08
  -h, --help    show this help

Examples:
  stack/update-stack.sh
  stack/update-stack.sh --apply --only 08
  stack/update-stack.sh --apply --rebuild --commit --push --cargo-metadata --check-coverage --from origin/refactor/audit-prep --start-at 08
EOF
}

apply=0
rebuild=0
commit_changes=0
push_changes=0
cargo_metadata=0
check_coverage=0
source_ref="refactor/audit-prep"
base_ref="origin/main"
start_at="${STACK_START_AT:-08}"
only=""

while (($#)); do
  case "$1" in
    --apply)
      apply=1
      ;;
    --rebuild)
      rebuild=1
      ;;
    --commit)
      commit_changes=1
      ;;
    --push)
      push_changes=1
      ;;
    --cargo-metadata)
      cargo_metadata=1
      ;;
    --check-final|--check-coverage)
      check_coverage=1
      ;;
    --from)
      source_ref="${2:?--from requires a ref}"
      shift
      ;;
    --base)
      base_ref="${2:?--base requires a ref}"
      shift
      ;;
    --start-at)
      start_at="${2:?--start-at requires an order number}"
      shift
      ;;
    --only)
      only="${2:?--only requires an order number}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

plan_file="stack/branches.tsv"
if [[ ! -f "$plan_file" ]]; then
  echo "missing $plan_file" >&2
  exit 1
fi

git rev-parse --verify "$source_ref^{commit}" >/dev/null
git rev-parse --verify "$base_ref^{commit}" >/dev/null

if ((apply)); then
  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "working tree is dirty; commit or stash before applying stack updates" >&2
    exit 1
  fi
fi

ensure_after() {
  local file="$1"
  local anchor="$2"
  local line="$3"

  if grep -Fxq "$line" "$file"; then
    return
  fi
  if ! grep -Fxq "$anchor" "$file"; then
    echo "anchor not found in $file: $anchor" >&2
    exit 1
  fi

  local tmp
  tmp="$(mktemp)"
  awk -v anchor="$anchor" -v line="$line" '
    { print }
    $0 == anchor { print line }
  ' "$file" > "$tmp"
  mv "$tmp" "$file"
}

pathspec_matches_ref() {
  local ref="$1"
  local pathspec="$2"

  [[ -n "$(git ls-tree -r --name-only "$ref" -- "$pathspec")" ]]
}

pathspec_matches_index() {
  local pathspec="$1"

  [[ -n "$(git ls-files -- "$pathspec")" ]]
}

restore_owned_paths() {
  local ref="$1"
  shift

  local existing_pathspecs=()
  local missing_pathspecs=()
  local existing_pathspec_count=0
  local missing_pathspec_count=0
  local pathspec

  for pathspec in "$@"; do
    if pathspec_matches_ref "$ref" "$pathspec" || pathspec_matches_index "$pathspec"; then
      existing_pathspecs+=("$pathspec")
      existing_pathspec_count=$((existing_pathspec_count + 1))
    else
      missing_pathspecs+=("$pathspec")
      missing_pathspec_count=$((missing_pathspec_count + 1))
    fi
  done

  if ((existing_pathspec_count)); then
    git restore --source "$ref" -- "${existing_pathspecs[@]}"
  fi

  if ((missing_pathspec_count)); then
    for pathspec in "${missing_pathspecs[@]}"; do
      echo "  skipped missing optional path: $pathspec"
    done
  fi
}

overlay_base_ref="09ed244b373f68152b09cc64fb9ed38ca19d6b61"

overlay_target_order() {
  local path="$1"

  case "$path" in
    crates/jolt-hyrax/*|\
    crates/jolt-dory-assist-verifier/*|\
    crates/jolt-claims/src/protocols/mod.rs|\
    crates/jolt-claims/src/protocols/dory_assist/*|\
    crates/jolt-field/Cargo.toml|\
    crates/jolt-field/src/lib.rs|\
    crates/jolt-field/src/arkworks/mod.rs|\
    crates/jolt-field/src/arkworks/bn254_fq.rs|\
    crates/jolt-crypto/Cargo.toml|\
    crates/jolt-crypto/src/lib.rs|\
    crates/jolt-crypto/src/commitment.rs|\
    crates/jolt-crypto/src/ec/mod.rs|\
    crates/jolt-crypto/src/ec/group.rs|\
    crates/jolt-crypto/src/ec/pairing.rs|\
    crates/jolt-crypto/src/ec/pedersen.rs|\
    crates/jolt-crypto/src/ec/bn254/mod.rs|\
    crates/jolt-crypto/src/ec/bn254/gt.rs|\
    crates/jolt-crypto/src/ec/grumpkin|\
    crates/jolt-crypto/src/ec/grumpkin/*|\
    crates/jolt-dory/src/types.rs|\
    crates/jolt-hyperkzg/src/types.rs|\
    crates/jolt-verifier/src/pcs_assist.rs|\
    crates/jolt-verifier/src/assist|\
    crates/jolt-verifier/src/assist/*|\
    crates/jolt-verifier/src/dory_assist.rs|\
    crates/jolt-verifier/tests/dory_assist|\
    crates/jolt-verifier/tests/dory_assist/*)
      printf '%s\n' 14
      ;;
    crates/jolt-r1cs/src/lib.rs|\
    crates/jolt-r1cs/Cargo.toml|\
    crates/jolt-r1cs/src/lowering.rs|\
    crates/jolt-r1cs/src/builder.rs|\
    crates/jolt-r1cs/src/nonnative.rs|\
    crates/jolt-r1cs/tests|\
    crates/jolt-r1cs/tests/*|\
    crates/jolt-wrapper/*|\
    crates/jolt-sumcheck/src/r1cs.rs|\
    crates/jolt-transcript/Cargo.toml|\
    crates/jolt-transcript/src/lib.rs|\
    crates/jolt-transcript/src/r1cs|\
    crates/jolt-transcript/src/r1cs/*|\
    crates/jolt-blindfold/src/r1cs.rs)
      printf '%s\n' 15
      ;;
    crates/jolt-claims/*|\
    crates/jolt-field/src/ring_core.rs|\
    crates/jolt-r1cs/src/constraints/field_constraints.rs|\
    crates/jolt-r1cs/src/constraints/jolt.rs|\
    crates/jolt-r1cs/src/constraints/mod.rs|\
    crates/jolt-riscv/src/flags.rs|\
    crates/jolt-riscv/src/lib.rs|\
    crates/jolt-verifier/*|\
    crates/jolt-blindfold/tests/jolt_claims_pipeline.rs|\
    specs/jolt-verifier-model-crate.md)
      printf '%s\n' 13
      ;;
  esac
}

filter_overlay_paths() {
  local input="$1"
  local output="$2"
  local path

  : > "$output"
  while IFS= read -r path; do
    if [[ -z "$(overlay_target_order "$path")" ]]; then
      printf '%s\n' "$path" >> "$output"
    fi
  done < "$input"
}

restore_overlay_base_paths() {
  local order="$1"
  local owned_paths_file="$2"
  local path
  local target_order
  local count=0

  git rev-parse --verify "$overlay_base_ref^{commit}" >/dev/null

  while IFS= read -r path; do
    target_order="$(overlay_target_order "$path")"
    if [[ -z "$target_order" || "$target_order" == "$order" ]]; then
      continue
    fi

    if pathspec_matches_ref "$overlay_base_ref" "$path"; then
      git restore --source "$overlay_base_ref" -- "$path"
    else
      rm -rf "$path"
      git rm -rf --quiet --ignore-unmatch -- "$path"
    fi
    count=$((count + 1))
  done < "$owned_paths_file"

  if ((count)); then
    echo "  restored overlay base for $count later-owned path(s)"
  fi
}

effective_owned_paths() {
  local order="$1"
  local pathspecs="$2"
  local output="$3"
  local current_paths=()
  read -r -a current_paths <<< "$pathspecs"

  if [[ -n "$pathspecs" ]]; then
    git diff --name-only "$base_ref...$source_ref" -- "${current_paths[@]}" | sort -u > "$output"
  else
    : > "$output"
  fi

  local after_current=0
  local later_order
  local later_branch
  local later_title
  local later_pathspecs
  local later_paths=()
  local later_file
  local remaining_file
  local filtered_later_file

  while IFS=$'\t' read -r later_order later_branch later_title later_pathspecs; do
    [[ -z "${later_order:-}" || "$later_order" == \#* ]] && continue

    if [[ "$later_order" == "$order" ]]; then
      after_current=1
      continue
    fi
    if ((!after_current)); then
      continue
    fi

    read -r -a later_paths <<< "$later_pathspecs"
    if [[ -z "$later_pathspecs" ]]; then
      continue
    fi

    later_file="$(mktemp)"
    remaining_file="$(mktemp)"
    filtered_later_file="$(mktemp)"
    git diff --name-only "$base_ref...$source_ref" -- "${later_paths[@]}" | sort -u > "$later_file"
    filter_overlay_paths "$later_file" "$filtered_later_file"
    comm -23 "$output" "$filtered_later_file" > "$remaining_file"
    mv "$remaining_file" "$output"
    rm -f "$later_file" "$filtered_later_file"
  done < "$plan_file"
}

apply_manifest_rules() {
  local order="$1"

  case "$order" in
    05)
      ensure_after Cargo.toml 'members = [' '  "crates/jolt-claims",'
      ensure_after Cargo.toml 'jolt-core = { path = "./jolt-core", default-features = false }' 'jolt-claims = { path = "./crates/jolt-claims" }'
      ;;
    06)
      ensure_after Cargo.toml 'jolt-poly = { path = "./crates/jolt-poly" }' 'jolt-r1cs = { path = "./crates/jolt-r1cs" }'
      ;;
    08)
      ensure_after Cargo.toml '  "crates/jolt-claims",' '  "crates/jolt-blindfold",'
      ensure_after Cargo.toml 'jolt-core = { path = "./jolt-core", default-features = false }' 'jolt-blindfold = { path = "./crates/jolt-blindfold" }'
      ;;
    09)
      ensure_after Cargo.toml '  "crates/jolt-openings",' '  "crates/jolt-verifier",'
      ensure_after Cargo.toml '  "examples/advice-demo/guest",' '  "examples/advice-consumer/guest",'
      ensure_after Cargo.toml 'jolt-openings = { path = "./crates/jolt-openings" }' 'jolt-verifier = { path = "./crates/jolt-verifier" }'
      ;;
    13)
      ensure_after crates/jolt-r1cs/Cargo.toml 'default = ["parallel"]' 'field-inline = ["jolt-claims/field-inline"]'
      ;;
    14)
      if pathspec_matches_ref "$source_ref" "crates/jolt-hyrax/Cargo.toml"; then
        ensure_after Cargo.toml '  "crates/jolt-dory",' '  "crates/jolt-hyrax",'
        ensure_after Cargo.toml 'jolt-openings = { path = "./crates/jolt-openings" }' 'jolt-hyrax = { path = "./crates/jolt-hyrax" }'
      fi
      if pathspec_matches_ref "$source_ref" "crates/jolt-dory-assist-verifier/Cargo.toml"; then
        local member_anchor='  "crates/jolt-dory",'
        local dependency_anchor='jolt-openings = { path = "./crates/jolt-openings" }'
        if grep -Fxq '  "crates/jolt-hyrax",' Cargo.toml; then
          member_anchor='  "crates/jolt-hyrax",'
        fi
        if grep -Fxq 'jolt-hyrax = { path = "./crates/jolt-hyrax" }' Cargo.toml; then
          dependency_anchor='jolt-hyrax = { path = "./crates/jolt-hyrax" }'
        fi
        ensure_after Cargo.toml "$member_anchor" '  "crates/jolt-dory-assist-verifier",'
        ensure_after Cargo.toml "$dependency_anchor" 'jolt-dory-assist-verifier = { path = "./crates/jolt-dory-assist-verifier" }'
      fi
      ;;
    15)
      if pathspec_matches_ref "$source_ref" "crates/jolt-wrapper/Cargo.toml"; then
        ensure_after Cargo.toml '  "crates/jolt-hyperkzg",' '  "crates/jolt-wrapper",'
        local dependency_anchor='jolt-openings = { path = "./crates/jolt-openings" }'
        if grep -Fxq 'jolt-hyrax = { path = "./crates/jolt-hyrax" }' Cargo.toml; then
          dependency_anchor='jolt-hyrax = { path = "./crates/jolt-hyrax" }'
        fi
        ensure_after Cargo.toml "$dependency_anchor" 'jolt-wrapper = { path = "./crates/jolt-wrapper" }'
      fi
      ;;
  esac
}

previous_branch="$base_ref"
matched=0
start_seen=0
updated_branches=()
updated_branch_count=0
coverage_file=""
if ((check_coverage)); then
  coverage_file="$(mktemp)"
  printf '%s\n' Cargo.lock Cargo.toml > "$coverage_file"
fi

while IFS=$'\t' read -r order branch title pathspecs; do
  [[ -z "${order:-}" || "$order" == \#* ]] && continue

  if [[ -n "$start_at" && "$start_seen" -eq 0 ]]; then
    if [[ "$order" != "$start_at" ]]; then
      if ((check_coverage)); then
        skipped_paths_file="$(mktemp)"
        effective_owned_paths "$order" "$pathspecs" "$skipped_paths_file"
        cat "$skipped_paths_file" >> "$coverage_file"
        rm -f "$skipped_paths_file"
      fi
      continue
    fi
    start_seen=1
  fi

  if [[ -n "$only" && "$order" != "$only" ]]; then
    previous_branch="$branch"
    continue
  fi

  matched=1
  echo
  echo "[$order] $branch"
  echo "  base:  $previous_branch"
  echo "  title: $title"
  echo "  paths: $pathspecs"

  owned_paths_file="$(mktemp)"
  effective_owned_paths "$order" "$pathspecs" "$owned_paths_file"

  if ((check_coverage)); then
    cat "$owned_paths_file" >> "$coverage_file"
  fi

  if ((!apply)); then
    previous_branch="$branch"
    continue
  fi

  if git show-ref --verify --quiet "refs/heads/$branch"; then
    if ((rebuild)); then
      git switch "$branch"
      git reset --hard "$previous_branch"
    else
      git switch "$branch"
      git merge --ff-only "$previous_branch"
    fi
  else
    git switch -c "$branch" "$previous_branch"
  fi

  owned_pathspecs=()
  owned_pathspec_count=0
  while IFS= read -r owned_pathspec; do
    owned_pathspecs+=("$owned_pathspec")
    owned_pathspec_count=$((owned_pathspec_count + 1))
  done < "$owned_paths_file"
  if ((owned_pathspec_count)); then
    restore_owned_paths "$source_ref" "${owned_pathspecs[@]}"
  fi
  restore_overlay_base_paths "$order" "$owned_paths_file"
  apply_manifest_rules "$order"

  if ((cargo_metadata)); then
    cargo metadata -q >/dev/null
  fi

  echo "  restored paths from $source_ref"

  if ((commit_changes)); then
    git add -A
    if git diff --cached --quiet; then
      echo "  no changes to commit"
    else
      git commit -m "$title"
    fi
  else
    echo "  review, then:"
    echo "    git add <paths>"
    echo "    git commit -m \"$title\""
  fi

  updated_branches+=("$branch")
  updated_branch_count=$((updated_branch_count + 1))

  previous_branch="$branch"
done < "$plan_file"

if [[ -n "$start_at" && "$start_seen" -eq 0 ]]; then
  echo "no stack item with start order $start_at" >&2
  exit 1
fi

if [[ -n "$only" && "$matched" -eq 0 ]]; then
  echo "no stack item with order $only" >&2
  exit 1
fi

if ((!apply)); then
  echo
  echo "dry run only; pass --apply to create/update branches"
  exit 0
fi

if ((check_coverage)) && [[ -z "$only" ]]; then
  echo
  echo "checking stack path coverage against $base_ref...$source_ref"
  all_paths="$(mktemp)"
  covered_paths="$(mktemp)"
  uncovered_paths="$(mktemp)"
  git diff --name-only "$base_ref...$source_ref" | sort -u > "$all_paths"
  sort -u "$coverage_file" > "$covered_paths"
  comm -23 "$all_paths" "$covered_paths" > "$uncovered_paths"
  if [[ -s "$uncovered_paths" ]]; then
    echo "source diff has paths not assigned to any stack slice:" >&2
    cat "$uncovered_paths" >&2
    exit 1
  fi
fi

if ((push_changes)); then
  echo
  echo "pushing $updated_branch_count stack branches"
  if ((updated_branch_count)); then
    for branch in "${updated_branches[@]}"; do
      remote_sha="$(git ls-remote --heads origin "$branch" | awk '{print $1}')"
      if [[ -n "$remote_sha" ]]; then
        git push --force-with-lease="refs/heads/$branch:$remote_sha" origin "$branch:refs/heads/$branch"
      else
        git push origin "$branch:refs/heads/$branch"
      fi
    done
  fi
fi
