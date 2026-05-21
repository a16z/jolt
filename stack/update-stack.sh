#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: stack/update-stack.sh [--apply] [--rebuild] [--commit] [--push] [--cargo-metadata] [--check-coverage] [--from REF] [--base REF] [--only NN]

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
  --only NN     update only one stack item, e.g. 08
  -h, --help    show this help

Examples:
  stack/update-stack.sh
  stack/update-stack.sh --apply --only 08
  stack/update-stack.sh --apply --rebuild --commit --push --cargo-metadata --check-coverage --from origin/refactor/audit-prep
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
  local pathspec

  for pathspec in "$@"; do
    if pathspec_matches_ref "$ref" "$pathspec" || pathspec_matches_index "$pathspec"; then
      existing_pathspecs+=("$pathspec")
    else
      missing_pathspecs+=("$pathspec")
    fi
  done

  if ((${#existing_pathspecs[@]})); then
    git restore --source "$ref" -- "${existing_pathspecs[@]}"
  fi

  for pathspec in "${missing_pathspecs[@]}"; do
    echo "  skipped missing optional path: $pathspec"
  done
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
  esac
}

previous_branch="$base_ref"
matched=0
updated_branches=()
coverage_file=""
if ((check_coverage)); then
  coverage_file="$(mktemp)"
  printf '%s\n' Cargo.lock Cargo.toml > "$coverage_file"
fi

while IFS=$'\t' read -r order branch title pathspecs; do
  [[ -z "${order:-}" || "$order" == \#* ]] && continue

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

  if ((check_coverage)); then
    # shellcheck disable=SC2086
    git diff --name-only "$base_ref...$source_ref" -- $pathspecs >> "$coverage_file"
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

  read -r -a owned_pathspecs <<< "$pathspecs"
  restore_owned_paths "$source_ref" "${owned_pathspecs[@]}"
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

  previous_branch="$branch"
done < "$plan_file"

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
  echo "pushing ${#updated_branches[@]} stack branches"
  for branch in "${updated_branches[@]}"; do
    remote_sha="$(git ls-remote --heads origin "$branch" | awk '{print $1}')"
    if [[ -n "$remote_sha" ]]; then
      git push --force-with-lease="refs/heads/$branch:$remote_sha" origin "$branch:refs/heads/$branch"
    else
      git push origin "$branch:refs/heads/$branch"
    fi
  done
fi
