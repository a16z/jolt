#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: stack/update-stack.sh [--from REF] [--base REF] [--check-coverage]

Prints the committed-bytecode stack plan. With --check-coverage, verifies that
every path changed by the source ref is assigned to at least one stack slice.

Options:
  --from REF        source ref for coverage checks (default: amir/bytecode-commitment-merged)
  --base REF        base ref for coverage checks (default: origin/main)
  --check-coverage  fail if a changed source path is not assigned to any slice
  -h, --help        show this help
EOF
}

source_ref="amir/bytecode-commitment-merged"
base_ref="origin/main"
check_coverage=0

while (($#)); do
  case "$1" in
    --from)
      source_ref="${2:?--from requires a ref}"
      shift
      ;;
    --base)
      base_ref="${2:?--base requires a ref}"
      shift
      ;;
    --check-coverage)
      check_coverage=1
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

previous_branch="main"
coverage_file=""
if ((check_coverage)); then
  coverage_file="$(mktemp)"
fi

while IFS=$'\t' read -r order branch title pathspecs; do
  [[ -z "${order:-}" || "$order" == \#* ]] && continue

  echo
  echo "[$order] $branch"
  echo "  base:  $previous_branch"
  echo "  title: $title"
  echo "  paths: $pathspecs"

  if ((check_coverage)); then
    # shellcheck disable=SC2086
    git diff --name-only "$base_ref...$source_ref" -- $pathspecs >> "$coverage_file"
  fi

  previous_branch="$branch"
done < "$plan_file"

if ((check_coverage)); then
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
  echo "all source paths are assigned to at least one stack slice"
fi
