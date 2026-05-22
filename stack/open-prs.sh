#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: stack/open-prs.sh [--apply] [--base BRANCH] [--source REF]

Creates or updates draft PRs for branches in stack/branches.tsv. Default mode is
a dry run.

Options:
  --apply       create/edit PRs; without this, print planned gh commands
  --base NAME   base branch for the first PR (default: main)
  --source REF  source branch named in PR bodies (default: refactor/audit-prep)
  -h, --help    show this help
EOF
}

apply=0
first_base="main"
source_ref="refactor/audit-prep"

while (($#)); do
  case "$1" in
    --apply)
      apply=1
      ;;
    --base)
      first_base="${2:?--base requires a branch name}"
      shift
      ;;
    --source)
      source_ref="${2:?--source requires a ref}"
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
base_branch="$first_base"

while IFS=$'\t' read -r order branch title pathspecs; do
  [[ -z "${order:-}" || "$order" == \#* ]] && continue

  body_file="$(mktemp)"
  cat > "$body_file" <<EOF
Generated stack PR from \`$source_ref\`.

Stack position: \`$order\`
Base branch: \`$base_branch\`
Generated branch: \`$branch\`

Owned paths:
\`\`\`
$pathspecs
\`\`\`

This PR is expected to be updated automatically when \`$source_ref\` is pushed.
EOF

  existing=""
  if ((apply)); then
    existing="$(gh pr list --head "$branch" --state open --json number --jq '.[0].number // empty')"
  fi

  if [[ -n "$existing" ]]; then
    echo "updating PR #$existing for $branch"
    if ((apply)); then
      gh pr edit "$existing" --title "$title" --base "$base_branch" --body-file "$body_file"
      rm -f "$body_file"
    fi
  else
    echo "creating draft PR for $branch -> $base_branch"
    if ((apply)); then
      gh pr create --draft --base "$base_branch" --head "$branch" --title "$title" --body-file "$body_file"
      rm -f "$body_file"
    else
      echo "  gh pr create --draft --base '$base_branch' --head '$branch' --title '$title' --body-file '$body_file'"
    fi
  fi

  base_branch="$branch"
done < "$plan_file"

if ((!apply)); then
  echo
  echo "dry run only; pass --apply to create/update PRs"
fi
