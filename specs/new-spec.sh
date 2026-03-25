#!/usr/bin/env bash
set -euo pipefail

SPECS_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ $# -eq 0 ]; then
  echo "Usage: $0 <feature-name>"
  echo "Example: $0 streaming-prover"
  exit 1
fi

FEATURE_NAME="$1"

if ! [[ "$FEATURE_NAME" =~ ^[a-z0-9]([a-z0-9-]*[a-z0-9])?$ ]]; then
  echo "Error: feature name must be lowercase alphanumeric with dashes (e.g., streaming-prover)"
  exit 1
fi

DATE_PREFIX="$(date +%Y-%m)"
FILENAME="${DATE_PREFIX}-${FEATURE_NAME}.md"
FILEPATH="${SPECS_DIR}/${FILENAME}"

if [ -f "$FILEPATH" ]; then
  echo "Error: ${FILEPATH} already exists."
  exit 1
fi

AUTHOR=$(gh api user --jq .login 2>/dev/null || git config user.name || echo "")

# Escape values for safe use in sed replacement strings
escape_sed() { printf '%s' "$1" | sed 's/[&/\]/\\&/g'; }
FEATURE_NAME_ESCAPED=$(escape_sed "$FEATURE_NAME")
AUTHOR_ESCAPED=$(escape_sed "$AUTHOR")

if [ -n "$AUTHOR" ]; then
  AUTHOR_CELL="@${AUTHOR_ESCAPED}"
else
  AUTHOR_CELL=""
fi

cp "${SPECS_DIR}/TEMPLATE.md" "$FILEPATH"
sed -i.bak "s/\[Feature Name\]/${FEATURE_NAME_ESCAPED}/" "$FILEPATH" && rm -f "${FILEPATH}.bak"
sed -i.bak "s/| Author(s)   |.*|$/| Author(s)   | ${AUTHOR_CELL} |/" "$FILEPATH" && rm -f "${FILEPATH}.bak"
sed -i.bak "s/Created     | YYYY-MM-DD/Created     | $(date +%Y-%m-%d)/" "$FILEPATH" && rm -f "${FILEPATH}.bak"

echo "Created ${FILEPATH}"
echo "Next: fill in the spec and open a PR."
