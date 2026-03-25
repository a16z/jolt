#!/usr/bin/env bash
set -euo pipefail

SPECS_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ $# -eq 0 ]; then
  echo "Usage: $0 <feature-name>"
  echo "Example: $0 streaming-prover"
  exit 1
fi

FEATURE_NAME="$1"
DATE_PREFIX="$(date +%Y-%m)"
FILENAME="${DATE_PREFIX}-${FEATURE_NAME}.md"
FILEPATH="${SPECS_DIR}/${FILENAME}"

if [ -f "$FILEPATH" ]; then
  echo "Error: ${FILEPATH} already exists."
  exit 1
fi

AUTHOR=$(gh api user --jq .login 2>/dev/null || git config user.name || echo "")

cp "${SPECS_DIR}/TEMPLATE.md" "$FILEPATH"
sed -i.bak "s/\[Feature Name\]/${FEATURE_NAME}/" "$FILEPATH" && rm -f "${FILEPATH}.bak"
sed -i.bak "s/| Author(s)   |.*|$/| Author(s)   | @${AUTHOR} |/" "$FILEPATH" && rm -f "${FILEPATH}.bak"
sed -i.bak "s/Created     | YYYY-MM-DD/Created     | $(date +%Y-%m-%d)/" "$FILEPATH" && rm -f "${FILEPATH}.bak"

echo "Created ${FILEPATH}"
echo "Next: fill in the spec and open a PR."
