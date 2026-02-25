#!/bin/bash
set -euo pipefail

TOOL="${1:-claude}"
REPO="a16z/jolt"
BRANCH="main"
BASE="https://raw.githubusercontent.com/$REPO/$BRANCH/agent-skills/jolt"

case "$TOOL" in
  claude) DEST="${2:-$HOME/.claude/skills/jolt}" ;;
  codex)  DEST="${2:-$HOME/.agents/skills/jolt}" ;;
  *)      echo "Usage: install.sh [claude|codex] [dest-dir]"; exit 1 ;;
esac

mkdir -p "$DEST"
curl -sfL "$BASE/SKILL.md" -o "$DEST/SKILL.md"
echo "Installed Jolt skill to $DEST"
