#!/bin/bash
set -euo pipefail

BASE="https://raw.githubusercontent.com/a16z/jolt/main/agent-skills/jolt"
installed=0

for dir in "$HOME/.claude" "$HOME/.codex"; do
  if [ -d "$dir" ]; then
    dest="$dir/skills/jolt"
    mkdir -p "$dest"
    curl -sfL "$BASE/SKILL.md" -o "$dest/SKILL.md"
    echo "Installed Jolt skill to $dest"
    installed=1
  fi
done

if [ "$installed" -eq 0 ]; then
  echo "No agent config found (~/.claude or ~/.codex). Install Claude Code or Codex first."
  exit 1
fi
