---
name: Never push to remote or force push
description: NEVER git push, git push --force, git reset --hard, or any destructive git operation
type: feedback
originSessionId: f0719e20-9c07-4478-bf0f-7ad6f4e1a5cf
---
NEVER push to remote, force push, or reset hard.

**Why:** The user explicitly requested this. Commits are local only until the user manually pushes. Destructive operations could lose work on shared branches.

**How to apply:** Always commit locally. Never run `git push`, `git push --force`, `git reset --hard`, `git checkout -- .`, or `git clean -f`. If you need to undo, create a revert commit instead.
