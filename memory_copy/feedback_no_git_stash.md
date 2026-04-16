---
name: No git stash
description: Never use git stash in agents or main conversation — user lost work to an unwanted stash
type: feedback
---

NEVER use `git stash` in any agent or the main conversation.

**Why:** An agent stashed the user's working tree without permission, causing confusion and lost context. The user had to manually pop it.

**How to apply:** When spawning agents, explicitly instruct them: "Do NOT run git stash, git checkout, git reset, or any destructive git commands." If an agent needs a clean working tree, use the `isolation: "worktree"` parameter instead.
