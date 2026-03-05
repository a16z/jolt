---
description: Analyze code changes since a base commit and update the Jolt docs (book/) accordingly
argument-hint: <base-commit-hash>
allowed-tools: Bash, Read, Edit, Write, Glob, Grep, Task
---

You are updating the Jolt documentation in `book/` based on code changes since a base commit.

The base commit is: $ARGUMENTS

If no base commit was provided, ask the user for one before proceeding.

## Step 1: Gather the diff

Run:
```
git diff $ARGUMENTS..HEAD -- '*.rs'
```

Also get a summary of changed files:
```
git diff --stat $ARGUMENTS..HEAD -- '*.rs'
```

Read the diff carefully. Focus on understanding:
- What functionality changed or was added
- Whether public APIs changed (new types, renamed functions, changed signatures)
- Whether new features, protocols, or subsystems were introduced
- Whether existing behavior was modified in user-visible ways

## Step 2: Read the docs outline

Read `book/src/SUMMARY.md` to understand the current documentation structure and what topics are already covered.

## Step 3: Triage — decide what needs updating

Not every code change requires a docs update. Apply this decision framework:

### Changes that DO NOT need docs updates:
- Internal refactors that don't change behavior or APIs
- Performance optimizations (unless they change usage patterns or add new features like new CLI flags)
- Bug fixes (unless they change documented behavior)
- Test changes
- CI/CD changes
- Dependency bumps
- Code style / formatting changes
- Changes to internal implementation details that aren't documented

### Changes that DO need docs updates:
- New user-facing features (new CLI commands, new macros, new configuration options)
- Changes to the proving/verification API that users interact with
- New architectural components that are significant enough to document
- Changes to existing documented behavior
- New optimizations or techniques that are novel and worth explaining in "How it works"
- Removed or deprecated features that are currently documented

### Changes that may warrant a NEW doc page:
- A major new subsystem or feature (e.g., a new commitment scheme, a new lookup argument)
- A new user-facing workflow (e.g., a new way to profile, a new SDK feature)

## Step 4: Report your findings

Before making any changes, present a summary to the user:

1. **Overview of code changes**: Brief description of what changed
2. **Docs verdict**: One of:
   - **No docs update needed** — explain why the changes don't affect documentation
   - **Existing pages need updates** — list which pages and what needs changing
   - **New page(s) warranted** — describe what the new page would cover and where it fits in SUMMARY.md
3. **Proposed changes**: For each doc file you plan to modify or create, describe specifically what you'll change

Wait for user confirmation before proceeding to Step 5.

## Step 5: Make the changes

For each doc update:

1. Read the full current content of the doc page
2. Make targeted edits — preserve the existing style, tone, and structure
3. For new pages:
   - Create the markdown file in the appropriate subdirectory under `book/src/`
   - Add the entry to `book/src/SUMMARY.md` in the correct location
4. Keep documentation accurate and concise — match the existing style in the book

### Writing style guidelines (match the existing book):
- Technical but accessible — assume the reader has basic cryptography/ZK knowledge
- Use concrete examples and code snippets where helpful
- Link to related pages within the book using relative paths
- Use standard mdBook conventions
