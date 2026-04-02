---
name: implement-spec
description: One-shot implementation from an approved spec
---

Implement the spec on this PR in a single pass.

## Process

1. **Read the spec**: Find the `specs/*.md` file in this PR (exclude `README.md` and `TEMPLATE.md`). Read it thoroughly.
2. **Read CLAUDE.md**: Understand the project's architecture, conventions, and testing requirements.
3. **Plan**: Based on the spec's Intent and Execution sections, determine the full set of changes needed. Consider:
   - Which files to create, modify, or remove
   - The order of changes (dependencies first)
   - How existing patterns and abstractions should be extended
4. **Implement**: Make all changes described in the spec. Follow the project's code style and conventions from CLAUDE.md.
5. **Evaluate**: Run the evaluation criteria from the spec:
   - Run any tests mentioned in the Evaluation section
   - Run the standard lint/test commands from CLAUDE.md
   - Fix any failures before committing
6. **Update spec status**: Change the spec's `Status` field from `proposed` or `approved` to `implemented`.
7. **Commit and push**: Create clear, well-scoped commits for the implementation. Push to the PR branch.

## Guidelines

- The spec is the source of truth. Implement what it says, not more.
- If something in the spec is ambiguous, flag it in a PR comment rather than guessing. But a `claude-approved` spec should have no ambiguities.
- Follow all conventions in CLAUDE.md (formatting, linting, testing).
- Do not add features, refactor code, or make improvements beyond what the spec describes.
- Performance is critical in this project. Profile before optimizing, but don't introduce regressions.
