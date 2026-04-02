---
name: analyze-spec
description: Deep-interview analysis of a spec file to find ambiguities before implementation
---

Perform a deep-interview-style analysis of the spec on this PR. Your goal: ensure the spec is clear enough for a one-shot implementation with zero clarifying questions.

## Process

1. **Read the spec**: Find the `specs/*.md` file in this PR (exclude `README.md` and `TEMPLATE.md`).
2. **Explore the codebase**: Identify areas of the codebase relevant to the spec's intent. Understand existing patterns, types, and abstractions that the implementation will interact with.
3. **Analyze for ambiguities**: Check each section of the spec:
   - **Intent**: Are the invariants precise? Are the types and abstractions well-defined? Could two engineers read this and build different things?
   - **Evaluation**: Are the success criteria testable and concrete? Do they cover the invariants from Intent? Are edge cases addressed?
   - **Execution**: If present, does the direction conflict with existing patterns? Are there missing considerations?
4. **Post questions**: For each ambiguity found, post a targeted review comment on the specific line of the spec. Each question should:
   - Expose a hidden assumption, missing invariant, or unclear boundary
   - Be specific enough that the answer directly improves the spec
   - Reference relevant codebase context when applicable
5. **Assess clarity**: After posting all questions, evaluate whether the spec is implementation-ready.

## When satisfied (no ambiguities remain)

1. Post a summary comment:
   ```
   **Spec analysis complete.**

   The spec is clear enough for one-shot implementation. Key points:
   - [brief summary of what will be built]
   - [key invariants that must hold]
   - [critical evaluation criteria]
   ```
2. Add the `claude-approved` label to the PR: `gh pr edit --add-label claude-approved`

## When NOT satisfied

Post your questions and end with:
```
**Spec analysis: questions remain.**

[N] ambiguities found that should be resolved before implementation.
Reply to the comments above, then ask me to analyze again.
```

Do NOT add the `claude-approved` label until all ambiguities are resolved.

## Guidelines

- Ask ONE question per review comment — no batching.
- Explore the codebase BEFORE asking questions about it. Never ask the author what the code already tells you.
- Focus on assumptions and invariants, not stylistic preferences.
- Reference specific files, types, or functions from the codebase when relevant.
- The spec doesn't need to be perfect — it needs to be unambiguous enough that the implementation is a mechanical follow-through.
