# Remote Mode — Single-Pass Output

Post a **single PR comment** with all findings:

```
**Spec Analysis: {spec title}**

| Dimension | Score | Gap |
|-----------|-------|-----|
| Goal | {s} | {gap or "Clear"} |
| Constraints | {s} | {gap or "Clear"} |
| Success Criteria | {s} | {gap or "Clear"} |
| Context | {s} | {gap or "Clear"} |
| **Ambiguity** | | **{score}%** |

{If ambiguity ≤ 20%:}
**Status: Approved** — The spec is clear enough for one-shot implementation.

**Summary:**
- {what will be built}
- {key invariants}
- {critical evaluation criteria}

**Next step:** Run `/implement-spec` to implement this spec:
- [Open in Claude Code (cloud)](https://claude.ai/code) — run `/implement-spec` on this branch
- Or run `/implement-spec` locally in Claude Code

{If ambiguity > 20%:}
**Status: Questions remain** — {n} ambiguities to resolve before implementation.

**Questions:**

**1. [{dimension}]** {question}

**2. [{dimension}]** {question}

...

> After addressing these questions, update the spec and re-add the `claude-spec-review-request` label.
```

If approved, add the label: `gh pr edit --add-label claude-spec-approved`

If NOT approved, do NOT add the label.

Single pass — no escalation. Either approve or list remaining questions.
