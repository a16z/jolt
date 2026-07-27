# Local Mode — Interactive Socratic Interview

Full iterative loop, one question at a time:

**Each round:**
1. Identify the dimension with the LOWEST clarity score
2. State why this dimension is the bottleneck
3. Ask ONE targeted question
4. Wait for the author's response
5. Re-score all dimensions
6. Report progress:

```
Round {n} complete.

| Dimension | Score | Weight | Weighted | Gap |
|-----------|-------|--------|----------|-----|
| Goal | {s} | 0.35 | {s*w} | {gap or "Clear"} |
| Constraints | {s} | 0.20 | {s*w} | {gap or "Clear"} |
| Success Criteria | {s} | 0.30 | {s*w} | {gap or "Clear"} |
| Context | {s} | 0.15 | {s*w} | {gap or "Clear"} |
| **Ambiguity** | | | **{score}%** | |

Next target: {weakest_dimension} — {rationale}
```

**Challenge modes:**

- **Round 4+ — Contrarian:** Challenge the spec's core assumption. "What if this constraint doesn't exist?" Test whether the framing is correct or habitual.
- **Round 6+ — Simplifier:** Probe whether complexity can be removed. "What's the simplest version that satisfies the invariants?"
- **Round 8+ — Ontologist (if ambiguity > 0.3):** "What IS this, really?" — find the essence.

Each mode is used ONCE.

**Stop conditions:**

- All dimensions at 0.9+: approve immediately
- Round 3+: allow early approval if the author says "enough", "looks good"
- Ambiguity stalls (same score ±0.05 for 3 rounds): activate Ontologist mode
- Round 10: soft warning about round count
- Round 15: hard cap

**When approved:** Print the summary and offer to update the spec with any refinements discovered during the interview.
