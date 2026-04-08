# Spec: [Feature Name]

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   |                                |
| Created     | YYYY-MM-DD                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

One paragraph: what is this feature and why does it matter? State the problem being solved, not just the solution.

## Intent

### Goal

What are we building? State the primary objective in one sentence without qualifiers. Define the key abstractions, types, and architectural boundaries this feature introduces or modifies.

### Invariants

What properties must hold? List the correctness, safety, or consistency invariants that the implementation must preserve. For ZK features, include prover/verifier consistency requirements.

### Non-Goals

What is explicitly out of scope? Listing non-goals prevents scope creep and clarifies the feature's boundaries.

## Evaluation

### Acceptance Criteria

Concrete, testable criteria. Each should be verifiable by a test, benchmark, or assertion.

- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3

### Testing Strategy

Which existing tests must continue passing? What new tests are needed? Specify both standard (`--features host`) and ZK (`--features host,zk`) mode requirements where applicable.

### Performance

What are the performance expectations? Specify benchmarks, acceptable regressions, memory budgets, or throughput targets. "No regression" is acceptable if there is a benchmark to verify against.

## Design

### Architecture

How does this feature fit into the existing system? Describe which modules, types, and abstractions are affected. Include a diagram if the interaction is non-trivial.

### Alternatives Considered

What other approaches were evaluated? Why was this design chosen over them? This section prevents re-litigating decisions during implementation review.

## Documentation

What changes to the Jolt book (`book/`) are required? List new pages, sections to update, or diagrams to add. If no documentation changes are needed, state why (e.g., internal refactor with no user-facing impact).

## Execution

Optional implementation direction — algorithmic approach, optimizations to consider, modules to touch. The implementer should be able to derive most of this from Intent and Evaluation.

## References

Links to papers, related specs, relevant issues/PRs, prior art.
