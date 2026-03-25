# Jolt Specs

This directory contains design specifications for major Jolt features. Every significant new feature or architectural change must begin as a spec, reviewed and approved by maintainers, before implementation begins.

## Philosophy

Jolt follows a **spec-driven, AI-native development** workflow. The core insight: code is cheap, trust is expensive. AI can generate substantial execution code, but human verification time is the scarce resource. We optimize for that by structuring work across three planes:

1. **Intent** -- What are we building, why, and what properties must hold? Intent lives in the spec, but it also manifests in code: types, schemas, data structures, abstractions, and architectural boundaries should document their own purpose. Good intent makes correctness legible.
2. **Execution** -- The bulk of the code that connects intent to evidence. Derivable, replaceable, regenerable. This is what AI is good at, steered by the spec author or a maintainer.
3. **Evaluation** -- Tests, benchmarks, assertions, proofs. How do we know the execution is correct? Evaluation should be independent of execution details and mechanically checkable.

Spec authors focus on **intent and evaluation**. Execution is downstream — it follows from what we want (intent) and how we'll know it works (evaluation). A spec may include some execution guidance (e.g. optimizations to consider, algorithmic direction), but the emphasis belongs on the other two planes.

## Workflow

1. **Author a spec** using the template: `./new-spec.sh my-feature-name`
2. **Open a PR** with the spec. Maintainers review it.
3. **Merge the spec.** A GitHub Action automatically creates a tracking issue.
4. **Implement.** The spec author or a maintainer steers the implementation (via AI or manually).
5. **Verify evaluation.** All evaluation criteria from the spec must be satisfied before the implementation PR merges.

## Template

See [TEMPLATE.md](TEMPLATE.md) for the spec template.
