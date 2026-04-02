# Jolt Specs

Design specifications for major Jolt features. Significant new features or architectural changes begin as a spec, reviewed and approved before implementation.

## Philosophy

Jolt follows a **spec-driven, AI-native development** workflow. Code is cheap, trust is expensive. AI can generate substantial code, but human verification time is the scarce resource. We optimize for that by structuring work across three planes:

1. **Intent** -- What are we building, why, and what properties must hold? Types, schemas, abstractions, and architectural boundaries should document their own purpose.
2. **Execution** -- The bulk of the code that connects intent to evidence. Derivable, replaceable, regenerable.
3. **Evaluation** -- Tests, benchmarks, assertions, proofs. Independent of execution details.

Spec authors focus on **intent and evaluation**. Execution is downstream.

## Workflow

A single PR carries a feature from spec to implementation:

1. **Create a spec** using `/new-spec <feature-name>` in Claude Code (or manually from `TEMPLATE.md`).
2. **Open a PR** with the spec. A GitHub Action auto-renames the file to `<PR#>-<name>.md` and adds the `spec` label.
3. **Analyze the spec.** A maintainer comments `@claude analyze` — Claude performs a deep-interview-style analysis, posting probing questions until it has zero ambiguity. When satisfied, Claude adds the `claude-approved` label.
4. **Review and approve.** Maintainers review the spec.
5. **Implement.** A maintainer comments `@claude implement` — Claude generates a one-shot implementation on the same branch.
6. **Review and merge.** Maintainers review the implementation. Spec status becomes `implemented`.

## Template

See [TEMPLATE.md](TEMPLATE.md) for the spec template.
