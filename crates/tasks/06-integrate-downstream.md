# Task 06: integrate-jolt-ir-downstream

**Status:** Not started
**Phase:** 4a (during jolt-zkvm implementation)
**Dependencies:** Tasks 01 + 02 (core IR + evaluate backend) at minimum. Task 03 (R1CS backend) for BlindFold.
**Blocks:** Completion of the refactoring

## Objective

Migrate `jolt-zkvm` to consume `jolt-ir` claim definitions, replacing the dual-implementation pattern.

## Architecture decision

Only `jolt-zkvm` depends on `jolt-ir`. The sumcheck and spartan crates remain generic protocol implementations:

- **jolt-sumcheck** — generic sumcheck protocol. Does not touch claim formulas. No `jolt-ir` dependency.
- **jolt-spartan** — generic Spartan IOP. Claim formulas for outer/shift/product sumchecks are defined in `jolt-zkvm`, not inside spartan. No `jolt-ir` dependency.
- **jolt-zkvm** — the orchestrator. Defines all claim formulas via `ClaimDefinition`, passes concrete evaluation results and R1CS constraints to sumcheck/spartan.

This keeps jolt-sumcheck and jolt-spartan reusable and the IR dependency contained to the one crate that actually defines claim formulas.

## Scope

### jolt-zkvm
- Add `jolt-ir` dependency
- Refactor `SumcheckInstanceParams`:
  - `claim_definition() -> ClaimDefinition` replaces `input_claim()` + `input_claim_constraint()`
  - `output_claim_definition() -> Option<ClaimDefinition>` replaces `output_claim_constraint()`
  - Default `input_claim()` calls `evaluate()` on the definition
- Migrate all ~20 implementations (booleanity, RAM r/w, registers, instruction RA, spartan outer/shift, claim reductions, etc.)
- BlindFold: `ZkStageData` stores `ClaimDefinition` instead of `OutputClaimConstraint`
- Spartan outer/shift claim formulas move from jolt-spartan into jolt-zkvm, expressed as `ClaimDefinition`s

### zklean-extractor (future)
- Replace `ClaimExpr` consumption with `jolt-ir::Expr::to_lean4()`

### gnark-transpiler (future, PR #1322)
- Replace `MemoizedCodeGen` for claim formulas with `jolt-ir::Expr::to_circuit()`

## Verification

- `muldiv` e2e test passes in both standard and ZK modes
- All existing sumcheck tests pass
- BlindFold R1CS constraints match (SoP normalization equivalent to hand-written)

## Reference

- `jolt-core/src/subprotocols/sumcheck_verifier.rs:49-69` — current `SumcheckInstanceParams`
- `jolt-core/src/subprotocols/booleanity.rs:87-161` — migration example
- `jolt-core/src/zkvm/ram/read_write_checking.rs:67-228` — migration example
- `jolt-core/src/subprotocols/blindfold/mod.rs:53-68` — `ZkStageData` with `OutputClaimConstraint`
