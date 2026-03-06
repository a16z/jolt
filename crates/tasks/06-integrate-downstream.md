# Task 06: integrate-jolt-ir-downstream

**Status:** Not started
**Phase:** 4a (during jolt-zkvm implementation)
**Dependencies:** Tasks 01–05 (core IR + all backends)
**Blocks:** Completion of the refactoring

## Objective

Migrate `jolt-zkvm` and `jolt-spartan` to consume `jolt-ir` claim definitions, replacing the dual-implementation pattern.

## Architecture

- **jolt-spartan** — depends on `jolt-ir`. Uses `jolt-ir`'s R1CS types as its native constraint representation. Claim formulas for outer/shift/product sumchecks are expressed as `Expr` and emitted via `SumOfProducts::emit_r1cs()`.
- **jolt-sumcheck** — generic sumcheck protocol. Does not touch claim formulas. No `jolt-ir` dependency.
- **jolt-zkvm** — the orchestrator. Defines all claim formulas via `ClaimDefinition`, passes concrete evaluation results to sumcheck and R1CS constraints to spartan.

## Scope

### jolt-spartan
- Add `jolt-ir` dependency
- Replace hand-written R1CS constraint types with `jolt-ir::R1csConstraint<F>` / `LinearCombination<F>`
- Spartan outer/shift claim formulas expressed as `Expr` → `SumOfProducts` → `emit_r1cs()`

### jolt-zkvm
- Add `jolt-ir` dependency
- Refactor `SumcheckInstanceParams`:
  - `claim_definition() -> ClaimDefinition` replaces `input_claim()` + `input_claim_constraint()`
  - `output_claim_definition() -> Option<ClaimDefinition>` replaces `output_claim_constraint()`
  - Default `input_claim()` calls `evaluate()` on the definition
- Migrate all ~20 implementations (booleanity, RAM r/w, registers, instruction RA, spartan outer/shift, claim reductions, etc.)
- BlindFold: `ZkStageData` stores `ClaimDefinition` instead of `OutputClaimConstraint`

### zklean-extractor (future)
- Replace `ClaimExpr` consumption with `jolt-ir::Expr::to_lean4()`

## Verification

- `muldiv` e2e test passes in both standard and ZK modes
- All existing sumcheck tests pass
- BlindFold R1CS constraints match (SoP normalization equivalent to hand-written)

## Reference

- `jolt-core/src/subprotocols/sumcheck_verifier.rs:49-69` — current `SumcheckInstanceParams`
- `jolt-core/src/subprotocols/booleanity.rs:87-161` — migration example
- `jolt-core/src/zkvm/ram/read_write_checking.rs:67-228` — migration example
- `jolt-core/src/subprotocols/blindfold/mod.rs:53-68` — `ZkStageData` with `OutputClaimConstraint`
- `jolt-core/src/subprotocols/blindfold/output_constraint.rs` — current `ValueSource`/`ProductTerm`/`OutputClaimConstraint` (replaced by `jolt-ir` types)
