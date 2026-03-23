# T22: Verifier Descriptor Wiring

**Status**: `[ ]` Not started
**Depends on**: T15 (Orchestrator), T16 (Verifier DAG)
**Blocks**: T23 (E2E muldiv with verify)
**Crate**: `jolt-zkvm`, `jolt-verifier`
**Estimated scope**: Large (~400 lines)

## Objective

Implement a `DescriptorSource` that builds `StageDescriptor`s matching the
prover's typed DAG challenge ordering. This connects the prover's output
(stage proofs) to the existing `jolt_verifier::verify()` function.

## Background

The jolt-verifier already has a complete `verify()` function that:
1. Verifies Spartan (S1)
2. For each stage, builds a `StageDescriptor` via `DescriptorSource`
3. Verifies the batched sumcheck proof against the descriptor
4. Checks `eq(point) × g(evals) == final_eval`
5. Accumulates opening claims
6. Verifies PCS opening proofs

The missing piece: a `DescriptorSource` that squeezes the same challenges
as the prover and builds correct `StageDescriptor`s.

## Existing Infrastructure

- **`StageDescriptor<F>`** (`jolt-verifier/stage.rs`): Describes one stage
  for verification. Fields: num_vars, degree, claimed_sum, eq_point,
  reverse_challenges, use_eq_plus_one, use_lt, output_expr, output_challenges,
  commitment_indices, extra_claims.

- **`DescriptorSource` trait** (`jolt-verifier/verifier.rs`): Builds
  descriptors lazily during verification. Has `init()` (receives r_x, r_y
  from Spartan) and `next_descriptor()` (receives prior_claims).

- **`ClosureSource`**: Adapter that builds all descriptors in `init()` via
  a closure.

- **`ClaimDefinition::evaluate()`** in jolt-ir: Evaluates symbolic formulas.

- **`input_claims.rs`** in jolt-zkvm: All input claim formulas, shared
  between prover and verifier.

## Deliverables

### 1. `TypedDagDescriptorSource` in jolt-zkvm

Implements `DescriptorSource<F, T>` using the same challenge squeeze order
as the prover stages:

```rust
pub struct TypedDagDescriptorSource<F: Field> {
    config: ProverConfig,
    tables: PolynomialTables<F>,  // for brute-force claimed sum computation
    descriptors: Vec<StageDescriptor<F>>,
    cursor: usize,
}
```

The `init()` method:
1. Receives `(r_x, r_y)` from Spartan verification
2. Evaluates virtual polynomials at `r_cycle` (same as `SpartanVirtualEvals`)
3. Squeezes all the same challenges as stages S2-S7
4. Computes claimed sums using the same input_claim formulas
5. Builds `StageDescriptor` for each stage with:
   - `eq_point` from challenge squeeze
   - `claimed_sum` from input_claim formula
   - `output_expr` from jolt-ir `ClaimDefinition`
   - `commitment_indices` mapping polynomial evaluations to commitments

### 2. Populate `SumcheckStageProof::evaluations`

Currently the prover returns empty `evaluations` vectors in stage proofs.
The verifier needs these to check `eq × g(evals) == final_eval`.

For each stage, extract the polynomial evaluations at the challenge point
and include them in the `SumcheckStageProof`. The ordering must match the
`commitment_indices` in the corresponding `StageDescriptor`.

### 3. Wire `verify()` in jolt-zkvm

Add a `verify()` function to jolt-zkvm that:
1. Creates a `TypedDagDescriptorSource`
2. Calls `jolt_verifier::verify(proof, vk, &mut source, &mut transcript)`
3. Returns `Ok(())` on success

### 4. Populate `commitment_indices`

Each `StageDescriptor` needs `commitment_indices` mapping evaluation slots
to indices in `proof.commitments`. The commitment ordering must match
what the prover produces in `prove()`.

Current prover commitment order:
```
[0] = ram_inc
[1] = rd_inc
[2..2+instr_d] = instruction_ra
[2+instr_d..2+instr_d+bytecode_d] = bytecode_ra
[...] = ram_ra
```

## Key Challenge: Prover/Verifier Evaluation Consistency

The verifier checks:
```
Σ_j α^j · pad_j · eq(eq_point_j, eval_point_j) · g_j(evals_j) == final_eval
```

For this to work:
- `claimed_sum` must match what the prover computed
- `eq_point` must be the same challenge point
- `output_expr` must encode the same formula
- `evaluations` must be the polynomial values at the challenge point
- `commitment_indices` must map to the correct commitments

## Reference

- `jolt_verifier::verify()`: `crates/jolt-verifier/src/verifier.rs:422-488`
- `verify_stage()`: `crates/jolt-verifier/src/verifier.rs:93-251`
- `DescriptorSource` trait: `crates/jolt-verifier/src/verifier.rs:292-310`
- `StageDescriptor`: `crates/jolt-verifier/src/stage.rs`
- `input_claims.rs`: `crates/jolt-zkvm/src/input_claims.rs`
- Prover stages: `crates/jolt-zkvm/src/stages.rs`

## Acceptance Criteria

- [ ] `TypedDagDescriptorSource` implements `DescriptorSource`
- [ ] Challenge squeeze order matches prover exactly
- [ ] Stage proofs carry correct polynomial evaluations
- [ ] `verify()` function in jolt-zkvm calls jolt_verifier::verify
- [ ] Commitment indices match prover ordering
- [ ] prove() → verify() round-trip succeeds for synthetic data
