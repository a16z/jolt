# T07: S1 Spartan Integration

**Status**: `[ ]` Not started
**Depends on**: T05 (Stage Output Types), T06 (Input Claim Formulas)
**Blocks**: T15 (Prove Orchestrator)
**Crate**: `jolt-spartan` + `jolt-zkvm`
**Estimated scope**: Medium (~200 lines)

## Objective

Clean integration of jolt-spartan's Spartan PIOP as Stage 1 of the typed DAG.
The Spartan outer sumcheck (with uni-skip) produces `SpartanOutput<F>` with
all virtual polynomial evaluations that downstream stages consume.

## Background

jolt-core's `prove_stage1`:
1. Runs `OuterUniSkipProver` for first-round polynomial
2. Runs `OuterRemainingStreamingSumcheck` for remaining rounds
3. Extracts virtual polynomial evaluations at `r_cycle`

jolt-spartan already exists as a separate crate. This task is about making
it return a typed `SpartanOutput<F>` and wiring it into jolt-zkvm.

## Deliverables

### 1. Modify jolt-spartan to return typed output

Either:
- Add a method that returns `SpartanOutput<F>` directly, OR
- Have jolt-zkvm's `prove_spartan()` function call jolt-spartan and extract
  the typed fields from the result

The key requirement is that all virtual polynomial evaluations at `r_cycle`
are available in the output.

### 2. `prove_spartan()` function in jolt-zkvm

```rust
fn prove_spartan<F, T, B>(
    tables: &PolynomialTables<F>,
    config: &ProverConfig,
    transcript: &mut T,
    backend: &Arc<B>,
) -> SpartanOutput<F>
```

This function:
1. Builds the R1CS key from config
2. Runs uni-skip for first round
3. Runs outer remaining sumcheck
4. Extracts `r_x`, `r_y` challenge vectors
5. Evaluates all virtual polynomials at `r_cycle = r_y[..log_T]`
6. Returns `SpartanOutput<F>`

### 3. Virtual evaluation extraction

The virtual evaluations come from evaluating the polynomial tables at
the Spartan challenges. These are the "R1CS outputs" that represent:
- `ram_read_value(r_cycle)`, `ram_write_value(r_cycle)`, etc.

Reference: jolt-core `zkvm/spartan/outer.rs::finalize_prover()`

## Reference

- jolt-core Stage 1: `jolt-core/src/zkvm/prover.rs::prove_stage1()`
- Outer sumcheck: `jolt-core/src/zkvm/spartan/outer.rs`
- Product virtual: `jolt-core/src/zkvm/spartan/product.rs`
- jolt-spartan crate: `crates/jolt-spartan/`

## Acceptance Criteria

- [ ] `prove_spartan()` returns `SpartanOutput<F>`
- [ ] All virtual evals populated correctly
- [ ] Uni-skip handled internally
- [ ] Unit test: synthetic R1CS → verify output evals match brute force
- [ ] `cargo clippy -p jolt-zkvm -p jolt-spartan` passes
