# jolt-sumcheck

Sumcheck protocol engine for the Jolt zkVM.

Part of the [Jolt](https://github.com/a16z/jolt) zkVM.

## Overview

This crate implements the sumcheck interactive proof protocol for multilinear polynomials. Given a claim `sum_{x in {0,1}^n} g(x) = S`, the prover convinces the verifier of the sum's correctness in `n` rounds, each reducing one variable. The crate supports both single-polynomial and batched (random linear combination) sumcheck.

## Public API

### Claims

- **`SumcheckClaim<F>`** -- A sumcheck claim specifying `num_vars`, `degree`, and `claimed_sum`.

### Proofs

- **`SumcheckProof<F>`** -- A sequence of univariate round polynomials, one per variable.

### Prover

- **`SumcheckCompute<F>`** -- Trait for witness polynomials. Provides `round_polynomial` and `bind`.
- **`SumcheckProver`** -- Drives the prover side: iterates rounds, queries the witness, produces a `SumcheckProof`.
- **`BatchedSumcheckProver`** -- Batches multiple sumcheck instances via random linear combination.
- **`StreamingSumcheckProver`** -- Memory-efficient streaming variant for large witnesses.
- **`SplitEqEvaluator`** -- Split-eq optimization for sqrt-cost sumcheck evaluation.

### Prefix-Suffix Decomposition

- **`PrefixSuffixEvaluator<F>`** -- Two-phase evaluator for tensor-decomposed polynomials `f(x) = Sum_i P_i(x_prefix) * S_i(x_suffix)`. Phase 1 operates on sqrt(N)-sized pair buffers; Phase 2 materializes suffix tables.
- **`PrefixSuffixTransition<F>`** -- State at the Phase 1 to Phase 2 transition, carrying prefix challenges and scalar evaluations.

### Streaming Types

- **`StreamingSumcheck`** -- Streaming sumcheck prover for memory-constrained settings.
- **`StreamingSumcheckWindow`** -- Window-based streaming interface.
- **`LinearSumcheckStage`** -- Linear-scan sumcheck stage.
- **`StreamingSchedule`** / **`HalfSplitSchedule`** / **`LinearOnlySchedule`** -- Schedule strategies for streaming.

### Verifier

- **`SumcheckVerifier`** -- Verifies a `SumcheckProof` against a claim.
- **`BatchedSumcheckVerifier`** -- Verifies a batched sumcheck proof.

### Reduction

- **`SumcheckReduction<F>`** -- Trait for sumcheck-based claim reductions.
- **`SumcheckWitnessBatch<F>`** -- Type alias for `(Vec<SumcheckClaim<F>>, Vec<Box<dyn SumcheckCompute<F>>>)`.

### Round Handlers

- **`RoundHandler<F>`** -- Prover-side strategy for absorbing round polynomials into the transcript.
- **`RoundVerifier<F>`** -- Verifier-side strategy for checking round data.
- **`ClearRoundHandler`** / **`ClearRoundVerifier`** -- Cleartext implementations. Committed-mode implementations live in `jolt-blindfold`.

### Errors

- **`SumcheckError`** -- Variants: `RoundCheckFailed`, `FinalEvalMismatch`, `DegreeBoundExceeded`, `WrongNumberOfRounds`.

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `parallel` | No | Enable rayon parallelism (also enables `jolt-poly/parallel`) |

## Dependency Position

```
jolt-field ─────┐
jolt-poly  ─────┼─> jolt-sumcheck -> jolt-blindfold, jolt-spartan, jolt-zkvm
jolt-transcript ┘
jolt-openings ──┘    (for ProverClaim/VerifierClaim used by SumcheckReduction)
```

## License

MIT
