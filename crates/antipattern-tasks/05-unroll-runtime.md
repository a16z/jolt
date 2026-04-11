# Task 05: Implement Granular Batched Ops in Runtime

## Status: TODO

## Goal
Implement the runtime handlers for all new ops defined in Task 04, wire up the compiler to emit them, and delete the old `Op::BatchedSumcheckRound`.

## Approach

### Phase A: Dual-emit (keep old + new)
1. Add runtime handlers for all new ops (each should be 5-20 lines, no branching)
2. Add a compiler flag or builder method to emit the new unrolled sequence
3. Verify both paths produce identical transcripts

### Phase B: Cut over
1. Switch the compiler to emit only the new ops
2. Run jolt-equivalence to verify parity
3. Delete `Op::BatchedSumcheckRound` and its ~300-line runtime handler

### Runtime State Changes
The batched sumcheck state currently lives in `RuntimeState`:
```rust
batch_instance_claims: Vec<Vec<F>>,
last_round_instance_evals: Vec<Vec<F>>,
segmented_outer_eqs: HashMap<(usize, usize), Vec<F>>,
```

With unrolled ops, this state is still needed but accessed in a more structured way:
- `BatchRoundBegin` resets `combined` accumulator, updates claims from prev round
- `BatchInactiveContribution` reads/writes `batch_instance_claims[batch][inst]`
- `InstanceReduce` stores evals in `last_round_instance_evals[inst]`
- `BatchAccumulateInstance` reads evals, extrapolates, accumulates into `combined`
- `BatchRoundFinalize` stores `combined` as `last_round_coeffs`

### Key Invariant
The transcript must be byte-identical before and after. The combined round polynomial is the same regardless of whether it's computed by one big handler or a sequence of small ones.

## Runtime Handlers (Sketch)

```rust
Op::BatchRoundBegin { batch, round, max_evals, bind_challenge } => {
    state.combined = vec![F::zero(); *max_evals];
    // Update claims from previous round
    if let Some(ch) = bind_challenge {
        let r = state.challenges[*ch];
        for (inst_idx, evals) in state.last_round_instance_evals.iter().enumerate() {
            if !evals.is_empty() {
                let poly = UnivariatePoly::interpolate_from_evals(evals);
                state.batch_instance_claims[*batch][inst_idx] = poly.evaluate(r);
            }
        }
    }
    state.last_round_instance_evals = vec![Vec::new(); ...];
}

Op::BatchInactiveContribution { batch, instance } => {
    let coeff = state.challenges[batch_coeff_for(batch, instance)];
    let two_inv = F::from_u64(2).inverse().unwrap();
    let half = state.batch_instance_claims[*batch][*instance] * two_inv;
    for slot in &mut state.combined { *slot += coeff * half; }
    state.batch_instance_claims[*batch][*instance] = half;
}

Op::InstanceReduce { batch, instance, kernel } => {
    let compiled = &executable.kernels[*kernel];
    let kdef = &module.prover.kernels[*kernel];
    let input_refs = ...; // from device_buffers
    state.last_round_instance_evals[*instance] = backend.reduce(compiled, &input_refs, &state.challenges);
}

Op::BatchAccumulateInstance { batch, instance, max_evals } => {
    let coeff = state.challenges[batch_coeff_for(batch, instance)];
    let evals = &state.last_round_instance_evals[*instance];
    let mut full = evals.clone();
    // Extrapolate if needed
    if full.len() < *max_evals {
        let poly = UnivariatePoly::interpolate_from_evals(&full);
        for s in full.len()..*max_evals {
            full.push(poly.evaluate(F::from_u64(s as u64)));
        }
    }
    for (i, &v) in full.iter().enumerate() {
        state.combined[i] += coeff * v;
    }
}

Op::BatchRoundFinalize { batch } => {
    state.last_round_coeffs = state.combined.clone();
}
```

## Test
```bash
# Phase A: verify dual-emit produces identical output
cargo nextest run -p jolt-equivalence transcript_divergence --cargo-quiet

# Phase B: verify after removing old op
cargo nextest run -p jolt-equivalence transcript_divergence --cargo-quiet
cargo nextest run -p jolt-equivalence --cargo-quiet
```

## Risk: Medium-High
This is the largest single change. The dual-emit approach mitigates risk by allowing comparison before cutover.

## Dependencies: Task 04
