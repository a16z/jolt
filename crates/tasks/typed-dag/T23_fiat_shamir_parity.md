# T23: Fiat-Shamir Transcript Parity

**Status**: `[ ]` Not started
**Depends on**: T19 (Multi-Phase), T20 (Uni-Skip), T21 (RA Virtual), T22 (Verifier)
**Blocks**: E2E muldiv with real Spartan
**Crate**: `jolt-zkvm`
**Estimated scope**: Large (~400 lines, mostly debugging)

## Objective

Ensure the prover's Fiat-Shamir transcript matches the verifier's
byte-for-byte. This means every challenge squeeze and transcript append
must happen in exactly the same order on both sides.

## Background

jolt-core's prover and verifier maintain transcript consistency through
`SumcheckInstanceParams` — each instance's `input_claim()` and
`output_claim()` are called in the same order by both sides.

In the typed DAG, the prover squeezes challenges in `stages.rs` and the
verifier squeezes them in the `DescriptorSource::init()`. These must match.

## Key Ordering Constraints

### Per-Stage Challenge Order

For each stage, the transcript state must match:

1. **Before stage**: All prior stages' round polys + evaluations appended
2. **Challenge squeeze**: Stage-specific challenges (γ, eq points, etc.)
3. **Sumcheck**: `BatchedSumcheckProver` appends claimed_sums, squeezes α,
   then appends round polys and squeezes per-round challenges
4. **After stage**: Polynomial evaluations appended to transcript

### Cross-Stage Data Appended

After each sumcheck proof, the prover must append evaluation claims to
the transcript (Fiat-Shamir binding). The verifier does the same at
`verifier.rs:459-461`:
```rust
for claim in &new_claims {
    claim.eval.append_to_transcript(transcript);
}
```

The prover must do the same between stages.

## Deliverables

### 1. Transcript Checkpointing

Add transcript state assertions between stages:
```rust
// After S2
let checkpoint_s2: F = transcript.challenge();
eprintln!("S2 checkpoint: {checkpoint_s2:?}");
```

Compare prover and verifier checkpoints to identify divergence points.

### 2. Evaluation Append After Each Stage

Currently the prover does NOT append evaluations to the transcript after
each stage. Add this:

```rust
// After each stage's sumcheck completes, append all evaluations
for eval in &stage_proof.evaluations {
    eval.append_to_transcript(&mut transcript);
}
```

This must match what the verifier does.

### 3. Challenge Ordering Audit

For each stage, document the exact transcript operations:

**S2 (5-instance batch):**
1. PV uni-skip: squeeze tau_high
2. RamRW: squeeze gamma
3. InstrLookupsCR: squeeze gamma
4. OutputCheck: squeeze r_address (log_K challenges)
5. Sumcheck: append 5 claimed_sums, squeeze alpha,
   then max_num_vars rounds of (append round_poly, squeeze challenge)
6. After: append evaluations

**S3 (3-instance batch):**
1. Shift: squeeze 5 gamma_powers
2. InstrInput: squeeze gamma
3. RegistersCR: squeeze gamma
4. Sumcheck: ...
5. After: append evaluations

(Document all 6 stages similarly)

### 4. End-to-End Transcript Comparison

Write a test that:
1. Runs `prove()` capturing the transcript state at each stage boundary
2. Runs `verify()` capturing the transcript state at each stage boundary
3. Asserts they match at every checkpoint

## Known Divergence Points

- **Uni-skip** (T20): The first-round polynomial for uni-skip has different
  degree than regular rounds, affecting transcript bytes
- **Multi-phase** (T19): Multi-phase instances interleave address and cycle
  challenges, which must match the verifier's StageDescriptor challenge order
- **Evaluation append order**: Must match between prover stages and verifier's
  `verify_stage` evaluation extraction

## Reference

- Verifier transcript ops: `crates/jolt-verifier/src/verifier.rs:450-471`
- BatchedSumcheckProver transcript: `crates/jolt-sumcheck/src/batched.rs:66-131`
- BatchedSumcheckVerifier transcript: `crates/jolt-sumcheck/src/batched.rs:205-237`
- ClearRoundHandler transcript: `crates/jolt-sumcheck/src/handler.rs:78-83`

## Acceptance Criteria

- [ ] Evaluation appends added after each prover stage
- [ ] Challenge ordering documented for all 6 stages
- [ ] Transcript checkpoint comparison test passes
- [ ] prove() → verify() succeeds for synthetic trace
- [ ] prove() → verify() succeeds for muldiv guest (T17)
