# jolt-compiler-v2 testing pattern

The v2 compiler should earn correctness incrementally. Every new protocol phase
needs a local compiler test, generated prover/verifier self-parity, and then a
bridge into `jolt-equivalence` for jolt-core parity.

## Gates

1. **IR/golden/schema tests**: run `cargo nextest run -p jolt-compiler-v2 --cargo-quiet`.
   These tests verify registered dialect parsing, schema validation, pass output,
   golden MLIR, SSA dataflow, and generated Rust compilation.
2. **Generated self-parity**: each phase emits prover and verifier Rust, runs
   them against the same proof object, and asserts transcript event streams
   match step-for-step, including absorbed bytes and post-event states. The
   commitment coverage includes `generated_commitment_prover_verifier_self_parity_runs`
   for a tiny CPU fixture and `pipeline_generated_commitment_prover_verifier_self_parity_runs`
   for the full protocol→party→compute→CPU pipeline at small domain sizes.
   Stage 1 coverage includes shape-proof verifier acceptance, synthetic
   remaining-sumcheck parity, and real R1CS-backed data parity through the
   generated prover and kernel-free generated verifier.
3. **Modular self-verify**: once v2 output is wired into the modular stack, run
   `cargo nextest run -p jolt-equivalence modular_self_verify --cargo-quiet`.
   This proves the generated prover output is accepted by the modular verifier.
4. **Transcript parity vs jolt-core**: run
   `cargo nextest run -p jolt-equivalence transcript_divergence --cargo-quiet`.
   This is the byte-identical Fiat-Shamir oracle.
   The v2 commitment slice also has a focused bridge:
   `cargo nextest run -p jolt-equivalence bolt_commitment_transcript_matches_jolt_core_append_serializable --cargo-quiet`.
   It extracts the Bolt CPU commitment program, runs prover/verifier commitment
   replay, and compares the transcript append bytes plus post-state sequence
   against jolt-core's `append_serializable` semantics.
5. **Proof acceptance by jolt-core**: run
   `cargo nextest run -p jolt-equivalence zkvm_proof_accepted --cargo-quiet`.
   This bootstraps soundness against the existing core verifier.

## Four per-stage parity oracles

Every wired protocol stage must pass the same four checks before the next stage
is considered unblocked:

1. **Bolt proof acceptance**: the generated/Bolt verifier accepts the proof
   objects produced by the generated/Bolt prover for that stage.
2. **Bolt transcript parity**: Bolt prover and Bolt verifier transcript states
   match step-for-step for the stage-local transcript operations.
3. **jolt-core proof acceptance**: jolt-core accepts a proof whose wired stage
   artifacts were produced by Bolt and spliced into the matching core proof
   prefix.
4. **jolt-core transcript/artifact parity**: Bolt matches jolt-core transcript
   states and observable stage artifacts through the stage boundary, including
   sumcheck polynomials, opening claims, commitments, and normalized opening
   points when those artifacts exist for the stage.

The first version applies this matrix to standard-mode stages. ZK/BlindFold
should reuse the same matrix when those dialects and proof objects are wired.

## Per-phase rule

For a new phase, add both party projections before treating the phase as done:

1. Define or extend protocol/concrete IR for the formal Bolt operation.
2. Project concrete IR into prover and verifier party IR.
3. Lower both parties through `compute` and `cpu`, carrying phase dataflow as
   SSA operands/results instead of symbol-token attributes.
4. Emit both Rust targets from `cpu`.
5. Add generated self-parity with transcript-state equality.
6. Wire the phase into `jolt-equivalence` as a dual path against the current
   modular/core implementation.

The commitment phase currently has executable generated self-parity on both a
small CPU fixture and the full generated compiler pipeline at small domain
sizes, plus a focused jolt-core transcript bridge in `jolt-equivalence`.

`tests/fixtures/jolt_protocol_chain_commitment_stage1.yaml` is the chain-level
fixture. It records the ordered commitment→Stage 1 components and the parity
gates that must keep passing as each new protocol phase is appended. The
generated chain test `generated_jolt_chain_commitment_then_stage1_self_parity_runs`
runs commitment and Stage 1 on the same transcript for prover and verifier.

Stage 2 currently has focused compiler and equivalence gates rather than a
longer chain fixture. The compiler gate
`cargo nextest run -p jolt-compiler-v2 stage2 --cargo-quiet` checks protocol,
concrete, party, compute, kernelized compute, CPU, and generated Rust fixtures,
including that verifier Rust does not import `jolt-kernels`. The equivalence
gate `cargo nextest run -p jolt-equivalence bolt_stage2 --cargo-quiet --no-fail-fast`
checks the four parity axes for the wired Stage 2 slice. The perf oracle is
`cargo run --release -p jolt-bench --bin bolt-stage1 -- --program sha2-chain --stage stage2 --log-t 16 --num-iters 16 --iters 3 --warmup 1 --json perf/bolt-stage2-last.json`.
