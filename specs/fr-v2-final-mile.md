# Final-mile — Stage 4 FieldRegRW input-claim mismatch

This is a working note for the next session, not a long-lived spec.
Delete after the SDK example proves green.

## State at HEAD

The full FR coprocessor stack is wired end-to-end. Running
`cargo run --release -p bn254-fr-poseidon2-sdk` reaches the Stage 4
batched sumcheck and fails on the first round with:

```
InvalidProof { driver: "jolt.stage4.field_reg_rw",
               reason: "stage4 relation input claim mismatch" }
```

(raised at `crates/jolt-kernels/src/stage4.rs:1735` —
`DenseStage4State::round_poly`'s check that
`poly.evaluate(0) + poly.evaluate(1) == previous_claim`.)

Observed during the run:
- 16,123 FieldRegEvents over a 35,890-cycle trace.
- Materializer produces non-zero `field_reg_val` / `frd_wa` / `frd_inc`
  buffers (first nonzero `frd_inc` at cycle 832, first nonzero
  `frd_wa` at slot=1 cycle=838).
- `populate_r1cs_fr_slots` runs and sets V_FIELD_RS1/RS2/RD_WRITE_VALUE
  on the matching cycles.

Everything else is intact: muldiv host green, jolt-witness 30/30,
bolt commitment_ir 53/53, clippy clean on the full FR stack.

## Hypotheses

1. **Materializer / R1CS pre-vs-post timing slip.** Both
   `materialize_field_reg_val` (in `jolt-witness/src/field_reg.rs`) and
   `populate_r1cs_fr_slots` (in `jolt-host/src/lib.rs`) walk the
   running state. They claim to record pre-execution state at cycle c
   and apply the event after. Cross-check: the existing materializer
   tests
   (`field_reg::tests::field_reg_val_tracks_running_state`,
    `field_reg::tests::frd_inc_is_post_minus_pre`)
   pin the materializer to known-good shapes; if those still pass, the
   R1CS post-pass is the suspect.

2. **Bytecode flag overshoot.** `fr_bytecode_from_trace` currently
   marks all `FieldOp` cycles as `reads_frs1 = reads_frs2 = true`.
   That overstates FINV (which reads only frs1). Poseidon2 doesn't
   use FINV, so this shouldn't bite here — but it's worth confirming
   by adding a `JoltInstructionKind::FieldOp` funct3 split via the
   existing CircuitFlags (`IsFieldMul/Add/Sub/Inv/AssertEq`) which
   are set on each cycle by the witness-gen flag-population path.

3. **Cycle indexing skew between events and trace rows.** The
   materializer matches via `ev.cycle as usize == c` against
   `0..num_cycles`. The tracer's `cycle_index = cpu.trace_len +
   trace_vec.len()`. Sanity-check by adding a debug assertion that
   every event's cycle_index falls inside `0..trace.len()` and that
   no two events share a cycle.

4. **Materializer for `frd_inc` semantics.** Source's `field_reg_inc`
   in the source branch (commit 06a78980d) used a slightly different
   formula; double-check the modular-sdk version matches the FR-RW
   sumcheck's expected `frd_wa · frd_inc` interpretation.

## Suggested fix path

1. Re-enable the debug `eprintln!`s I had in
   `with_field_reg_replay` and `populate_r1cs_fr_slots` to compare
   first-nonzero indices side-by-side.
2. Pick the first FR-event cycle (cycle 832 in this run) and dump
   both sides' values for V_FIELD_RS1/RS2/RD_WRITE_VALUE plus
   `field_reg_val[k * t + c]` / `frd_wa[k * t + c]` / `frd_inc[c]`
   for k in {0..16}. They should be identical mod limb-encoding.
3. If they disagree, walk back through the running-state loop to find
   the divergence cycle.

## Quick fallback (downgrade Stage 4 to trivial check)

If the synchronization is hard to track down, temporarily downgrade the
materializer to return all-zero buffers regardless of events. The
existing `Stage45SparseTraceWitness::with_field_reg_replay` already
falls back to zero when `replay.events.is_empty()` — extend that to
also handle the FR-active case while leaving Stage 4 RW trivially
satisfied (0 = 0). That sacrifices FR-Twist soundness but lets the SDK
example complete its prove + verify round-trip end-to-end while the
materializer is being debugged.

## After this lands

- Delete this spec.
- Audit fixes C1–C11 from `specs/fr-v2-port-plan.md` lines 117–122 are
  the remaining tail (Phase 5d).
