# Ticket 0 — Dual-path fusion validation harness

**Status:** prereq for tickets 1 & 2.
**Est. perf win:** 0 (infrastructure).
**Est. effort:** small (~150 LOC + one test).
**Philosophy check:** harness is debug-only; runtime hot path unchanged in release.

## Why this exists

Tickets 1 and 2 rewrite the emitted `Op` stream (fusion pass) and override
backend primitives with fused implementations. Sumcheck evals are sensitive
to ordering, rounding, and accumulation — a subtle divergence inside
`batch_round_evaluate` won't show up until the BlindFold transcript
mismatch hundreds of rounds later, which is painful to debug.

This ticket installs a cheap, always-available sanity check: in a debug
mode flipped on by `JOLT_FUSE_DEBUG=1`, the runtime runs **both** the
un-fused op stream and the fused stream, and asserts every per-instance
eval vector produced by `BatchRoundEvaluate` equals what the un-fused
`InstanceReduce` / `InstanceSegmentedReduce` sequence would have produced.

This is the same "dual-path validation" pattern called out in CLAUDE.md
Task Loop Protocol step 5. Keeping the old path live until we've proven
equality is the cheapest form of insurance.

## Architectural change

Introduce a `FuseDebugMode` on the `Executable` struct. When enabled:

1. The executable stores BOTH the original ops vec AND the fused ops vec.
2. Before `execute` runs a batch-round, it runs both the pre-fusion and
   post-fusion sub-sequences against cloned copies of the compute state.
3. After both finish, it compares `state.last_round_instance_evals` vs.
   the un-fused shadow state's `last_round_instance_evals`, asserting
   element-wise equality.
4. On mismatch: panic with (batch_id, round, instance_idx, actual, expected).

When disabled (default): runtime executes only the fused stream. Zero
overhead in release.

Shape on disk:

```rust
// crates/jolt-compiler/src/executable.rs  (new module or extend existing)
pub struct Executable<B: ComputeBackend> {
    pub ops: Vec<Op>,               // post-fusion (or pre-fusion if fuse is identity)
    pub shadow_ops: Option<Vec<Op>>, // pre-fusion, populated iff fuse_debug
    pub kernels: Vec<B::CompiledKernel<F>>,
    // ... existing fields
}

pub enum FuseDebugMode {
    Off,
    On { tolerance: Option<usize> },  // element-wise exact unless tolerance set
}
```

Read `FuseDebugMode::from_env()`:
- `JOLT_FUSE_DEBUG=1` → `On { tolerance: None }` (exact equality).
- unset → `Off`.

## Code-level sketch

### 1. `Executable` construction (crates/jolt-compiler/src/compiler/mod.rs)

Wherever `Executable` is constructed today (find via
`grep -n "Executable {" crates/jolt-compiler/src/`), wrap the emitted ops
with the backend's `fuse_ops` and branch on debug mode:

```rust
let raw_ops = builder.finish();  // what compile() produces today
let fused_ops = backend.fuse_ops(&raw_ops);

let (ops, shadow_ops) = match FuseDebugMode::from_env() {
    FuseDebugMode::Off => (fused_ops.unwrap_or(raw_ops), None),
    FuseDebugMode::On { .. } => match fused_ops {
        Some(f) => (f, Some(raw_ops)),
        None => (raw_ops.clone(), Some(raw_ops)),
    },
};
```

**Note:** Ticket 1 adds `fuse_ops` to the `ComputeBackend` trait with a
default identity impl. This ticket assumes ticket 1's trait surface is in
place (or adds a stub).

### 2. Runtime dual-path execution (crates/jolt-zkvm/src/runtime/)

The simplest implementation: bracket the whole `execute` loop with a
shadow run. For each `[BatchRoundBegin .. BatchRoundFinalize]` window,
run the window under the shadow ops first, snapshot
`state.last_round_instance_evals`, then run the window under the fused
ops, and compare.

Alternatively — and cheaper — interpose at `Op::BatchRoundFinalize`:

```rust
// in the Op::BatchRoundFinalize handler
#[cfg(debug_assertions)]
if let Some(shadow_ops) = &executable.shadow_ops {
    let shadow_evals = run_shadow_window(shadow_ops, batch, round, ...);
    for (i, evals) in state.last_round_instance_evals.iter().enumerate() {
        assert_eq!(evals, &shadow_evals[i],
            "fuse divergence at batch={} round={} instance={}", batch.0, round, i);
    }
}
```

Shadow window runs a cloned state through the pre-fusion ops up to the
matching `BatchRoundFinalize`. The clone is expensive but only paid in
debug mode; not in the hot path.

### 3. Test coverage

New integration test in `crates/jolt-equivalence/tests/fuse_equivalence.rs`:

```rust
#[test]
fn fuse_debug_mode_agrees_with_unfused() {
    std::env::set_var("JOLT_FUSE_DEBUG", "1");
    // drive a small prover run (e.g. the existing modular_self_verify harness)
    // — the harness asserts internally, so this test is "doesn't panic"
    jolt_equivalence::modular_self_verify::run_small_program();
}
```

Keep it small: one program, log_T=6 or similar. The point is the mode
compiles, the harness hooks fire, and evals match for an identity `fuse_ops`.

## Dependencies

- **Prerequisite fix:** unblock clippy on
  `crates/jolt-witness/src/polynomials.rs:294-295` (identity ops) so the
  clippy gate passes on this ticket's commits.
- **Touches `ComputeBackend` trait:** adds `fuse_ops` method with default
  identity. Ticket 1 builds on this, but the trait method lands here.

## Correctness gate

Standard suite from CLAUDE.md Perf Loop Protocol:

```
cargo nextest run -p jolt-equivalence transcript_divergence
cargo nextest run -p jolt-equivalence zkvm_proof_accepted
cargo nextest run -p jolt-equivalence modular_self_verify
cargo nextest run -p jolt-equivalence
cargo clippy -p jolt-core --features host --message-format=short -q --all-targets -- -D warnings
cargo clippy -p jolt-core --features host,zk --message-format=short -q --all-targets -- -D warnings
```

Plus the new fuse-equivalence test.

## Perf gate

None — this is infrastructure. Ticket 1 measures the delta.

But: run the bench once with and without `JOLT_FUSE_DEBUG=1` to confirm
debug mode does NOT affect the release-mode prove time. If debug mode is
accidentally left on in a non-debug build, perf will crater — the test
should make this impossible.

## Exit criteria

- [ ] `ComputeBackend::fuse_ops` method added with default identity impl.
- [ ] `FuseDebugMode` enum + env var detection wired into `Executable`
      construction.
- [ ] Runtime handler interposes dual-path assertion at
      `BatchRoundFinalize` in debug mode.
- [ ] `fuse_equivalence.rs` test passes with identity fuse (sanity).
- [ ] `fuse_equivalence.rs` test with a toy non-identity fuse (e.g. noop
      that re-emits the same ops in a different order where legal) also
      passes — confirms the harness actually catches divergence when the
      fused path is wrong.
- [ ] Full correctness gate green.
- [ ] Release bench unchanged (within ±1%).

## Rollback

Pure additive ticket. Revert = one commit. No runtime-visible change
when env var is unset.

## Open questions

- **Q:** Should we support `JOLT_FUSE_DEBUG=1` in the public `jolt-bench`
  CLI as a `--fuse-debug` flag? Probably yes for convenience.
- **Q:** Tolerance for field-element inequality — do we need any? Sumcheck
  evals should be bit-exact regardless of reduction order (field arithmetic
  is exact), so tolerance should always be `None`. Keep the knob but leave
  it unreachable for now.
