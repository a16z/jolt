# Modular-stack perf attack: architectural tickets

This folder holds self-contained specs for the architectural attacks on the
~18× sumcheck-inner-loop gap between the modular stack and `jolt-core` on
sha2-chain @ log_T=16. Each ticket is designed to be picked up with a fresh
context window; the ticket itself carries enough background, file pointers,
and exit criteria to execute without rereading the rest of the codebase.

## Why this folder exists

The Perf Loop Protocol (CLAUDE.md "Perf Loop Protocol") optimizes per-iter
by reading a trace, picking the top span, and attacking it — great for
kernel-level wins but it was churning on reverts when the remaining gap is
architectural, not kernel-local. The rigorous decomposition lives at
`perf/report_tools/kernel_gap_memo.md` (sumcheck inner loop = 54.9% of
modular CPU, 28.7× over core). That memo projects a 6.4× CPU reduction if
the four architectural fixes below land — enough to close modular wall
time to ~3× of core.

This folder turns the memo's fixes into independently-executable tickets
that preserve the ML-compiler philosophy (compiler emits unrolled
primitives; backend fuses).

## Design principles (carried into every ticket)

1. **Handlers stay ≤30 LOC and protocol-unaware** (`CLAUDE.md` Task Loop
   Protocol). Fusion happens in the backend via a rewrite pass, never
   inside a handler.
2. **Compiler emits unrolled primitives.** Tickets never change what the
   compiler emits; they add new `Op` variants and a backend-side pass that
   rewrites the emitted stream.
3. **Fusion pass is opt-in per backend.** Default `ComputeBackend::fuse_ops`
   is the identity. Only CPU opts in for ticket 1/2. Other backends (a
   future GPU) write their own passes.
4. **Dual-path correctness gate before perf gate.** Every ticket ships a
   debug-only mode that runs the un-fused + fused paths in parallel and
   asserts evals equal. Flipped on via `JOLT_FUSE_DEBUG=1` env var and off
   in release. Ticket 0 installs the harness.
5. **Perf gate lives in the existing Perf Loop.** Each ticket's exit
   criteria map onto `perf/baseline-modular-best.json` with the same
   `--program sha2-chain --log-t 16` measurement.

## Ticket index

| # | Title | Depends on | Est. CPU win | File |
|---|-------|------------|--------------|------|
| 0 | Dual-path fusion validation harness | — | 0 (prereq) | [00-validation-harness.md](00-validation-harness.md) |
| 1 | Reduce-side fusion: `fuse_ops` hook + CPU `batch_round_evaluate` override | 0 | ~47s (factor B+partial C) | [01-reduce-fusion.md](01-reduce-fusion.md) |
| 2 | Bind-side fusion: `Op::BatchBind` + CPU fused bind | 0 | ~19s (factor C continued) | [02-bind-fusion.md](02-bind-fusion.md) |
| 3 | Persistent per-Executable state (scratch + shared eq handle) | 1 | ~4s (factor A) | [03-persistent-state.md](03-persistent-state.md) |
| 4 | Shape-preserving Compact/OneHot tags on `DeviceBuffer` | 1, 3 | ~15s (factor D) | [04-shape-preserving-bufs.md](04-shape-preserving-bufs.md) |

Order is **topological**, not strictly-linear: 1 and 2 can proceed in parallel
after 0, 3 and 4 unlock after 1 lands. Picking 1 first maximizes win-per-line.

## Shared glossary

- **Sumcheck inner loop** — per-round `reduce` (eval at grid points) +
  `bind` (in-place fold) calls across N batched instances. The hot loop.
- **Batch/batch-round** — `Op::BatchRoundBegin .. BatchRoundFinalize`
  range; the compiler emits one per protocol round. ~120 instances × 20
  rounds for sha2-chain log_T=16.
- **InstanceReduce / InstanceSegmentedReduce** — single-instance reduce
  primitives the compiler emits today; targets for ticket 1.
- **BatchRoundEvaluate** — variable-arity reduce op that already exists
  in `Op` enum (`jolt-compiler/src/module.rs:1264`) with a handler and a
  byte-identical trait default. Target sink for ticket 1's fusion pass.
- **InstanceBind** — single-instance bind primitive; target for ticket 2.
- **BatchBind** — does NOT exist yet; ticket 2 adds it.
- **`fuse_ops`** — trait method on `ComputeBackend` added by ticket 1;
  default is identity. Analogous to XLA `HloFusion` pass.

## Invariants every ticket preserves

1. Handler in `crates/jolt-zkvm/src/runtime/handlers.rs` stays ≤30 LOC.
2. No change to the compiler emission logic in
   `crates/jolt-compiler/src/builder.rs`. Tickets only add Op variants and
   backend-side passes.
3. `modular_self_verify`, `transcript_divergence`, `zkvm_proof_accepted`
   pass at every intermediate commit.
4. `cargo clippy -p jolt-core --features host,zk` clean.
5. If a ticket regresses or is flat, the perf-loop commit is `journal:
   <ticket> reverted (<reason>)` per CLAUDE.md Perf Loop protocol.

## Current baseline numbers

Refresh once at the start of each ticket (see `perf/last-iter.json`):

- `perf/baseline-core.json` — frozen jolt-core reference
  (sha2-chain log_T=16, prove_ms ≈ 3900)
- `perf/baseline-modular-best.json` — monotone ratchet
  (as of 2026-04-22: prove_ms ≈ 72638, ratio ≈ 18.7×)
- `perf/report_tools/kernel_gap_memo.md` — rigorous CPU decomposition

## Known prerequisite: clippy block on jolt-witness

`crates/jolt-witness/src/polynomials.rs:294-295` has two clippy errors
(identity ops — `* 0` and `+ 0` or similar that clippy flagged as "always
returns zero" / "has no effect"). These block the correctness gate
(`cargo clippy ... -D warnings`) so every ticket must either fix them as
a first step or depend on a separate fix-up commit. Easiest path: open the
file, reduce the identity ops clippy suggests.

## How to resume a ticket in a fresh context

1. Read this README.
2. Read the ticket file you're attacking (it's self-contained).
3. Read `perf/report_tools/kernel_gap_memo.md` §1 (bottleneck table) and
   the specific factor the ticket attacks.
4. Re-run the baseline measurement:
   ```
   cargo run --release -p jolt-bench -- --program sha2-chain \
     --num-iters 16 --log-t 16 --iters 1 --warmup 1 \
     --json perf/last-iter.json
   ```
5. Run correctness gate once before starting (so any regression is
   attributable to your change):
   ```
   cargo nextest run -p jolt-equivalence transcript_divergence
   cargo nextest run -p jolt-equivalence modular_self_verify
   cargo clippy -p jolt-core --features host,zk -- -D warnings
   ```
6. Implement following the ticket's step list. Commit once at end per
   Perf Loop commit discipline.
