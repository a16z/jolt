# Akita large-trace follow-up screens

Date: 2026-07-31
Machine: Apple M4 Max
Branch: `perf/packed-onehot`

## Objective

Screen the next trace-backed Akita prover and memory candidates after
task-local D128 opening rotations. Promote only changes that improve the
affected prover span without increasing the analytical peak or retained
trace-scaled state. A memory candidate must remove a named live owner and
must not reproducibly slow the affected stages.

## Frozen workload

- `sha2_chain_akita_perf`
- K256 (`PERF_LOG_K_CHUNK=8`)
- instruction RA virtual chunk size 32
- Akita fp128 field and the accepted D128 policy at `T = 2^28`
- release build, proof verification enabled

Kernel and arithmetic candidates are screened at `T = 2^22`. Planner
candidates use the exact packed opening layout with 41 variables and one
physical polynomial. A candidate advances to `T = 2^26` only after a positive
affected-span signal at `T = 2^22`.

## Decision rules

1. Keep protocol, transcript, claims, and verifier behavior unchanged unless
   the experiment explicitly targets a protocol change.
2. Require focused proof or output equivalence before timing.
3. Attribute performance to directly affected spans; whole-proof motion is
   supporting evidence only.
4. Reject a memory cut if its affected-stage regression is larger than the
   work it removes.
5. Revert rejected production code completely.
6. Retain named traces and record their SHA-256 digests.
