# PERF-5 lane 1 — HyperKZG folds and honest gate clock

Date: 2026-09-03. Base: `e3dad486d`. Code commit: `080b0c9f2`.
Machine: Mac mini M4, 10 Rayon threads. Fixture:
`fibonacci_2_18_blake3.bin`, `k = 32`, `N = 2^23`.

## Result

- Honest online wall: **~40.3 s -> 37.523 s**.
- HyperKZG fold commitments: **8.250 s -> 5.760 s**, or
  **0.686705 us/point** over `N - 2` committed points. The `<=4.5 s` lane
  target was not met.
- Claimed-point evaluation: **0.509 s -> 0**. `HyperKZG::open` checks the
  supplied claim against its final two-entry fold.
- Proof: **7,488 B payload / 7,628 B bincode / 352 B statement**.
- Verifier: **127,884 Fr mul / 5,048,805 gas**, unchanged.

## Idle gate

The measurement lock was acquired before waiting. Command-start load was
`3.95 / 11.31 / 20.64`. The online clock starts after deterministic setup,
reference layout, verifier keys, and negative-key setup; its reported loads
were `8.72 / 10.49 / 18.98` at start and `9.94 / 10.57 / 18.66` at end. The
offline work immediately before the online interval accounts for the higher
one-minute load shown there.

| clock | value |
|---|---:|
| honest online wall | 37.523 s |
| online phase sum | 37.518 s |
| gap | 0.005 s / 0.013% |
| process CPU | 274.380 s |
| CPU / wall | 7.312 |

### Printed phases

The before column is the idle planning run in `plan-prover-time.md`. Its old
`adapt_r` bucket mixed online work with reference/key construction; that row's
delta is a clock correction, not a production saving.

| phase | before ms | after ms | delta ms |
|---|---:|---:|---:|
| deterministic SRS (offline) | 7,867 | 8,161 | +294 |
| hash key/profile (offline) | 409 | 160 | -249 |
| wrapper preparation | 595 | 550 | -45 |
| `adapt_r` / T1-R stream setup | 2,778 | 270 | -2,508 |
| T2 adaptation | 1,353 | 1,261 | -92 |
| fixed-key commitments (offline) | 1,034 | 973 | -61 |
| phase 1a commitment | 1,783 | 1,945 | +162 |
| T2 phase 1b commitment | 1,148 | 1,050 | -98 |
| T2 phase 2a commitment | 7,435 | 7,271 | -164 |
| T2 phase 2b commitment | 101 | 101 | 0 |
| CopyLink helpers | 2,890 | 2,960 | +70 |
| T2 phase 2c + helpers | 348 | 344 | -4 |
| T2 finish | 481 | 457 | -24 |
| member construction | 1,766 | 1,983 | +217 |
| proof stages/opening | 22,406 | 19,326 | -3,080 |
| verifier | 25 | 25 | 0 |

### HyperKZG split

| phase | before ms | after ms | delta ms |
|---|---:|---:|---:|
| fold materialization | 46 | 94 | +48 |
| fold commitments | 8,250 | 5,760 | -2,490 |
| claimed-point evaluation | 509 | 0 | -509 |

The fold commitments now run sequentially at the outer level; each
`kzg_commit` retains its internally parallel MSM. The measured MSM sequence is
slower than the plan's quotient-rate projection, so no `<=4.5 s` claim is
made.

## Review-4 minors

1. The real gate pins the complete challenge-count vector
   `[39, 23, 1, 3, 232]`.
2. `state_in` and the statement public segment are coupled by
   `hash_public_statement`; the key constructor cannot change one while
   retaining the other. At fixed T1 challenges, changing `state_in` changes
   the wiring input claim while all pinned commitments stay fixed. The
   end-to-end changed-statement negative rejects at
   `SpartanError::OuterFinalClaim`.
3. `T2Challenges::from_transcript` is crate-private. The matrix-only verifier
   counter was removed; total `fr_mul` accounting is unchanged.

## Gates

| command | result |
|---|---|
| `cargo check -p jolt-wrapper --all-targets --features prover-fixtures` | pass |
| `cargo fmt` | pass |
| wrapper + HyperKZG clippy, all targets, warnings denied | pass |
| `cargo nextest run -p jolt-hyperkzg -p jolt-wrapper --cargo-quiet` | 89/89 pass |
| locked `real_wrapper --no-capture` | 1/1 pass; all tampers reject |

The temporary fold timer was removed after the locked run.
