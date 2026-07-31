# Akita Stage 5 memory-traffic results

Date: 2026-07-30 EDT

## Outcome

One structural optimization landed as `f25ca8e65`: the instruction read-RAF
prover now stores its per-table cycle buckets as `u32` indices instead of
64-bit `usize` values.

At `T = 2^28`, the exact retained bucket capacity falls from
1,689,155,368 to 844,577,684 bytes, a reduction of 805.45 MiB. The buckets
are live through the address phases of Stage 5 and are reread by every suffix
preparation. Widening an index at the final vector access does not add a pass
or another allocation.

| `2^28` metric | Parent | Compact buckets | Change |
|---|---:|---:|---:|
| Grouped bucket bytes | 1,689,155,368 | 844,577,684 | -50.0% |
| Read-RAF initialization | 2.487 s | 2.156 s | -13.3% |
| Suffix preparation | 4.386 s | 4.312 s | -1.7% |
| All read-RAF phase initialization | 8.403 s | 8.382 s | -0.3% |
| Stage 5 | 15.726 s | 15.479 s | -1.6% |
| Prover | 189.876 s | 194.002 s | not attributed |
| Maximum RSS | 80.781 GiB | 81.513 GiB | run variance |
| Process swaps | 0 | 0 | unchanged |

The proof verified. Unchanged commitment and opening spans were 3.03 seconds
slower in the candidate run, so the 4.13-second whole-prover movement is not
evidence against the Stage-5 change. Likewise, the process high-water occurs
outside this bucket lifetime and moved opposite the exact live-owner cut.

The previous D128 Stage-5 ownership ceiling was 67.846 GiB. Removing
0.786574 GiB changes it to approximately 67.059 GiB, or from about 271.38 to
268.24 analytical bytes per cycle at `2^28`.

At `2^26`, read-RAF initialization fell from 0.697–0.701 seconds in the two
controls to 0.587 seconds. Stage 5 remained inside the control range, the
proof verified, maximum RSS was 36,258,349,056 bytes, and the process
reported zero swaps.

## Rejected candidates

### Source-aware increment zero lanes

The prover identified all-zero raw increment subcubes without comparing field
elements, then factored their common Booleanity term. Compute plus ingest was
6.3% and 1.9% slower in two `2^22` screens. The branch and source inspection
cost outweighed the removed products, so the source was restored.

### Fused read-RAF condensation

Seven post-phase `u_evals` updates were moved into the existing RAF-Q scan.
The old Q-plus-update total was 78.737 ms; the fused Q scan was 78.744 ms.
This moved rather than removed the cost and did not justify a larger run.

## Validation

- randomized instruction read-RAF sumcheck test;
- standard, ZK, and Akita muldiv suites;
- exact K256 proofs at `2^22`, `2^26`, and `2^28`;
- warning-denying clippy for the Akita target and both required workspace
  feature modes;
- `cargo fmt` and `git diff --check`.

## Retained traces

- `benchmark-runs/perfetto_traces/akita_stage6_increment_zero_22a.json`
- `benchmark-runs/perfetto_traces/akita_stage6_increment_zero_22b.json`
- `benchmark-runs/perfetto_traces/akita_readraf_fused_condense_22a.json`
- `benchmark-runs/perfetto_traces/akita_readraf_u32_buckets_22a.json`
- `benchmark-runs/perfetto_traces/akita_readraf_u32_buckets_22b.json`
- `benchmark-runs/perfetto_traces/akita_readraf_u32_buckets_26.json`
- `benchmark-runs/perfetto_traces/akita_28_u32_buckets.json`
