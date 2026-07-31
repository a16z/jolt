# Akita D128 commitment: narrow rank accumulators

Date: 2026-07-30 EDT

Machine: Apple M4 Max

Workload: forced-K256 `sha2_chain_akita_perf`, `T = 2^28`

## Result

Commit `309a81df4` changes only the D128/K256 rank-tiled root kernel. D64
continues to use the carry-free `Fp128x8i32` accumulator.

For D128, each rank now accumulates directly into canonical Fp128 ring
coefficients. This changes the active rank state from

```text
29 columns * 128 coefficients * 32 bytes = 118,784 bytes (116 KiB)
```

to

```text
29 columns * 128 coefficients * 16 bytes = 59,392 bytes (58 KiB).
```

The new loop also avoids expanding every source coefficient from one Fp128
value into eight i32 limbs. Each shifted source and destination traversal
therefore moves half as many accumulator bytes. The cost is a canonical
Solinas add or subtract for each coefficient instead of a carry-free NEON
add followed by one reduction at the end of the tile.

## Measurements

| Run | Root accumulation | Commitment | Evaluation proof | Prover | Maximum RSS |
|---|---:|---:|---:|---:|---:|
| Frozen parent | 87.61 s | 87.93 s | 33.32 s | 199.25 s | 83.812 GiB |
| Narrow observed | 81.91 s | 82.21 s | 31.73 s | 190.82 s | unavailable |
| Narrow recheck | 78.59 s | 78.89 s | 32.20 s | 188.52 s | 82.496 GiB |
| Adjacent wide control | 86.35 s | 86.66 s | 33.01 s | 198.05 s | 80.930 GiB |
| Narrow final | 82.49 s | 82.80 s | 32.92 s | 193.70 s | 81.080 GiB |

The final adjacent pair improves root accumulation by 4.47%. All three
narrow observations beat both controls; their median is 81.91 seconds,
5.14% below the adjacent control. Relative to the previously accepted final
trace, the final narrow run improves root accumulation by 5.84% and whole
prover time by 2.79%.

The result is accepted for this M4 workload as a prover-time optimization.
It is revalidated, but not transferred to another machine or workload. It
is not credited with a global RSS reduction: the final candidate is 0.15
GiB above the adjacent control, the recheck is 1.57 GiB above it, and both
are below the 83.812 GiB frozen baseline. Those differences are much larger
than the sub-megabyte analytical reduction across concurrently active rank
accumulators and reflect ordinary whole-process RSS variation. Every
RSS-capable run stayed below 90 GiB and reported zero process swaps.

## Why the trade wins at D128

The wide D128 rank state is just small enough to suggest L1 residency, but
not after including the current source ring, decoded lanes, masks, loop
state, and neighboring worker activity. Direct Fp128 accumulation cuts the
rank state from 116 to 58 KiB and halves the bytes touched for every shifted
coefficient. On this M4 Max, that saved more time than the optimized Fp128
carry chains cost.

This does not imply the same policy for D64. Its wide rank state is already
58 KiB, and D64 remains selected through the `2^27` packed shape. The
implementation therefore dispatches to narrow accumulation only when
`D == 128`.

## Validation

- 48 `jolt-akita` tests passed, including the D128/K256 round trip.
- Natural, forced-K256, and committed-program Akita muldiv proofs passed.
- The exact `2^28` candidate proof passed and verified three times.
- Standard and ZK muldiv suites passed.
- Both required clippy modes and `cargo fmt` passed.
- `git diff --check` passed.

## Traces

| Trace | Purpose |
|---|---|
| `benchmark-runs/perfetto_traces/akita_28_d128_reduced_accumulator.json` | first narrow observation |
| `benchmark-runs/perfetto_traces/akita_28_d128_reduced_recheck.json` | narrow RSS recheck |
| `benchmark-runs/perfetto_traces/akita_28_d128_wide_control.json` | adjacent restored-parent control |
| `benchmark-runs/perfetto_traces/akita_28_d128_reduced_final.json` | accepted final narrow run |

Raw logs and parsed span tables are under
`benchmark-runs/akita-d128-commit-loop-2026-07-30/logs`.
