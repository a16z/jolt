# Akita Fp128 deferred-accumulator results

Date: 2026-07-31 EDT

Machine: Apple M4 Max

## Outcome

The D128 commitment candidate is accepted in Jolt commit `12427d2d0`. It
reduces the exact `T = 2^28` root-accumulation span from 84.75 to 59.73
seconds, a 29.5% improvement, and reduces the complete prover from 190.42 to
163.60 seconds.

The general Fp128 delayed-product flag is rejected as a standalone PIOP
optimization. It helps long dot products but not the prover's dominant
two- and three-product shapes.

## Accepted D128 carry deferral

The previous D128 kernel accumulated each shifted coefficient in canonical
Fp128 form. Every add or subtract therefore applied the pseudo-Mersenne
correction. The new task-local ring stores two wrapping `u64` limbs and one
signed `i16` wrap counter per coefficient:

```text
exact sum = low128 + wraps * 2^128
          = low128 + wraps * C (mod 2^128 - C).
```

Carry and borrow tracking is enough inside the tile. Canonical reduction and
the `wraps * C` correction happen once when each A rank is flushed.

The 8,192-row tile statically bounds `|wraps|` by 8,192. The implementation
uses 18 bytes per coefficient rather than 16:

| Active rank state | Bytes |
|---|---:|
| Canonical Fp128: `29 * 128 * 16` | 59,392 |
| Carry-deferred: `29 * 128 * 18` | 66,816 |
| Difference per running task | 7,424 |

This is task-local scratch, not persistent proof state. Even with 16 concurrent
workers, the analytical increase is under 0.12 MiB. Setup, witness, proof, and
opening-hint sizes are unchanged.

### Focused kernel

The probe used the exact production operation count: 8,192 rows, two D128
source rings per row, 29 destination columns, and uniformly distributed K256
shifts. Every output was checked against canonical ring accumulation.

| Trial | Canonical | Carry-deferred | Ratio |
|---:|---:|---:|---:|
| 1 | 17.624 ms | 11.827 ms | 0.671 |
| 2 | 17.649 ms | 11.931 ms | 0.676 |
| 3 | 17.525 ms | 11.918 ms | 0.680 |
| 4 | 17.468 ms | 11.900 ms | 0.681 |
| 5 | 17.474 ms | 11.768 ms | 0.673 |
| 6 | 17.520 ms | 11.883 ms | 0.678 |

The warmup-excluded medians are approximately 17.52 and 11.88 ms, a 32.2%
kernel reduction.

### Complete D128 commitment

An adjacent pair used Jolt `cb5589fbc`, Akita `a56b933c`, `T = 2^28`, K256,
D128, 29 semantic columns, and the production planner.

| Variant | Complete commitment | Change |
|---|---:|---:|
| Canonical parent | 167.698 s | — |
| Carry-deferred candidate | 113.764 s | -32.16% |

This isolated benchmark includes the complete streaming commitment rather than
only the accumulation subspan. Both variants were built with the same release
profile and dependency revisions.

### Full prover

The comparison parent is the immediately preceding accepted fold-wave trace.

| `T = 2^28` metric | Parent | Carry-deferred | Change |
|---|---:|---:|---:|
| `trace_onehot_commit_accumulate` | 84.75 s | 59.73 s | -29.5% |
| Complete commitment | 85.06 s | 60.05 s | -29.4% |
| Evaluation proof | 26.91 s | 26.37 s | -2.0% |
| Complete prover | 190.42 s | 163.60 s | -14.1% |
| Maximum RSS | 79.311 GiB | 78.708 GiB | -0.603 GiB |
| Process swaps | 0 | 0 | unchanged |

The proof verified. The evaluation-proof movement is treated as run noise, not
as a gain from this commitment-only change. RSS did not regress; its observed
decrease is much larger than the sub-megabyte analytical scratch difference
and is likewise not attributed to the patch.

The policy selects D128 only when the packed K256 polynomial has at least 41
variables. With the current 32-column prefix this begins at `T = 2^28`.
`T = 2^26` and `T = 2^27` remain on D64 and do not execute this kernel.

## Rejected delayed product sums

The experimental Akita field change made full Fp128 product accumulation exact
over batches smaller than `2^64` and passed all field/prover differential
tests. Long dot products improved substantially:

| Terms | Eager canonical | Delayed product |
|---:|---:|---:|
| 16 | 43.785 ns | 27.297 ns |
| 64 | 173.65 ns | 90.55 ns |
| 256 | 700.94 ns | 349.22 ns |

The actual small prover shapes did not:

| Terms | Eager canonical | Delayed product | Change |
|---:|---:|---:|---:|
| 2 | 5.810 ns | 7.328 ns | +26.1% |
| 3 | 8.509 ns | 8.986 ns | +5.6% |
| 4 | 11.222 ns | 10.503 ns | -6.4% |

The exact digit-range and sparse extension-ring benchmarks were flat, and a
paired traced `T = 2^22` proof moved the backend batched prover by only 0.8%.
The product flag is therefore not pinned. Experimental Akita commit
`13bdc58d` remains isolated and unpushed for reproduction.

## Validation

- full-bound carry/borrow differential test over 8,192 shifts;
- accumulator-clear and reuse test;
- D128/K256 commit/open round trip;
- all 49 enabled `jolt-akita` tests;
- natural, forced-K256, and committed-program Akita muldiv proofs;
- standard and ZK Dory muldiv suites;
- exact `T = 2^28` proof verification;
- maximum RSS below 79 GiB with zero swaps;
- scoped `jolt-akita` warning-denying Clippy;
- workspace warning-denying Clippy with `host` and `host,zk`;
- `cargo fmt --check` and `git diff --check`.

## Traces

| Trace | Purpose | SHA-256 |
|---|---|---|
| `benchmark-runs/perfetto_traces/akita_28_fold_waves.json` | accepted parent | `bd8f1970c81b02596aebc5cbe2a3ebf7e7eee865022cde9eae20820f8dc9ca2f` |
| `benchmark-runs/perfetto_traces/akita_28_deferred_carries.json` | accepted candidate | `5c2da1657128cf77c7b826f1ee9036c109eb565d2d34e59a7489a635f6047f38` |
| `benchmark-runs/perfetto_traces/akita_22_fp128_delayed_baseline.json` | rejected product-sum parent | `af8aa13328abb45ef2d73400b020e1122cdda86b14082d3c927a2dfc24fa9a1a` |
| `benchmark-runs/perfetto_traces/akita_22_fp128_delayed_candidate.json` | rejected product-sum candidate | `b2b7e49c7dab59a292213a970a7309ca50dda8c02aefd83f555f37e506a771a7` |
