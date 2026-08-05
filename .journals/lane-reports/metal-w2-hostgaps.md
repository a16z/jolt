# Metal wave 2: host-serial gaps

## Audit

Ranked by the largest attributable host-serial span in the fresh `2^27`
trace:

| rank | stage / routine | isolated wall | disposition |
|---:|---|---:|---|
| 1 | st4 `RegistersRWC::prepare` and scan/allocation boundaries | 2.449 s prepare; 4.761 s total sampled-zero | Larger ceiling, but the repeated scan/bind boundary is not one isolated routine. |
| 2 | st3 `InstructionInput::prove_batch` round 0 | 2.140 s | Mixed host message construction and dense device write; no reusable isolated kernel. |
| 3 | st7 Hamming-weight pushforward preparation | 1.887 s | Selected: 99.7% of st7, one transcript-free routine, with an existing exact split-eq implementation to reuse. |

## Isolated objective

Added `jolt-eval` objective/bench `hamming_weight_pushforward`:

- `2^22` real 48-byte `InstructionCycleRow` values;
- production `2^27` one-hot geometry: 8-bit chunks and 16/2/3
  instruction/bytecode/RAM columns;
- deterministic hot/cold bytecode and RAM distribution;
- setup-time oracle: full `eq(r, ·)` materialization plus direct per-row
  scatter, compared exactly against every output bucket before timing;
- timed body: the stage-7 pushforward only; no prover or transcript.

## Change

Extracted stage 6a's one-hot selector and split-eq deferred-bucket algorithm
into `optimized::one_hot_pushforward`. Booleanity keeps its balanced split.
Stage 7 chooses four outer blocks per Rayon worker: inner eq weights accumulate
by addition, then one outer multiplication is applied per reduced bucket.
This removes the old per-row eq multiplication while retaining enough blocks
for work stealing.

## Timing decision

Criterion, 10 flat samples per arm, 12 s measurement window:

| arm | median | 95% median CI |
|---|---:|---:|
| before | 79.816671 ms | 78.355649–80.282463 ms |
| after | 46.795669 ms | 46.535044–47.512235 ms |

Result: **33.021002 ms removed, 41.3711% reduction, 1.7056x speedup**.
The confidence intervals do not overlap; retain.

Applying the isolated ratio to the measured 1.887 s stage-7 prepare estimates
1.106 s after, **0.781 s stage gain**. Against the 71.77 s flagship proof,
the estimated whole-proof gain is **1.09%** (`71.77 -> 70.99 s`). This is a
stage-calibrated estimate, not an end-to-end measurement.

## Verification

- `cargo check --message-format=short -p jolt-eval`
- targeted `cargo nextest`: Hamming reduction reference parity plus three
  booleanity address parity geometries, **4/4 passed**
- `cargo clippy -p jolt-eval --all-targets ... -D warnings`
- `cargo fmt --all`
- no end-to-end prover run, per lane scope
