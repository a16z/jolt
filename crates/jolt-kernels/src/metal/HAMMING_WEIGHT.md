# Hamming-weight claim reduction on Metal

Stage 7 reuses the production Booleanity address pushforward to build the
Hamming-weight `G` tables. The shader and selector ABI do not change. The Metal
backend retains the stage-6b `BooleanityRows` allocation through stage 7, evaluates
the pushforward at Hamming's newly sampled cycle point, reads 29 small tables once,
and runs the eight address rounds and Fiat--Shamir on the host.

## Exact computation

For each one-hot column `i` and address `k`, Hamming preparation needs

```text
G_i[k] = sum_j eq(r_cycle, j) [hot_i(row_j) = k].
```

This is the same operation and production selector order as Booleanity address: 16
instruction bytes, two bytecode bytes, two RAM bytes, eight fused-increment bytes,
and the fused-increment carry. The point is not reusable: `r_cycle` has `log_t`
coordinates (26 at the target `2^26` trace) and is known only after stage 6b
finishes. The resident rows and the shader are reusable.

The Akita relation represents the implicit default lane separately. After readback,
the backend sets `G_i[0] = 0` for all 29 tables, matching the optimized kernel. It
then constructs the existing `W_i` tables and baseline delta, and uses the existing
host round kernel. Output openings, round order, challenges, and transcript bytes
remain unchanged.

## Residency and failure behavior

Stage 5 prepares the row plane when any enabled downstream consumer needs it. An
admitted Hamming consumer can be the reason for that producer-side upload; raw PIOP
timing includes the upload. When the Hamming cutoff admits the trace, stage 6b leaves
an `Arc` clone of the `BooleanityRows` allocation for stage 7 even if the Booleanity
cycle itself selects CPU. The clone must have the same allocation identity, row
count, and Metal device registry. Hamming takes that clone, so the 2.5-GiB row plane
is released after its invocation completes.

Missing rows, an ineligible trace, unsupported geometry, or a capacity rejection
before command submission selects the optimized CPU kernel. The Hamming adapter
itself performs no row upload; its equal-input local benchmark starts with the same
pre-existing rows on both arms. A failure after command submission returns a Metal
compute error; it cannot retry after consuming device state.

The stage-5 producer, stage-6b retention path, and stage-7 Metal consumer form one
residency-coupled slot family. `with_metal_compute` installs them coherently. A
backend assembled by replacing slots individually must preserve that family:
installing the producer and retention slots while leaving stage 7 on the optimized
CPU leaves the private row carry allocated until the `ProofSession` is dropped.

The integration phase changed the Hamming host adapter, row lifetime, configuration,
evaluator, and tests without changing the accepted `booleanity_address.metal`
shader. Shader tuning, if measurement requires it, is a separate logged phase.

The implementation map is:

| Requirement | Code owner | Acceptance evidence |
|---|---|---|
| Canonical selectors and bucket-zero recentering | optimized Hamming Metal plan | independent 29-table parity test |
| Stage-6b to stage-7 row lifetime | Metal Booleanity adapter | allocation-identity lifecycle test |
| One pushforward and CPU fallback | Metal Hamming adapter | complete-member parity and fallback tests |
| Unchanged rounds and output mapping | existing optimized Hamming kernel | reference lockstep and transcript parity |
| Performance and resource gates | fixed local evaluator and PIOP evaluator | append-only run plus production validation |

The nested CPU row-source span is implemented. The equal-input production CPU
denominator has not yet been measured; promotion waits for that measurement. The
span does not change the algorithm or the timed Metal boundary.

## Throughput ceiling

At `T = 2^26` and `K = 256`, the retained pushforward presents `29*T = 1.946`
billion selector-row opportunities. The evaluator reports this rate and the smaller
rate of structurally nonzero recentered contributions separately: an opportunity is
excluded from the latter when its optional selector is absent or its selected lane
is bucket zero. With the baseline width of six selectors per tile, the command uses
five tiles; a width `w` uses `ceil(29 / w)`. Logical traffic and owned scratch at the
baseline width are identical to Booleanity address:

| Quantity | Value |
|---|---:|
| Resident-row reads | 12.500 GiB |
| Logical partial write plus read | 0.453 GiB |
| Owned buffers beyond rows | 51,007,720 bytes |
| Result readback | 118,784 bytes |
| Cache-optimistic traffic floor at 420.68 GiB/s | 30.8 ms |

The promoted primitive measured 111.635 ms in the full proof working set. Hamming
adds bucket-zero recentering, `W_i` construction, and about 0.8 ms of host rounds.
Before measurement, 120--150 ms is the planning range and 115--130 ms is the stretch
target; retained-row lifetime and stage-7 thermal state can move it. The latest raw
optimized CPU member median is about 594.2 ms, giving a provisional 118.8-ms 5x
budget. Promotion uses the measured equal-input CPU denominator. Expected PIOP
recovery is 0.40--0.48 s.

## Existing schema-1 evaluator and promotion

The standalone local scalar is the median paired wall-time ratio between a
production-shaped shared-row CPU mirror and the complete Metal member at `2^26`;
larger is better. The production PIOP evaluator supplies the real comparison between
the optimized `PrepareKernel` and the deployed Metal adapter. One excluded warmup
precedes five alternating local pairs. Resident-row construction is outside both
local timed members. The Metal member includes equality-table preparation, command
encoding/submission/completion, readback, recentering, `W_i` construction, all host
rounds, and unattributed timer remainder.

Promotion requires:

- equality of all `29*256` recentered `G` values;
- equality of every round polynomial, challenge, final claim, output opening, and
  transcript state;
- one retained row allocation, one command completion, `ceil(29 / w)` ordered
  tile/finalize pairs for selector width `w` (five at width six), one readback, and
  no Hamming-adapter row upload;
- exact buffer, pipeline, threadgroup, component-accounting, and sample-cardinality
  guards;
- at least five alternating `2^26` pairs and at least 4x local paired speedup;
- a separate five-pair production PIOP validation with all proofs verified.

Four times is this existing-run contract's floor. Tuning continues when the complete
member remains more than 10% above its measured traffic/host floor or a tested
configuration shows a clear gain beyond the noise threshold. The initial foreground
budget is 12 trials or 30 minutes. A clean result below 4x keeps this slot on CPU and
records the failure.
Any fresh v2 successor must instead require at least 5x local speedup and the v2
holdout and transfer evidence; it cannot promote from this schema-1 run.
