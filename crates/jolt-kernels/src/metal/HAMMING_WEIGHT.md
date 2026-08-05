# Hamming-weight claim reduction on Metal

Stage 7 will reuse the production Booleanity address pushforward to build the
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
and the fused-increment carry. The point is not reusable: `r_cycle` is the 26-round
Booleanity output point and is known only after stage 6b finishes. The resident rows
and the shader are reusable.

The Akita relation represents the implicit default lane separately. After readback,
the backend sets `G_i[0] = 0` for all 29 tables, matching the optimized kernel. It
then constructs the existing `W_i` tables and baseline delta, and uses the existing
host round kernel. Output openings, round order, challenges, and transcript bytes
remain unchanged.

## Residency and failure behavior

Stage 5 prepares the row plane when any enabled downstream consumer needs it. When
the Hamming cutoff admits the trace, stage 6b leaves an `Arc` clone of that
`BooleanityRows` allocation for stage 7 even if the Booleanity cycle itself selects
CPU. The clone must have the same allocation identity, row count, and Metal device
registry. Hamming takes that clone, so the 2.5-GiB row plane is released after its
invocation completes.

Missing rows, an ineligible trace, unsupported geometry, or a capacity rejection
before command submission selects the optimized CPU kernel. No row upload is
allowed solely for Hamming. A failure after command submission returns a Metal
compute error; it cannot retry after consuming device state.

The first integration phase may change the Hamming host adapter, row lifetime,
configuration, evaluator, and tests. It may not change the accepted
`booleanity_address.metal` shader. Shader tuning, if measurement requires it, starts
a new logged phase.

The implementation map is:

| Requirement | Code owner | Acceptance evidence |
|---|---|---|
| Canonical selectors and bucket-zero recentering | optimized Hamming Metal plan | independent 29-table parity test |
| Stage-6b to stage-7 row lifetime | Metal Booleanity adapter | allocation-identity lifecycle test |
| One pushforward and CPU fallback | Metal Hamming adapter | complete-member parity and fallback tests |
| Unchanged rounds and output mapping | existing optimized Hamming kernel | reference lockstep and transcript parity |
| Performance and resource gates | fixed local evaluator and PIOP evaluator | append-only run plus production validation |

The only unresolved measurement fact is the equal-input CPU denominator. A nested
CPU row-source span will measure it before promotion; it does not change the
algorithm or the timed Metal boundary.

## Throughput ceiling

At `T = 2^26` and `K = 256`, the retained pushforward performs `29*T = 1.946`
billion useful selector contributions in five selector tiles. Its logical traffic
and owned scratch are identical to Booleanity address:

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
optimized CPU member median is about 594.2 ms, giving a provisional 148.5-ms 4x
budget and a 118.8-ms 5x target. Promotion uses the measured equal-input CPU
denominator. Expected PIOP recovery is 0.40--0.48 s.

## Fixed evaluator and promotion

The local scalar is the median of paired optimized-CPU member wall time divided by
complete Metal-hybrid member wall time at `2^26`; larger is better. One excluded
warmup precedes five alternating pairs. Resident-row construction is outside both
timed members. The Metal member includes equality-table preparation, command
encoding/submission/completion, readback, recentering, `W_i` construction, all host
rounds, and unattributed timer remainder.

Promotion requires:

- equality of all `29*256` recentered `G` values;
- equality of every round polynomial, challenge, final claim, output opening, and
  transcript state;
- one retained row allocation, one command completion, five ordered
  tile/finalize pairs, one readback, and no new row upload;
- exact buffer, pipeline, threadgroup, component-accounting, and sample-cardinality
  guards;
- at least five alternating `2^26` pairs and at least 4x local paired speedup;
- a separate five-pair production PIOP validation with all proofs verified.

Four times is a floor. Tuning continues when the complete member remains more than
10% above its measured traffic/host floor or a tested configuration shows a clear
gain beyond the noise threshold. The initial foreground budget is 12 trials or 30
minutes. A clean result below 4x keeps this slot on CPU and records the failure.
