# Akita per-column default-lane probe

## Question

For the single prefix-packed `OneHotTrace` polynomial at K = 256, would
choosing one implicit lane per semantic column remove materially more unit
support than the current fixed implicit lane zero?

This is a representation diagnostic, not a protocol implementation. The
commitment, statement, transcript, sumchecks, and verifier remain unchanged
during the probe.

## Frozen setup

- Jolt parent: `1b72ad289`
- Workload: `sha2-chain`
- Backend configuration: K = 256, virtual lookup chunk = 32
- Current support for column `i`: `active_i - count_i[0]`
- Best fixed-default support: `active_i - max_lane count_i[lane]`
- Exact additional removable support:
  `sum_i(max_lane count_i[lane] - count_i[0])`

Instruction, bytecode, and balanced-increment columns are active on every
cycle. RAM columns use their semantic activation mask; their result is
reported separately because reconstructing a nonzero implicit RAM lane
requires the already-proved RAM activation evaluation.

## Decision rule

1. Run one small forced-K256 count probe.
2. Stop as negative below 5% additional support reduction.
3. Escalate to an exact 2^26 count only at 10% or better. A 5–10% result is
   escalated only if one family has a materially different distribution that
   could grow with trace length.
4. Implement no protocol change unless the exact 2^26 reduction is at least
   15% and a conservative projection against the measured 22.36-second
   commitment is at least one second.
5. If the count gate passes, time a commitment-only representation prototype
   before changing claim reduction or verifier logic.

Count time is excluded from prover timing. Temporary instrumentation must be
removed after the result is recorded.

## Results

Rejected before protocol implementation.

The release smoke at 2^16 and the frozen small screen at 2^22 agreed:

| K256 count probe | 2^16 | 2^22 |
|---|---:|---:|
| Current zero-default support | 867,531 | 56,148,929 |
| Best per-column-default support | 865,528 | 56,018,928 |
| Additional removable units | 2,003 | 130,001 |
| Additional support reduction | 0.230885% | **0.231529%** |

At 2^22, lane zero was already the modal lane for every always-active
column:

| Family | Current support | Modal support | Reduction |
|---|---:|---:|---:|
| Instruction (16 columns) | 31,746,115 | 31,746,115 | 0 |
| Balanced increment (9 columns) | 17,395,419 | 17,395,419 | 0 |
| Bytecode (2 columns) | 6,795,923 | 6,795,923 | 0 |
| RAM (2 columns) | 211,472 | 81,471 | 130,001 |

Only the RAM columns preferred nonzero defaults:

- `RamRa(0)`: lane 22 occurred on 105,582 of 105,748 active rows.
- `RamRa(1)`: lane 148 occurred on 24,443 of 105,748 active rows.

The two independent scales have essentially identical total ratios, and
`sha2-chain` repeats the same kernel. Linear scaling to 2^26 predicts about
2.08 million removable units out of 898.38 million current units. Even the
deliberately generous assumption that the entire measured 22.364-second
commitment scales with unit support projects only 0.052 seconds. Fixed
commitment work makes the realistic saving smaller.

This misses the 5% stop gate by more than 20x and the one-second projection
gate by about 19x. No 2^26 count, commitment-only prototype, or protocol
change is warranted. The useful conclusion is stronger than “the mode gain
is small”: for every large family, the proposed optimization selects the
lane already virtualized by the current protocol.

## Commands and artifact

The commands below record the temporary diagnostic invocation. The
`PERF_MODAL_DEFAULT_PROBE` and `PERF_TRACE_STEM` hooks were deliberately
removed from `packed.rs` after measurement, so rerunning them at the final
HEAD requires restoring the count-only instrumentation first.

Smoke:

```bash
PERF_LOG_T=16 PERF_MODAL_DEFAULT_PROBE=1 \
  cargo nextest run --release -p jolt-prover-legacy --features akita \
  -E 'test(sha2_chain_akita_perf)' --run-ignored all --no-capture --cargo-quiet
```

Frozen screen:

```bash
PERF_LOG_T=22 PERF_MODAL_DEFAULT_PROBE=1 PERF_TRACE=1 \
PERF_TRACE_STEM=akita-modal-default-k256-2e22 \
  cargo nextest run --release -p jolt-prover-legacy --features akita \
  -E 'test(sha2_chain_akita_perf)' --run-ignored all --no-capture --cargo-quiet
```

Trace: `benchmark-runs/perfetto_traces/akita-modal-default-k256-2e22.json`.
