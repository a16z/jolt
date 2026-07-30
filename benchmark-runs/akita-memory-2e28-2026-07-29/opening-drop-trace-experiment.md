# Late trace-release experiment

Date: 2026-07-30 EDT

## Question

Can the packed prover release its compact execution trace before Akita opening
without rebuilding data or changing the proof?

`JoltTraceRow` is statically 64 bytes. At `T = 2^26`, the retained
`Arc<Vec<JoltTraceRow>>` therefore owns exactly 4 GiB; at `2^28`, it would own
16 GiB.

The Akita commitment hint does not refer back to these rows. During
commitment, `JoltOneHotTraceRows` derives and owns the packed byte lanes that
the opening consumes. All sumcheck provers that clone the trace finish by
Stage 7. The candidate saves the public trace length, verifies in debug builds
that only one `Arc` owner remains, and drops that owner before reconstruction
and opening.

K256 (`PERF_LOG_K_CHUNK=8`), virtual chunk size 32, protocol, transcript,
workload, setup retention, and opening implementation remained fixed. The
control is the three-round InstructionInput implementation at commit
`2d08372ec`; the accepted implementation is commit `7afb90166`.

## Protocol boundary

This is an ownership change only. The commitment already consumed the trace
to construct:

- the packed byte-lane cache retained in the Akita hint
- `RaIndices`, consumed through Stage 7
- fused-increment columns, consumed through Stage 7

After Stage 7, reconstruction reads advice and preprocessed program data, and
the main opening reads the packed cache in the hint. Neither reads
`JoltTraceRow`. The proof still carries the saved trace length, and no
transcript message, claim, polynomial, challenge, or verifier input changes.

## `2^22` promotion screen

| Variant | Prove | Packed opening |
|---|---:|---:|
| Control A | 5.77 s | 2.64 s |
| Control B | 5.72 s | 2.64 s |
| Release trace | 5.77 s | 2.64 s |

The screen was exactly neutral and the debug ownership assertion passed.

## `2^26` target

| Metric | Control | Release trace | Difference |
|---|---:|---:|---:|
| Prove | 53.59 s | 53.66 s | +0.07 s (+0.13%) |
| Verify | 191.05 ms | 194.67 ms | +3.62 ms |
| Packed opening | 11.04 s | 11.11 s | +0.07 s |
| Maximum RSS | 40.029 GB | 39.923 GB | -0.106 GB |
| Swaps | 0 | 0 | unchanged |

The process-wide maximum moves little because it now occurs in an earlier
Stage-3/4 transient, before the trace can be released. The phase-aligned RSS
timeline shows the intended late cut:

| RSS sample | Control | Release trace | Difference |
|---|---:|---:|---:|
| Opening maximum | 36.29 GiB | 33.18 GiB | -3.11 GiB |
| Opening end | 23.95 GiB | 19.94 GiB | -4.01 GiB |
| Whole-proof sampled maximum | 36.29 GiB | 34.16 GiB | -2.13 GiB |

The opening-end result matches the exact 4 GiB allocation. Sampling at
0.5-second intervals catches some opening scratch allocation before or during
the drop, so its phase maximum shows 3.11 GiB rather than the full 4 GiB.

The sampled global peak moves from packed opening to Stage 6b:

| Phase | Release-trace maximum |
|---|---:|
| Stage 3 | 33.60 GiB |
| Stage 4 | 32.25 GiB |
| Stage 6b | 34.16 GiB |
| Packed opening | 33.18 GiB |

No reconstruction pass, copy, conversion, or re-read was added. The 70 ms
whole-proof and opening movements are below run-to-run noise and do not
indicate a regression.

## Validation and outcome

The candidate is accepted as commit `7afb90166`.

Validation:

- debug assertion confirms one final trace owner after Stage 7
- 456/456 `jolt-prover-legacy` tests with `host,akita`
- all-target warning-denying clippy on `jolt-prover-legacy` with `host`,
  `host,zk`, and `host,akita`
- formatting and diff checks
- K256 `2^22` promotion screen
- quiet-gated K256 `2^26` target with proof verification and zero swaps

The workspace-wide clippy command remains independently blocked by the
untracked debug test `crates/jolt-akita/tests/schedule_probe.rs`; that file was
not modified.

## Retained traces and logs

Primary traces:

- `benchmark-runs/perfetto_traces/mem-svo3-2e22-b.json`
- `benchmark-runs/perfetto_traces/mem-drop-trace-2e22.json`
- `benchmark-runs/perfetto_traces/mem-svo3-2e26.json`
- `benchmark-runs/perfetto_traces/mem-drop-trace-2e26.json`

Target logs and RSS samples:

- `logs/svo3-2e26.log` / `logs/svo3-2e26.rss`
- `logs/drop-trace-2e26.log` / `logs/drop-trace-2e26.rss`
