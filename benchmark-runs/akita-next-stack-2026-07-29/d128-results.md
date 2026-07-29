# D128/K256 commitment-only falsifier

Date: 2026-07-29 EDT
Code parent: `c70524e4b`

## Result

Reject the full D128 protocol port for the current speed objective. On the
exact `2^26` SHA2-chain `OneHotTrace` source, D128 reduced commitment from
24.296 s to 21.444 s: 2.852 s, or 11.7%. The frozen promotion gate required
at least 5 s or 25%.

| Metric | D64/K256 | D128/K256 | Delta |
|---|---:|---:|---:|
| Root `n_a` | 6 | 3 | -50% |
| Positions per block | 2,097,152 | 524,288 | -75% |
| Live blocks | 4,096 | 8,192 | +100% |
| Setup envelope | 12 GiB | 3 GiB | -9 GiB |
| Setup generation | 2.20 s | 0.528 s | -1.67 s |
| Full commitment | 24.296 s | 21.444 s | -2.852 s |
| `TracePackedOneHot::commit_inner` | 23.683 s | 20.829 s | -2.853 s |
| `trace_onehot_commit_accumulate` | 23.666 s | 20.798 s | -2.868 s |

Setup generation is outside the prover number used for the 2x-Dory target.
Even counting it, setup plus commitment falls from about 26.50 s to 21.97 s,
a 4.52 s reduction that still misses the absolute gate.

## Why halving `n_a` did not halve commitment

At K256, each trace row occupies four D64 rings or two D128 rings. The D64
root loads `4 × 6 = 24` A rings per row; D128 loads `2 × 3 = 6`. Because a
D128 ring has twice as many coefficients, this approximately halves A
coefficient traffic.

The shift-accumulation work does not halve. For each live semantic lane, the
two configurations both update 384 wide coefficients:

- D64: `n_a × D = 6 × 64 = 384`
- D128: `n_a × D = 3 × 128 = 384`

D128 also uses twice as many live blocks. The experiment therefore removes a
large fraction of setup/A-load traffic but leaves the principal
coefficient-accumulation work essentially unchanged. The measured 11.7% win
is consistent with that model.

## Controls and evaluator corrections

The custom Jolt config used the upstream fp128 D128 policy but overrode the
one-hot chunk to exactly K256; upstream `D128OneHot` defaults to K1 and would
have tested a different protocol. Planner resolution at the packed
`2^39`-variable singleton shape produced `n_a = 6` for D64 and `n_a = 3` for
D128.

A first synthetic-source full run was excluded because its lazy
`fill_row` recomputed random-looking lanes inside the timed kernel, inflating
D64 commitment to 45.37 s. The decisive runs instead constructed the same
`JoltOneHotTraceRows` cache as `prove_packed` from the same 67,108,864-cycle
SHA2-chain trace, then committed it once under each ring configuration.
The D64 control's 24.30 s agrees with the 23–24 s commitment spans in the
full-prover traces.

The temporary D128 config, benchmark API, kernel dispatch, and perf-harness
branch were removed after the falsifier failed. The branch remains on D64;
no verifier, transcript, proof format, or protocol code was ported.

## Correctness and traces

Before measurement, the D128 rank-tiled commitment was checked against
Akita's materialized K256 one-hot commitment on a small instance. The
commitment-only harness completed successfully for both target runs.

The decisive traces are:

- `benchmark-runs/perfetto_traces/akita-d128-probe-d64.json`
- `benchmark-runs/perfetto_traces/akita-d128-probe-d128.json`

The screen and excluded synthetic-run logs remain under
`benchmark-runs/akita-next-stack-2026-07-29/`.
