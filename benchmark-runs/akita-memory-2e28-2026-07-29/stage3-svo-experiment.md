# Instruction-input small-value experiment

Date: 2026-07-30 EDT

## Question

Can Stage 3 avoid expanding eight compact instruction-input columns to
`Fp128` at `T / 2` without changing the sumcheck or slowing the prover?

The previous implementation used `CompactPolynomial`, which materialized a
field vector for every column at the first low-to-high bind. The candidate
keeps the original `bool`, `u64`, and `i128` values for three rounds and
materializes the same bound polynomials directly at `T / 8`.

K256 (`PERF_LOG_K_CHUNK=8`), virtual chunk size 32, protocol, transcript,
workload, and benchmark harness remained fixed. The control is commit
`1355fab03`; the accepted implementation is commit `2d08372ec`.

## Algebra and protocol boundary

For one pair of small coefficients, the existing first bind computes

`bind(a, b, r) = a + r * (b - a)`.

Applying three low-to-high binds to eight consecutive coefficients is the
same whether each intermediate vector is stored or the eight-coefficient
block is evaluated when read. For example, after challenges `r0`, `r1`, and
`r2`, the candidate computes:

```
x0 = bind(a0, a1, r0)    x1 = bind(a2, a3, r0)
x2 = bind(a4, a5, r0)    x3 = bind(a6, a7, r0)
y0 = bind(x0, x1, r1)    y1 = bind(x2, x3, r1)
z0 = bind(y0, y1, r2)
```

`z0` is exactly the coefficient that ordinary sequential binding stores at
the corresponding `T / 8` index. The first three round messages read these
logical intermediate coefficients on demand; after ingesting the third
challenge, the prover stores all `z` values and resumes the existing field
bind loop.

This changes only a prover-side representation. Sumcheck messages, challenge
order, final claims, cached openings, transcript contents, and verifier code
are unchanged. A focused test compares `bool`, `u64`, and signed `i128`
polynomials against `CompactPolynomial` before every bind and at the final
claim.

The three-round delay is enabled only for 16-byte fields. BN254/Dory retains
its previous first-bind materialization point so this Akita experiment does
not alter the Dory comparison baseline.

## Structural result

At `T = 2^26`, the eight compact source columns occupy 2.53125 GiB:

- four bit-packed `Vec<bool>` columns: 0.03125 GiB total
- three `u64` columns: 1.5 GiB total
- one `i128` column: 1 GiB

The old first bind allocated eight `T / 2` Fp128 vectors, or 4 GiB. The new
third bind allocates eight `T / 8` vectors, or 1 GiB. Thus the retained field
state after round three is exactly 3 GiB smaller. It also avoids making the
4 GiB and 2 GiB intermediate field generations resident during the first two
binds.

All eight polynomials are dropped at the end of Stage 3. The late benefit is
therefore allocator behavior rather than a forgotten live owner: avoiding the
large transient generations leaves fewer resident pages behind for Stage 4.

## Performance screens

### `2^22`

Two same-build controls and two candidates were retained:

| Variant | Stage 3 | InstructionInput message + bind | Whole proof |
|---|---:|---:|---:|
| Control C | 124.713 ms | 84.268 ms | 5.916 s |
| Control D | 132.516 ms | 87.865 ms | 5.755 s |
| Three-round A | 96.583 ms | 52.736 ms | 5.77 s |
| Three-round B | 108.277 ms | 66.913 ms | 5.72 s |

The Stage 3 baseline-to-Stage 4 baseline RSS increase fell from
0.39–0.40 GB to 0.08 GB. Stage 4 round-zero RSS fell from 4.55 GB to
4.16–4.23 GB.

### `2^26`

| Metric | Control | Three-round | Difference |
|---|---:|---:|---:|
| Prove | 53.35 s | 53.59 s | +0.24 s (+0.45%) |
| Maximum RSS | 44.157 GB | 40.029 GB | -4.128 GB (-9.35%) |
| Stage 3 | 1.127052 s | 1.099730 s | -27.322 ms |
| Stage 4 | 2.421274 s | 2.374468 s | -46.806 ms |
| Commit | 22.431457 s | 22.709290 s | +277.833 ms |
| Packed opening | 11.147609 s | 11.036208 s | -111.401 ms |

The directly affected InstructionInput work improved:

| Span | Control | Three-round |
|---|---:|---:|
| Initialization | 100.866 ms | 103.053 ms |
| Round messages | 330.672 ms | 518.643 ms |
| Challenge binding | 466.773 ms | 245.058 ms |
| Message + bind | 797.445 ms | 763.701 ms |

On-demand reads add 187.971 ms to message generation, but skipping the first
two full-field binds removes 221.715 ms. The directly affected aggregate
improves by 33.744 ms (-4.23%). The whole-proof movement is within ordinary
noise and is dominated by an unchanged 278 ms commitment movement, so no
whole-prover speedup or regression is claimed.

The run reported zero swaps. Internal RSS markers put Stage 4 baseline at
27.44 GB versus 32.23 GB for the control, a 4.79 GB reduction. `/usr/bin/time`
reported a 4.128 GB reduction in process maximum RSS.

The sampled global peak moved:

| Phase | Control maximum | Three-round maximum |
|---|---:|---:|
| Stage 3 | 37.95 GiB | 26.87 GiB |
| Stage 4 | 36.22 GiB | 32.87 GiB |
| Stage 6b | 34.18 GiB | 35.29 GiB |
| Packed opening | 36.52 GiB | 36.29 GiB |

The control peaked in Stage 3. The candidate peaks during packed opening,
which is therefore the next memory target.

## Validation and outcome

The candidate is accepted as commit `2d08372ec`.

Validation:

- focused equivalence through every bind for `bool`, `u64`, and `i128`
- 456/456 `jolt-prover-legacy` tests with `host,akita` before the final
  Akita-only scope guard
- Akita muldiv suite after the scope guard
- standard and ZK Dory muldiv suites after the scope guard
- all-target warning-denying clippy on `jolt-prover-legacy` with `host`,
  `host,zk`, and `host,akita`
- formatting and diff checks

The workspace-wide clippy command is independently blocked by the untracked
debug test `crates/jolt-akita/tests/schedule_probe.rs`; that file was not
modified.

## Retained traces and logs

Primary traces:

- `benchmark-runs/perfetto_traces/mem-packed-delta-2e22-c.json`
- `benchmark-runs/perfetto_traces/mem-packed-delta-2e22-d.json`
- `benchmark-runs/perfetto_traces/mem-svo3-2e22.json`
- `benchmark-runs/perfetto_traces/mem-svo3-2e22-b.json`
- `benchmark-runs/perfetto_traces/mem-packed-delta-2e26-b.json`
- `benchmark-runs/perfetto_traces/mem-svo3-2e26.json`

Target logs and RSS samples:

- `logs/packed-delta-2e26-b.log` / `logs/packed-delta-2e26-b.rss`
- `logs/svo3-2e26.log` / `logs/svo3-2e26.rss`
