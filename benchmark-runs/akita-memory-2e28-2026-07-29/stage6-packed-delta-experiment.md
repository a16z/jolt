# Packed fused-delta experiment

Date: 2026-07-30 EDT

## Question

Can the Stage 6 fused-increment stream use sign-magnitude storage instead of
`i128` coefficients without slowing the bytecode read-RAF sumcheck?

The fused value is the difference of two `u64` machine words, so its magnitude
is at most `2^64 - 1`. The candidate stores one `u64` magnitude and one sign
bit per cycle, then decodes signed values only where the Stage 6 address and
cycle provers consume them.

K256 (`PERF_LOG_K_CHUNK=8`), virtual chunk size 32, protocol, batching,
transcript, SHA-256 chain workload, and benchmark harness remained fixed. The
frozen control is the dense-inc implementation at commit `1f3652bc4`.

## Structural result

Storage falls from 16 B/cycle to 8.125 B/cycle:

| Trace size | Old `Vec<i128>` | Packed stream | Bytes removed |
|---|---:|---:|---:|
| `2^22` | 64 MiB | 32.5 MiB | 31.5 MiB |
| `2^26` | 1 GiB | 520 MiB | 504 MiB |
| `2^28` | 4 GiB | 2.03125 GiB | 1.96875 GiB |

The saving is live from fused-delta construction through the first cycle bind.
After that bind, both representations retain the same `T / 2` field elements.
The process-wide RSS maximum occurs in another phase, so maximum RSS is not
expected to fall by the full 504 MiB at `2^26`.

The representation and binding change is prover-local. Sumcheck identities,
claims, transcript messages, openings, and verifier code are unchanged.

## Kernel gate

A `2^20`-element Criterion benchmark first tested two possible first-bind
kernels. Values included zeros, both signs, and full-width `u64` magnitudes.

| First-bind kernel | Median | Relative to current |
|---|---:|---:|
| Current generic `CompactPolynomial<i128>` | 523.94 µs | baseline |
| Naive sign-magnitude, two `mul_u64` calls | 550.24 µs | +5.0% |
| One-multiply kernel over `i128` storage | 442.30 µs | -15.6% |
| One-multiply kernel over packed storage | 442.09 µs | -15.6% |

The two-`mul_u64` formulation was rejected. The accepted kernel reconstructs
each signed pair and preserves the current interpolation:

`a + r * (b - a)` or `a - r * (a - b)`.

Thus it still performs one magnitude multiplication per unequal pair. The
packed and old-storage versions of that kernel were indistinguishable, showing
that sign decoding does not add measurable first-bind cost when performed in
64-cycle blocks. Standalone field conversion was also neutral (303.01 versus
295.41 µs medians with overlapping ranges).

## Prover measurements

The first target candidate decoded sign bits by global index in each of nine
one-hot passes. It was correct, but increased
`fused_inc_one_hot_columns` from 192.845 to 221.359 ms. The final version
decodes each 64-cycle sign word once per block, bringing that span to
188.810 ms.

### `2^22` screens

The final tuned screen measured 5.75 s prove with a 465.261 ms
Stage 6a+6b+7 aggregate. The two controls measured 5.60–5.66 s prove and
453.698–461.848 ms for the same aggregate. The changed bytecode-cycle spans
were in the control range:

| Span | Control range | Tuned candidate |
|---|---:|---:|
| Fused one-hot construction | 10.357–10.708 ms | 11.242 ms |
| Bytecode cycle messages | 38.167–38.685 ms | 38.404 ms |
| Bytecode cycle binds | 13.211–13.810 ms | 13.928 ms |

The whole screen was slowed by unchanged work, including a 2.660-second
packed-opening span. It was used only as a promotion screen, not as the target
decision.

### `2^26` target

| Variant | Prove | Stage 6a | Stage 6b | Stage 7 | Affected aggregate | Max RSS |
|---|---:|---:|---:|---:|---:|---:|
| Control | 53.84 s | 941.098 ms | 5.102098 s | 599.175 ms | 6.642371 s | 44.310 GB |
| Packed, untuned | 53.75 s | 956.878 ms | 5.191196 s | 630.242 ms | 6.778316 s | 44.304 GB |
| Packed, word-aligned | 53.35 s | 927.822 ms | 5.158092 s | 611.335 ms | 6.697249 s | 44.157 GB |

The tuned affected aggregate is 54.9 ms (+0.83%) above the control. The sum of
the directly changed or adjacent instrumented spans improves by 37.6 ms:

| Span | Control | Tuned |
|---|---:|---:|
| Fused-delta construction | 46.267 ms | 40.917 ms |
| Fused one-hot construction | 192.845 ms | 188.810 ms |
| Bytecode address initialization | 328.169 ms | 293.134 ms |
| Bytecode cycle initialization | 29.737 ms | 30.647 ms |
| Bytecode cycle messages | 581.848 ms | 588.368 ms |
| Bytecode cycle binds | 113.170 ms | 112.577 ms |

The remaining aggregate movement is in unchanged batched instances and is
below the campaign's no-regression threshold. The 0.49-second whole-proof
improvement is not attributed to this change: the unchanged commitment span
alone improved by 0.57 seconds.

Maximum RSS moved from 44.310 to 44.157 GB with zero swaps. Another phase sets
that maximum, so the accepted memory claim remains the exact 504 MiB Stage 6
working-set reduction rather than the noisy 153 MB headline movement.

## Correctness and outcome

The candidate is accepted as commit `1355fab03`.

Validation:

- boundary round trips across `[-(2^64-1), 2^64-1]`
- coefficient equality before binding and after every bind through the final
  sumcheck claim
- 455/455 `jolt-prover-legacy` tests with `host,akita`
- standard and ZK Dory muldiv suites
- all-target warning-denying clippy with `host`, `host,zk`, and `host,akita`
- formatting and diff checks

## Retained traces and logs

Primary K256 traces:

- `benchmark-runs/perfetto_traces/mem-packed-delta-2e22-c.json`
- `benchmark-runs/perfetto_traces/mem-packed-delta-2e22-d.json`
- `benchmark-runs/perfetto_traces/mem-packed-delta-2e26-b.json`

The untuned traces are retained as
`mem-packed-delta-2e22.json`, `mem-packed-delta-2e22-b.json`, and
`mem-packed-delta-2e26.json`. Two initially misconfigured K16 screens are
clearly labeled `mem-packed-delta-k16-2e22.json` and
`mem-packed-delta-k16-2e22-b.json`; they are not used in the K256 decision.

Target logs and RSS samples:

- `logs/packed-delta-2e26.log` / `logs/packed-delta-2e26.rss` (untuned)
- `logs/packed-delta-2e26-b.log` / `logs/packed-delta-2e26-b.rss` (accepted)
