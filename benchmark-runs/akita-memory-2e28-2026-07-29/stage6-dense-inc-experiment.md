# Dense fused-increment lane experiment

Date: 2026-07-30 EDT

## Question

Can the nine K256 fused-increment one-hot columns use one byte per cycle
without slowing the Stage 6/7 kernels that consume them?

The existing columns stored `Option<u8>`. Every fused-increment lane is
present, so the option state was unused, but it doubled each index from one to
two bytes. The candidate keeps the existing column-major layout and adds an
all-present source to `RaPolynomial`; it does not use the rejected row-major
access pattern.

Frozen controls: K256 (`PERF_LOG_K_CHUNK=8`), virtual chunk size 32, protocol,
batching, transcript, SHA-256 chain workload, and benchmark harness. The
control is commit `39bc6ce38`.

## Structural result

K256 represents the 64-bit unsigned increment with eight byte chunks plus one
carry lane. Replacing nine `Option<u8>` columns with nine `u8` columns removes
9 B/cycle while those columns are live:

| Trace size | Bytes removed |
|---|---:|
| `2^22` | 36 MiB |
| `2^26` | 576 MiB |
| `2^28` | 2.25 GiB |

The dense and sparse RA sources produce identical coefficients through all
three specialized binding rounds and after materialization. A focused test
covers lanes 0 and 255 and both binding orders.

## Performance

The affected aggregate is `prove_stage6a_lattice + prove_stage6b_lattice +
prove_stage7_lattice`, using inclusive Perfetto spans.

### `2^22` screens

| Variant | Prove | Stage 6a | Stage 6b | Stage 7 | Affected aggregate | Max RSS |
|---|---:|---:|---:|---:|---:|---:|
| Control | 5.62 s | 104.793 ms | 327.054 ms | 43.967 ms | 475.814 ms | 14.624 GB |
| Dense A | 5.67 s | 88.577 ms | 330.870 ms | 42.401 ms | 461.848 ms | — |
| Dense B | 5.60 s | 91.571 ms | 323.285 ms | 38.842 ms | 453.698 ms | 14.829 GB |

The affected aggregate improves by 2.9% and 4.6% in the two screens. The RSS
maximum occurs outside the shortened allocation's lifetime and is noisy at
this scale; the byte saving above is structural, not inferred from maximum
RSS.

### `2^26` target

| Variant | Prove | Stage 6a | Stage 6b | Stage 7 | Affected aggregate | Max RSS |
|---|---:|---:|---:|---:|---:|---:|
| Control | 53.48 s | 926.195 ms | 5.133042 s | 620.308 ms | 6.679545 s | 44.304 GB |
| Dense | 53.84 s | 941.098 ms | 5.102098 s | 599.175 ms | 6.642371 s | 44.310 GB |

The target-scale affected aggregate improves by 37.2 ms (0.56%), so the
candidate passes the no-regression gate. The 0.36-second whole-proof movement
does not come from the modified work: the unchanged commitment span alone was
0.260 seconds slower. The packed-opening span was unchanged to measurement
precision.

The headline RSS is also unchanged because another phase determines the
process maximum. This result should be credited as a 576 MiB Stage 6/7
working-set reduction at `2^26`, not as a new global-RSS low.

## Correctness and outcome

The candidate is accepted as commit `1f3652bc4`. It changes only the prover's
index representation; transcript messages, claims, openings, and verifier
logic are unchanged.

Validation:

- 453/453 `jolt-prover-legacy` tests with `host,akita`
- dense/sparse RA equivalence through every specialized round and both binding
  orders
- standard and ZK Dory muldiv suites
- scoped all-target clippy with `host`, `host,zk`, and `host,akita`
- formatting and diff checks

## Retained traces

- `benchmark-runs/perfetto_traces/mem-dense-inc-2e22.json`
- `benchmark-runs/perfetto_traces/mem-dense-inc-2e22-b.json`
- `benchmark-runs/perfetto_traces/mem-dense-inc-2e26.json`
