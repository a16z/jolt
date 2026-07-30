# Stage 6 row-major instruction RA experiment

Date: 2026-07-29 EDT

## Question

Can the instruction RA virtualization prover consume the retained row-major
`RaIndices` directly for its first three cycle rounds instead of gathering 16
`Option<u8>` columns at K256?

This is a prover representation change. The sumcheck schedule, round messages,
Fiat-Shamir transcript, committed polynomials, and verifier are unchanged.

## Prediction and gate

- Exact transient allocation removed: 32 B/cycle, or 2 GiB at `2^26` and
  8 GiB at `2^28`.
- Likely performance effect: neutral or faster. Each product evaluation reads
  all 16 instruction chunks for two nearby rows, which should favor the
  existing row-major layout over 16 independent column streams.
- Reject on any proof/parity failure, a repeated focused regression above 2%,
  a full-prover regression above the 0.48-second noise band, or a higher
  target-scale memory requirement attributable to the candidate.
- Promote to `2^26` only after a K256 `2^22` screen.

Frozen controls: K256 (`PERF_LOG_K_CHUNK=8`), virtual chunk size 32,
packed-one-hot protocol, batching, transcript, workload, and benchmark harness.

## Representation

The old path gathered one `Arc<Vec<Option<u8>>>` per instruction chunk, then
constructed 16 independent `RaPolynomial`s. The accepted path gives
`SharedRaPolynomials` a contiguous-family offset and keeps a shared
`Arc<Vec<RaIndices>>` through the first three bindings. Lightweight
single-polynomial views let the existing product-sum kernels read either
representation.

Both representations use the same per-polynomial equality tables. For each
cycle `j` and instruction chunk `i`, the new coefficient is

```text
table[i][ra_indices[j].instruction[i]]
```

which is exactly the value previously obtained from transposed column
`H_indices[i][j]`. After the third challenge, both paths materialize the same
field polynomials of length `T / 8`; all subsequent binding and opening code is
unchanged.

The focused equivalence test compares every coefficient before each binding and
every final claim against the transposed implementation at K256. The full Akita
suite, forced-K256 end-to-end proof, standard Dory proof, and ZK Dory proof
provide protocol-level coverage.

## Measurements

All runs use the SHA-256 chain harness and forced K256. Times below are
inclusive Perfetto span totals.

### `2^22` screen

| Variant | Prove | Stage 6b | Instruction init | Instruction messages | Instruction bind | Max RSS |
|---|---:|---:|---:|---:|---:|---:|
| Compact-row control | 5.95 s | 379.338 ms | 8.729 ms | 78.715 ms | 51.447 ms | 14.741 GB |
| Row-major candidate A | 5.76 s | 343.973 ms | 0.286 ms | 67.466 ms | 21.758 ms | not sampled |
| Row-major candidate B | 5.58 s | 325.839 ms | 0.207 ms | 65.963 ms | 19.296 ms | 14.628 GB |

The repeat reduces Stage 6b by 14.1%. Observed maximum RSS falls by
approximately 113 MB; the exact allocation prediction is 134 MB.

### `2^26` target

| Variant | Prove | Stage 6b | Instruction init | Instruction messages | Instruction bind | Max RSS |
|---|---:|---:|---:|---:|---:|---:|
| Compact-row control | 54.95 s | 5.568556 s | 74.376 ms | 1.342602 s | 462.344 ms | 44.244 GB |
| Row-major candidate | 52.63 s | 5.128936 s | 0.193 ms | 1.200088 s | 338.313 ms | 44.325 GB |

Stage 6b improves by 7.9%, instruction message work by 10.6%, and the
instruction transpose disappears. The full proof is 2.32 seconds faster in
this pair, although a single whole-prover pair is not enough to attribute all
of that movement to this change.

Headline maximum RSS is unchanged within run noise because another phase sets
the process maximum at `2^26`. This experiment therefore does not claim a new
global RSS low. It removes a structurally exact 2 GiB Stage-6 allocation, which
still matters for phase-local pressure and for larger traces where phase
ordering may change.

## Outcome

Accepted as commit `0be326e83`. The target-scale affected phase is faster, the
full prover does not regress, and the 16 transposed instruction columns are no
longer allocated. No protocol or verifier change is involved.

Validation:

- 451/451 `jolt-prover-legacy` tests with `host,akita`
- standard and ZK Dory muldiv suites
- scoped all-target clippy with `host`, `host,zk`, and `host,akita`
- formatting and diff checks

## Retained traces

- `benchmark-runs/perfetto_traces/mem-ra-row-2e22.json`
- `benchmark-runs/perfetto_traces/mem-ra-row-2e22-b.json`
- `benchmark-runs/perfetto_traces/mem-ra-row-2e26.json`

