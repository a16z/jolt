# Remaining Stage 6 RA transpose experiments

Date: 2026-07-29 EDT

## Question

After removing the dominant 16-column instruction RA transpose, should the
bytecode and RAM RA provers also read the retained row-major `RaIndices`
directly during their first three cycle rounds?

The possible memory saving is smaller than for instruction RA, and the access
density is worse: each prover reads only its own few bytes from a 54-byte row.
The two families were therefore tested separately.

Frozen controls: K256 (`PERF_LOG_K_CHUNK=8`), virtual chunk size 32, protocol,
batching, transcript, SHA-256 chain workload, and benchmark harness. The
control is commit `0be326e83`.

## Bytecode: rejected

The candidate replaced bytecode's `Option<u8>` columns with a
`SharedRaPolynomials` view at the bytecode-family offset. It passed all Akita
end-to-end checks, but both `2^22` traces showed slower bytecode message work:

| Variant | Prove | Stage 6b | Bytecode init | Bytecode messages | Bytecode bind | Max RSS |
|---|---:|---:|---:|---:|---:|---:|
| Control | 5.58 s | 325.839 ms | 2.077 ms | 38.697 ms | 12.560 ms | 14.628 GB |
| Candidate A | 5.62 s | 327.658 ms | 0.339 ms | 43.232 ms | 12.433 ms | 14.587 GB |
| Candidate B | 5.61 s | 325.622 ms | 0.367 ms | 39.828 ms | 10.718 ms | 14.559 GB |

Message work regressed by 11.7% and 2.9%. Saved initialization and binding made
the total Stage-6 result neutral, but this is the expected sparse-access
failure mode: a bytecode chunk read pulls a full RA row instead of a compact
column stream. The code was reverted before the RAM experiment.

The better follow-up is a dense column representation. Bytecode indices are
always present, so `Vec<u8>` can preserve the existing contiguous kernel while
halving the current `Option<u8>` column bytes.

## RAM: accepted

RAM also uses a sparse subset of each row, but its shared binding path recovered
enough work to offset the slower coefficient reads. The candidate retains the
row-major source for three rounds, preserves `None` as the logical zero
coefficient, and materializes the same field polynomials at `T / 8`.

### `2^22` screen

| Variant | Prove | Stage 6b | RAM init | RAM messages | RAM bind | Max RSS |
|---|---:|---:|---:|---:|---:|---:|
| Control | 5.58 s | 325.839 ms | 1.968 ms | 8.857 ms | 6.576 ms | 14.628 GB |
| Candidate A | 5.63 s | 318.122 ms | 0.047 ms | 10.209 ms | 4.704 ms | 14.869 GB |
| Candidate B | 5.62 s | 327.054 ms | 0.028 ms | 11.167 ms | 5.131 ms | 14.624 GB |

RAM's init+message+bind aggregate improves from 17.401 ms to 14.960 and
16.326 ms. Stage 6b is -2.4% and +0.4% across the two screens.

### `2^26` target

| Variant | Prove | Stage 6b | RAM init | RAM messages | RAM bind | Max RSS |
|---|---:|---:|---:|---:|---:|---:|
| Control | 52.63 s | 5.128936 s | 19.671 ms | 114.989 ms | 83.465 ms | 44.325 GB |
| Candidate | 53.48 s | 5.133042 s | 0.063 ms | 134.545 ms | 66.427 ms | 44.304 GB |

The modified phase changes by +0.08%. RAM's aggregate improves from 218.125 to
201.035 ms. The 0.85-second whole-proof difference comes from unchanged spans:
commitment was +0.50 seconds and packed opening +0.31 seconds in the candidate
run.

Maximum RSS remains controlled by another phase. The structural saving during
the first three Stage-6b rounds is `2 * ram_d` bytes per cycle: the removed
`Option<u8>` column for each RAM chunk. No global-RSS reduction is claimed.

## Correctness and outcome

The RAM variant is accepted as commit `39bc6ce38`. A focused K256 equivalence
test covers the nonzero RAM-family offset, present indices, absent indices, all
binding states, and final claims against the transposed implementation.

Validation:

- 452/452 `jolt-prover-legacy` tests with `host,akita`
- standard and ZK Dory muldiv suites
- scoped all-target clippy with `host`, `host,zk`, and `host,akita`
- formatting and diff checks

No protocol or verifier change is involved.

## Retained traces

- `benchmark-runs/perfetto_traces/mem-ra-bytecode-2e22.json`
- `benchmark-runs/perfetto_traces/mem-ra-bytecode-2e22-b.json`
- `benchmark-runs/perfetto_traces/mem-ra-ram-2e22.json`
- `benchmark-runs/perfetto_traces/mem-ra-ram-2e22-b.json`
- `benchmark-runs/perfetto_traces/mem-ra-ram-2e26.json`

