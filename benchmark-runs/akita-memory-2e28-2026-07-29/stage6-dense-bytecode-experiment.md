# Dense bytecode RA experiment

Date: 2026-07-30 EDT

## Question

Can bytecode RA keep its fast column-major reads while replacing each
`Option<u8>` index with an all-present `u8`?

This is distinct from the rejected row-major experiment. The candidate kept
the same blocked transpose and contiguous per-column sumcheck access, changed
only the stored element type, and used the dense `RaPolynomial` source landed
for fused-increment lanes.

Frozen controls: K256 (`PERF_LOG_K_CHUNK=8`), virtual chunk size 32, protocol,
batching, transcript, SHA-256 chain workload, and benchmark harness. The
control is commit `1f3652bc4`.

## Structural result

The benchmark workload has two bytecode chunks. A dense representation would
remove 2 B/cycle during the first three Stage-6b rounds:

| Trace size | Bytes removed |
|---|---:|
| `2^22` | 8 MiB |
| `2^26` | 128 MiB |
| `2^28` | 512 MiB |

The blocked gather became faster, but the target-scale sumcheck did not.

## Performance

Bytecode work below is the inclusive sum of cycle-prover initialization,
message generation, and challenge ingestion.

### `2^22` screens

| Variant | Prove | Stage 6b | Bytecode work | Bytecode gather | Max RSS |
|---|---:|---:|---:|---:|---:|
| Control | 5.60 s | 323.285 ms | 53.947 ms | 1.771 ms | 14.829 GB |
| Dense A | 5.66 s | 329.571 ms | 52.190 ms | 1.521 ms | 14.909 GB |
| Dense B | 5.65 s | 334.577 ms | 53.684 ms | 1.414 ms | 14.939 GB |

The gather improves by 14–20%, and bytecode work is 3.3% better and 0.5%
better in the two screens. Total Stage 6b nevertheless moves upward by 1.9%
and 3.5%, so the small screens do not establish a safe win.

### `2^26` adjudication

| Variant | Prove | Stage 6b | Bytecode init | Bytecode messages | Bytecode bind | Bytecode work | Max RSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Control | 53.84 s | 5.102098 s | 29.737 ms | 581.848 ms | 113.170 ms | 724.755 ms | 44.310 GB |
| Dense A | 54.75 s | 5.197425 s | 17.840 ms | 597.827 ms | 110.612 ms | 726.279 ms | 44.305 GB |
| Dense B | 55.24 s | 5.246110 s | 18.269 ms | 612.575 ms | 107.090 ms | 737.934 ms | 44.305 GB |

Initialization improves by about 12 ms, but bytecode message generation
regresses by 16 and 31 ms. Stage 6b regresses by 95 ms (1.87%) and 144 ms
(2.82%) in the two target runs. The maximum-RSS values are unchanged because
another phase sets the peak.

Unchanged phases also moved: commitment plus packed opening account for
0.702 seconds of Dense A's 0.902-second whole-proof delta, and commitment
alone accounts for 1.122 seconds of Dense B's 1.401-second delta. Those
unrelated movements do not erase the repeated Stage-6b regression.

## Outcome

Rejected. All candidate code was reverted and no implementation commit was
created. The 512 MiB projected `2^28` saving is too small to justify a
repeatable target-scale slowdown.

A future retry would need a source-specialized dense RA kernel that
demonstrates a message-generation win in isolation, not another representation
swap through the current runtime source enum.

## Retained traces

- `benchmark-runs/perfetto_traces/mem-dense-bytecode-2e22.json`
- `benchmark-runs/perfetto_traces/mem-dense-bytecode-2e22-b.json`
- `benchmark-runs/perfetto_traces/mem-dense-bytecode-2e26.json`
- `benchmark-runs/perfetto_traces/mem-dense-bytecode-2e26-b.json`
