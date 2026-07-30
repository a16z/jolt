# Deferred RA-row materialization

Date: 2026-07-30 EDT

## Question

Can Akita avoid retaining `RaIndices` during commitment and Stages 1–5
without changing the K256 Stage-6 representation or its hot kernels?

Before this change, the initial row-cache pass built both representations
needed later in the proof:

- 29 packed one-hot lane bytes per cycle for commitment and opening
- one 54-byte `RaIndices` row per cycle for Stages 6–7

Stages 1–5 do not read `RaIndices`. The candidate builds only the packed lanes
and one RAM-validity byte per cycle, then reconstructs the same `RaIndices`
rows immediately after Stage 5.

## Expected result

The early retained saving is 53 B/cycle:

| Trace size | Expected saving |
|---|---:|
| `2^22` | 0.207 GiB |
| `2^26` | 3.3125 GiB |
| `2^28` | 13.25 GiB |

Stage 6 and later should have the same memory footprint because the original
row representation is restored before its first consumer. The additional
work is one contiguous read of 30 B/cycle and a deferred write of the same
54-byte rows; a measurable Stage-6 or whole-prover regression would reject
the candidate.

## Correctness

Packed lanes preserve every instruction, bytecode, and RAM chunk. The only
information they cannot encode by themselves is the distinction between a
non-memory row and a memory row whose chunk value is zero. A one-byte
RAM-validity sidecar preserves that distinction. `RaIndices::from_trace_row`
derives all RAM chunks from one optional remapped address, so they are either
all `None` or all `Some`.

The parity test reconstructs the deferred rows at K16 and K256 and compares
them with `RaIndices::from_trace_row` on a real trace. The full Akita package
suite, forced-K256 end-to-end proof, standard Dory muldiv tests, and ZK Dory
muldiv tests all passed.

## Measurements

### `2^22` screen

| Run | Prove | Initial cache | Deferred rows | Commitment | Stage 6b |
|---|---:|---:|---:|---:|---:|
| Control | 5.765 s | 25.498 ms | — | 1.575 s | 324.133 ms |
| Candidate A | 5.778 s | 20.015 ms | 9.487 ms | 1.568 s | 334.960 ms |
| Candidate B | 5.683 s | 21.343 ms | 9.385 ms | 1.559 s | 336.622 ms |

The direct representation work increases by 4.0–5.2 ms. Stage-6b values
remain inside the 332.5–349.5 ms range of adjacent accepted-control traces.
Stages 1–5 are 0.22–0.26 GiB lower, close to the 0.207 GiB retained-byte
prediction.

### `2^26` target

| Variant | Prove | Commitment | Stage 6b | Maximum RSS |
|---|---:|---:|---:|---:|
| Control | 53.66 s | 22.75 s | 5.27 s | 39.923 GB |
| Deferred rows | 53.54 s | 22.49 s | 5.19 s | 38.924 GB |

The retained commitment plateau falls by approximately 3.50 GiB, consistent
with removing 3.3125 GiB of live RA rows plus allocator effects. The exact
process maximum falls by 0.999 GB. It does not fall by the full retained-byte
amount because a short commitment transient and Stage 6 occur after or while
other large allocations are live; Stage 6 deliberately rematerializes the
same rows.

The 0.5-second sampler caught different Stage-6 transients: 34.16 GiB in the
control and 35.80 GiB in the candidate. This does not establish a Stage-6
increase. `/usr/bin/time` observed a lower process maximum, the Stage-6
representation and kernels are unchanged, and the Stage-6 duration moved in
the favorable direction. No speedup is claimed.

## Outcome

Accepted in commit `5d1ff81a1`. The change removes a large unused lifetime,
keeps K256 and the protocol fixed, and shows no performance regression at
either screen or target scale.

The result is strongest as an early-lifetime and capacity improvement. It
reduces the `2^28` retained working set by 13.25 GiB until Stage 6, but it does
not address the current Stage-6b peak.

## Retained traces

- `benchmark-runs/perfetto_traces/mem-defer-ra-2e22.json`
- `benchmark-runs/perfetto_traces/mem-defer-ra-2e22-b.json`
- `benchmark-runs/perfetto_traces/mem-defer-ra-2e26.json`

The target log and RSS series are in
`benchmark-runs/akita-memory-2e28-2026-07-29/logs/defer-ra-2e26.log` and
`benchmark-runs/akita-memory-2e28-2026-07-29/logs/defer-ra-2e26.rss`.
