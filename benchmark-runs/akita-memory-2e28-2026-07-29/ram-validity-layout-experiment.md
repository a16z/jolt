# RA row-level RAM validity

Date: 2026-07-30 EDT

## Question

Can `RaIndices` store one RAM-presence flag per cycle instead of repeating the
same `Option<u8>` tag in all eight RAM slots?

All RAM chunks come from one optional remapped address, so they are either all
present or all absent. Replacing `[Option<u8>; 8]` with `[u8; 8]` plus one
`bool` preserves the information while shrinking each row from 54 bytes to 47.

The experiment kept K256, the proof protocol, the three-round delayed RA
representation, and all Stage-6 algorithms fixed. A size test, RAM
absent-versus-zero tests, shared/transposed equivalence tests, packed-cache
parity, and a forced-K256 end-to-end proof passed.

## Expected result

The 47-byte layout removes 7 B/cycle:

| Trace size | Expected saving |
|---|---:|
| `2^22` | 28 MiB |
| `2^26` | 448 MiB |
| `2^28` | 1.75 GiB |

The first target runs showed that the odd stride hurt `compute_all_G`. A
second variant aligned the row to eight bytes, producing a 48-byte stride and
a 6 B/cycle saving: 384 MiB at `2^26` and 1.5 GiB at `2^28`.

The candidate had to preserve the directly affected
`compute_all_G_from_ra_indices + SharedRaRound3::bind` aggregate. A lower RSS
headline could not compensate for a repeatable regression there.

## `2^22` screens

The unaligned 47-byte layout was neutral in the broad Stage-6/7 aggregate,
but its two `G` spans were mixed at 76.4 and 86.3 ms versus 70.1 and 80.8 ms
in the parent. Materialization improved to 27.1–27.3 ms from 31.3–32.4 ms.

The aligned 48-byte layout repaired the small-scale signal:

| Variant | Prove | Two `G` spans | Three materializations |
|---|---:|---:|---:|
| Parent A | 5.778 s | 70.118 ms | 32.414 ms |
| Parent B | 5.683 s | 80.818 ms | 31.265 ms |
| Aligned A | 5.733 s | 73.900 ms | 28.266 ms |
| Aligned B | 5.727 s | 67.257 ms | 28.511 ms |
| Aligned C | 5.675 s | 72.184 ms | 27.809 ms |

One aligned Stage-6b screen measured 393 ms and was treated as an outlier;
the other two measured 335 and 331 ms.

## `2^26` results

| Variant | Prove | Stage 6a+6b+7 | `G` + materialization | Maximum RSS |
|---|---:|---:|---:|---:|
| 54-byte parent | 53.543 s | 6.759 s | 1.591 s | 38.924 GB |
| 47-byte A | 54.270 s | 6.961 s | 1.644 s | 38.475 GB |
| 47-byte B | 54.104 s | 6.815 s | 1.646 s | 38.415 GB |
| 48-byte aligned | 54.091 s | 6.868 s | 1.695 s | 38.461 GB |

Both 47-byte runs reduce maximum RSS by 0.449–0.509 GB, consistent with the
448 MiB structural saving. Their directly affected aggregate regresses by
53–54 ms, or 3.3–3.4%.

The aligned target reduces maximum RSS by 0.463 GB, but the direct aggregate
regresses by 104 ms, or 6.5%. Its favorable small-scale result did not
transfer. Unchanged commitment and opening spans account for about 0.32
seconds of its 0.55-second whole-proof movement, but they do not erase the
focused regression.

## Outcome

Rejected. The memory savings are real, but neither representation meets the
no-performance-regression requirement at `2^26`. All source and test changes
were reverted; no implementation commit was created.

This rules out compacting the current fixed row by changing only its RAM tag
layout. A future compact RA source needs a kernel and layout designed
together, with target-scale `G` and materialization measurements.

## Retained traces

- Invalid harness check: `mem-ram-valid-invalid-k16-2e22.json`
- 47-byte screens: `mem-ram-valid-2e22.json`,
  `mem-ram-valid-2e22-b.json`
- 47-byte targets: `mem-ram-valid-2e26.json`,
  `mem-ram-valid-2e26-b.json`
- 48-byte screens: `mem-ram-valid-a8-2e22.json`,
  `mem-ram-valid-a8-2e22-b.json`, `mem-ram-valid-a8-2e22-c.json`
- 48-byte target: `mem-ram-valid-a8-2e26.json`

The target logs and RSS series are under
`benchmark-runs/akita-memory-2e28-2026-07-29/logs/` with matching
`ram-valid-2e26`, `ram-valid-2e26-b`, and `ram-valid-a8-2e26` names.
