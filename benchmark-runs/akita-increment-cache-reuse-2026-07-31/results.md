# Packed increment-cache reuse results

Date: 2026-07-31 EDT

Machine: Apple M4 Max

Implementation: Jolt `011d96f77`

## Outcome

Accepted. Stage 6 and Stage 7 now read the fused-increment one-hot lanes from
the packed row cache that commitment and opening already retain. The change
removes nine trace-length byte vectors: 576 MiB at `T = 2^26` and 2.25 GiB at
`T = 2^28`.

The affected aggregate improved by 5.5% at `T = 2^22`, 1.35% across two
`T = 2^26` runs, and 8.2% in the promoted `T = 2^28` run. The large proof
finished in 152.655506 seconds with zero swaps. The current process peak still
occurs before the removed columns are allocated, so this is a Stage 6/7
live-memory reduction rather than a measured peak-RSS reduction.

## What changed

`JoltOneHotTraceRows` stores 29 hot-lane bytes per trace row. Nine adjacent
lanes are the eight balanced increment chunks and their carry lane. Before
this change, the packed source remained live for the later opening, while the
prover derived those same nine lanes again into separate
`Arc<Vec<u8>>` columns after Stage 5.

The candidate:

1. shares the packed row cache through an `Arc`;
2. represents the nine columns as an offset and stride over that cache;
3. lets the pushforward kernel read the row-major values directly; and
4. lets the cycle Booleanity prover construct strided dense RA polynomials
   over the same storage.

The old materializer and its production-only map helper are removed. No
committed data or verifier code changes.

### Value equivalence

For trace row `j` and increment column `c`, the old value was computed from
the fused delta:

```text
old[c][j] = increment_hot_lane(delta[j], c).
```

The packed cache was already populated by the same functions:

```text
rows[j * 29 + increment_offset + c]
    = increment_hot_lane(delta[j], c).
```

The candidate replaces the first lookup with the second. Therefore the
pushforward

```text
G_c(k) = sum_j eq(r_cycle, j) * [hot_lane_c(j) = k]
```

and every RA-polynomial coefficient are unchanged. A differential unit test
compares contiguous and strided RA polynomials before each bind and at the
final claim for both low-to-high and high-to-low binding.

## Analytical memory

The removed owner is exactly `9T` bytes:

| Trace length | Removed bytes | Binary size |
|---:|---:|---:|
| `2^22` | 37,748,736 | 36 MiB |
| `2^26` | 603,979,776 | 576 MiB |
| `2^28` | 2,415,919,104 | 2.25 GiB |

In bytes-per-cycle terms, Stage 6 and Stage 7 lose **9 B/cycle** of retained
state. The packed 29-byte row cache is not new: the Akita opening hint already
keeps it alive from commitment through the final opening.

After the accepted compact read-RAF buckets, the current `T = 2^28`
source-owned model puts the Stage 5 transition at about 67.059 GiB. The
Stage 6b transition remained 63.846 GiB; this change lowers it to
61.596 GiB, or from 255.383 to 246.383 B/cycle. The roughly 268.24 B/cycle
Stage 5 peak is unchanged. Lowering peak RSS still requires deleting or
shortening an owner live during Stage 5.

## Focused layout screen

A temporary exact-shape benchmark evaluated the nine fp128 Booleanity
products over `T = 2^22` rows. After warmup, the prior column-major layout
took roughly 10.1--10.4 ms and the row-major strided layout took
7.48--7.90 ms, about 25% less. Every output matched. The temporary benchmark
was removed before the production commit.

## Integrated performance

The affected aggregate includes the removed column build, fused-delta build,
and the complete Stage 6a, Stage 6b, and Stage 7 spans.

### `T = 2^22`

| Variant | Prover | Column build | Affected aggregate |
|---|---:|---:|---:|
| Control 1 | 4.709299 s | 0.011362 s | 0.488260 s |
| Control 2 | 4.773994 s | 0.010581 s | 0.497822 s |
| Candidate 1 | 4.659094 s | 0 | 0.473318 s |
| Candidate 2 | 4.628571 s | 0 | 0.458317 s |
| **Control mean** | **4.741647 s** | **0.010972 s** | **0.493041 s** |
| **Candidate mean** | **4.643833 s** | **0** | **0.465818 s** |

The localized mean falls by 0.027224 seconds, or 5.5%. Whole-prover time
falls by 2.1%.

### `T = 2^26`

| Variant | Prover | Commitment | Column build | Affected aggregate | Peak RSS |
|---|---:|---:|---:|---:|---:|
| Control 1 | 45.368753 s | 16.340658 s | 0.179140 s | 6.851724 s | 33.8208 GiB |
| Control 2 | 45.397429 s | 16.333641 s | 0.185047 s | 6.696306 s | 33.8131 GiB |
| Candidate 1 | 45.519607 s | 16.784271 s | 0 | 6.534126 s | 33.7902 GiB |
| Candidate 2 | 46.868442 s | 17.225726 s | 0 | 6.831100 s | 33.7690 GiB |
| **Control mean** | **45.383091 s** | **16.337150 s** | **0.182094 s** | **6.774015 s** | **33.8169 GiB** |
| **Candidate mean** | **46.194025 s** | **17.004999 s** | **0** | **6.682613 s** | **33.7796 GiB** |

The affected mean improves by 0.091402 seconds, or 1.35%. Whole-prover time
is 1.79% slower, but the unchanged commitment alone is 0.667849 seconds
slower in the candidate pair and explains most of that motion. The changed
stages do not regress.

The measured RSS mean falls by only 0.037 GiB rather than the analytical
576 MiB. That is expected because `/usr/bin/time` reports the earlier process
maximum, not Stage 6 live memory. All four runs report zero swaps.

### `T = 2^28`

| Metric | Control mean | Candidate | Change |
|---|---:|---:|---:|
| Prover | 159.508540 s | 152.655506 s | -4.3% |
| Commitment | 58.818831 s | 57.351768 s | -2.5% |
| Column build | 0.795602 s | 0 | removed |
| Stage 6a | 3.677146 s | 3.274467 s | -11.0% |
| Stage 6b | 21.383224 s | 20.346159 s | -4.8% |
| Stage 7 | 2.380985 s | 2.299639 s | -3.4% |
| Affected aggregate | 28.317985 s | 25.994312 s | **-8.2%** |
| Batched opening | 22.282583 s | 21.559880 s | -3.2% |

The control is the two accepted task-local-rotation runs. The candidate's
unmodified commitment and opening also ran faster, so the 6.85-second
whole-prover difference should not all be attributed to this change. The
causal claims rest on the repeated smaller screens; the exact large run shows
that strided access does not create a cache cliff at the target size.

The candidate used 81,025,875,968 bytes, or 75.461 GiB, of maximum RSS and
reported zero swaps. The two earlier control runs measured 80.141 GiB with a
release rebuild and 74.937 GiB without one. Since Stage 5 sets all three
maxima, the point measurements neither expose nor contradict the exact
2.25 GiB Stage 6/7 deletion.

## Validation

Passed:

- strided-versus-contiguous RA equivalence in both binding orders;
- the complete lattice Booleanity round-loop test;
- all 49 `jolt-akita` tests;
- natural, forced-K256, and committed-program Akita `muldiv` proofs;
- standard and ZK Dory `muldiv` suites;
- exact `T = 2^22`, `T = 2^26`, and `T = 2^28` proof verification;
- scoped `jolt-akita` and legacy `host,akita` Clippy;
- workspace Clippy with `host`;
- workspace Clippy with `host,zk`;
- `cargo fmt --check` and `git diff --check`.

## Retained traces

All files are in `benchmark-runs/perfetto_traces/`.

| Trace | Purpose | SHA-256 |
|---|---|---|
| `akita_22_inc_row_major.json` | small candidate A | `7abfb3289e691c8dcf52676a7f876f0d644358283a1dc4af2cf02e237e2e97bb` |
| `akita_22_inc_row_major_repeat.json` | small candidate B | `b6cfdae12c6c2e6c1e30d58864f2fa6550cd0f07d8c403231a3abc20661ac03d` |
| `akita_26_inc_row_major.json` | intermediate candidate A | `abc47fb74e7ead0588aa90031a0c7bde45104fc40e4628f7ecc0ac96d1b06a47` |
| `akita_26_inc_row_major_repeat.json` | intermediate candidate B | `9c549e621949f0019c9fcc22d7530953c543a3338599e2014145d59ffa96f1be` |
| `akita_28_inc_row_major.json` | exact-target candidate | `7b588a026daba3943453369dd5fb8963e72cda78079a0311e796c59e9e740803` |
