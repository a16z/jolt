# Akita `2^28` structural memory cuts

Date: 2026-07-30 EDT

## Outcome

The first three targets from the
[analytical memory model](analytical-memory-model.md) are accepted. They
remove a different dominant allocation from commit, Stage 6b, and the root
evaluation proof:

| Window | Change | Exact `2^28` reduction |
|---|---|---:|
| Commit | Negacyclic-only packed-row NTT cache | 23.515625 GiB |
| Stage 6b | Release the compact trace at its final reader | 16 GiB |
| Root evaluation | Stream capacity-safe `t_hat`/`z` chunks | 47.03125 GiB |

The reductions do not add: they affect different lifetime maxima. After all
three, the conservative structural ceiling is Stage 5 at no more than
81.15 GiB, or 324.6 B/cycle at `2^28`. This leaves 13.85 GiB below the
machine's 95 GiB hard limit and 8.85 GiB below the 90 GiB working target.

No commitment, proof, transcript, verifier, or protocol message changed.

## 1. Negacyclic-only packed-row cache

`digit_rows` performs only a negacyclic matrix product, but its shared cache
stored both the negacyclic and cyclic transforms. Under the production Q128,
`D=64` profile:

```text
one transform = 64 coefficients * 5 CRT primes * 4 bytes = 1,280 bytes/ring
old cache     = 2,560 bytes/ring
```

The new cache contract keeps negacyclic-only and both-transform slots
separate. A previously warmed both-transform slot can satisfy a negacyclic
request, but a cyclic consumer can never receive the weaker slot.

The production Q128 unit test pins three facts:

1. `cyc` is absent;
2. the optional i16 tail is absent;
3. cache bytes equal `rings * 1,280`.

At `2^26`, the commit slot is exactly 5 GiB instead of 10 GiB. Its build span
fell from 525 ms to 384 ms. The process maximum did not move materially
because the limiting phase occurred later.

At `2^28`, the commit projection changes from 90.51953 to 67.00391 GiB.

## 2. Compact-trace final reader

The 64-byte `JoltTraceRow` vector was retained through Stage 7 even though its
last read occurs at the beginning of Stage 6b:

```text
HammingBooleanitySumcheckProver::initialize(params, &trace)
```

That initializer materializes the RAM-access Boolean polynomial. Every later
Stage-6b, Stage-7, reconstruction, and opening operation consumes derived
state instead of the trace.

The prover now captures `trace_len`, initializes the Hamming prover, and
immediately drops its trace owner. The retained trace events report:

| `T` | Released bytes | Remaining `Arc` owners |
|---:|---:|---:|
| `2^22` | 268,435,456 | 0 |
| `2^26` | 4,294,967,296 | 0 |

Thus this is an actual freeable allocation at the boundary, not merely a
cleared field with another hidden owner.

At `2^28`, the exact 16 GiB reduction moves the Stage-6b transition from
87.18164 to 71.18164 GiB.

## 3. Capacity-safe streamed root quotients

The root relation already had a streamed setup source, but its `t_hat` arm
required the entire vector to fit one CRT accumulator. At `2^26`:

```text
t_len  = 1,056,768 rings
t_safe = false
```

That single Boolean sent the relation back to the cached path, which rounded
the request to 4,194,304 rings and built a 10 GiB both-transform cache.

The streamed kernel now partitions every unsafe role into independently safe
CRT chunks:

```text
sum_j A[j] * t[j]
  = sum_chunks reduce_CRT(sum_{j in chunk} A[j] * t[j])
```

Each chunk is converted back to the exact field ring before the chunk results
are added. This is the same arithmetic and capacity argument used by the
existing cached kernel and streamed `z` arm.

The production-shaped test forces the Q128/D64 boundary with 512 `t_hat`
terms against a 511-term worst-case capacity, exercises D, B, and A
simultaneously, and compares both materialized and seed-derived streamed
sources with the cached kernel.

The target trace shows:

```text
z_len=2,097,152   z_chunk_width=461
t_len=1,056,768   t_chunk_width=16,381
```

There is no 4,194,304-ring cache build. The largest later both-transform slot
is 262,144 rings, or 0.625 GiB.

At `2^28`, this removes the modeled 47.03125 GiB root fallback. The
evaluation-proof ceiling becomes about 33.3 GiB, set by an earlier
root-preparation window rather than the quotient.

## Target measurements

All runs used the same release harness, printed K256, verified the proof, and
reported zero swaps.

| Revision | Prove | Commit | Stage 6b | Opening | Maximum RSS | Gross B/cycle |
|---|---:|---:|---:|---:|---:|---:|
| Pre-cut control | 53.697 s | 22.685 s | 5.241 s | 11.124 s | 38.876 GB | 579.29 |
| Negacyclic cache | 54.379 s | 23.113 s | 5.278 s | 11.105 s | 38.824 GB | 578.53 |
| Early trace release | 54.023 s | 23.062 s | 5.222 s | 10.977 s | **36.264 GB** | **540.38** |
| Streamed quotient | **53.784 s** | 23.132 s | 5.275 s | **10.547 s** | 36.363 GB | 541.85 |

The whole-prover variation follows unchanged commitment and PIOP spans. The
directly affected measurements show no regression:

- the negacyclic cache build is 141 ms faster;
- early trace release adds no scan or conversion;
- streamed quotient opening is 429 ms faster than its immediate control.

The final run's small maximum-RSS increase over the early-release run is
noise in the unchanged Stage-6 window. The root allocation removed by the
third change occurs later and therefore cannot lower that process maximum.

## Why `2^26` B/cycle cannot be extrapolated

The final observed `2^26` maximum is 541.85 B/cycle, but multiplying it by
four would be the wrong `2^28` forecast:

- the setup is 12 GiB at `2^26` and 18.8125 GiB at `2^28`, so its
  contribution falls from 192 to 75.25 B/cycle;
- schedule ranks, matrix rounding, program state, allocator arenas, and
  thread stacks do not scale as `4T`;
- the measured maximum includes resident logically dead pages, while the
  structural model counts live ownership and necessary construction overlap.

The post-cut source ceiling is 324.6 B/cycle. The remaining question is
whether background-owned and allocator-resident state can be kept within its
8.85 GiB working reserve.

## Commits

- Akita `b3c9bc50`: negacyclic-only packed-row cache
- Jolt `8232e5828`: pin that Akita cache change
- Jolt `a6c5ed811`: release compact trace at its final reader
- Akita `32caef7c`: stream CRT-chunked quotient roles
- Jolt `095ae7eb5`: pin the streamed quotient change

## Validation

- `cargo nextest run -p akita-prover --cargo-quiet`: 271 passed, 5 skipped
- `cargo clippy -p akita-prover --all-targets -- -D warnings`: passed
- `cargo nextest run -p jolt-akita --cargo-quiet`: 43 passed, 5 skipped
- `cargo nextest run -p jolt-prover-legacy muldiv --cargo-quiet --features
  host,akita`: 3 passed
- forced-K256 `2^26` proofs: three passed and verified, zero swaps

## Retained traces

- `benchmark-runs/perfetto_traces/mem-neg-ntt-2e22.json`
- `benchmark-runs/perfetto_traces/mem-neg-ntt-2e26.json`
- `benchmark-runs/perfetto_traces/mem-trace-early-2e22.json`
- `benchmark-runs/perfetto_traces/mem-trace-early-2e26.json`
- `benchmark-runs/perfetto_traces/mem-stream-t-2e22.json`
- `benchmark-runs/perfetto_traces/mem-stream-t-2e26.json`
