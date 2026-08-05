# Booleanity address pushforward

This note freezes the Metal architecture and evaluator contract for
`BooleanityAddressPhase`. It covers only the exact cycle-to-address
pushforward. The eight address rounds and Fiat--Shamir remain on the host.

## Measured boundary

The clean five-pair `2^26` Fibonacci holdout at revision
`ea5acd23019ca4d8045e6cade3b321601b41019a` measured:

| Quantity | Median |
|---|---:|
| Optimized CPU member | 939.917 ms |
| Current Metal-backend member | 942.588 ms |
| Current local speedup | 0.997x |
| Optimized CPU PIOP | 19,330.984 ms |
| Metal-hybrid PIOP | 8,667.648 ms |
| PIOP speedup | 2.203x |

The current Metal backend still selects the optimized CPU address kernel. In one
representative accepted trace, preparation took 960.230 ms and all eight rounds
together took 0.698 ms. The only material target is therefore preparation.

Those values describe the PIOP holdout at the pre-integration revision, not the
standalone kernel evaluator's denominator. The pre-ABI-audit `2^26` standalone
run at `inner_log2 = 15`, six selectors per tile, and 1,024 tile threads used
five alternating pairs and measured:

| Quantity | Median |
|---|---:|
| Standalone optimized CPU mirror | 194.97 ms |
| Complete Metal-hybrid member | 59.02 ms |
| Median paired speedup | 3.255x |

The CPU and Metal member medians are reported independently, so their ratio
need not equal the median of the five paired speedups. That run was exact for
its inputs and missed the 4x floor, but it used the old noncanonical selector
schedule (16 ascending instruction chunks, one bytecode chunk, and three RAM
chunks). It is mechanism-search evidence only. It cannot satisfy the current
evaluator's production-selector guard.

The corrected canonical run
`benchmark-runs/metal-autoresearch/booleanity-address-v2` uses 16 descending
instruction chunks, two bytecode chunks, two RAM chunks, and nine fused-inc
selectors. Across five independent evaluator executions, each containing five
alternating pairs, its controller median was 3.620x. A representative execution
measured a 198.833-ms CPU median, a 54.952-ms Metal median, a 3.618 ratio of
medians, and a 3.604 median paired speedup. It was exact and specialized, but
`promotion.local_eligible` was false because it did not clear 4x. This is the
current equal-input baseline.

Geometry probes found 512 tile threads faster than 1,024. Before compile-time
selector specialization, its best single sample was about 55.18 ms; after
specialization, the best single sample was about 53.53 ms. These samples also
predate the selector correction and are exploratory observations, not promotion
evidence. The `inner_log2 = 14`,
`inner_log2 = 16`, and three-selector tile probes were slower. Native 64-bit
atomic fetch-add was also rejected by the runtime Metal compiler, so it is not
an available accumulator candidate on this toolchain. The production default
is now 512 tile threads; it still requires a clean alternating five-pair run.

At production geometry, `T = 2^26`, `K = 256`, and there are 29 checked sparse
columns in ABI order: 16 instruction chunks at shifts `120, 112, ..., 0`, two
bytecode chunks at shifts `8, 0`, two RAM chunks at shifts `8, 0`, eight
fused-inc chunks at shifts `0, 8, ..., 56`, and the fused-inc carry. The device
already holds the required 40-byte cycle row for the stage-6b Booleanity kernel.

## Exact computation

For selector `i` and address bucket `k`, preparation produces

```text
G_i[k] = sum_j eq(reference_cycle, j) [hot_i(row_j) = k].
```

Split the 26-variable equality point after the first 11 variables:

```text
j = x_out * 2^15 + x_in
eq(reference_cycle, j) = E_out[x_out] * E_in[x_in].
```

For each `x_out`, the device accumulates `E_in[x_in]` by selector and hot
bucket, reduces each bucket exactly, multiplies it by `E_out[x_out]` once, and
writes a compact partial. A final reduction sums the partials over `x_out`.
This is the same regrouping as the optimized CPU kernel; no protocol value or
round schedule changes.

The split is intentionally not balanced. An inner block of `2^15 = 32,768`
rows matches the retained direct-address accumulator geometry, reduces partial
storage by 4x relative to a `13 + 13` split, and keeps the exact overflow count
within the existing bound.

## Threadgroup design

One field accumulator uses five atomic `u32` words: four wrapping limbs and a
fifth word counting `2^128` carries. Six selectors therefore occupy

```text
6 selectors * 256 buckets * 5 words * 4 bytes = 30,720 bytes.
```

That fits the M4 Max's 32 KiB threadgroup store. One threadgroup processes one
`x_out` block for one six-selector tile. Both 512 and 1,024 threads are admitted
by the compiled pipeline, but 512 is the measured default: 16 SIMD groups share
the accumulator while thousands of independent threadgroups supply device-wide
parallelism at target size. The compiled pipeline's execution width, maximum
threads, static memory, dynamic memory, and total threadgroup memory are part of
the evaluator fingerprint. Register pressure and measured residency remain the
principal hardware risks. Metal does not expose enough occupancy data here to
derive resident threadgroups per core from the 30 KiB allocation alone, so this
is an admission ceiling, not a demonstrated occupancy figure. Five selector
tiles cover all 29 columns.

Each input term is canonical and a bucket receives at most `2^15` terms. The
fifth word is therefore at most `2^15`; multiplying it by the Solinas offset
fits in `u64`. Mapping that carry count through `2^128 = 0xffffa7f7 (mod p)`
and reducing the low limbs gives the exact field sum. There is no lossy field
atomic and no dependence on row ordering.

The tile output layout is `[x_out][local_selector][bucket]`. It gives each tile
a contiguous write and lets the final kernel use 256 bucket lanes per selector.
Four shards per bucket keep the finalizer at 1,024 threads; its 16 KiB dynamic
shared reduction is below the accumulator footprint. Each tile is finalized
before the next tile overwrites the same 48-MiB partial buffer. All five
tile/finalize pairs remain in one command buffer, so there is one host wait.

## Traffic and arithmetic ceiling

For `P = 29`, selector tile width `S = 6`, `I = 2^15`, and
`O = T / I`, the modeled work is

```text
selector_tiles = ceil(P / S)
row_bytes       = 40 * T * selector_tiles
e_in_bytes      = 16 * T * selector_tiles       # cache-logical
partial_bytes   = 16 * P * K * O
field_adds      = P * T
bucket_muls     = P * K * O.
```

At `T = 2^26`, `O = 2,048`:

| Quantity | Value |
|---|---:|
| Selector tiles | 5 |
| Resident-row reads | 12.500 GiB |
| Cache-logical `E_in` reads | 5.000 GiB |
| Logical partials across five tiles | 0.2266 GiB |
| Owned reusable partial buffer | 0.0469 GiB |
| Logical partial write plus read | 0.4531 GiB |
| Exact field additions | 1,946,157,056 |
| Nominal four-limb atomics | 7,784,628,224 |
| Post-bucket field multiplications | 15,204,352 |
| Final output | 118,784 bytes |

The cache-optimistic traffic is 12.953 GiB, or 30.8 ms at the measured
420.68-GiB/s copy roof. Charging every repeated `E_in` access to DRAM gives
17.953 GiB and a 42.7-ms floor, although `E_in` is only 512 KiB and is shared
by every outer block.

The closest measured primitive is the heat-soaked direct-address accumulator:
14.69 ms for `2^26` rows with up to three five-limb field additions per row.
Scaling its field-add count by `29 / 3` gives about 142 ms. This is not a
throughput guarantee--the new kernel has five row passes and more tile
initialization--but it is the conservative empirical planning term. The 15.2
million bucket multiplications are below 1 ms at the retained 16.4-Gmul/s
whole-device field roof when fully distributed; the tile shader performs them
before writing partials so the six-group finalizer does not serialize
multiplication onto only part of the GPU.

The initial PIOP-member planning budgets from the 939.917-ms CPU holdout were:

| Local target | Complete wall budget |
|---|---:|
| 4x architecture floor | 234.98 ms |
| 5x working target | 187.98 ms |
| Match current cycle Booleanity (5.768x) | 162.95 ms |

The closed standalone evaluator now uses the optimized CPU mirror as its local
denominator. In the representative canonical run, the 198.833-ms CPU median
sets a 49.71-ms 4x wall budget (39.77 ms for 5x). The 54.952-ms Metal median
therefore does not promote,
even though it is far below the original architecture budget. Four times is not
a stopping point: once a configuration clears 4x cleanly, tuning continues if
the component timings and roofline show a clear larger gain.

At `2^28`, the resident rows occupy 10 GiB, the reusable six-selector partial
buffer grows to 0.1875 GiB, and the five tiles move 0.9063 GiB of logical
partials. No unsegmented buffer approaches the observed 80.64-GiB Metal limit.
Aggregate proof residency remains the admission constraint.

## Residency and host boundary

Stage 5 already parks `BooleanityRows` for the bytecode and Booleanity cycle
members. Stage 6a borrows the same allocation and leaves it parked for stage
6b. The address path must record the allocation identity and may not upload,
project, repack, or consume a second row plane.

Only these new buffers are owned by address preparation:

- 29 eight-byte selector descriptors;
- the 512-KiB `E_in` and 32-KiB `E_out` tables at `2^26`;
- one 48-MiB reusable six-selector partial table;
- the 118,784-byte result table.

After one command buffer completes, the result table is read once. The host
constructs the existing optimized address kernel from those masses, proves the
eight tiny `K`-domain rounds, draws Fiat--Shamir challenges, and performs the
normal final relation check. No transcript operation occurs on the device.

Before dispatch, unsupported geometry or working-set admission may select the
optimized CPU kernel. After a command is committed, any failure aborts the proof;
it does not retry from partially executed device state.

## Fixed evaluator

The corrected standalone evaluator freezes the production 29 selectors in the
exact PIOP ABI order above, with mixed hot/cold bytecode and RAM rows, signed
fused increments, and adversarial field values. Selector count, kind, shift,
and order are an `all_exact` guard. Its CPU denominator is a standalone
optimized mirror: parallel `TensorEqTable` pushforward using
`AkitaAccumulator`, followed by the same eight host address rounds used by the
Metal arm. It is not a timed invocation of the production CPU kernel.

The primary Metal metric is complete member wall time. Its exclusive timed
components are weight-table preparation/allocation/upload, one command's
encoding/submission/completion wait, one result readback, the host rounds, and a
reported unattributed timer remainder. The GPU timestamp interval is nested
inside command wall time and is never added as another component. Resident-row
creation and row construction are outside both timed arms; the same resident row
allocation is reused for every Metal sample. One exact CPU/Metal warmup pair is
reported separately and excluded from the alternating samples.

The production evaluator reports two member ratios. The raw service ratio uses
the deployed CPU and Metal members exactly as they occur inside PIOP. The CPU
backend currently reconstructs its packed rows in that member, while Metal
reuses the stage-5 allocation, so this ratio includes the residency dividend.
For the equal-input diagnostic, a required nested
`OptimizedBooleanityAddress::row_source` span wraps only
`shared_instruction_rows`; its duration is subtracted from the CPU member. The
Metal arm must contain no such span. Full PIOP and raw service timings are never
adjusted. The standalone equal-input result remains the authoritative 4x search
gate because subtraction is an attribution control, not a third resident-CPU
measurement arm.

Promotion requires:

- every one of the `29 * 256` masses equals the independent optimized CPU
  mirror, and every warmup/timed mass vector has exactly that length;
- all four sampled evaluations in each of eight address rounds, the round
  polynomials, host challenges, final claim, and transcript state match;
- exactly `ceil(29 / S)` ordered tile/finalize pairs (five when `S = 6`), one
  command completion, one result readback, and no per-row contribution buffer;
- the input row allocation identity remains stable and is reused by stage 6b;
- the requested/effective `2^15` split, selector tile width, thread width,
  pipeline execution limits, static and dynamic scratch bytes, buffer sizes,
  specialization status, and command completion are reported;
- every raw member/component array has the declared cardinality and reconciles
  to its enclosing wall timer;
- a local promotion candidate uses at least five alternating pairs at `2^26` or
  above and clears the 4x paired-speedup floor;
- final promotion also survives the separate production PIOP holdout.

`all_exact` covers correctness, accounting, geometry, pipeline, and resource
integrity only. A correct exploratory run below `2^26`, with fewer than five
pairs, or below 4x remains a valid search result. The separate
`promotion.local_eligible` field combines `all_exact` with those three
performance gates.

The oracle has explicit limits. The Rust pushforward is independent of the
Metal shader and is the primary exactness check. Both arms call the same local
host-round implementation, so agreement of round evaluations, challenges, and
transcript state checks deterministic propagation from the masses; it is not an
independent second implementation of the sumcheck. The standalone evaluator
also infers one-command, one-readback, and no-per-row-buffer behavior from the
fixed invocation API rather than runtime command counters. Production tracing
and the PIOP holdout provide those integration checks.

Against the earlier accepted PIOP medians, replacing the 942.588-ms member by a
4x, 5x, or 5.768x result projected PIOP speedups of roughly 2.429x, 2.443x, and
2.451x. The production holdout must recompute this projection after integration;
the standalone mirror is not substituted into the PIOP total. This kernel is
necessary but cannot complete the 4x portfolio goal by itself.

## Criterion microbenchmark

The reusable prepared-kernel boundary is available through the standard
`jolt-kernels` Criterion bench:

```bash
JOLT_SOLINAS_BENCH_FAMILY=booleanity-address \
JOLT_METAL_BOOLEANITY_ADDRESS_LOG_N=26 \
cargo bench -p jolt-kernels --features metal --bench metal_solinas
```

Rows and invocation storage are prepared once. Each timed iteration executes
the five tile/finalize pairs and performs the allocation-free 118,784-byte
readback. Before timing, all masses are checked once against the parallel CPU
pushforward. Criterion throughput counts the `29 * T` useful selector-row field
additions; the benchmark also reports logical row traffic and effective launch
geometry. Sequence preparation, row construction, and host address rounds are
outside this microbenchmark and remain covered by the standalone complete-member
evaluator.

## Excluded first candidates

- A full `P * K` threadgroup table needs 148,480 bytes and cannot reside.
- One selector per dispatch rereads the 2.5-GiB row plane 29 times.
- A per-row contribution stream materializes the selector expansion and loses
  the useful-operations-per-read advantage.
- Per-row multiplication by a materialized full equality table discards the
  tensor factorization and adds a large resident weight plane.
- Moving the eight address rounds to Metal adds synchronization around host
  Fiat--Shamir to work over only 256 initial entries.

Those alternatives can be revisited only if the fixed evaluator falsifies the
six-selector exact-accumulator schedule.
