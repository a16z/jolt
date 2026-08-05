# Booleanity address pushforward

This note freezes the Metal architecture and evaluator contract for
`BooleanityAddressPhase`. It covers only the exact cycle-to-address
pushforward. The eight address rounds and Fiat--Shamir remain on the host.

## Measured boundary

The clean five-pair `2^26` Fibonacci production holdout at revision
`b4da2261a022820acfb4e9263e23a01795b78bb2` measured:

| Quantity | Median |
|---|---:|
| Optimized CPU equal-input member | 929.140 ms |
| Optimized CPU raw service member | 955.465 ms |
| Metal member | 111.635 ms |
| Equal-input paired speedup | 8.453x |
| Raw service paired speedup | 8.702x |
| Optimized CPU PIOP | 19,281.782 ms |
| Metal-hybrid PIOP | 7,630.265 ms |
| Paired PIOP speedup | 2.513x |
| PIOP plus backend witness preparation | 2.400x |

All five CPU proofs and all five Metal proofs verified. Both execution-order
strata cleared 4x: the median was 7.340x when Metal ran first and 8.581x when
the optimized CPU ran first. The raw service denominator includes CPU row
materialization; the equal-input metric subtracts only its attributed row-source
span. PIOP timing is never adjusted. Ratios of independently reported medians
need not equal medians of paired ratios.

The closed standalone search is recorded in
`benchmark-runs/metal-autoresearch/booleanity-address-v3`. Its accepted
`trial-009` used five independent evaluator executions, each with five
alternating pairs. Across those executions, the median CPU member was
207.125 ms, the median Metal member was 35.724 ms, and the controller's median
paired speedup was 5.552x. Every mass, address round, Fiat--Shamir challenge,
final claim, and transcript state matched. The production gate promoted the
same source snapshot.

The search began from a 3.665x canonical baseline. Six-way and pairwise atomic
interleaving lost to register pressure. Aggregating the three-valued signed
carry was useful, but the decisive improvement was exploiting exact, checked
common cases in the high fused-increment bytes. Selectors 24--26 share one
thread-local bucket-zero subtotal when all three computed hot indices are zero;
otherwise all three take the general atomic path. The subtotal is flushed into
three disjoint selector tables. The carry selector independently accumulates
its `-1`, `0`, and `+1` bins. This changes only how field sums are grouped.

Geometry probes retained `inner_log2 = 15`, six selectors per tile, 512 tile
threads, and 1,024 finalizer threads. The `inner_log2 = 14`, `inner_log2 = 16`,
three-selector tiling, four-word deferred sums, and native 64-bit atomics were
rejected by measurement or the runtime compiler. The production holdout
validated the retained geometry and one-command, one-readback contract.

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

The final production tile specializes five fused-increment selectors without
changing their tables. It computes the biased increment and signed carry once.
If the exact hot indices for shifts 32, 40, and 48 are all zero, their common
weight enters one thread-local five-word deferred sum; otherwise the three
weights take their ordinary selector-local atomic paths. At the end of the
thread's inner scan, that one subtotal is merged into bucket zero of all three
disjoint selector tables. The signed carry similarly enters one of three local
deferred sums and is flushed to buckets 255, 0, or 1. Reducing subsets before
merging them is exact because addition in the field is associative, and the
five-word representation retains every `2^128` carry before Solinas reduction.

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

The nominal atomic count describes the fully general direct-scatter path. On
the fixed target workload, the shared high-byte subtotal and three carry
subtotals replace four per-row selector chains. Including their per-worker
flushes leaves about 6.74 billion four-limb threadgroup atomics, roughly 13.5%
fewer than the direct schedule. The speedup is larger than that count suggests
because the removed high-byte and carry updates target a handful of highly
contended buckets.

The cache-optimistic traffic is 12.953 GiB, or 30.8 ms at the measured
420.68-GiB/s copy roof. Charging every repeated `E_in` access to DRAM gives
17.953 GiB and a 42.7-ms floor, although `E_in` is only 512 KiB and is shared
by every outer block.

The accepted standalone trial measured a 33.49-ms median GPU-active interval,
only 2.7 ms above the cache-optimistic 30.8-ms traffic floor. Its complete
35.72-ms member includes command wall time, weight preparation, readback, and
host rounds. The 15.2 million bucket multiplications remain below 1 ms at the
retained 16.4-Gmul/s whole-device field roof when fully distributed. On this
workload the accepted shader is therefore close to the optimistic traffic roof;
the production PIOP trace is slower because it runs in the full resident proof
working set and after other heat-producing members.

The final equal-input budgets and results are:

| Boundary | 4x budget | Measured member | Paired speedup |
|---|---:|---:|---:|
| Standalone exact mirror | 51.78 ms | 35.72 ms | 5.552x |
| Production PIOP member | 232.28 ms | 111.64 ms | 8.453x |

Four times remained a floor rather than a cap: the search continued through
the 4.393x and 4.797x accepted parents until the 5.552x standalone winner. The
separate production relation then cleared both order strata and promoted it.

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

The production member is now 1.46% of the 7.630-s Metal PIOP. Even removing it
entirely could save only that share, so more local tuning is not a useful route to
the portfolio target. The holdout moved from the 2.203x pre-integration checkpoint
to 2.513x. Reaching 4x at the current 19.282-s CPU denominator requires reducing
Metal PIOP to at most 4.820 s, a further 2.810-s saving from the remaining kernels.

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
