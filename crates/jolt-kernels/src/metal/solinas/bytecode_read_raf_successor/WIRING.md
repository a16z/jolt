# Bytecode read/RAF address successor

This packet replaces the unregistered run-arena proposal with a fixed
address-major topology. One Metal threadgroup owns each bytecode address, so
the nine pushforward outputs are written once and never updated with global
atomics. The optimized host kernel keeps the 13 address rounds and every
Fiat--Shamir operation. Nothing in this directory is registered or compiled.

The uncovered slot is `bytecode_read_raf_address` in stage 6a. The existing
stage-6b cycle kernel is a separate implementation and remains unchanged.

## Exact boundary

The packed Akita relation has `N = 2^log_T` cycle rows, `K = 2^13` padded
bytecode addresses, five base stages, and four fused-increment stages. For a
stage cycle point `r_s`, pushed PC `pc(j)`, and fused increment `inc(j)`, the
device produces

```text
F_s(k) = sum_{j: pc(j) = k} eq(r_s, j)                    s = 0..4
F_s(k) = sum_{j: pc(j) = k} eq(r_s, j) * inc(j)           s = 5..8.
```

An absent mapped PC pushes to address zero. This is the address-phase rule;
the cycle phase instead treats an absent mapping as a cold zero row. The two
rules must not share a convenience decoder.

The nine value sources are

```text
T0, T1, T2, T3, T4, T5, T5, 1 - T5, 1 - T5.
```

For batching challenge `gamma`, the outer stage weights are `gamma^0` through
`gamma^8`. Stage 0 has within-stage RAF weight `gamma^9`; stage 2 has
`gamma^8`; all other RAF weights are zero. The entry product has weight
`gamma^11`. The address summand is

```text
sum_s gamma^s * F_s(k) * (Val_s(k) + raf_s * Int(k))
  + gamma^11 * EntryTrace(k) * EntryExpected(k).
```

It has degree two in each address variable. The host computes `q(0)` and
`q(2)`, reconstructs `q(1) = previous_claim - q(0)`, absorbs the polynomial,
draws the next challenge, and binds low-to-high. The verifier opening point is
the reverse of the 13 encountered challenges. After the final bind, the
kernel returns the same `intermediate` scalar as the optimized CPU kernel.
Committed-program mode additionally returns the six raw bound `T0..T5`
values; the complemented uses of `T5` are not separate output claims.

The device boundary is therefore nine canonical Akita tables of `K` elements.
The host boundary is the existing address-round state. No challenge, round
message, transcript state, stage-value table, `Int` table, or entry table is
owned by Metal. A GPU address-round tail would add 13 host/device waits to
tables totaling less than 2 MiB and is rejected.

Source anchors are:

- `optimized/bytecode_read_raf.rs`, `OptimizedBytecodeReadRafAddress` and
  `AddressKernel`, for the production algebra and output contract;
- `reference/bytecode_read_raf.rs` for the direct definition;
- `metal/solinas/booleanity/mod.rs` for the resident 40-byte row allocation;
- `metal/instruction_read_raf.rs` for stage-5 production and session lifetime;
- verifier stage 6a for the reversed address output point.

`oracle.rs` uses its own canonical `2^128 - 0xffff_a7f7` arithmetic, equality
tables, direct row scan, topology scan, round messages, binds, and final
output. It imports none of those implementations.

## Frozen denominator and target

The optimized-CPU evidence is
`benchmark-runs/metal-piop-eval/20260806-133709-697013`, revision
`5f520c21e338632aa0bf5936ceb02be6c22fa40f`, binary SHA-256
`a8b5f918c4a86ebdd2e4be3da10511ea071df4ea4949a23e02e5b286397d0e8b`.
It is a production-eligible Fibonacci `log_T = 26` run on macOS 26.6, M4 Max,
with 16 Rayon threads. The five complete address-member samples are

```text
172.796544, 198.165708, 181.211502, 190.915958, 198.945292 ms.
```

The median is `190,915,958 ns`. Strict integer ceilings are:

| Goal | Complete address member |
| --- | ---: |
| 5x floor | `38,183,191 ns` |
| 8x stretch | `23,864,494 ns` |

Median CPU prepare is `182,930,333 ns`. The retained sum of the 13
`prove_round` spans is `7,918,251 ns`. That second number is the only current
host-tail proxy; it excludes shell construction, equality-table work,
`finish_rounds`, output construction, and transcript work. Complete alternating
member wall remains the acceptance metric.

The existing cycle member has CPU and Metal medians of `1,004.692916 ms` and
`160.876418 ms`. The paired address-plus-cycle CPU median is `1,203.638208 ms`,
with an 8x ceiling of `150.454776 ms`. The cycle Metal median alone exceeds
that ceiling, so this address design cannot establish 8x for the combined
read/RAF pair. If the address member lands at its 5x ceiling, the current pair
projects to about 6.05x. The cycle member must improve independently.

## Resident inputs and aliases

The authoritative source is the stage-5 `BooleanityRows` allocation. Each row
contains five `u64` words:

```text
0  lookup index low
1  lookup index high
2  remapped RAM address plus one
3  fused-increment magnitude
4  mapped PC plus one and flags
```

The low 56 bits of word 4 are `mapped_pc + 1`; zero means absent. Bit 63 is
the fused-increment sign. Bits 56--62 carry instruction table and RAF metadata
and are ignored only after masking. Word 3 permits the full `u64` magnitude.

The stage-6a adapter borrows the allocation, checks length, device registry,
and allocation identity, and leaves it in the `ProofSession`. Booleanity
address, bytecode cycle, Booleanity cycle, and the admitted stage-7 consumer
remain aliases of the same row allocation. No successor path may repack or
upload those rows.

The topology owner is a second session object with these immutable buffers:

```text
cell[k * O + j_hi] = { start: u16, count: u16 }
inner_sign[j_hi * I + slot] = inner_index | (negative << 31)
magnitude[j_hi * I + slot] = abs(inc)
```

Here `I = 2^15` and `O = N / I`. `start` is local to one `I`-row outer block.
Both `start` and `count` are at most 32,768, so the packed `u32` cell is exact.
The low 15 bits of `inner_sign` contain the inner index; all reserved bits must
be zero. PC is implicit in the address-major cell, and outer index is implicit
in the layout position. The occurrence streams therefore omit the original
row index and every unused row word.

Topology is transcript-independent. Build it as soon as the stage-5 row
allocation exists, but record its complete wall and active time. At the local
member boundary it is a resident producer; in the whole-PIOP ledger its work
is charged exactly once. Moving the build earlier is not a speedup unless its
incremental PIOP wall is reduced or hidden behind work that does not compete
for the same GPU resources.

The six host value tables alias `T5` exactly as the stage map above specifies.
Metal pushforward output never aliases row, topology, equality, or host value
storage. Failed admission selects the complete optimized CPU address member
before transcript mutation.

## Topology construction

One 1,024-thread group owns one 32,768-row outer block. It uses exactly 8,192
threadgroup `atomic_uint` bins, or 32 KiB:

1. clear the bins;
2. scan the block and count masked pushed PCs;
3. scan the 8,192 counts, write address-major packed cells, and replace counts
   with local scatter cursors;
4. scan the same block again and scatter `inner_sign` plus `magnitude` into the
   outer block's compact streams;
5. publish one completed-group count and fail-closed invalid-row status.

All count and cursor contention is in threadgroup memory. There is no global
count matrix, descriptor list, bidirectional arena, indirect dispatch, or
global output atomic. Every cell and occurrence slot is overwritten, so the
buffers do not need a full-size clear.

This is the only mechanism reused from the earlier CSR proposal: grouping by
`(outer, address)` is algebraically valid because `E_hi_s[outer]` is constant
over the run. The new layout separately charges its own storage and two row
passes. The prior compact-run arena and nine-accumulator output atomics are not
reused.

## Address-owned worker

One 256-thread group owns one address. It reads the `O` packed cells in tiles
of at most 2,048 entries. At log 26 that is one 8-KiB layout tile; at log 28 it
is four tiles. Partial output storage is 1,152 bytes for eight SIMDgroups by
nine fields, keeping static plus dynamic threadgroup memory below 12 KiB.

The initial short threshold is one SIMD width, `tau = 32`:

- For a short run, one lane owns one `(outer, address)` cell. A SIMDgroup
  processes up to 32 cells from consecutive outers. It reduces address
  contributions once per stage for the batch, not once per run.
- For a long run, one SIMDgroup owns the cell. Lanes stride across its
  occurrences and reduce the nine inner sums. After reduction, stages are
  spread across lanes 0--8 so the nine `E_hi` products issue as one masked
  field-product sequence instead of nine lane-zero sequences.

Stages are tiled `5 + 4`. The base tile reads only `inner_sign`; the fused tile
reads `inner_sign` and `magnitude`. Five field accumulators per lane replace the
nine-accumulator structural state. The second occurrence-index pass costs
`4N` requested bytes, about `0.594 ms` at the retained copy rate, before cache
effects. That is the accepted price for a materially lower spill risk.

For each run, the worker forms

```text
base_sum_s  = sum E_lo_s[inner]                         s < 5
fused_sum_s = sum E_lo_s[inner] * signed_magnitude     s >= 5
contribution_s = E_hi_s[outer] * sum_s.
```

The address-owning group accumulates contributions over outer blocks and
writes its nine canonical fields once. Device-generated split-equality tables
use the big-endian split

```text
eq(r_s, j) = E_hi_s[j >> 15] * E_lo_s[j & (2^15 - 1)].
```

The host uploads only nine 26-field points. Equality expansion and the worker
are encoded in one command buffer, followed by one completion wait before the
host address rounds.

## Exact work model

Let `U_s` and `U_l` be short and long nonempty cells. Let `B_s` be the number
of `(address, 32 consecutive outer cells)` batches containing a short run.
Define

```text
P_s = 32 * sum over short batches(max short count in the batch)
P_l = sum over long runs(32 * ceil(count / 32)).
```

The topology census records all six values, not only total `U = U_s + U_l`.
The exact selected schedule has:

```text
useful signed-u64 products       4N
useful full-field products       9U
issued signed-u64 product lanes  4(P_s + P_l)
issued full-product lanes        288 B_s + 32 U_l
issued accumulation add lanes    9(P_s + P_l)
issued SIMD reduction add lanes  1440(B_s + U_l) + 256K
topology threadgroup atomics      2N
equality-generation products      18(I + O - 2).
```

The full-product issued count is lower than `32 * 9U_l` because the long path
assigns nine stages to different lanes. The model rejects inconsistent census
aggregates. Per-cell validation remains a producer invariant; aggregate counts
alone cannot prove a realizable topology.

At log 26, the two useful-product extremes are:

| Topology | Runs | Issued signed lanes | Issued full lanes |
| --- | ---: | ---: | ---: |
| one full run per outer | 2,048 | 268,435,456 | 65,536 |
| four rows in every cell | 16,777,216 | 268,435,456 | 150,994,944 |

The dense case has `B_s = 524,288` and 754,974,720 batch-reduction lanes before
the small final address reduction. A sparse placement can have much worse
short-lane padding than either table row. That is why the actual census is a
pre-implementation gate.

## Movement lower bounds

Keep three traffic views separate.

Topology construction requests

```text
pass 1 PC word                     8N
pass 2 PC/sign plus magnitude     16N
compact stream writes             12N
packed cell writes               4OK
-------------------------------------
shader-requested total            36N + 4OK.
```

The unique minimum is `28N + 4OK` if the repeated PC word is retained. A
conservative cache-line sensitivity charges two complete 40-byte row pulls,
or `92N + 4OK` including stream writes. These are not interchangeable roofs.

The worker requests

```text
packed layout                      4OK
compact streams                    16N
E_lo requests                     144N
E_hi requests                     144U
output writes                     144K.
```

The nine `E_lo` tables are only 4.5 MiB at log 26, and the `E_hi` tables are
288 KiB. The absolute physical minimum is the successor-owned allocation
touched once: layout, compact streams, both equality sets, and output. Hardware
counters must decide whether repeated equality requests stay on chip.

At log 26:

| View | Bytes | Floor at 451.701710520 GB/s |
| --- | ---: | ---: |
| topology requested | 2,483,027,968 | 5.497052 ms |
| topology unique minimum | 1,946,157,056 | 4.308501 ms |
| topology uncached-row sensitivity | 6,241,124,352 | 13.816916 ms |
| worker physical unique minimum | 878,608,384 | 1.945108 ms |
| worker requested, minimum `U` | 10,806,001,664 | 23.922871 ms |
| worker requested, maximum `U` | 13,221,625,856 | 29.270701 ms |

The requested worker rows are a cache sensitivity, not a DRAM lower bound.
Promotion needs GPU counter evidence for external bytes, L2 hit behavior, and
the row-builder's second pass.

## Arithmetic, launch, and occupancy ceilings

Retained M4 Max controls are:

| Control | Rate |
| --- | ---: |
| streaming copy | 451.701710520 GB/s |
| isolated signed-magnitude 128-by-64 chain | 70.417 Gterm/s |
| isolated full-field chain | 45.709 Gproduct/s |
| signed-u64 admission floor | 26.272 Gterm/s |
| register-constrained full-field control | 18.10 Gproduct/s |
| conservative command boundary | 0.141 ms |

The isolated rates are theoretical arithmetic ceilings for this worker, not
matched results. The five-accumulator worker must retain at least the admission
rates in the same binary. Field-add, SIMD-reduction, and threadgroup-atomic
rates are still missing. `model.rs` therefore marks every projection
incomplete and reports only a necessary product-plus-traffic screen.

Including equality expansion, the product-only 100%-roof floors are:

| Rates | Minimum runs | Dense four-per-cell |
| --- | ---: | ---: |
| isolated peaks | 3.827227 ms | 7.129190 ms |
| admission floors | 10.255792 ms | 18.594434 ms |

The 80%-of-roof worker caps are 4.784034--8.911488 ms at the isolated peaks
and 12.819740--23.243043 ms at the admission rates. The latter dense value
leaves no comfortable standalone 5x margin once a separately charged topology
build and host tail are added. The actual topology and matched add/atomic
controls decide feasibility.

After the retained `7.918251 ms` host-round proxy and one command boundary, a
resident worker has `30.123940 ms` of 5x headroom and `15.805243 ms` of 8x
headroom. Charging a second topology command leaves `29.982940 ms` and
`15.664243 ms` for topology plus worker active caps.

The product/traffic-only charged screens span about 19.86--23.98 ms at the
isolated rates and 27.89--38.32 ms at the admission rates. The dense
admission row narrowly misses 5x before add and atomic costs; it is a
falsification warning, not a performance prediction. The dense isolated row
lands just above the strict 8x ceiling, so clear measured headroom must be
pursued but 8x is not pre-claimed.

The topology kernel has 2,048 groups at log 26. Its 32-KiB threadgroup-memory
request likely limits it to one group per core, still exposing 32 SIMDgroups
per resident group and more than 50 waves across 40 cores. The worker has
8,192 groups and eight SIMDgroups per group. Its structural live state is five
four-word field accumulators, one output accumulator for a stage lane, a
signed-u64 product temporary, and loop state. The target is at least two
resident worker groups per core with no spills. Compiler capture, not the
source estimate, determines the occupancy limiter. If five accumulators spill,
the next experiment is a `3 + 2 + 4` stage tiling; it must re-price the extra
occurrence pass.

## Capacity and cutoff

The new topology is smaller than the prior occurrence-plus-run arena while
also removing output atomics.

| Storage | log 26 | log 28 |
| --- | ---: | ---: |
| packed cells | 67,108,864 | 268,435,456 |
| inner/sign stream | 268,435,456 | 1,073,741,824 |
| magnitude stream | 536,870,912 | 2,147,483,648 |
| equality tables | 5,013,504 | 5,898,240 |
| pushforwards | 1,179,648 | 1,179,648 |
| successor-owned total | 878,608,384 | 3,496,738,816 |
| shared resident rows | 2,684,354,560 | 10,737,418,240 |
| aggregate | 3,562,962,944 | 14,234,157,056 |

Admission checks every buffer against `maxBufferLength`, the aggregate against
the recommended working set, execution width 32, 1,024 builder threads, 32 KiB
builder memory, and the worker pipeline limits. Log 28 uses four outer-layout
tiles rather than a 32-KiB per-address tile.

The initial CPU cutoff is `2^20` rows. The frozen linear extrapolation of CPU
prepare plus the retained host-round proxy is 10.777 ms at `2^20`, 13.635 ms
at `2^21`, and 19.351 ms at `2^22`; those are planning values only. Freeze the
real cutoff from alternating complete-member pairs at `2^19`, `2^20`, and
`2^21`. Below cutoff, or for any unsupported relation shape, use the complete
optimized CPU member.

## Implementation ladder and falsifiers

1. **Topology census.** Add no shader. On the exact log-26 Fibonacci rows,
   record `U_s`, `U_l`, `B_s`, `P_s`, `P_l`, count histograms, short padding,
   and address support per outer block. Reject aggregate-only synthetic data.
2. **Producer slice.** Implement the fixed-cell builder and compare every
   cell and compact occurrence against `oracle.rs`. Record requested and
   counter-measured bytes, threadgroup atomic throughput, register count,
   spills, and active time.
3. **Worker slice.** Use prebuilt topology and prebuilt equality tables. Check
   all nine pushforwards against both direct and topology oracles, including
   absent PC, PC 0, PC 8191, zero increment, and both signs of `u64::MAX`.
   Capture short and long paths separately.
4. **Complete address member.** Expose a narrow optimized host shell accepting
   nine precomputed pushforwards. Compare all 13 messages and challenges,
   final `intermediate`, raw committed `val_stages`, and proof verification.
5. **Resident integration.** Publish topology from stage 5, preserve row
   aliases through stage 7, and add allocation/upload/identity telemetry.
   Alternate complete CPU and Metal members, then alternate whole PIOPs.
6. **Only if measured:** compact active short cells by address when fixed
   batches lose to padding. The compaction must win complete wall by at least
   3%; descriptor creation and storage are charged. Do not carry forward the
   previous run arena by default.

Pre-registered falsifiers are:

- actual short or long issued-row padding exceeds 1.25x useful rows and the
  compact-cell screen predicts at least 5% complete-member gain;
- the topology builder exceeds its topology-aware 80%-roof cap, or threadgroup
  atomics dominate after the second row scan is cached;
- the worker exceeds its topology-aware 80%-roof cap, spills, or exposes fewer
  than two resident groups per core;
- counter-measured worker external traffic exceeds twice the unique minimum
  and the complete member no longer clears 5x;
- producer plus worker plus host wall exceeds `38,183,191 ns` in any of five
  alternating log-26 pairs;
- complete proof parity fails, any reserved topology bit is nonzero, a PC is
  outside `0..8192`, status is incomplete, or an allocation identity changes;
- log-28 storage exceeds an admitted device limit;
- a fixed-layout result clears 5x but measured matched ceilings make 8x or a
  further 5% PIOP gain clearly attainable. In that case continue rather than
  stop at the floor.

No protocol change is proposed. The only ownership change is a
transcript-independent resident topology whose full PIOP cost remains
observable.
