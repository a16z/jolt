# Bytecode read/RAF address successor v2

The production target is a fixed address-major topology with one Metal
threadgroup owning each bytecode address. It writes the nine pushforward
outputs once, without global output atomics, while the optimized host shell
retains all 13 address rounds and every Fiat--Shamir operation. The existing
CSR implementation is only an immediate asynchronous shadow and topology
control; it is not a candidate for the final backend. Nothing in this
successor packet is registered as a kernel, compiled into the production Metal
source, or GPU-executed.

The uncovered slot is `bytecode_read_raf_address` in stage 6a. The existing
stage-6b cycle kernel is separate and remains unchanged. The fixed owner must
fit a `27.735322 ms` complete-address budget at Fibonacci `log_T = 26` to let
the measured read/RAF family reach 7x. A standalone 5x result is necessary but
is not the architecture target.

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

Stage 6a constructs this relation from the stage-1 through stage-5 output
points, then draws the six bytecode challenges before calling the batched
sumcheck prover. `begin_batch` and every subsequent round remain host-owned:
the host absorbs the combined round polynomial and draws the next challenge.
Submitting or joining a shadow command must not absorb a transcript value,
draw a challenge, or advance protocol state.

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

## Measured denominator and wall budgets

All numbers in this section are measurements. The frozen artifact is
`benchmark-runs/metal-piop-eval/20260807-072030-939267/result.json`, revision
`2447f0f619680d8e938300a26f9a0fc916aaaeb6`, binary SHA-256
`ae4859e9e6e39725d5b76412fe75639c102cbb5a7a8d490d95be0e365fcee197`.
It is a clean, acceptance-eligible Fibonacci `log_T = 26` production run on an
M4 Max with 16 Rayon threads. The optimized CPU address samples, extracted
from `BytecodeReadRafAddressPhase`, are

```text
210.167999, 194.299459, 180.933333, 204.085127, 209.662249 ms.
```

The median is `204.085127 ms`. The same five pairs give a CPU cycle median of
`1,033.187499 ms`, a Metal cycle median of `147.619957 ms`, and a paired CPU
address-plus-cycle median of `1,227.486958 ms`. The current hybrid family,
which still runs address on the CPU and cycle on Metal, has a `342.566540 ms`
median and a 3.58321x ratio of medians.

| Measured-denominator goal | Total ceiling | Address wall available after the measured Metal cycle |
| --- | ---: | ---: |
| standalone address 5x | `40.817025 ms` | `40.817025 ms` |
| standalone address 7x | `29.155018 ms` | `29.155018 ms` |
| read/RAF family 5x | `245.497391 ms` | `97.877434 ms` |
| read/RAF family 7x | `175.355279 ms` | `27.735322 ms` |

The table uses strict whole-nanosecond caps, displayed in milliseconds.
The fixed address-owner target is the last cell: complete address wall at or
below `27.735322 ms`. The standalone 5x cap of `40.817025 ms` is the minimum
promotion bar, not permission to stop. These are complete-member wall budgets,
not GPU-active budgets. If the accepted cycle denominator changes, refresh all
four derived ceilings from one new clean alternating artifact.

The `7.918251 ms` sum of 13 host `prove_round` spans is a measured proxy from
the older 2026-08-06 artifact, not from the frozen denominator above. It is
retained only for analytical screens until the narrow host shell is measured;
it excludes shell construction, equality work, `finish_rounds`, output claims,
and transcript overhead.

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
upload those rows. Stage-5 resident-row admission must include the bytecode
address cutoff; otherwise address-only Metal configurations can discard the
producer before stage 6a.

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

## Phase 1: CSR asynchronous shadow and census control

The tracked `metal/solinas/bytecode_read_raf` CSR path is exact and executable,
but today it is probe/test-only and assembled outside the production shader
source. Its retained screening measurements are controls, not projections for
the fixed successor:

| Measured CSR control at log 26 | Result |
| --- | ---: |
| isolated full-width worker GPU active | `9.164542 ms` |
| isolated exact-u64 worker GPU active | `9.365500 ms` |
| one-active-address complete slice | `20.941375 ms` |
| 26-active-address CSR-only GPU active | `11.419042 ms` |
| 26-active-address complete slice | `29.109917 ms` |

The full-width isolated worker beat exact-u64 by 2.15% in that CSR layout.
The 26-address fixture had 53,248 runs and maximum run length 1,261. None of
these measurements used the production Fibonacci topology or the production
host shell, so they establish no address speedup. In particular, the
`29.109917 ms` slice already exceeds the `27.735322 ms` family target before
host rounds; CSR is not the endpoint.

The smallest production shadow reuses the stage-5 row handle and changes no
protocol output:

1. After the nine stage points are known, upload only the existing split
   equality tables and submit one CSR-build, indirect-worker, finalize command.
   Row upload bytes and row allocations must both be zero.
2. Split the synchronous CSR runtime into `submit` and `join`. Run the current
   optimized CPU address preparation while Metal is in flight; the CPU tables
   remain authoritative.
3. Join once after CPU preparation. Require `completed_before_join = true` for
   recurring shadow runs, record any exposed join wait, and compare all
   `9 * 8192 = 73,728` field elements exactly.
4. Read or device-reduce the CSR run descriptors. They contain outer, address,
   and count, so they can be reclassified at the fixed successor's `tau = 32`
   even if the CSR worker keeps its own threshold.

The production census record is versioned by `log_T`, `I`, `O`, `K`, and
`tau`, and contains:

```text
U_s, U_l                         short and long nonempty cells
S_s, S_l                         useful occurrences in each class
B_s                              short batches of 32 consecutive outers
P_s = 32 * sum(max short count per batch)
P_l = sum(32 * ceil(long count / 32))
max_count, count_histogram
active_addresses_per_outer histogram and min/median/p95/max
short_lane_padding = P_s / S_s
long_lane_padding  = P_l / S_l
```

A padding ratio is `null`, not zero, when its class has no occurrences.

The shadow also records command completion, GPU active/wall time, output
readback bytes, descriptor/census readback bytes, device registry ID, stage-5
and stage-6a storage IDs, and row allocation/upload counts. It must not absorb
or draw Fiat--Shamir state, delay CPU preparation, or replace a CPU value. Once
the real census and repeated full-table parity are captured, stop tuning CSR
and implement the fixed owner.

## Topology construction

One 1,024-thread group owns one 32,768-row outer block. The bring-up control
uses exactly 8,192 threadgroup `atomic_uint` bins, or 32 KiB:

1. clear the bins;
2. scan the block and count masked pushed PCs;
3. scan the 8,192 counts, write address-major packed cells, and replace counts
   with local scatter cursors;
4. scan the same block again and scatter `inner_sign` plus `magnitude` into the
   outer block's compact streams;
5. publish one completed-group count and fail-closed invalid-row status.

That two-scan builder is a control, not the target dataflow. The production
path publishes the same per-outer address counts while the stage-5 row
producer is already visiting each row. The topology builder then starts at
step 3, scatters from the authoritative resident rows, and charges count
production exactly once in the whole-PIOP ledger. If producer counts require a
second row walk, duplicate row pack, or a second row upload, they do not count
as reuse and the two-scan control remains the honest model.

All count and cursor contention is in threadgroup memory. There is no global
count matrix, descriptor list, bidirectional arena, indirect dispatch, or
global output atomic. Every cell and occurrence slot is overwritten, so the
buffers do not need a full-size clear.

This is the only mechanism reused from the earlier CSR proposal: grouping by
`(outer, address)` is algebraically valid because `E_hi_s[outer]` is constant
over the run. The new layout separately charges its own storage and two row
passes in the control, or the producer-count handoff plus one scatter pass in
the target. The prior compact-run arena and nine-accumulator output atomics are
not reused.

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

Let `U_s` and `U_l` be short and long nonempty cells, and let `S_s` and `S_l`
be their useful occurrence totals. Let `B_s` be the number of `(address, 32
consecutive outer cells)` batches containing a short run. Define

```text
P_s = 32 * sum over short batches(max short count in the batch)
P_l = sum over long runs(32 * ceil(count / 32)).
```

The topology census records these values and the distributions specified in
the shadow contract, not only total `U = U_s + U_l`. The exact selected
schedule has:

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

## Analytical movement lower bounds

The byte counts below are analytical. The `451.701710520 GB/s` denominator is
a measured M4 Max streaming-copy rate. Keep three traffic views separate.

Topology construction requests

```text
pass 1 PC word                     8N
pass 2 PC/sign plus magnitude     16N
compact stream writes             12N
packed cell writes               4OK
-------------------------------------
shader-requested total            36N + 4OK.
```

The producer-count target removes the first standalone PC scan and is modeled
as `28N + 4OK`. A conservative cache-line sensitivity charges two complete
40-byte row pulls, or `92N + 4OK` including stream writes. These are not
interchangeable roofs; producer work hidden under another span still remains
charged once in whole-PIOP accounting.

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
| two-scan topology request | 2,483,027,968 | 5.497052 ms |
| producer-count topology target | 1,946,157,056 | 4.308501 ms |
| topology uncached-row sensitivity | 6,241,124,352 | 13.816916 ms |
| worker physical unique minimum | 878,608,384 | 1.945108 ms |
| worker requested, minimum `U` | 10,806,001,664 | 23.922871 ms |
| worker requested, maximum `U` | 13,221,625,856 | 29.270701 ms |

The requested worker rows are a cache sensitivity, not a DRAM lower bound.
Promotion needs GPU counter evidence for external bytes, L2 hit behavior, and
the row-builder's scatter pass. At 80% of the measured copy rate, the
producer-count topology and worker-unique caps are `5.385626 ms` and
`2.431385 ms`, respectively.

## Measured controls and analytical ceiling

The following are measured M4 Max controls retained from the existing probe
artifacts. They were not measured inside the fixed five-accumulator worker:

| Measured control | Rate |
| --- | ---: |
| streaming copy | 451.701710520 GB/s |
| isolated signed-magnitude 128-by-64 chain | 70.417 Gterm/s |
| isolated full-field chain | 45.709 Gproduct/s |
| signed-u64 admission control | 26.272 Gterm/s |
| register-constrained full-field control | 18.10 Gproduct/s |
| conservative command boundary | 0.141 ms |

The projections below are analytical. They use the synthetic 26-active-address
census (`U = 53,248`, maximum run 1,261), producer-count topology, the stale
`7.918251 ms` host proxy, and two command boundaries. They include product and
traffic screens but exclude matched field additions, SIMD reductions,
threadgroup atomics, register loss, shell construction, transcript overhead,
and counter-measured external traffic.

| Analytical rate assumption | 80%-roof complete-address projection |
| --- | ---: |
| exact-u64 at isolated controls | `18.489 ms` |
| full-width at isolated full-field control | about `21.105 ms` |
| exact-u64 at admission controls | `26.718 ms` |

The `26.718 ms` row leaves about 1.0 ms under the `27.735322 ms` target before
the omitted work. It shows that the fixed topology can plausibly reach the
target only if production topology, addition/reduction rates, caching, and
occupancy are favorable; it is not a latency prediction. Replace the
synthetic census with the shadow's Fibonacci census and add matched controls
before shader promotion. The implemented result must land within 80% of the
recomputed topology-aware roof, or the model must be revised with counters.

At log 26 the topology control launches 2,048 groups. Its 32-KiB
threadgroup-memory request is expected to limit it to one group per core, but
that group exposes 32 SIMDgroups and more than 50 waves across 40 cores. The
worker launches 8,192 groups of eight SIMDgroups. Its default `5 + 4` tiling
has five four-word field accumulators per lane; the acceptance target is at
least two resident worker groups per core with no spills. Compiler/ISA capture,
not the source estimate, decides occupancy. If five accumulators spill, test a
`3 + 2 + 4` tiling only after pricing the extra occurrence pass.

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

Use `2^20` only as an initial experimental cutoff. Freeze the production
switchover from alternating complete CPU/Metal member pairs at `2^19` through
`2^23`; the first scale whose lower-confidence result consistently favors
Metal becomes the cutoff. Below it, on capacity rejection, or for any
unsupported relation shape, select the complete optimized CPU member before
transcript mutation. Log 27 and log 28 are required capacity and scaling
checks even though log 26 is the promotion denominator.

## Implementation slices and production seams

1. **Async CSR shadow.** Register the existing CSR shader only for an explicit
   shadow mode, split its runtime into submit/join, borrow `BooleanityRows`,
   compare full outputs, and emit the census above. No CSR output reaches the
   prover.
2. **Narrow host shell.** In `optimized/bytecode_read_raf.rs`, separate CPU
   pushforward preparation from construction of the private address-round
   shell. The latter accepts nine precomputed canonical tables plus the entry
   PC and owns the unchanged 13 host rounds. Both CPU and Metal paths use it.
3. **Fixed producer and worker.** Implement the address-major cells, compact
   streams, device equality expansion, and address-owned worker. First validate
   a two-scan builder; then wire producer counts and charge their real source.
4. **Resident backend.** Add an address config/slot in `metal/backend.rs` and
   `with_metal_compute`; include its cutoff in the stage-5 admission logic in
   `metal/instruction_read_raf.rs`; borrow the same row allocation in stage 6a.
5. **Source and organization.** Replace the mixed
   `metal/bytecode_read_raf.rs` with
   `metal/bytecode_read_raf/{mod.rs,address.rs,cycle.rs}`. Put the accepted
   address ABI/runtime/shader under one Solinas module, register its fragment
   and pipelines in `metal/solinas/source.rs`, and add the address slot to
   `metal/kernel_registry.json`. Keep CSR named as experimental shadow until
   retirement.
6. **Evaluator.** Add explicit address mode/cutoff controls and lifecycle spans
   to `jolt-prover/examples/modular_benchmark.rs`. Extend
   `scripts/metal_piop_eval.py` and its tests with the standalone address
   member, address-plus-cycle family metric, census, allocation identity,
   zero-row-upload, async completion, and proof guards.

`crates/jolt-prover/src/stages/stage6a.rs` is the prover integration point. The
verifier relation, transcript schedule, proof shape, and public protocol do not
change.

## Exact parity oracle

The independent oracle uses canonical arithmetic modulo
`2^128 - 0xffff_a7f7` and imports no Jolt field, polynomial, sumcheck, or
optimized implementation. Before performance work, it checks absent PC, PC 0,
PC 8191, invalid PC fail-closed behavior, every metadata flag position, zero
increment, both signs of `u64::MAX`, and the entry row.

Promotion parity is exact, with no tolerance:

- compare every fixed cell and compact occurrence against the topology oracle;
- compare all 73,728 GPU pushforward fields with both the direct oracle and the
  optimized CPU preparation;
- starting from the same previous claim, feed identical challenges through all
  13 rounds and compare every univariate coefficient and every drawn
  challenge;
- compare final `intermediate` and all six raw committed `T0..T5` claims;
- verify complete optimized and hybrid proofs and retain canonical output
  checksums.

Any invalid status, nonzero reserved topology bit, PC outside `0..8191`,
incomplete producer count, allocation identity change, transcript mutation by
the shadow, or parity mismatch fails closed to the complete CPU member before
the first protocol message.

## Promotion and kill gates

The fixed backend promotes only after five alternating log-26 CPU/Metal pairs
from one clean binary. Both order strata must be present, relative MAD must be
at most 3%, neither first sample may decide the winner, no capacity fallback
may occur, and all parity and lifecycle guards above must pass. Every complete
Metal address sample must be at most `27.735322 ms`; `40.817025 ms` is only the
standalone research floor. The result must also be within 80% of its
Fibonacci-census roof and pass log-27/log-28 capacity and scaling runs.

Architecture kills are pre-registered:

- CSR stops after census/parity. If its real-census 80%-roof projection or any
  complete sample exceeds `27.735322 ms`, do not optimize it as a backend.
- Add a compact active-cell fallback to the fixed owner only if short-lane
  padding exceeds 1.25x and the fully charged model predicts at least 5%
  complete-wall improvement.
- Reject a fixed worker that spills, exposes fewer than two resident groups per
  core, or drives counter-measured external worker traffic above twice the
  `878,608,384`-byte physical unique minimum.
- Reject a layout that cannot admit log 28. Recompute rather than waive a roof
  when field-add, reduction, atomic, or cache counters invalidate the model.

Keep the variant search bounded:

- `5 + 4` stage tiling is the default. A single-pass nine-accumulator worker
  saves one `4N` index pass, about `0.594 ms` at the measured copy rate, but is
  retained only if it has no spills, preserves two resident groups per core,
  and wins complete wall by at least 3%.
- Retest full-width and exact-u64 arithmetic inside the fixed five-accumulator
  worker. The old CSR full-width result won by only 2.15%; keep exact-u64 only
  if it wins complete wall by at least 3%, otherwise retain full-width.

If the fixed backend clears the target while matched counters show another 5%
PIOP gain is clearly attainable, continue rather than stopping at the bar.

## Retirement after promotion

On the Metal-selected path, remove CPU `shared_instruction_rows`
reconstruction, both CPU `stage_pushforwards` trace walks, host equality-table
construction/upload, any duplicate resident-row allocation, and the CSR first
PC scan. Retire the CSR occurrence-index gather, run arena, indirect output
atomics, and finalize materialization once the fixed owner replaces it. Keep
the complete CPU implementation for cutoff and capacity fallback.

No protocol change is proposed. As a later internal-only experiment, the two
`T5` stages and two `1 - T5` stages may be preweighted into seven host-round
tables; this must preserve the raw nine-table parity boundary and all six
committed claims, and is deferred until the fixed nine-table path promotes.
