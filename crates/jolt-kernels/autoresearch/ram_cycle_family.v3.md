# RAM cycle family v3: one owner, adaptive sparse execution

Status: production design packet. This file changes no protocol or backend
registration and contains no performance claim for unimplemented code. Exact
facts, measured observations, and projections are labeled separately.

## Decision

Build one proof-scoped `RamCycleFamilyOwner` during
`jolt_prover::backend_witness_prepare`. It is the only component allowed to
walk the RAM witness or publish RAM Metal storage. It owns the compact access
and increment streams, the two sparse merge graphs, the final-memory table,
and any optional dense projection selected from the same census. The four
members borrow typed leases from that owner; none may scan the witness, pack a
private row, or upload a second copy.

Execution is adaptive:

- `RamReadWriteChecking` uses a sparse cycle frontier, with a topology-fixed
  Metal prefix only while the frontier is large enough to amortize a round
  trip, then one host-sparse continuation through the address phase.
- `RamValCheck` uses the shared access/increment union tree. The retained
  Fibonacci topology is a host-sparse case, not a GPU-occupancy case. Larger
  topologies may use the same Metal-prefix/host-tail seam.
- `RamRafEvaluation` uses the exact host-sparse pushforward at low support,
  direct Metal at medium support, and a producer-bucketed Metal histogram at
  high support. The six-scan dense kernel remains a compatibility fallback.
- `RamOutputCheck` remains a resident CPU leaf at `K = 8192`. A Metal dispatch
  is admissible only inside a command and wait that another stage-2 member
  already owns, and only if an ablation measures the incremental cost below
  the CPU selector.

This is a hybrid Metal backend. Choosing CPU for a topology that cannot fill
the device is part of the architecture, not an error path. Every choice is
fixed from transcript-independent geometry, census, and a machine profile
before the first member polynomial is absorbed.

## Evidence classes

The rest of this packet uses three labels:

- **Exact** means algebraic identity, integer accounting from a named ABI, or
  an observation copied from a retained artifact.
- **Measured** means a timing or rate observed under the artifact's stated
  conditions. It is not a hardware specification or a future-kernel result.
- **Projection** means a model or implementation budget. It cannot promote a
  kernel without paired measurements.

The target artifact is
`benchmark-runs/metal-piop-eval/20260807-103715-208977/result.json`, revision
`2ed9ce265f00ca06120a7d4a46fb979ee07919b8`. It is one optimized-first
diagnostic pair at `log_T = 26`, `log_K = 13`, and 16 Rayon threads. Both
proofs verified, but the run is not acceptance-eligible.

The older five-pair artifacts and focused Criterion controls remain useful for
strict denominators. Promotion uses the smaller of the preregistered cap below
and one fifth of the same-revision paired CPU median. It never uses a slower
CPU run to loosen a cap.

## Fixed protocol boundary

The owner and execution lanes do not change any relation, challenge, degree,
or opening orientation.

`RamReadWriteChecking` proves, over address `k` and cycle `j`,

```text
eq(tau_low, j) * ra(k, j)
  * (val(k, j) + gamma * (val(k, j) + RamInc(j))).
```

It has 26 low-to-high cycle rounds followed by 13 low-to-high address rounds,
degree three, and outputs `(val, ra, inc)`. For cycle challenges `c_i` and
address challenges `a_i`, its opening point remains
`[reverse(a_0..a_12) || reverse(c_0..c_25)]`.

`RamValCheck` proves

```text
RamInc(j) * RamRa(r_address, j) * (LT(j, r_cycle) + gamma).
```

It has 26 low-to-high cycle rounds and degree three. Each message exposes the
same evaluations at `0`, `2`, and `3`; the running claim supplies the value at
`1`. Its final `LtCyclePlusGamma` check and the staged advice/program-image
opening order remain host-owned.

`RamRafEvaluation` computes

```text
R(k) = sum_{j: address(j) = k} eq(tau_low, j)
sum_k (lowest_address + 8k) * R(k) = ram_address_spartan.
```

Its 13 address rounds are degree two. The device or host pushforward produces
only `R`; the existing affine host continuation produces the round
polynomials, `RamRa`, and `UnmapAddress`.

`RamOutputCheck` proves

```text
eq(output_address, k) * io_mask(k) * (val_final(k) - val_io(k)).
```

It has 13 low-to-high address rounds and degree three. Under the certified
production layout, its first ten messages are exactly zero and its final
three rounds operate on eight folded values.

Fiat--Shamir stays on the host. A device command may end at a challenge
boundary but may not cross it.

## Current target-scale facts

At the target scale:

```text
T = 2^26 = 67,108,864
K = 2^13 = 8,192
A = remapped access records = 190
D = nonzero RamInc cycles = 77
P = adjacent increment-active pairs = 74
```

The retained `RamAccessCensus` has 27 levels, including leaves and root:

```text
sum E_b, b=0..26 = 2,665 distinct (block,address) states
sum G_b, b=0..26 =   826 nonempty cycle blocks
sum D_b, b=0..26 =   826 grouped 256-lane tiles.
```

These are **exact observations** from the target trace. Since each leaf is one
access at one cycle, `E_0 = G_0 = D_0 = A = 190`. Nonempty support has one root,
so the cycle-message totals are therefore

```text
V = sum E_b, b=1..26 = 2,475
H = sum G_b, b=1..26 =   636
sum G_b, b=0..25     =   825.
```

`D_b = G_b` at every level follows from equality of the aggregate sums and
the pointwise inequality `D_b >= G_b`. Every group contains at most 256
entries. More importantly, every level contains at most `A = 190` entries.
A 256-thread flat RAM read/write message therefore launches one threadgroup
per round on this trace. No shader tuning can turn that topology into a
device-occupancy workload.

The current shared collector writes these **exact** target-scale host payloads:

| Current payload | Bytes |
| --- | ---: |
| `u32[T]` addresses | 268,435,456 |
| `u64[T]` pre-values | 536,870,912 |
| `u64[T]` post-values | 536,870,912 |
| Three dense host columns total | 1,342,177,280 |
| Current resident `u32[T]` Metal address plane | 268,435,456 |
| `24A` access tape | 4,560 |
| `8D + 16D` increment activity | 1,848 |

The tape is currently retained only for `A <= 2^18`; `records() == None` is an
explicit sparse-lane rejection, never permission to rescan in a member.

### Current member walls and hard budgets

The latest diagnostic pair attributes these exact complete-member walls:

| Member | Optimized CPU | Recorded Metal arm | What the Metal arm does today |
| --- | ---: | ---: | --- |
| `RamReadWriteChecking` | 86,979,214 ns | 70,577,004 ns | submits RAF, then runs optimized CPU |
| `RamValCheck` | 278,143,584 ns | 267,514,794 ns | optimized CPU remains authoritative; Metal checks round 0 as a shadow |
| `RamRafEvaluation` | 80,615,083 ns | 208,127 ns | authoritative Metal command completed under overlap, then affine host tail |
| `RamOutputCheck` | 578,334 ns | 811,919 ns | optimized CPU in both arms |

The RAF member seam does not include the overlapped service as visible wall.
The same target trace recorded 8,540,917 ns of RAF GPU-active work,
667,808,042 ns of overlap, and a 24,708 ns ticket join. The later clean
resident control measured 6,996,225 ns for setup plus service and 30,475,833 ns
for a standalone 256-MiB address upload. Both views must remain visible.

Stricter five-pair or focused controls set the promotion budgets:

| Member | Denominator used for the fixed cap | Hard 5x wall | Pursuit wall |
| --- | ---: | ---: | ---: |
| `RamReadWriteChecking` | 86,979,214 ns latest diagnostic | 17,395,842 ns | 8,697,921 ns (10x) |
| `RamValCheck` | 234,656,875 ns five-pair median | 46,931,375 ns | 29,332,109 ns (8x) |
| `RamRafEvaluation` | 74,870,252 ns five-pair median | 14,974,050 ns | 9,358,781 ns (8x) |
| `RamOutputCheck` | 276,100 ns focused deferred CPU | 55,220 ns | 32,000 ns selected experiment |

The four hard caps sum to 79,356,487 ns, but every member must pass its own
cap. A fast RAF result cannot excuse a CPU placeholder elsewhere.

The later `5210979cb` sparse transfer installed the target selector and
measured the exact retained topology again. At log 26, read/write moved from
90.832168 ms on optimized CPU to 6.402416 ms on the host-sparse lane
(14.187171x), while value check moved from 246.143121 ms to 19.003418 ms
(12.952571x). These are one-pair diagnostics, not promotion evidence, but both
already clear the fixed member caps. Metal cycle rounds remain disabled for
this topology; the high-support crossover below is a separate candidate.

## One producer and one ownership DAG

`ProofSession` stores one value per concrete `TypeId` for the proof lifetime.
Use that seam directly: park one `Arc<RamCycleFamilyOwner>`, and let prepared
kernels clone typed leases into its ranges. Do not park sibling owner types
whose payloads can drift or be regenerated independently.

```text
one owner preparation
  -> one T-row cycle-witness chunk stream
     -> checked access/increment collector
        -> 24-byte access records
        -> sparse nonzero increment records
        -> optional dense CPU fallback columns
  -> one K-row RamValFinal read
     -> checked native u64[K] final-memory table
  -> one bottom-up pass over the compact records
     -> address-sensitive RW merge DAG: E_b
     -> address-free access/increment union DAG: G^u_b
     -> final touched-address list
     -> RAF compact view; optional high-support buckets
  -> optional census-selected dense projections, written directly
  -> RamCycleFamilyOwner + immutable RamCycleFamilyReceipt

stage 2
  owner -> read/write lease -> sparse cycle state -> host address tail
        -> RAF lease        -> host/direct/bucket pushforward -> host tail
        -> final-memory lease -> resident CPU output check

stage 4
  owner -> union-tree lease -> sparse value check

after stage 4
  drop the family owner unless another registered consumer has an explicit lease
```

The cycle collector extends the existing `RamAccessColumns::shared` traversal
rather than wrapping it in another pass. `RamValFinal` has address-domain
shape, so the owner reads its `K` entries once, checks that every field value
is a canonical `u64` word, and publishes the one native table shared by
read/write and output check. On Apple unified memory, compact Metal input
ranges are allocated in `StorageModeShared` and filled through their final
mapped address. `new_buffer_with_data`, a host pack followed by upload, and a
second scan of either domain are forbidden.

The first migration may retain the three existing host columns so optimized
CPU fallback remains available without rescanning. Their 1.342-GB write is
charged once in the producer diagnostic. Once all sparse paths are complete,
removing those columns is a separate, measurable migration. “No duplicate
scan” does not mean hiding the one unavoidable witness traversal.

### Owner payload

Keep the existing proven ABIs where possible:

```text
RamAccessRecord       24 B  {cycle:u32, address:u32, pre:u64, post:u64}
RamIncrementRecord    24 B  parallel u64 cycle and i128 increment storage
RamRwMergeEvent       32 B  low/high state ids, group id, absent checkpoints
RamRwGroupEvent       16 B  low/high increment ids and cycle-pair id
RamBlockMerge          8 B  low/high block-frontier ids
RamBlockLeaf           8 B  access and increment record ids
LevelRange             8 B  level offset and length
RamValFinalNative      8 B  one little-endian u64 per RAM word
```

Under those declared sizes, the target trace's base sparse payload is exactly
`24A + 24D + 8K = 71,944` bytes. The read/write projection adds
`32V + 16H = 89,376` bytes. A reusable block merge adds `8H = 5,088` bytes.
The reusable block leaves and level ranges bring the allocation-unique owner
payload to at most **169,848 bytes**. The current physical buffer layout is at
most **171,144 bytes** before allocator padding and receipt metadata.

The 169,848-byte value is a **projection conditional on the declared ABI**,
not a compiled allocation observation. A future ABI must replace the equation
with its physical allocation lengths. It is nevertheless the correct scale:
the target path does not need a 256-MiB address plane or a 1-GiB value-check
row.

An optional dense `u16[T]` metadata projection is allowed only for a
high-support value-check lane selected by the census. It encodes
`address + 1`, zero for no address, and increment sign; nonzero magnitudes stay
in the compact increment stream. Its physical target size is 134,217,728
bytes plus `12D` bytes and topology descriptors. It must be written directly
by the owner and is absent on the retained Fibonacci path.

### Receipt invariants

`RamCycleFamilyReceipt` is immutable and travels with every lease. It records:

- schema version, witness/source generation, `log_T`, `log_K`, device registry
  identity, and owner generation;
- `A`, `D`, every `E_b`, `G_b`, union `G^u_b`, and grouped tile count `D_b`;
- content digests, storage identities, byte offsets, lengths, and lifetimes for
  every published range;
- strict record order, complete non-sentinel access coverage, address bounds,
  and exactly one record per remapped cycle;
- exact equality between sparse increment activity and authoritative `RamInc`,
  including a nonzero increment on an unremapped/raw-zero cycle;
- access checkpoint continuity, final-memory reconstruction, and the native
  final-table source digest;
- witness rows visited, physical bytes written, host-upload bytes,
  full-domain-copy bytes, peak scratch, producer wall, owner join wall, and
  allocation count; and
- the preselected lane and cycle cutoff for each consumer.

The increment certificate is stronger than today's one-way flags: `RamInc`
remains authoritative even when no access record exists. The current
`prepare_witness` hook does not receive checked public I/O, so output-check
prepare derives a separate `RamOutputCertificate` from the immutable native
table and the relation's `PublicIoMemory`. It checks every public-mask cell
before transcript mutation and records the owner generation and table digest.
An aligned mask alone is not a certificate.

Before any member polynomial is absorbed, an absent or invalid receipt selects
optimized CPU. After the first polynomial is absorbed, a storage mismatch,
command error, or status bit is `SumcheckError::ComputeBackend`; replay from a
fresh CPU state would change the transcript and is forbidden. Every pending
ticket retains its borrowed allocations and waits on drop.

## Target-scale work and ceilings

The measured controls used only for projection are 451.701710520 GB/s for a
streaming copy, 18.1 billion full-field products/s for the register-constrained
Solinas class, 141 us for the retained command/wait control, and about 2 us per
host Fiat--Shamir round. These are observations, not M4 Max specifications.

### Read/write

For a flat cycle level, the successor model performs exactly
`8E_b + 2G_b` useful field products. A grouped level performs
`6E_b + 4G_b`. The target totals are therefore:

```text
all-flat products    = 8V + 2H = 21,072
all-grouped products = 6V + 4H = 17,394.
```

Grouped arithmetic saves only 3,678 products but expands the target from one
flat partial per round to 636 group partials across the cycle phase. The flat
schedule is the target choice. Its product roof is 1.164 us.

Using the current cycle-only traffic equation and the exact target aggregates,
the cache-case traffic before reduction-tree writes is 481,824 bytes; the
group-tuple miss case is 570,096 bytes. The final-address count cancels between
state and handoff terms. The copy-roof floors are 1.067 us and 1.262 us. These
are **derived lower bounds**, not wall predictions.

The existing all-Metal-cycle projection has about 5.82 ms of fixed setup,
wait, Fiat--Shamir, and CPU-address-tail control before useful device work.
The target launches one threadgroup per round, so the selected target lane is
host sparse (`cycle_cutoff = 0`). For larger traces, choose one monotone Metal
prefix and one host continuation:

```text
cycle_cutoff = argmin_b(
    measured Metal rounds [0,b)
  + one shared-state handoff
  + measured host-sparse rounds [b,26)
  + measured host address tail)
```

The choice is frozen before round 0. A provisional screen keeps host sparse
when `A <= 2^15` or every candidate Metal round has fewer than 32 flat
threadgroups. The latter is `E_b < 8192` at width 256. Calibration replaces
both constants with alternating measurements; the hard 17,395,842-ns member
cap still applies.

Metal passes for an admitted prefix are:

1. `ram_rw_seed`: records to initial `(ra,val)` and increment frontiers.
2. `ram_rw_flat_bind_message`: bind the prior challenge, form both Gruen hints,
   and write each parent once. Round 0 omits the bind.
3. `ram_rw_grouped_bind_message`: optional topology-selected alternative for
   wide groups; apply the shared split-eq weight after segmented reduction.
4. `ram_reduce_pair`: 32-way ping-pong reduction to `[q(0), q_infinity]`.
5. One host read, cubic reconstruction, absorb, and challenge per round.
6. At the cutoff, the host binds the last challenge itself and continues from
   the shared sparse frontier. No terminal GPU bind command is needed.

There is no per-round allocation, topology construction, or factor copy.

### Value check

The retained dense low-level sequence has an **exact** 537,083,904 useful
products, 6,447,842,552 accounted bytes, a 2,685,665,280-byte resident set,
and a 2,097,152-byte CPU handoff. It measured 31.106 ms without host
Fiat--Shamir, excluding a separately observed 113.846958-ms row upload. That
is useful dense-control evidence, but it is not the target architecture.

On the target, `increment_compatible = true`, so every increment cycle is in
the access tree and the exact union count is `G^u = 826`; the conservative
generic bound remains `G^u <= 826 + 27D = 2,905`. Accounting for split-LT
construction and rebinding, plus address-equality evaluation only at the 190
access leaves, gives:

```text
useful products = 54,711 + 2*H_lo = 54,737..55,983
logical bytes   = 1,679,720..1,699,656.
```

The upper compute and traffic floors are 3.09 us and 3.76 us. With at most 190
live blocks in any round, command/wait latency still dominates by roughly two
orders of magnitude; host sparse is the target lane.

The host and Metal implementations share one algorithm and frontier ABI:

1. Build `eq(r_address)` once (`16K = 131,072` bytes) and the three split-LT
   tables once (`48*8192 = 393,216` bytes).
2. `ram_val_seed` gathers one RA leaf from each access record and one increment
   leaf from the authoritative increment stream, including increment-only
   raw-zero cycles.
3. `ram_val_bind_message` walks the union parents, binds prior RA/increment
   state, evaluates the three products at `0`, `2`, and `3` while values are in
   registers, and writes each parent once.
4. `ram_reduce_triple` reduces three columns and returns 48 bytes.
5. The host absorbs the polynomial and supplies the next shared stage-4
   challenge. At the chosen cutoff it takes the frontier once and completes
   the same sparse loop locally.
6. The host performs the final `LtCyclePlusGamma` derived-table check and emits
   openings in the existing curated order.

The provisional host screen is `G^u <= 2^15` or fewer than 32 256-lane
threadgroups in every candidate round. A dense owner projection may compete
only above that screen and only if its producer-inclusive diagnostic improves
on the sparse lane. The timed Metal arm never replays an authoritative CPU
shadow.

### RAF evaluation

The host-sparse target algorithm has an **exact** count:

```text
split equality tables       34,814 products
190-record pushforward         190 products
affine 13-round tail         24,599 products
total                        59,603 products.
```

Its exact working payload is `8A + 16*(32768+2048) + 16K = 689,648` bytes.
This lane is selected for the retained trace.

The direct Metal lane borrows `(cycle,address)` from the owner, performs one
product per record, uses deferred global additions, finalizes 8,192 fields,
and returns a 128-KiB `R` table in one command. At `A=190`, its conservative
external service payload is 598,976 bytes, but a 141-us command control still
dominates its roughly 1.3-us traffic floor.

Use the preregistered provisional selector until one same-binary crossover
sweep replaces it:

```text
A <= 2^15                         -> host sparse
2^15 < A <= 2^20 and Q >= A/8    -> direct Metal
otherwise, valid bucket view     -> bucketed Metal
otherwise, resident dense plane  -> retained dense Metal
otherwise                         -> optimized CPU
```

The bucketed pass launches only nonempty `(outer,address-tile)` buckets,
reads each packed record once, aggregates in 27,520 bytes of threadgroup
memory, and performs one global deferred add per live subtotal. Direct and
bucketed paths each use one producer dispatch, one finalizer dispatch, one
command buffer, and one wait. The 13 address rounds remain on the host.

Submit a high-support RAF command during read/write preparation, but keep it
in a separate command buffer. RAF and read/write have different result
deadlines; sharing an owner is useful, making the first read/write message
wait behind a histogram is not. On the target host-sparse lane, compute and
park `R` during preparation with no GPU submission.

### Output check

The selected host-weight schedule has exact work of 1,023 host field products,
8,192 native 128-by-64 contributions, one 64-field reduction, and a
three-round eight-element tail. A hypothetical coalesced GPU partial pass has
100,352 perfect-cache semantic bytes, one dispatch, and a 1,024-byte readback.

The retained standalone Metal control measured 830,270 ns wall and 52,321 ns
active, while the focused CPU member is 276,100 ns. The standalone GPU wall is
already 15.0 times the 55,220-ns 5x budget. Therefore:

- first implement the wide-accumulator resident CPU fold over the owner's
  native `u64[K]` range and target 55,220 ns, pursuing 32,000 ns;
- emit the ten certified zero messages and update the low weights on the host;
- do not create a Metal command, wait, allocation, or `RamValFinal` upload;
- reopen the one-dispatch Metal partial only when stage 2 has an unavoidable
  batch-owned command at the first active output round. Promotion then charges
  incremental active and host time from an otherwise identical ablation.

Calling this a CPU leaf is intentional. A standalone Metal kernel cannot meet
the current wall budget at this geometry.

## Fusion boundaries

Fusion is permitted where it removes movement without altering challenge
order:

- Fuse witness extraction, owner certificate checks, and writes to final
  owner ranges in the one producer traversal. Derive the public-I/O certificate
  once at output-check prepare, where checked public memory is available.
- Fuse a member's bind with its next message and parent write.
- Fuse message columns while their inputs are already in registers.
- Reuse one `eq(r_address)` table and one native final-memory allocation by
  identity.
- Coalesce independent dispatches into a batch-owned command only when they
  consume the same already-known challenge and the control arm owns the same
  commit and wait.

The following are forbidden:

- Challenge-dependent state may not cross stage 2 to stage 4. The stage-4
  gamma is drawn only after intervening transcript work.
- Read/write and RAF arithmetic are not one shader. Their deadlines and
  high-support occupancy differ.
- Output-check's certified zero messages do not authorize changing its
  relation or sharing read/write's polynomial.
- A command may not compute a future round before the host batch absorbs the
  current combined polynomial and returns its challenge.
- Producer construction outside the primary PIOP is not free. Report PIOP,
  backend preparation, and their sum.

## Exact oracle and parity plan

Correctness promotion has five independent layers.

1. **Owner oracle.** On small fixtures, rescan the authoritative witness with
   a simple reference collector. Compare every record and increment; rebuild
   dense address/pre/post columns, final memory, both merge graphs, per-level
   counts, and digests. Exercise no access, one access, repeated addresses,
   raw-address-zero stores, negative increments, record-retention overflow,
   and malformed order/range cases.
2. **Read/write oracle.** Use the existing unfactored `K*T` oracle, which does
   not call sparse event, Gruen, or shader helpers. Compare raw
   `[q(0),q(infinity)]`, reconstructed coefficients, every challenge, each
   cutoff frontier, the final `(val,ra,inc)`, opening orientation, and the
   derived `EqCycle` check for host-sparse and Metal-prefix lanes.
3. **Value-check oracle.** Materialize dense `RamInc`, `RamRa`, and
   `LT+gamma` on small domains and evaluate products directly. Compare all
   `0/1/2/3` message values, every union bind, the CPU handoff, final factors,
   `LtCyclePlusGamma`, and staged opening order. Include a load beside a store,
   two stores in one pair, increment-only raw-zero support, and nonzero initial
   memory.
4. **RAF and output oracles.** Compare coordinate-by-coordinate equality
   against compact, direct, bucketed, and dense RAF pushforwards, then run the
   independent affine tail. Compare output-check's full dense 13-round oracle
   with zero deferral, resident CPU deferral, and any coalesced partial path;
   mutate cells inside and outside the public mask to test certification.
5. **Protocol lockstep.** Run each member against optimized CPU with identical
   supplied challenges and compare every polynomial, output point, output
   claim, derived table, and transcript byte sequence. Then verify a complete
   clear Akita/Metal proof and keep the CPU backend's `host,zk` proof
   regressions green; the workspace intentionally forbids `akita` and `zk`
   together. Final-output equality alone is insufficient.

Performance evidence is accepted only after parity. Each candidate records
command count, waits, readback bytes, GPU active, complete member wall,
producer-inclusive wall, storage identities, allocations, spills, compiled
registers, resident SIMDgroups, inactive lanes, external bytes, and atomic
stalls. Five alternating log-26 pairs and an untuned adjacent-size holdout are
required for promotion.

## Ordered implementation slices

### Slice 1: owner plus authoritative sparse value check

This is the highest-leverage slice. Publish `RamCycleFamilyOwner` and its
receipt from the existing witness traversal, build the union DAG once, and
replace `RamValCheck`'s round-0 shadow with an authoritative host-sparse kernel
using the production `SumcheckKernel` seam. The retained trace has at most
55,983 planned products against a 234.7--278.1-ms CPU member, so this
establishes the largest likely wall reduction while validating cross-stage
owner lifetime, raw-zero increments, and the frontier ABI that a later Metal
prefix will use.

Gates: no second witness scan or upload, exact clear-mode Akita/Metal lockstep,
green CPU `host,zk` regressions, no CPU shadow in the timed arm, complete member
at or below 46,931,375 ns, and owner preparation reported separately. If the
host-sparse member already clears the pursuit wall, keep it as the target
crossover rather than forcing one-threadgroup Metal rounds.

### Slice 2: sparse read/write and arbitrary cycle handoff

Reuse the owner address DAG and union increment frontier. Implement the
host-sparse cycle/address kernel first, then the same frontier ABI in
`ram_rw_flat_bind_message`; add grouped levels only where the per-level model
selects them. Measure cutoffs including zero rather than assuming all 26 cycle
rounds belong on Metal. Gate at 17,395,842 ns and pursue 8,697,921 ns.

### Slice 3: adaptive RAF without a target dense plane

Wire the exact host-sparse lane, then direct and bucketed Metal projections for
crossover controls. Preserve the current dense command until same-binary
parity and overlap traces pass. The target must stop allocating/uploading the
256-MiB plane once no other registered consumer leases it. Gate at 14,974,050
ns and report the fresh sparse CPU denominator as well as the historical one.

### Slice 4: resident output leaf

Publish the certified native final-memory range and implement the
wide-accumulator CPU fold with no allocation. Gate at 55,220 ns and pursue
32,000 ns. Do not write the Metal partial shader into the production path
until a qualifying batch-owned command exists.

### Slice 5: high-support Metal prefixes and command coalescing

Run topology sweeps around the provisional crossovers. Add value-check and
read/write Metal prefixes only where alternating measurements beat the host
sparse lane and their complete members clear the fixed caps. Add a stage-level
command aggregator only after per-member kernels are correct; use dispatch
ablation to ensure shared command/wait costs are charged once.

## Stop conditions

Stop a candidate rather than tuning around any of these failures:

- it requires a second full-domain scan, private owner, repack, or upload;
- target support launches too little work to beat the measured host-sparse
  lane, after one launch-width control;
- a lower bound already exceeds the member's 5x cap;
- a spill adds more than 10% of modeled external traffic;
- a receipt or allocation identity changes between producer and consumer;
- fallback would occur after transcript mutation; or
- a faster sparse algorithm applies equally to the CPU control but the
  comparison retains the obsolete dense denominator.

The family is complete only when all four member caps pass under one
owner-normalized evaluator and the full PIOP improves. The expected target
shape is mostly host sparse; Metal earns work on the higher-support holdouts,
where occupancy is available and the same ownership contract prevents memory
traffic from erasing the gain.
