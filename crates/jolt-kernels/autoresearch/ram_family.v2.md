# RAM family v2: one sparse owner, five missing consumers

Status: design only. The numbers below are planning gates, not promotion
evidence. No protocol, transcript, kernel, harness, or backend registration is
changed by this document.

The RAM family should not be implemented as six independent dense ports. The
production prover already creates one `RamAccessTape`, and the retained log-26
Fibonacci run has only 190 remapped accesses. A challenge-independent sparse
cycle topology derived once from that tape can serve the five slow relations.
`RamRafEvaluation` is the exception: it is already an authoritative,
asynchronous Metal kernel and remains the control while the missing consumers
are built.

## Frozen evaluator and targets

The planning artifact is
`benchmark-runs/metal-piop-eval/20260807-103715-208977/result.json`, revision
`2ed9ce265f00ca06120a7d4a46fb979ee07919b8`, at `log_T = 26` with 16 Rayon
threads. It contains one optimized-first pair, is marked diagnostic, and is
not acceptance-eligible. Its values are exact span observations, but its zero
MAD and order-stratum fields carry no statistical information. Promotion
still requires at least five alternating paired samples from one clean,
stable binary.

The complete-member boundary is the union of that relation's `prepare`,
`prove_round`, `finish_rounds`, and `output_claims` seams. Batch Fiat-Shamir is
outside these seams in both arms. Converting the recorded optimized walls to
integer nanoseconds and taking `floor(cpu_ns / 5)` gives:

| Relation | Optimized CPU (ns) | Recorded Metal arm (ns) | Current ratio | Hard 5x cap (ns) | 10x pursuit (ns) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `RamRaVirtualization` | 286,463,081 | 273,819,294 | 1.046x | 57,292,616 | 28,646,308 |
| `RamValCheck` | 278,143,584 | 267,514,794 | 1.040x | 55,628,716 | 27,814,358 |
| `RamHammingBooleanity` | 137,156,459 | 117,499,379 | 1.167x | 27,431,291 | 13,715,645 |
| `RamReadWriteChecking` | 86,979,214 | 70,577,004 | 1.232x | 17,395,842 | 8,697,921 |
| `RamRafEvaluation` | 80,615,083 | 208,127 | 387.336x | 16,123,016 | 8,061,508 |
| `RamRaClaimReduction` | 41,259,834 | 35,558,002 | 1.160x | 8,251,966 | 4,125,983 |

The six CPU walls sum to exactly `910,617,255 ns`. The hard fused-family wall
is therefore `182,123,451 ns`; the pursuit wall is `91,061,725 ns` (10x).
The recorded Metal-arm sum is `765,176,600 ns`, or 1.190x. Removing the one
real Metal relation leaves the five missing relations at only 1.085x. This is
why the family is a development priority even though RAF itself looks fast.

The frozen caps prevent benchmark drift from lowering the bar. A promotion run
uses the stricter of the table's cap and one fifth of its same-revision paired
CPU median. Every relation must clear its standalone 5x cap as well as the
family cap; excess speed in one member cannot excuse a CPU placeholder in
another.

### One owner charge

The family metric is

```text
unique RAM owner preparation
+ union of the six owner-exclusive member-seam intervals
```

not six owner preparations plus six member walls. The current optimized path
constructs `RamAccessColumns`, `RamAccessValues`, `RamIncrementActivity`, and
`RamAccessTape` on the first RAM request, normally inside
`RamReadWriteChecking::prepare`. The current Metal path moves address-plane
creation into backend witness preparation. A future evaluator must expose a
single `RamFamilyOwner::prepare` span and charge it once to both arms, even if
its physical placement differs. A nested owner span is subtracted from its
member before the two terms are added. Until that normalization exists, the
recorded 1.190x family ratio is favorable to Metal and is diagnostic only.

Asynchronous device work is charged on the critical path, not again as GPU
active time. The RAF command is submitted during RAM read/write preparation
and joined before RAF output is used. Its active time may overlap the
read/write member; the family interval union counts that overlap once. A
completed-command guard, GPU-active telemetry, and the full PIOP wall remain
mandatory so hidden unfinished work cannot manufacture a speedup.

## Production truth

The near-1x values are expected from the installed backend, not evidence that
the GPU is intrinsically slow.

| Relation | Production Metal behavior |
| --- | --- |
| `RamRaVirtualization` | Slot is not replaced by `with_metal_compute`; optimized CPU runs. |
| `RamHammingBooleanity` | Slot is not replaced; optimized CPU runs. |
| `RamRaClaimReduction` | Slot is not replaced; optimized CPU runs. |
| `RamReadWriteChecking` | The adapter submits RAF early, then returns the optimized CPU read/write kernel. |
| `RamValCheck` | The optimized CPU kernel is authoritative. Metal computes only a round-0 shadow and compares it with CPU. |
| `RamRafEvaluation` | Authoritative Metal pushforward plus a small affine host tail. |

The artifact's companion Metal trace records RAF at 190 accessed rows, 76 live
subtotals, `4,096,625 ns` GPU active, and only `208,127 ns` in its visible
member seams because it completed during useful overlap. The same trace
records the value-check shadow at 77 active increments, 74 active pairs, a
1,776-byte upload, and `11,224,500 ns` GPU active. That shadow is a useful
correctness probe, but replaying the whole CPU relation prevents a speedup.

## Shared data model

At the target scale:

```text
T = 2^26 = 67,108,864 cycles
K = 2^13 = 8,192 RAM words
b = log2(T) = 26
d = ceil(log2(K) / committed_chunk_bits) = ceil(13 / 8) = 2
```

The production `RamAccessRecord` is exactly 24 bytes:

```text
cycle: u32, address: u32, pre_value: u64, post_value: u64
```

The tape retains records only while `A <= 2^18`; otherwise `records()` is
`None` and the sparse path must fall back explicitly. For the retained trace:

```text
A  = remapped accesses                         = 190
D  = nonzero RAM increments                    = 77
P  = adjacent pairs containing an increment    = 74
Ea = sum of distinct (block, address) entries  = 2,665
Ga = sum of active cycle blocks over 27 levels = 826
```

`Ea` and `Ga` come from the committed sparse census. An increment at raw
address zero may not appear in the access tape, so increment consumers use a
union topology. The current trace does not emit that union count; the safe
planning bound is

```text
Gu <= Ga + (b + 1)D = 2,905.
```

The resident sparse owner should contain sorted records, nonzero increment
activity, level offsets, and compact parent/child merge maps. It is immutable
and challenge-independent. Each relation owns only two ping-pong field
frontiers and small message buffers. A 16-byte planning node gives
`16 * Gu <= 46,480 bytes` on this trace. The ABI may change, but any candidate
must update the byte equation rather than hiding metadata in an unreported
allocation.

Sparse admission also requires complete producer certificates. Reconstructing
increments from `post_value - pre_value` requires `increment_compatible`.
Using access records as the Hamming support requires a checked equivalence
between `RamHammingWeight` and a non-sentinel remapped address; the current
one-way `ram_ra_compatible` flag alone is not that equivalence. The owner must
publish the stronger one-bit certificate (or retain an explicit sparse
Hamming bit) before Hamming booleanity may use the tape.

The current owner also materializes three dense host columns and a dense Metal
address plane:

| Current shared storage | Target-scale bytes |
| --- | ---: |
| Host addresses (`u32[T]`) | 268,435,456 |
| Host pre-values (`u64[T]`) | 536,870,912 |
| Host post-values (`u64[T]`) | 536,870,912 |
| Metal address plane (`u32[T]`) | 268,435,456 |
| Retained tape at its cap (`24 * 2^18`) | 6,291,456 |
| Retained tape in this trace (`24A`) | 4,560 |
| Increment activity in this trace (`24D`) | 1,848 |

The first implementation reuses these committed objects; it does not claim an
end-to-end memory win from work already done outside PIOP. The sparse consumer
path adds no `T`-field table and does not upload another full-domain row
buffer. Once all consumers are stable, eliminating the dense producer can be
a separate end-to-end project. Rewriting RAF first would trade a proven
critical-path win for producer work and is deliberately deferred.

### Cross-stage resident values

The stage-2 batch produces the shared RAM address point. Stage 4 can build
`eq(r_address, k)` once (`16K = 131,072 bytes`) and retain the same storage
identity through stage-5 claim reduction. Stage-6b RA virtualization uses the
same reduced address and can borrow its two 256-entry chunk tables
(`2 * 256 * 16 = 8,192 bytes`). Value check alone needs the three split-LT
tables (`3 * 8,192 * 16 = 393,216 bytes`), and read/write needs `val_init`
(`16K = 131,072 bytes`).

Including tape, activity, a 16-byte union topology, address equality, split
LT, `val_init`, and chunk equality gives an optimistic unique persistent set
of at most `716,440 bytes` for this trace. Per-relation frontiers are reused
sequentially and are counted at peak, not summed across stages. The existing
RAF plane and its 851,968-byte sequence workspace remain separate and keep
the physical family peak near the current 256-MiB plane until RAF's producer
is revisited.

Every resident object carries cycle geometry, address geometry, device
registry, allocation identity, byte length, and source-generation identity.
Stage 6b is the final consumer and releases the sparse and dense address
owners. A matching value with a different allocation identity is not reuse.

## Kernel architecture and ceilings

The sparse topology is a tree of active cycle blocks. A round reads the two
child values when present, treats a missing child as the relation's implicit
zero or checkpoint value, emits the member polynomial, and writes one parent
frontier. The host supplies the next sumcheck challenge. This keeps useful
work proportional to sparse topology rather than `T`.

The operation and traffic figures below are deliberately optimistic planning
bounds. A “product” is one full Solinas field multiplication. Traffic counts
frontier fields and one touch of each unique table; it excludes cache-line
overfetch, command traffic, and spills. Compiled counters replace these bounds
at promotion.

| Relation | Sparse state and round algebra | Useful-product planning bound | Optimistic logical bytes |
| --- | --- | ---: | ---: |
| `RamRaClaimReduction` | One address-folded RA frontier and three cycle-equality frontiers; emit the three gamma-weighted linear terms together. | `12Ga = 9,912` | `192Ga + 16K = 289,664` |
| `RamHammingBooleanity` | One sparse Hamming frontier plus its equality weight; missing blocks are zero. | `5Ga = 4,130` | `96Ga = 79,296` |
| `RamRaVirtualization` | Two chunk-selector frontiers plus equality; evaluate all four degree-3 samples while values are in registers. | `14Ga = 11,564` | `144Ga + 8,192 = 127,136` |
| `RamValCheck` | Sparse increment and RA frontiers over the union topology; retain split LT, emit samples at 0, 2, and 3 together. | `12Gu <= 34,860` | `144Gu + 16K + 48sqrt(T) <= 942,608` |
| `RamReadWriteChecking` | Preserve sparse value checkpoints by `(block, address)` and replace the dense increment polynomial with sparse activity; keep the small address phase on the host unless measurement favors Metal. | `24Ea + 8K + D(b+1) <= 131,498` | `192Ea + 64A + 16K + 24D = 656,760` |

Together, the five missing relations have at most 191,964 planned useful
products and 2,095,464 bytes (2.10 MB, 2.00 MiB) of optimistic logical traffic
on this trace. The retained M4 Max controls in the existing RAF model are 18.1
Gproduct/s and 451,701,710,520 B/s. They give an arithmetic floor of about
10.6 microseconds and a copy-roof floor of about 4.6 microseconds. These are
not wall-time predictions.
They show that launch, synchronization, host challenge boundaries, allocation,
and fallback behavior dominate this workload. A kernel that scans a dense
cycle table has selected the wrong algorithm even if its occupancy is high.

### Relation-specific schedule

`RamRaClaimReduction` is the simplest pilot. It gathers
`eq(r_address)[address]` once per access and advances the three cycle-equality
terms in the same sparse pass. One command returns both sampled values for the
round; no dense `Q` table and no two `T`-element regathers are allowed.

`RamHammingBooleanity` and `RamRaVirtualization` are in the same stage-6b
batch and receive the same batch round challenge. One command should emit both
member messages and advance both frontiers, followed by one wait and one small
readback per round. Their algebra remains separate in the output ABI so the
host driver can preserve canonical member order. If the fused shader causes
spills or lowers either member's occupancy, split the pipelines but retain one
command buffer, one wait, and one owner.

`RamValCheck` uses nonzero increments as the driving support but must retain
RA values in sibling blocks: an inactive increment endpoint can contribute a
cross term after interpolation. The union tree supplies that halo exactly.
The current round-0 shadow is replaced only after the sparse path is
authoritative; promotion forbids computing the same round on CPU for
comparison in the timed arm.

`RamReadWriteChecking` already has the correct sparse matrix for RA and value,
but still materializes and binds a `T`-field increment polynomial. The new
path takes increments from the owner, carries them beside sparse matrix
groups, and retains the `K = 8,192` address tail on the host initially. It must
preserve raw-zero increments even though those rows have no RA entry.

`RamRafEvaluation` stays unchanged. Its current shader reads the 256-MiB
address plane six times (`6 * 4T = 1,610,612,736 bytes`) to perform only 76
live outer/address products on this workload, but the command is already
hidden behind stage-2 work and clears its standalone cap by a wide margin. It
is a regression and contention control, not the next optimization target.

## Host Fiat-Shamir boundaries

Fiat-Shamir remains entirely on the host.

- Stage 2 draws its batch gammas and output-check reference point before
  preparation. RAM read/write has 26 cycle and 13 address rounds; RAF's 13
  address rounds share the stage-2 batch schedule. RAF may still be submitted
  early and joined when its first output is needed.
- Stage 4 stages advice and program-image values, then draws the labeled RAM
  value-check gamma. No value-check command may use gamma before that draw.
- Stage 5 draws the RAM claim-reduction gamma after the complete stage-4
  transcript boundary.
- Stage 6b performs its promoted host draws before the batch. Hamming
  booleanity and RA virtualization can share a device command because they
  consume the same subsequent batch challenge, not because their claims are
  merged.

There are 117 relevant host challenge boundaries across the four batches
(`39 + 26 + 26 + 26`). A device path therefore uses persistent buffers and
zero per-round allocation. At sparse sizes, a CPU sparse frontier may beat 117
GPU round trips. The Metal backend must benchmark both and choose a fixed,
telemetry-visible crossover from `(A, D, Ga, Gu, Ea)`. Hybrid CPU execution is
the intended result below that crossover, not a fallback disguised as a Metal
measurement.

Cross-stage fusion of challenge-dependent work is forbidden. The owner,
topology, and address equality tables may cross stages because their
provenance is explicit; round state may not cross an intervening transcript
draw.

## First three implementation slices

Each slice gets one analysis packet, one parity evaluator, and one performance
evaluator. Stop at its gate before adding the next relation.

### Slice 1: sparse owner plus claim-reduction pilot

Add a typed resident sparse view over `RamAccessTape` and
`RamIncrementActivity`, with level offsets, merge maps, geometry, identities,
and an explicit `records() == None` rejection. Wire only
`RamRaClaimReduction` first. This tests owner lifetime, sparse equality
propagation, host round trips, and stage-4 address-equality reuse with the
lowest-degree relation.

Hard gates:

- exact polynomial, challenge, final claim, and opening parity for no access,
  one hot address, colliding siblings, randomized addresses, retention at
  `2^18`, and explicit fallback above it;
- one sparse-owner allocation/generation (the existing dense RAF plane is
  attributed separately), zero `T`-field allocation or upload, zero per-round
  allocation, and one command completion per active round;
- complete member at or below `8,251,966 ns`; pursue `4,125,983 ns`;
- if Metal is slower than the sparse CPU control after one launch-width pass,
  install the CPU crossover and stop tuning the GPU at that topology; and
- kill this representation if correctness requires expanding a dense cycle
  table or if compiled spill bytes exceed 10% of modeled traffic.

### Slice 2: fused stage-6b pair

Reuse the accepted owner to implement `RamHammingBooleanity` and
`RamRaVirtualization` in one per-round command/readback lifecycle. Build chunk
equality directly from the retained address point and release the family owner
after the final bound output.

Hard gates:

- each standalone cap clears: `27,431,291 ns` for Hamming and `57,292,616 ns`
  for virtualization;
- the pair clears its exact combined 5x cap of `84,723,908 ns`; pursue half
  that wall until launch latency, not field work, is demonstrated as the
  limit;
- exactly one wait/readback per shared batch round, canonical two-member
  output order, and no device Fiat-Shamir;
- fused and split-pipeline controls agree bit-for-bit; retain fusion only if
  it does not regress either member by more than 10%; and
- reject any candidate that rereads the dense 256-MiB address plane each
  round, spills, or runs an optimized CPU shadow in the timed arm.

### Slice 3: authoritative sparse value check

Replace the round-0 shadow with the union-topology value-check sequence. First
benchmark the complete sparse CPU member; then benchmark a persistent Metal
frontier with the same evaluator. Use the faster path on each preregistered
topology. Retain address equality from stage 4 for slice 1's stage-5 consumer.

Hard gates:

- complete member at or below `55,628,716 ns`; pursue `27,814,358 ns`;
- round 0 on the retained `P = 74` topology must beat the current
  `11,224,500 ns` shadow active time before any dense-tail work is added;
- parity covers an increment beside a load, two stores in one pair, sparse
  stores in different high blocks, a raw-address-zero store, and nonzero
  initial RAM;
- zero authoritative CPU replay in the Metal arm, no full-domain row scan,
  and at most one message readback per round; and
- if the union halo grows beyond the registered `Gu` model, fall back before
  allocation rather than silently switching to dense state.

After these slices, rerun the fixed family evaluator. `RamReadWriteChecking`
is next and remains mandatory for across-the-board 5x: its dense increment
table must be removed and its standalone wall must clear `17,395,842 ns`
(`8,697,921 ns` pursuit). RAF is touched only for a regression, provenance,
or contention failure.

## Promotion checklist

A family result is promotable only when all of the following are true:

- five or more alternating paired log-26 samples, clean source, stable binary,
  fixed 16-thread CPU control, and both proofs verified;
- every standalone relation clears 5x and the owner-normalized family wall is
  at most `182,123,451 ns`;
- the result reports the `91,061,725 ns` pursuit decision instead of stopping
  automatically at 5x when the sparse ceilings remain far below wall time;
- owner construction is charged once, all consumers report the same source
  generation, and allocation reuse is proven by identity;
- CPU fallback, sparse CPU hybrid, Metal dispatches, command completion,
  readback bytes, GPU active time, spills, and peak residency are all
  attributed per relation;
- no timed Metal arm performs a full optimized CPU shadow; and
- the complete PIOP wall improves consistently with the family result. A fast
  seam result that merely moves work into backend preparation or an
  unobserved command does not pass.
