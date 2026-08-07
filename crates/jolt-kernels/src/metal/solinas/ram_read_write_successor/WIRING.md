# RAM read/write successor design

This packet is an analysis-only successor to `../ram_read_write/`. It does not
register source or change the backend. The predecessor's sparse relation and
oracle remain the correctness base. This packet changes the ownership boundary
and message schedule that should be implemented after one missing observation:
the exact log-26 Fibonacci topology census.

## Decision

Implement after the census gate, using:

1. one pre-PIOP RAM access/topology owner shared by RAM members;
2. Metal for all 26 cycle rounds and optimized CPU for 13 address rounds; and
3. a per-level choice between the predecessor's flat message and a grouped
   message that factors the split-eq weight outside the address sum.

Do not implement one grouped schedule for every level. When a level has one
address entry per live cycle block (`E_b = G_b`), grouping creates one reduction
tile for every event and saves no products. The flat schedule wins there. Wide
groups can use the grouped schedule and remove two field products per address
entry at the cost of two products per group.

No cross-protocol arithmetic fusion is recommended. Reuse the witness walk,
access plane, compact records, and allocation identities where their layouts
match, but keep member-specific state and every Fiat-Shamir boundary separate.

## Exact boundary

The relation over address `k` and cycle `j` is

```text
eq(tau_low, j) * ra(k, j)
  * (val(k, j) + gamma * (val(k, j) + RamInc(j))).
```

It is degree three. Cycle variables bind low-to-high for 26 rounds, followed by
13 low-to-high address binds. For challenges `c_0..c_25` and `a_0..a_12`, the
opening point remains

```text
[reverse(a_0..a_12) || reverse(c_0..c_25)].
```

The device owns:

- compact access records and the transcript-independent merge topology;
- sparse `(ra, val)` state and the bound increment state;
- all 26 cycle messages and the final cycle challenge bind; and
- message reductions to the two Gruen hints `[q(0), q_infinity]`.

The host owns:

- the authoritative `GruenSplitEqPolynomial` and cubic reconstruction;
- every Fiat-Shamir absorb and challenge;
- the 13 address messages, final `(val, ra, inc)` claims, and derived-eq check;
- admission and CPU fallback before transcript mutation; and
- resource publication and lifetime checks.

The cycle handoff returns one `(ra, val)` state per touched address, one bound
increment scalar, and the sorted address list already owned by the topology.
No cycle state returns to the host between rounds; only two field hints do.

The compact record is the predecessor's 24-byte tuple:

```text
(cycle: u32, address: u32, pre_value: u64, post_value: u64).
```

It is valid only if the producer proves

```text
RamInc[j] = post_value - pre_value  on an emitted record
RamInc[j] = 0                       otherwise.
```

`RamInc` remains authoritative. A failure selects optimized CPU before a round
message is observed.

### Frozen CPU denominator

The control is the five alternating optimized runs in
`benchmark-runs/metal-piop-eval/20260806-133709-697013`, revision
`5f520c21e338632aa0bf5936ceb02be6c22fa40f`, Fibonacci `log_T = 26`,
`log_K = 13`, and 16 Rayon threads:

```text
158.786294 ms
154.710378 ms
148.525711 ms
152.095581 ms
160.704373 ms
```

The complete-member median is `154.710378 ms`. The 5x ceiling is
`30.942075 ms`; the 8x stretch ceiling is `19.338797 ms`.

The sample whose complete member equals the median decomposes as:

| Component | Wall time |
|---|---:|
| prepare | 110.985042 ms |
| 26 cycle rounds | 41.641794 ms |
| 13 address rounds | 2.082250 ms |
| finish plus output | 0.001292 ms |

Across the five runs, the address-round median is `1.793751 ms`. Preparation,
not the sumcheck arithmetic, is 71.7% of the control. A production design that
builds the same vectors again inside `RamReadWriteChecking::prepare` cannot hit
the goal consistently.

## Owner and stage ordering

`backend_witness_prepare` runs before the PIOP. The Metal backend already uses
it to collect `RamAccessColumns` and upload the address plane for RAM RAF. Extend
that same witness walk to publish the RAM successor owner. Do not scan `T` again.

The owner contains:

- dense address-plane allocation and generation;
- compact record allocation and record count;
- `E_b`, `G_b`, and `D_b` for every cycle level, where `D_b` is the sum of
  `ceil(entries_in_group / threads)`;
- selected flat/grouped descriptor offsets for every level;
- the sorted final-address list;
- increment-invariant result and source identities;
- `log_T`, `log_K`, device registry, buffer lengths, and maximum resident bytes;
- producer CPU wall, GPU active wall, upload bytes, and PIOP join wall.

Preparation may finish asynchronously, but the stage-2 prepare must validate
and join the owner before returning a Metal kernel. The primary PIOP comparison
charges only this join because the agreed evaluator excludes backend witness
preparation. The diagnostic comparison reports the complete producer wall.
Both numbers are required; moving work outside the PIOP is not evidence that it
became free.

Stage 2 prepares all five members before its 39-round batch. RAM read/write is
the only 39-round member. Product remainder and instruction claim reduction are
active on the 26-round suffix; RAM RAF and output check are active for the last
13 rounds. RAM RAF is currently submitted from RAM read/write prepare because
`tau_low` first exists there, then joined at its later active window.

Do not combine the RAM RAF command with the RAM read/write first command. They
have different outputs and latency deadlines. Queue order must let RAM RAF use
otherwise idle device time without delaying the first RAM read/write message.

The generic prover currently delivers a challenge with the next
`prove_round`, and calls members in declaration order. A later orchestration
experiment may add a protocol-neutral post-challenge submit hook and evaluate
RAM read/write last in the next round. That can hide command-service gaps under
other members' host work. It cannot hide GPU-active time when other stage-2
members already saturate the same device, so the standalone member still has to
clear 5x without overlap credit.

## Sparse counts and useful work

After cycle bind `b`, define:

```text
E_b = sum over nonempty 2^b-cycle blocks of distinct addresses
G_b = number of nonempty 2^b-cycle blocks
D_b = sum over those blocks of ceil(distinct_addresses / threads)
V   = sum(E_b, b = 1..26)
H   = sum(G_b, b = 1..26).
```

The real census must come from the exact access tape. Access count alone is not
enough: a hot-address trace has `E_b = G_b`, while a high-entropy trace can have
`E_b` thousands of times larger than `G_b` at late levels.

### Flat level

The predecessor computes a split-eq weight for each group, then applies it to
both hints for every address entry. Including the state and increment binds:

```text
products_flat(b) = 8 E_b + 2 G_b.
```

Its flat launch reduces `ceil(E_b / threads)` partial pairs and stays efficient
when groups are small.

### Grouped level

All entries in one cycle block share the same split-eq weight and increment
pair. Sum the two unweighted inner hints across addresses, then apply the weight
once after the segmented reduction:

```text
message inner             4 E_b
state bind                2 E_b
eq weight + two outputs   3 G_b
increment bind            1 G_b
--------------------------------
products_grouped(b)       6 E_b + 4 G_b.
```

The algebra is checked independently in `oracle.rs`. The grouped form saves
`2(E_b - G_b)` field products. It also needs `D_b` tile partials and a grouped
reduction, so product savings alone do not select it.

`model.rs` prices both forms at every level and selects the lower local
compute/traffic floor. The selector is fixed by the topology before any
transcript challenge; it does not change the protocol or data-dependent proof
behavior.

## Traffic lower bound

The retained measured roofs are 451.701710520 GB/s for streaming copies and
18.1 G full field products/s for the register-constrained Solinas kernel class.
These are measured controls, not M4 Max specification peaks.

Common traffic includes:

```text
record initialization       72 E_0 bytes
first message state read    32 E_0 bytes
state binds                 32 sum(E_(b-1) + E_b) bytes
increment endpoints         32 sum(G_(b-1)) bytes
message reductions          exact partial-tree bytes
cycle hint readback          32 * 26 bytes
final handoff                32 E_26 + 16 bytes.
```

The flat path retains 32-byte events and the predecessor's 48-byte group tuple.
The grouped path removes that tuple and can use a 24-byte event because group
identity comes from the group span. Its segmented reduction traffic is

```text
64 D_b + 32 G_b + global_reduce(G_b) bytes.
```

The 24-byte event is a design target, not a compiled ABI result. If Metal pads
it or generates uncoalesced 24-byte loads, re-price it as 32 bytes before the
candidate is admitted. Descriptor compression is rejected if the added address
arithmetic or split loads raise the measured active floor.

The model reports two independent floors:

```text
compute_floor = total_products / 18.1 Gproduct/s
traffic_floor = total_bytes / 451.701710520 GB/s
active_floor  = max(compute_floor, traffic_floor).
```

The optimistic complete-member floor adds active time to fixed latency. The
conservative projection adds compute and traffic. The fixed primary charge at
the cycle-to-CPU cutoff is:

```text
owner join
+ one setup wait
+ 26 round waits
+ one handoff wait
+ 1.793751 ms CPU address tail
+ 39 * 2 us host Fiat-Shamir.
```

With a zero owner join this is `5.819751 ms`. Operational intensity flips from
traffic- to compute-bound at about

```text
18.1e9 / 451.701710520e9 = 0.04007 product/byte,
```

or 24.96 bytes per full product. The exact census determines the side of this
boundary.

## Occupancy and launch design

The flat kernel remains one linear event grid. The grouped kernel assigns
contiguous tiles to a group, accumulates two field values per lane, reduces to
one pair per tile, then reduces tile pairs within the group before applying the
weight. State binds remain fused with the next message and write each
destination once.

Start with 256 threads. The two field accumulators, two endpoint states,
increment pair, gamma, and transient Solinas product make register pressure a
structural limit. Do not add four persistent weighted/unweighted accumulators
to avoid a reduction pass without pricing the lower occupancy. Promotion needs:

- compiled register allocation and spill bytes for both schedules;
- resident SIMDgroups and limiting resource;
- execution width and maximum threads;
- achieved product/s and bytes/s by level;
- command-buffer wall, GPU active, join wall, and readback wall; and
- cache counters or a cold/warm comparison for descriptors and split-eq tables.

The grouped schedule is rejected at a level if it spills, if its group-tail
imbalance leaves fewer than two resident threadgroups per GPU core for a
material share of work, or if its measured active time exceeds the flat level.
The selector should support a flat fallback without rebuilding state.

No allocation, pipeline creation, topology construction, or split-eq vector
copy occurs inside a round. All buffers ping-pong through the owner.

## Reuse across RAM members

The owner has three consumers:

| Consumer | Reuse | Lifetime |
|---|---|---|
| stage-2 RAM read/write | compact records, RW topology, final addresses | through stage 2 |
| stage-4 RAM val check | dense `IncrementAccessRow` plane from the same witness walk | through stage 4 |
| stage-6b RAM RA virtualization | dense address plane and compact access positions | through stage 6b |

RAM RAF already reuses the dense address plane. RAM output check works over the
small `val_final` address table and should stay separate. RAM RA claim reduction
uses opening claims, not this merge state.

Sharing changes resource attribution, not equations. Each consumer records the
same owner generation and allocation identities. The first consumer does not
claim all producer cost in the PIOP-only metric; the diagnostic evaluator reports
producer wall once, and a fused RAM-family report amortizes it once across the
three consumers. A standalone member report must show both zero-amortization and
full-charge ratios.

RAM val check binds one cycle scalar per block, not one state per
`(block, address)`, so the RAM read/write merge descriptors are not reusable
there. Its win is direct production of the agreed dense native-row ABI, not
forcing two different relations through one topology. Do not retain the
predecessor's topology plus a successor copy. Release the RW topology after
stage 2 and value-bearing producer data after stage 4; retain only the address
plane needed by stage 6b.

## Adjustment candidates

In priority order:

1. **Move the producer boundary.** Extend the existing RAM witness walk and
   publish one owner before the PIOP. Correctness is unchanged; diagnostic wall
   must include it.
2. **Choose flat or grouped per level.** This lowers products only when
   `E_b > G_b` and avoids grouped reduction explosion on hot levels.
3. **Remove the 48-byte group tuple on grouped levels.** Group spans supply
   increment endpoints and split-eq pair once per group. Correctness follows
   from the group-factor oracle.
4. **Compress grouped events from 32 to 24 bytes.** Keep low/high state indices
   and two raw endpoint checkpoints; group and pair move to the span. Admit only
   after compiled load evidence.
5. **Post-challenge async submit.** A prover-engine hook may overlap service
   gaps with other stage-2 host work. It is protocol-neutral but affects shared
   orchestration and needs its own review.
6. **Cross-member producer reuse.** Emit RAM val check's dense native rows in
   the same witness walk, while keeping its cycle-only state separate. Do not
   fuse transcripts or opening claims.

Rejected for the first implementation:

- all 13 address rounds on Metal: the retained CPU tail is below the fixed
  per-round wait charge before device arithmetic;
- a grouped-only cycle kernel: it loses on `E_b = G_b` levels;
- dynamic GPU sorting or hashing each round: it changes the movement bound and
  adds synchronization when the topology is transcript-independent;
- deriving increments without validating the dense committed `RamInc`; and
- claiming producer work is free because it moved before the PIOP.

## Pre-registered bars

These bars predate successor shader work:

| Phase | Required result |
|---|---:|
| real-census perfect-overlap projection | <= 27.847867 ms |
| real-census conservative 8x projection | <= 17.404917 ms |
| PIOP owner join | <= 0.500 ms |
| added backend-witness preparation | <= 125.000 ms diagnostic wall |
| first executable complete-member screening | <= 27.847867 ms |
| five-pair complete-member promotion | <= 30.942075 ms median |
| eight-x stretch | <= 19.338797 ms median |

The 27.847867 ms screening bar is 90% of the 5x ceiling and leaves margin for
alternating-run noise. The 17.404917 ms census bar is 90% of the 8x ceiling.
Failing the first census bar rejects this mechanism before shader work. Passing
5x but missing 8x promotes only if the measured model shows no remaining flat
or grouped level with at least 5% complete-member headroom.

Promotion also requires:

- exact round polynomials, challenges, final values, output claims, and derived
  eq validation against optimized CPU;
- empty, one-access, hot-address, alternating-address, high-entropy, nonzero
  initial memory, nonzero increments, and invalid-increment fallback fixtures;
- no transcript mutation before fallback is impossible;
- no round allocation and no resource-identity mismatch;
- five alternating log-26 pairs, an untuned adjacent-size holdout, and log 27;
- standalone attribution with producer excluded and fully charged; and
- RAM-family fused attribution with the producer charged exactly once.

## Next action

Instrument the existing pre-PIOP RAM witness walk to emit the exact `E_b`,
`G_b`, and `D_b` census without building or registering successor shaders. Feed
it to `model.rs`. If the 27.847867 ms projection gate passes, implement only the
selected level schedules and the owner join. If it fails, stop this topology
mechanism and evaluate a different representation before writing Metal source.

The delegated lane ran no compiler or GPU work. The root's isolated `rustc`
check passes all six model and oracle tests; no shader is registered yet.
