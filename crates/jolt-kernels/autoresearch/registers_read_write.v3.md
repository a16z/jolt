# Registers read/write v3: one CSR-native resident sequence

## Decision

`RegistersReadWriteChecking` will use one certified CSR-256 owner for cycle
rounds 0--8, materialize a dense state while emitting round 8, run fused dense
bind/message rounds 9--25, and finish the seven register-address rounds on the
optimized CPU. The sparse and dense regimes are **one backend member and one
ownership state machine**, but not one dispatch: host Fiat--Shamir is a hard
boundary between every round. The round-8 entry point is the only sparse/dense
junction and writes the dense state while its source data is live.

The old 40-byte row-major prefix is not a production dependency. At log 26 its
single row plane is 2,684,354,560 bytes and its two local active medians total
73.050847 ms. The allocation is legal on the recorded device, whose runtime
`maxBufferLength` is 86,586,540,032 bytes; the rejection is about avoidable
traffic and lifecycle wall, not a fictitious 2-GiB hardware limit. The v3 owner
instead stores the 1,239,649,860-byte CSR plus a 1,073,741,824-byte canonical
`rd_inc` plane. The existing dense round-9 kernel remains a useful negative
control: it reached 95.5% of its traffic roof but took 5.862 ms versus 3.968 ms
for the isolated CPU round. Therefore a dense representation before the
topology has contracted is ruled out, while the post-round-8 dense
representation is retained only as part of the complete resident sequence.

The modeled complete-member range is 6.7--8.8x at log 26. The lower end treats
all 47.333 GB of logical requests as DRAM traffic at 80% of the measured copy
rate; the upper end uses cache-unique traffic and the measured arithmetic
rates. Five-times is the hard promotion floor, six-times is the initial
production target, and eight-times remains active while the cache-aware result
is within reach.

## Exact protocol and CPU boundary

For register `k`, cycle `j`, and the host challenge `gamma`, define

```text
ra(k,j) = gamma * rs1_ra(k,j) + gamma^2 * rs2_ra(k,j)

s(k,j) = eq(r_cycle,j) *
         (ra(k,j) * val(k,j)
          + rd_wa(k,j) * (val(k,j) + rd_inc(j))).
```

The input claim is

```text
rd_write_value + gamma * rs1_value + gamma^2 * rs2_value.
```

At log 26 and `K = 128`, rounds 0--25 bind cycle variables low-to-high and
rounds 26--32 bind the seven register variables low-to-high. Every round is
degree three. A cycle command returns the quadratic inner endpoints
`[q(0), q(infinity)]`; the host calls the existing Gruen constructor with the
running claim to obtain the canonical cubic, absorbs its exact bytes, and
draws the next challenge. No command hashes, draws a challenge, or crosses a
challenge dependency.

After challenges `c_0..c_25` and `a_0..a_6`, the opening point is

```text
reverse([a_0..a_6]) || reverse([c_0..c_25]).
```

Output order is `registers_val`, `rs1_ra`, `rs2_ra`, `rd_wa`, `rd_inc`.
After round 25, the host reads two dense rows and two increment values, binds
`c_25`, and uses `OptimizedRegistersReadWrite`'s dense address algorithm for
rounds 26--32 and the terminal bind. The device evaluates the rs1/rs2 one-hot
openings from the retained CSR after the address point is known. The CPU
implementation remains the byte oracle for all 33 round polynomials, running
claims, the five outputs, derived `EqCycle`, and transcript bytes. A Metal
preflight failure selects the complete CPU member before round 0; a failure
after the first command is submitted aborts the proof rather than rebuilding
CPU sparse state.

## Owner, memory layout, and command graph

The stage-1 composite witness traversal produces these immutable device
planes directly in their final allocations:

```text
start_values    u64[blocks * 128]
rs1_offsets     u32[blocks * 128 + 1]
rs2_offsets     u32[blocks * 128 + 1]
rd_offsets      u32[blocks * 128 + 1]
rs1_positions   u8[rs1_events]
rs2_positions   u8[rs2_events]
rd_positions    u8[rd_events]
rd_post_values  u64[rd_events]
rd_inc          Fp128[T]
```

Blocks contain 256 cycles. The frozen analytical log-26 fixture has 262,144
blocks, 33,554,432 block/register columns, 59,652,323 rs1 events, 55,924,053
rs2 events, and 50,331,648 rd events. CSR storage is exactly 1,239,649,860
bytes and the integrated CSR-plus-increment producer writes 2,313,391,684
bytes. It validates register indices, monotone offsets, event order, every
read value, every write pre-value, carried state across blocks, source
generation, and all allocation sizes before publishing.

The sequence owns two increment arenas (1,073,741,824 and 536,870,912 bytes),
two 16-byte-aligned AoS dense arenas of `(val, ra, wa)` (1,610,612,736 and
805,306,368 bytes), coefficient/equality tables, and reduction scratch. The
modeled peak resident set is 5,276,143,172 bytes. No round allocates a buffer,
uploads rows, rebuilds CSR, or repacks the sparse/dense boundary.

Ownership advances through private, one-shot types:

```text
RegisterProducerIdentity
  -> CertifiedRegisterOwnerReceipt
  -> PreparedRegistersRwReceipt
  -> RegistersRwRoundReceipt { next_round, challenge_digest, command_serial }
  -> RegisterStage5OwnerReceipt
```

Every plane uses `AllocationReceipt<PlaneKind>` containing device-registry id,
allocation identity, initialized generation, elements, and bytes. Receipt
constructors are private and reject zero or duplicate identities, stale
generations, wrong geometry, missing completion, and a changed ordered-prefix
digest. `ProofSession::take` consumes the stage-1 owner at stage 4; the
sequence's residue parks only the stage-5 planes after output completion.
Cloning a Metal buffer handle does not mint another owner or permit a second
producer charge.

The command graph is:

```text
certified CSR + rd_inc
  -> raw round 0
  -> [host FS]
  -> raw bind/message rounds 1..7
  -> [host FS after each]
  -> raw round 8 + dense materialization
  -> [host FS]
  -> dense bind/message rounds 9..25
  -> CPU bind c25 + address rounds 26..32
  -> CSR rs1/rs2 output scan
  -> five output claims + stage-5 owner
```

For raw rounds, one 128-thread group contains four SIMD groups. Each SIMD group
walks one residue class of `x_in`; its 32 lanes own four adjacent register
columns, so CSR headers and event streams stay coalesced. It accumulates two
endpoints without a threadgroup barrier per `x_in`, then the four SIMD-group
partials are reduced once and multiplied by `e_out`. Prefix-coefficient tables
are generated on device after each host challenge and remain read-only for the
message pass. Round 8 also writes one 48-byte dense cell per
256-cycle-block/register. Raw threadgroup storage is limited to cooperative
offset windows and endpoint scratch (under 4 KiB); its structural live floor
is 22 scalar words. Dense rounds use one lane per register, bind source rows
into the destination, and form the next message before eviction. Their
structural live floor is 52 scalar words and threadgroup storage is 288 bytes.
Promotion requires no local-memory spill and at least two resident
128-thread groups per core for both regimes; emitted-code/Instruments evidence
must replace these source-level occupancy estimates.

## Log-26 work, traffic, and roof

Counts below are exact for the deterministic analytical row generator, not for
the production Fibonacci witness. The round-0/1 census uses its exact period
`lcm(128,4,5,6,7,9,512) = 161,280` and
`2^26 = 416 * 161,280 + 16,384`; the implementation must regenerate these
figures with a checked model before shader admission. “Full” includes full
field products and coefficient/increment update-equivalents. “Half” is a
field-by-raw-`u64` or signed-delta product. Cache-unique bytes count distinct
streaming data per round; requested bytes count all logical shader loads even
when coefficient/equality tables hit cache.

| Raw slice | Full | Half | Cache-unique bytes | Requested bytes |
|---|---:|---:|---:|---:|
| round 0 | 167,788,547 | 80,211,070 | 4,461,613,324 | 6,284,941,504 |
| round 1 | 100,679,693 | 100,317,099 | 2,850,968,148 | 4,727,590,040 |
| rounds 2--4 | 73,450,494 | 413,961,160 | 5,130,314,732 | 11,997,976,968 |
| rounds 5--7 | 9,288,816 | 364,709,709 | 3,899,204,236 | 9,773,005,624 |
| round 8 + junction | 868,480 | 141,656,012 | 2,869,819,780 | 5,140,505,912 |
| **raw total** | **352,076,030** | **1,100,855,050** | **19,211,920,220** | **37,924,020,048** |

The round-8 row includes 50,069,504 materialization delta products and the
1,610,612,736-byte dense write. Its independent active cap remains
7.941690 ms. Complete execution is:

| Phase | Full | Half | Cache-unique bytes | Requested bytes | Intensity (products/cache byte) |
|---|---:|---:|---:|---:|---:|
| raw 0--8 | 352,076,030 | 1,100,855,050 | 19,211,920,220 | 37,924,020,048 | 0.075627 |
| dense 9--25 | 135,085,048 | 0 | 4,847,746,576 | 4,849,843,264 | 0.027866 |
| output scan | 262,144 | 0 | 395,816,832 | 2,246,087,416 | 0.000662 |
| **execution total** | **487,423,222** | **1,100,855,050** | **24,455,483,628** | **45,019,950,728** | **0.064946** |

Including producer writes, the optimistic lifecycle device traffic is
26,768,875,312 bytes and logical requested traffic is 47,333,342,412 bytes.
The retained M4 Max controls are 451,701,710,520 B/s copy bandwidth,
18.10 G full products/s, and 26.272 G half-width products/s. The larger of
the arithmetic and cache-unique traffic floors is divided by 0.80:

| Phase | Compute floor | Traffic floor | 80%-roof cap |
|---|---:|---:|---:|
| producer | 0 | 5.122 ms | 6.402 ms |
| raw 0--8 | 61.354 ms | 42.532 ms | 76.692 ms |
| dense 9--25 | 7.463 ms | 10.732 ms | 13.415 ms |
| output scan | 0.014 ms | 0.876 ms | 1.095 ms |

Adding the retained 8.756582-ms host-FS/wait/readback/CPU-tail reserve gives a
106.361-ms cache-aware projection: 8.79x against the frozen 934.665875-ms CPU
median. Treating every logical request as DRAM traffic, applying the same 80%
factor, and adding host work gives 139.743 ms or 6.69x. The frozen complete
budgets are 186.933175 ms (5x), 155.777645 ms (6x), and 116.833234 ms (8x).
The cache-aware model has 10.472 ms of headroom to 8x.

The official PIOP span follows `piop_goal.v2.json` and excludes this
transcript-independent owner production. The standalone member evaluator
charges the producer anyway, while the PIOP evaluator reports its wall time as
the existing secondary backend-witness metric. It may not disappear from both
views or be charged twice in one view.

The five-pair frozen artifact is
`benchmark-runs/metal-piop-eval/20260806-133709-697013/result.json`, revision
`5f520c21e`, with CPU/Metal-fallback
medians 934.666/940.177 ms. The latest stable one-pair diagnostic at revision
`d63bc7d97` measured 958.552/965.251 ms. Thus 106--140 ms would remove
85.5--89.0% of the current fallback wall. These are feasibility projections,
not achieved speedups. Production admission must replace the analytical
census and retained rates with same-record observations.

## Hybrid, validation, and evaluator

Routing is all CPU or the complete Metal/CPU hybrid. Start with CPU below log
25 and Metal at or above log 25, then freeze the crossover from an untuned
alternating sweep over logs 20, 22, 24, 25, 26, and 27. Capacity or receipt
rejection occurs before transcript mutation. No partial Metal run may fall
back.

Required parity covers empty access sets; every register index; rs1/rs2/rd
collisions; missing even and odd children; repeated writes; `gamma = 0, 1`
and random; zero, maximum, and signed-boundary increments; `u64::MAX` values;
odd and even small logs; stale devices, generations, allocation identities,
and challenge digests; and intentionally inconsistent value flow, which must
select CPU before round 0. For every valid case, compare the independent dense
relation oracle, `OptimizedRegistersReadWrite`, and the Metal model at every
endpoint, cubic byte string, running claim, dense junction cell, opening point,
derived scalar, and output. A standard Akita/Metal `muldiv` proof must verify.
Because the workspace forbids the `akita` and `zk` features together, the
`host,zk` checks remain on the CPU backend and must continue to pass with the
Metal slot unavailable.

The Criterion groups are
`registers_read_write_v3/{producer,raw_round,junction,dense_round,output,complete}`.
Each invocation prepares once, reuses allocations, checks parity before timing,
and reports command wall, GPU active, producer wall, host FS/wait, readback,
CPU tail, allocations, peak bytes, cache/DRAM bytes, registers, occupancy, and
spills. Cheap iteration uses small exact fixtures and log-22 resident rounds;
log 26 is the ranking scale. The root controller alone runs Cargo or Metal,
with evaluator concurrency one and unchanged source/binary fingerprints.

Promotion requires five alternating log-26 CPU/Metal pairs from one binary,
both run-order strata at least 5x, median complete wall at most 186.933175 ms,
every proof verified, raw/dense/output active medians within their independent
caps, and an untuned log-27 transfer. If the complete result clears 6x and the
raw phase is cache-bound below its arithmetic cap, continue toward 8x. Kill or
redesign this architecture if the round-8 junction exceeds 7.941690 ms, raw
active exceeds 76.692 ms, any buffer spills or exceeds the runtime
`maxBufferLength`, the owner needs a second trace scan/upload, or the complete
member cannot clear 5x.

## Implementation slices

1. Register a checked v3 census/model and an independent CSR-to-dense oracle;
   freeze the numbers above before MSL changes.
2. Implement the typed device owner and stage-1 shadow producer. CPU remains
   authoritative; record incremental construction, validation, first-touch,
   identities, and peak memory.
3. Implement one parameterized raw MSL family and the round-8 junction first.
   Clear parity, the 7.941690-ms junction cap, spills, and occupancy before
   filling rounds 0--7.
4. Add the host-FS sequence adapter with preallocated equality, coefficient,
   increment, partial, and dense arenas. Shadow every round against CPU; do not
   register the backend slot.
5. Move the accepted dense bind/message and output logic behind the same v3
   owner. The old standalone dense runtime remains a negative control and is
   not called by production.
6. Add the CPU address tail, terminal outputs, one-shot stage-5 residue, and
   complete-member Criterion evaluator. Register the backend only after exact
   shadow parity.
7. Run the crossover sweep, one-pair log-26 diagnostic, five-pair validation,
   and log-27 transfer. If cache evidence supports it, price a round-9
   materialization cutoff; it is a new phase, not an unrecorded tuning change.

The canonical implementation layout is a high-level
`metal/registers_read_write.rs` adapter and
`metal/solinas/registers_read_write_v3/{abi,owner,sequence,model,oracle}` with
`shaders/{raw,dense,output}.metal`. Older bridge/dense/successor packets are
evidence inputs only; production code must not import their admission types or
duplicate their ownership layers.
