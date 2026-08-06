# RAM value-check successor design

Status: corrected static design packet. Nothing in this directory is registered,
compiled, or measured. The existing low-level `ram_val_check/` implementation
remains the only device evidence. The shader here is an unmeasured prototype,
not a promoted candidate.

## Frozen relation

For cycle `j`, the optimized prover sums

```text
s(j) = RamInc(j) * RamRa(r_address, j) * (LT(j, r_cycle) + gamma).
```

`RamRa` is zero on a cycle without a remappable address and otherwise equals
`eq(r_address, remapped_address[j])`. A read has a nonzero `RamRa` but zero
`RamInc`. The sumcheck has 26 low-to-high rounds and degree three. Each message
returns evaluations at `t = 0, 2, 3`; the engine supplies `s(1)` from the
preceding claim.

The symbolic input and output expressions remain:

```text
RamVal + gamma * RamValFinal - (1 + gamma) * ValInit
ValInit = InitEval - sum(InitSelector_i * contribution_i)

(LtCyclePlusGamma) * RamInc * RamRa.
```

Untrusted-advice, trusted-advice, and program-image openings are dual-role. The
adapter echoes those input cells exactly and returns `RamRa` and `RamInc`; the
stage-4 coordinator retains its hand-written global opening order.

Fiat-Shamir stays on the host. The empty labeled append
`b"ram_val_check_gamma"` precedes the gamma draw. Every round challenge remains
shared with the stage-4 batch. `TraceDimensions::cycle_opening_point` performs
the verifier-side reversal, so the shader binds low-to-high without another
reversal.

The retained LT factorization is

```text
LT(j, r_cycle) + gamma
  = lt_high[j_high] + eq_high[j_high] * lt_low[j_low].
```

Gamma belongs in `lt_high`. At `2^26`, both halves have 13 variables, so
`lt_low`, `lt_high`, and `eq_high` contain 8192 fields each.

## Screening evidence, not promotion evidence

`screening_evidence.json` is the durable provenance fixture for every retained
number used here. It records:

- the CPU artifact, SHA-256, revision, extraction selector, alternating pair
  order, five trace hashes, and five samples;
- the tracked predecessor Metal evidence and its SHA-256;
- the exact machine/geometry controls and limitations.

The frozen CPU samples are 240.056416, 274.334163, 232.004456, 234.656875,
and 229.820624 ms. Their median is 234.656875 ms, giving screening caps of
46,931,375 ns at 5x and 29,332,109 ns at 8x. The attributed seam is the union of
`RamValCheck::{prepare, prove_round, finish_rounds, output_claims}`. It excludes
the batch coordinator's between-call Fiat-Shamir time.

Those caps rank experiments only. The CPU artifact, predecessor Metal control,
and successor source are different revisions. Promotion requires a new tracked
artifact with five alternating paired CPU/Metal samples from one current
revision and the same boundary. `speed_screen_decision` derives the ratio from
those paired medians rather than comparing a new candidate against the frozen
cap. A passing speed screen is not promotion. `admission_decision` additionally
requires activity provenance, all three phase bars, and compiled resource
evidence described below.

## Common producer contract

There is no admissible producer in production today:

- `RamAccessColumns` owns host `Vec<u32>` addresses, not a Metal allocation;
- the current 40-byte stage-5 Booleanity row has remapped address and fused
  increment but no Store selector and is produced after RAM value-check;
- the increment-claim common row is another static proposal;
- the predecessor RAM value-check benchmark packed and uploaded a dedicated
  1-GiB allocation.

The successor requires one producer-owned base row:

```text
IncrementAccessRow (16 bytes)
  increment_magnitude: u64
  remapped_ram_address: u32       // u32::MAX means no remappable address
  flags: u32
    bit 0: selected increment is nonnegative
    bit 1: Store selected RamInc rather than RdInc
```

The physical row is constructed only from this checked composite source:

```text
IncrementAccessSource
  remapped_ram_address: Option<u64>
  store: bool
  ram_increment: i128
  rd_increment: i128
```

Before discarding either delta, release-mode construction requires

```text
store      => rd_increment == 0 and selected = ram_increment
not store  => ram_increment == 0 and selected = rd_increment.
```

This matches the increment-claim packet. A future shared module must own this
single ABI; the two sumchecks must not publish private lookalike types.
Construction also receives the configured `u32` address-domain length, requires
a nonzero power of two below the sentinel, checks the `u64 -> u32` conversion,
and rejects every non-sentinel address outside the table before publishing the
packed row.

### Address-zero semantics

Raw address zero remaps to `None`, including on a store. Such a store is valid:
the row retains the Store bit and RAM delta, while RAM value-check maps its
`RamRa` to zero. Increment claim reduction can still consume the delta. The ABI,
Rust oracle, and MSL validator therefore permit bit 1 together with the
no-address sentinel. A sentinel in `Some(address)` remains a collision, and any
non-sentinel address must be within the address table.

### Ownership and construction

The intended owner allocates the final `StorageModeShared` buffer during
transcript-independent `backend_witness_prepare` and fills it directly from a
typed witness window. It records `A` and `S` in the same pass. For an owned
random-access trace this requires no full-domain temporary and no
`new_buffer_with_data` upload. A re-emulating source must stream the checked
composite directly into the final destination and report any chunk scratch; a
retained Spartan row is not a substitute because it lacks remapped address and
the non-store rd pre-state needed to reconstruct both deltas.

The base replaces the Metal backend's separate address representation. Stage 5
adds a 24-byte lookup/PC payload, preserving 40 bytes per cycle after that
point. RAM value-check, increment reduction, and the terminal Booleanity/Hamming
consumer borrow the same allocation identity.

Promotion requires:

- one nonzero storage identity at creation, RAM value-check, and terminal use;
- exactly one allocation and `2^26` rows written;
- zero row-upload, full-domain-copy, and full-domain-temporary bytes;
- an explicit peak-byte counter for any bounded streaming scratch, rejected if
  it reaches a full 16-byte row domain;
- release-checked Store/RamInc/RdInc exclusivity;
- parity for store, load, no-access, and raw-address-zero store rows;
- retention through the final registered consumer.

The PIOP metric follows the project contract and excludes transcript-independent
backend representation materialization. The diagnostic must also report the
full 1-GiB producer write and PIOP plus backend preparation. Zero upload does
not mean zero memory traffic.

## Selected first-message mechanism

For a pair `(2y, 2y+1)`, if both endpoint RAM increments are zero, their
multilinear bind is zero at every `t`. All six inner products at `t = 0, 2, 3`
can be omitted. This does not erase a load's `RamRa` when its neighbor is a
store, so interpolation cross terms remain intact. A raw-address-zero store is
still active when its RAM delta is nonzero because its neighbor can supply a
nonzero interpolated `RamRa`.

The first shader reads and validates every 16-byte row. Each SIMD32 iteration
owns 32 adjacent pairs and uses `simd_any` before address/LT gathers and field
products. One SIMD32 threadgroup owns each high block. There is no threadgroup
scratch or barrier.

Let

```text
N = 2^26
H = 8192
A = pairs with at least one nonzero RAM increment
S = 64-cycle chunks with at least one nonzero RAM increment.
```

Then

```text
logical field products             = 6A + 6H
SIMD32-equivalent product slots     = 6 * 32 * S + 6 * 32 * H
compulsory native-row bytes         = 16N = 1,073,741,824.
```

The `6H` epilogue products execute under lane zero. Until a compiled capture
demonstrates a scalar execution path or the epilogue is repacked across lanes,
the throughput model charges 32 slots per product. `A` alone is insufficient:
scattered stores can make every SIMD iteration active.

## Host launch and reduction contract

`abi.rs` encodes the host-side validation contract. Before the first dispatch:

- parameters are power-of-two and satisfy `high_blocks * low_length == rows`;
- reserved words are zero and the sentinel is `u32::MAX`;
- dispatch is exactly `high_blocks` threadgroups by 32 threads;
- row, address, LT, partial, and one-word status buffers meet their minimum
  element lengths;
- the atomic status word is zero.

Each reduction step requires

```text
output_count = ceil(input_count / 32)
columns = 3
threadgroups = output_count
threads per threadgroup = 32.
```

Input and output are column-major buffers of `3 * input_count` and
`3 * output_count` fields. Each reducer binding carries a nonzero storage
identity plus a byte offset and length. Field ranges are 16-byte aligned and the
status range is 4-byte aligned. Host validation checks range-end arithmetic,
minimum byte lengths, and rejects any input/output intersection or status
intersection in the same storage allocation. Adjacent nonoverlapping ranges in
one allocation are valid. The shader additionally guards
`output_index >= output_count`, so accidental overdispatch cannot write out of
bounds, but host validation still rejects it.

The host clears status before submission, waits for command-buffer completion,
then checks it before consuming partials or issuing a dependent step. Bit 0 is
unsupported parameters; bit 1 is an invalid row. Any nonzero value discards the
candidate result.

## Resource inventory

The explicit outer shader scope keeps six field accumulators, endpoint values,
deltas, and one field result live. That inventory is not a peak-register
envelope: inlined `solinas_mul_wide` also carries an eight-limb wide product,
folding state, carry words, and correction values. There is no source-only
occupancy conclusion.

The candidate is resource-valid only after a same-revision compiled capture
records nonzero binary and capture hashes. Each of the sparse-first, native,
and dense phases records allocated registers, observed resident SIMD groups per
core, and a preregistered minimum residency. Admission rejects a missing field,
residency below that minimum, or any device-memory spill bytes. If the compiler
retains materially more state than expected, the next bounded experiment is a
three-sample lane split; it is not preselected because it adds shuffles and
duplicate table work.

## Resident hybrid schedule

The complete intended schedule preserves the ten-bind cutoff:

1. sparse first message from the common native rows;
2. one native bind plus message, materializing `N/2` dense `(RamInc, RamRa)` rows;
3. nine fused dense bind-and-message transitions;
4. read the `2^16` two-field state once;
5. run the remaining 15 messages and final bind in the optimized CPU tail;
6. validate `LtCyclePlusGamma`, echo contribution cells, and return
   `RamRa`/`RamInc` in canonical member order.

After every GPU message, the host constructs the member polynomial, lets the
stage-4 batch absorb it, and supplies the next challenge. There is no device
Fiat-Shamir or protocol change.

Only step 1 and its reducer are present in `shader.metal`. Steps 2--6 are a
required completion contract, not implemented successor code.

## Exact target-scale work and traffic

For dense activity (`A = N/2`, `S = N/64`):

| Phase | Logical products | SIMD32-equivalent slots | Large-state bytes |
| --- | ---: | ---: | ---: |
| First message | 201,375,744 | 202,899,456 | 1,073,741,824 |
| Native bind + message | 167,821,312 | 169,345,024 | 2,147,483,648 |
| Nine dense transitions | 167,886,848 | 187,170,816 | 3,214,934,016 |
| GPU prefix | 537,083,904 | 559,415,296 | 6,436,159,488 |

The last three dense transitions have only 16, 8, and 4 low pairs per high
block. Their ten inner product instructions still occupy SIMD32 slots in masked
lanes; the slot column charges that underfill as well as the lane-zero epilogue.

The remaining mandatory accounted traffic is:

| Item | Bytes |
| --- | ---: |
| Three-column partial writes/reductions for one message | 811,824 |
| Three-column partial writes/reductions for 11 messages | 8,930,064 |
| Initial address/LT table writes | 524,288 |
| Ten LT-low bound-prefix writes | 130,944 |
| CPU-tail handoff | 2,097,152 |
| Eleven 48-byte host message reads | 528 |
| Eleven status clears and reads | 88 |
| Accounted compulsory total | 6,447,842,552 |

The first-message traffic floor includes its row scan, 811,824 reduction bytes,
and an eight-byte status clear/read. Repeated small-table cache-line fills are
not invented; they must be observed.

The requested resident buffer footprint is 2,685,665,284 bytes:

| Allocation | Bytes |
| --- | ---: |
| Producer-owned native base | 1,073,741,824 |
| Dense arena A | 1,073,741,824 |
| Dense arena B | 536,870,912 |
| Address and split-LT tables | 524,288 |
| Two three-column partial buffers | 786,432 |
| Atomic status word | 4 |

Sequence-owned scratch excluding the attached base is 1,611,923,460 bytes.

## Independent roof and heuristic screen

The optimistic controls are 451,701,710,520 B/s and 32.33 G full-field
products/s. The retained 18.10 Gproduct/s six-accumulator result is a matched
diagnostic, not a hard compute ceiling.

For any exact activity record, `model.rs` computes

```text
compute_floor = SIMD32-equivalent slots / 32.33 Gproduct/s
traffic_floor = accounted compulsory bytes / 451.701710520 GB/s
optimistic_floor = max(compute_floor, traffic_floor)
80%-roof bar = optimistic_floor / 0.80.
```

Dense first-message bounds are 6.275888 ms compute, 2.378902 ms traffic, and a
7.844860-ms 80%-roof bar. Dense full-prefix bounds are 17.303288 ms compute,
14.274559 ms traffic, and a 21.629110-ms 80%-roof bar.

Complete admission uses three separately timed bars. Their byte boundaries
include each phase's large state, reducers, status traffic, 48-byte message
readbacks, and bound-table writes. The dense phase also includes the 2-MiB CPU
tail handoff. Initial challenge-table construction remains backend preparation.
All source rows and initial tables are resident before these timers. The first
timer ends after its status and message readback. The native timer starts before
the first LT bind and ends after that message readback. The dense timer starts
before the first dense bind and ends after the tail state is host-readable.

| Dense phase | Accounted bytes | Compute floor | Traffic floor | 80%-roof bar |
| --- | ---: | ---: | ---: | ---: |
| Sparse first message | 1,074,553,704 | 6.275888 ms | 2.378902 ms | 7.844860 ms |
| Native bind + message | 2,148,361,064 | 5.238015 ms | 4.756150 ms | 6.547519 ms |
| Nine dense transitions + handoff | 3,224,403,496 | 5.789385 ms | 7.138347 ms | 8.922934 ms |

The producer records `A` and `S`, so the sparse-first bar is recomputed from
the observed activity rather than the dense row above. Its provenance record
must contain nonzero revision, artifact, and trace hashes and must match the
producer's storage identity, row count, `A`, and `S`. Five nonzero
exact-boundary latency samples for every phase must have a median no greater
than its recomputed bar.

The predecessor's 7.918792-ms dense first message is slightly outside that
first-phase bar. It is evidence about a different shader, not a waiver: the
successor must meet the bar or register a measured structural term before its
result is called occupancy-satisfied.

At the conservatively rounded `q = 0.600` screen (`S = 629,146`):

| Boundary | SIMD32 slots | Compute floor | Traffic floor | 80%-roof bar |
| --- | ---: | ---: | ---: | ---: |
| First message | 122,368,896 | 3.784996 ms | 2.378902 ms | 4.731245 ms |
| Full GPU prefix | 478,884,736 | 14.812396 ms | 14.274559 ms | 18.515495 ms |

The retained historical interpolation is still useful for experiment ordering:

```text
heuristic_first(q) = old row-only traffic floor
                   + q * (old dense observation - old row-only traffic floor)
heuristic_hybrid(q) = 31.106 ms - 7.918792 ms + heuristic_first(q).
```

It projects 5.702117 ms for the first phase and 28.889325 ms for the hybrid at
`q = 0.600`. That first-phase projection is slower than the 4.731245-ms
80%-roof bar, so it predicts experiment value, not occupancy satisfaction. The
interpolation is neither a lower nor an upper bound and cannot reject the
mechanism. Activity classes mean only:

- `q <= 0.600`: target-scale priority after a small parity/proxy run;
- `0.600 < q < 0.680`: proxy first;
- `q >= 0.680`: low priority, not impossible.

Only an analytical lower bound crossing a cap or a measured fixed-candidate
proxy may kill the mechanism. Passing 5x does not stop the search; an integrated
result below 8x remains `PassFiveXPursueEightX`.

## Evidence sequence

1. Commit this corrected analysis and its screening provenance before compiling
   or measuring the prototype.
2. Add the checked common producer and record `A`, `S`, storage identities,
   copy/upload counters, source kind, and lifetime without running this shader.
   Bind the activity record to nonzero revision, artifact, and trace hashes.
3. Reject if producer validation fails. Otherwise use activity only to choose
   experiment priority.
4. Compile the first slice and compare all three columns with the independent
   dense oracle on signed limits, modulus edges, loads next to stores,
   raw-address-zero stores, no-access rows, and challenges 0/1/-1/random.
5. Run a small proxy for parity and active-SIMD behavior before `2^26`.
6. Capture allocated registers, spills, resident SIMD groups, and cache behavior
   for all three phases. Register a nonzero residency minimum before promotion.
7. Wire the hybrid tail and stage-4 adapter. Run five alternating paired CPU and
   Metal proofs in clear and ZK modes, preserving host Fiat-Shamir and checking
   output point, output claim, and transcript parity.
8. Record five exact-boundary latency samples for each phase and compare their
   medians with the activity-derived 80%-roof bars.
9. Promote only from a tracked same-revision artifact after all fail-closed
   admission gates pass. Report PIOP plus backend preparation separately.

## Open evidence

- The common producer and shared ABI owner are absent.
- Fibonacci `A` and `S` are unknown.
- The first slice has not been compiled or run.
- Register allocation, spills, residency, and cache behavior are unknown.
- No activity-provenance record or per-phase boundary-latency samples exist.
- Steps 2--6 of the hybrid schedule and the proof-stage adapter are absent.
- The exact CPU tail, host synchronization, and adapter costs require integrated
  measurement.
- Clear/ZK proof and transcript parity are untested.

No protocol amendment is proposed.
