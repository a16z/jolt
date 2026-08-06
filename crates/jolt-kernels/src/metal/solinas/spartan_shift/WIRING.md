# Spartan shift Metal design

This directory contains the checked geometry and independent host oracle,
registered Metal entry points, a standalone resident-buffer runtime, and its
microbenchmark for the stage-3 `SpartanShift` member. It does not yet select a
PIOP backend or change the protocol.

The current result is a provisional resident-service kernel, not an accepted
PIOP result. Exact GPU parity passes and a warmed residency control clears 5x,
but the corrected first-use host-written path does not. The production resident
producer, reusable cross-proof workspace, and fairly attributed complete-member
evaluator are still missing.

## Decision

Use a two-command resident hybrid, not a literal 26-round device prover:

1. Before round 0, scan compact native cycle planes once and build the four
   `P * Q` prefix tables on the GPU. Read back four `sqrt(T)`-sized `Q` tables.
2. Prove the low-variable half of the sumcheck on the CPU. Fiat--Shamir remains
   in the generated batch driver.
3. Once the last low challenge is known, scan the same resident planes once
   more and fold the five witness columns to `sqrt(T)` dense tables. Read those
   tables back.
4. Prove the high-variable half and return the five openings on the CPU.

At `T = 2^26`, each large command has thousands of independent threadgroups,
while all host tables contain only 8192 fields per column. The two full-domain
passes are unavoidable under host Fiat--Shamir unless the device stores a much
larger challenge-independent transform. A per-round Metal port is rejected:
26 submissions add latency and repeatedly move shrinking field tables without
removing either full-domain pass.

The implementation carries both prefix-build variants as controls:

- **mixed:** form the gamma-combined outer value with one 128-by-64 product,
  then perform two full-field products per row for the current/successor sums;
- **expanded:** pre-scale five high weights and use only 128-by-64 products.

The target sweep retained mixed at 64 build threads and a 128-row high tile.
Expanded was slower across the useful target configurations and remains a
correctness/control path rather than a production candidate. Revisit it only
if a new half-width primitive materially changes the matched arithmetic roof.

## Frozen CPU denominator

The development denominator is the five optimized-CPU traces in
`benchmark-runs/metal-piop-eval/20260806-133709-697013`, revision
`5f520c21e338632aa0bf5936ceb02be6c22fa40f`, `log_T = 26`, with 16 Rayon
threads. Summing this member's `prepare`, every `prove_round`,
`finish_rounds`, and `output_claims` spans gives:

```text
131.051624 ms, 131.584500 ms, 129.304918 ms, 130.343291 ms, 134.289502 ms
```

The median is `131,051,624 ns`. The hard 5x cap, rounded down, is
`26,210,324 ns`; the 8x stretch cap is `16,381,453 ns`. Host Fiat--Shamir
spans sit outside these member spans and must either remain excluded from both
arms or be charged to both arms in final paired evidence.

The median sample's internal shape is particularly useful:

| component | optimized CPU wall |
|---|---:|
| prepare: collect native rows and build four Q tables | 116.075167 ms |
| prefix rounds 0--12 | 0.583040 ms |
| prefix-to-dense transition | 13.739417 ms |
| suffix rounds 13--25 | 0.653083 ms |
| finish and outputs | 0.000917 ms |
| complete member | 131.051624 ms |

This is why the design offloads exactly the prepare and transition scans while
leaving the tiny round ladders on the host. The GPU numerator must include the
incremental cost of producing its resident planes. Moving that cost to stage 1
does not make it free.

## Exact protocol boundary

For cycle `j`, define

```text
outer(j) = upc(j)
         + gamma * pc(j)
         + gamma^2 * is_virtual(j)
         + gamma^3 * is_first(j)

product(j) = gamma^4 * (1 - is_noop(j)).
```

The degree-two summand is

```text
eq+1(r_outer, j) * outer(j)
  + eq+1(r_product, j) * product(j).
```

`r_outer` is stage 2's product uni-skip `tau_low`; `r_product` is the product
remainder opening point. Both are big-endian and have `log_T` coordinates.
The input claim is the gamma-fold of the five upstream `Next*` openings. The
output consists, in canonical order, of openings of:

```text
UnexpandedPC, PC, VirtualInstruction, IsFirstInSequence, IsNoop.
```

Every output point is the reverse of the low-to-high round-challenge list.
The generated stage-3 batch combines this member with InstructionInput and
RegistersClaimReduction, absorbs the combined round polynomial, and draws the
shared challenge. A shader must never hash, absorb, draw, or independently
advance the transcript.

No protocol change is proposed. In particular, the five output values cannot
be replaced by this member's gamma combination: stage 6 folds them under a
later, independent challenge. `UnexpandedPC` aliases InstructionInput at the
same point, but exploiting that alias requires a typed sibling-output seam and
is lower priority than the self-contained five-output path.

## Prefix--suffix algebra

Let

```text
n = log2(T)
suffix_vars = floor(n / 2)
prefix_vars = n - suffix_vars
H = 2^suffix_vars
P = 2^prefix_vars
j = x_hi * P + x_lo.
```

For either reference point `r = r_hi || r_lo`, the existing exact
decomposition is

```text
eq+1(r, (x_hi, x_lo))
  = p0(x_lo) * s0(x_hi) + p1(x_lo) * s1(x_hi),

p0 = eq+1(r_lo, .)
s0 = eq(r_hi, .)
p1 = product(r_lo) * delta_0(.)
s1 = eq+1(r_hi, .).
```

The shader uses the equivalent successor identity

```text
sum_x s1(x) * value(x) = sum_x eq(r_hi, x) * value(x + 1),
value(H) = 0.
```

Thus one high weight serves both current and successor accumulations. The
zero at `H` is essential: `eq+1` has no wraparound. Applied to the outer and
product terms, this produces the four pairs, in fixed order:

```text
(outer.p0, outer.q_current)
(outer.p1, outer.q_successor)
(product.p0, product.q_current)
(product.p1, product.q_successor).
```

The host computes each degree-two round at `t = 0` and `t = 2`; `s(1)` comes
from the previous-claim hint. It binds all eight P/Q tables low-to-high. For
the target `P = 8192`, the complete prefix message-and-bind core has exactly
`16P - 24 = 131,048` full-field products.

After the thirteenth prefix challenge, reverse those challenges to obtain the
big-endian point and materialize

```text
w[x_lo] = eq(reverse(prefix_challenges), x_lo).
```

The second command emits five column-major high tables:

```text
column_dense[x_hi]
  = sum_x_lo w[x_lo] * column(x_hi * P + x_lo).
```

The host recombines each partially bound `eq+1` table from its two suffix
components exactly as the optimized CPU kernel does, then proves the remaining
high rounds. The seven-table suffix core has `19H - 19 = 155,629` products at
the target, excluding lower-order equality-table construction.

## Resident source contract

The preferred producer exposes structure-of-arrays storage:

| plane | ABI | target bytes |
|---|---:|---:|
| unexpanded PC | native little-endian `u64[T]` | 512 MiB |
| PC | native little-endian `u64[T]` | 512 MiB |
| current flags | three `u32` bitplanes per 32 cycles | 24 MiB |

The three flag bitplanes are `VirtualInstruction`, `IsFirstInSequence`, and
`InstructionFlags::IsNoop`, in that order. They are current-cycle flags, not
the `Next*` values consumed by the input claim. All padding rows must match the
canonical witness; no satisfied-witness shortcut is allowed.

The owner carries the row count, device-registry identity, and an
`(allocation_identity, byte_len)` descriptor for each of the three buffers. The
checked model accepts the active context's expected registry identity and
rejects a foreign device, a zero or duplicate allocation identity, an incorrect
byte length, or missing exact-current-flags certification. The isolated type
does not own a `metal::Buffer`; the production adapter must construct the
metadata from the actual buffers before validation. This member borrows the
allocations through its second command and releases them afterward. It performs
no conversion, allocation-sized copy, or upload in its measured resident path.

The natural production seam is the existing stage-1 Metal witness traversal
that writes `SpartanOuterUniskipRows`. That traversal already has both PC
values and both current circuit flags. `SpartanOuterRow` does not currently
retain `InstructionFlags::IsNoop`, so integration must either add that
protocol-neutral witness field or read it from the same source row while
packing. A second full witness traversal is a diagnostic fallback, not the
production residency claim.

The existing producer parallelizes over individual rows, but the packed flag
destination has one word for 32 rows. Production therefore changes the work
partition to aligned 32-row chunks. One worker owns one flag word, extracts the
chunk's rows, writes the same 32 positions in both `u64` planes, and emits the
three flag masks with ordinary stores. Atomic OR, a temporary byte-per-row flag
array, and a second packing pass are rejected because they either add
contention or violate the one-traversal boundary. `pack_flag_word` is the host
oracle for this ownership rule; unused bits in the final word remain zero.

At the target the checked producer plan has 67,108,864 row extractions,
2,097,152 independently owned flag chunks, 1,073,741,824 value bytes written,
25,165,824 flag bytes written, and 1,098,907,648 total output bytes. Allocation,
row extraction, these writes, and shared-buffer visibility all count toward the
3.5-ms producer gate.

The current 48-byte InstructionInput rows and 112-byte outer residual rows are
not acceptable aliases. Reading one useful word at those strides would move
about 10 GiB over two scans and the residual owner is released before stage 3.
Likewise, reconstructing current `is_noop` by shifting stage-2 `NextIsNoop`
loses cycle zero unless another exact source is retained.

Producer cost is accounted as follows:

- standalone comparison: charge the incremental allocation and fill to the
  Metal member;
- resident PIOP comparison: attribute the same fixed producer boundary to both
  arms and separately report its incremental wall and bytes;
- never compare an uncharged resident Metal numerator with a CPU denominator
  that includes row collection.

After command 2, no protocol consumer needs the full native planes. The host
dense state produces the five scalar openings and their common point; stage 6
consumes only those values. The allocation owner may therefore be released
after the fold unless InstructionInput explicitly shares the unexpanded-PC
plane. In that case the last typed owner releases it. Scratch may be returned
to the stage arena immediately after each command, but allocator reuse must not
be described as data reuse.

## Device schedule

### Command 1: prefix Q build

The retained geometry uses 64 threads per threadgroup and 128 high rows per
tile. One thread owns one `x_lo`; adjacent lanes therefore read adjacent cycle
rows at every high step. At the target there are 128 low threadgroups times 64
high tiles, or 8192 threadgroups. Each thread carries four field accumulators
and writes one four-field partial. A second dispatch in the same command buffer
reduces the 64 partials for every `x_lo` and writes four Q tables.

The mixed kernel carries the current outer value across its high loop. A tile
loads one successor halo row, so only the 63 internal tile boundaries repeat.
The expanded kernel pre-scales high weights by the gamma powers and replaces
the dependent outer/full multiplication pair with direct half-width products.
Both produce identical partial and Q layouts.

### Command 2: native prefix fold

Dispatch one threadgroup per `x_hi`. At the retained width 32 each lane visits
256 contiguous low rows and accumulates the five output columns. PC columns
use canonical 128-by-64 multiplication. Boolean columns conditionally add the
field weight. SIMD reductions use five fields per SIMD group, or 80 bytes of
dynamic threadgroup memory. The command writes five 8192-element tables and
waits once before the CPU suffix.

The two commands read back `4P + 5H = 73,728` fields, exactly 1,179,648 bytes.
Shared-buffer visibility and host cache effects count in wall time even if no
explicit copy occurs.

The initial evaluator runs both commands synchronously. A later stage-level
optimization may submit command 1 as a primer and join it at round 0, allowing
sibling preparation to overlap. Command 2 can overlap sibling round-13 work
only through an explicit two-phase batch-driver hook. Such overlap preserves
Fiat--Shamir because all member polynomials are combined before absorption, but
it is an end-to-end scheduling win, not permission to remove either command's
active time from standalone reporting.

The draft ABI is fixed as follows. Every field buffer contains canonical
little-endian 16-byte `SolinasFp128` cells. `SpartanShiftParams` is four
little-endian `u32` words: `P`, `H`, high-tile elements, high-tile count.

| entry point | buffer slots | dynamic threadgroup slot |
|---|---|---|
| `solinas_spartan_shift_build_mixed_partials` | 0 upc, 1 pc, 2 flags, 3 `[gamma,gamma^2,gamma^3]`, 4 `[eq_outer,eq_product*gamma^4]`, 5 partials, 6 params | none |
| `solinas_spartan_shift_build_expanded_partials` | 0 upc, 1 pc, 2 flags, 3 `[eq_outer,eq_outer*gamma,eq_outer*gamma^2,eq_outer*gamma^3,eq_product*gamma^4]`, 4 partials, 5 params | none |
| `solinas_spartan_shift_reduce_prefix` | 0 partials, 1 four column-major Q tables, 2 params | none |
| `solinas_spartan_shift_fold_native` | 0 upc, 1 pc, 2 flags, 3 low weights, 4 five column-major outputs, 5 params | 0: five fields per SIMD group |

Concatenate the source after the offset-specialized `fp128.metal` and
`simd_reduce.metal`. The half-width helper is locally namespaced until the
independent half-width probe is promoted; integration should then share one
reviewed implementation rather than retain duplicate arithmetic.

## Storage and exact target work

With 64 high tiles, target private storage is:

| allocation | bytes |
|---|---:|
| borrowed native value planes | 1,073,741,824 |
| borrowed current-flag bitplanes | 25,165,824 |
| four-field prefix partials | 33,554,432 |
| four Q tables | 524,288 |
| five dense output tables | 655,360 |
| high weights, low weights, and parameters | below 1 MiB |

The local planning model totals 1,134,035,024 resident bytes for mixed and
1,134,428,240 bytes for expanded when all scratch is preallocated. The largest
single source buffer is 512 MiB, which avoids the 2-GiB single-buffer problem
at log 26 and leaves a viable SoA layout through log 28.

The cache-unique lower bound counts each native plane, cached weight table,
partial write/read, and output once. The coalesced-halo model also counts 16
value bytes for every repeated boundary row and one 12-byte packed flag word
for every 32-row block touched at a boundary. At the target that adds 8,257,536
value bytes and 193,536 packed-flag bytes to either build. It still assumes the
small weight tables are served from cache; it is not an instruction-issued byte
count. Expanded has the same repeated native rows as mixed; its additional
high-weight references remain inside that explicit cache assumption.

| phase and traffic model | bytes | floor at 420.68 GiB/s |
|---|---:|---:|
| mixed prefix, cache unique | 1,166,802,944 | 2.583127 ms |
| mixed prefix, coalesced halos | 1,175,254,016 | 2.601837 ms |
| expanded prefix, cache unique | 1,167,196,160 | 2.583998 ms |
| expanded prefix, coalesced halos | 1,175,647,232 | 2.602707 ms |
| native low fold, cache unique | 1,099,694,080 | 2.434558 ms |
| mixed build plus fold, coalesced | 2,274,948,096 | 5.036395 ms |
| expanded build plus fold, coalesced | 2,275,341,312 | 5.037265 ms |

Under the accepted half-width and full-width arithmetic controls, both build
variants and the fold remain compute-bound. This conclusion is conditional on
the unpromoted half-width primitive clearing its floor and on cache capture
confirming the weight-table assumption. Weight tables are at most 640 KiB.

The retained same-machine controls are 18.10 Gproduct/s for a multi-accumulator
full-field kernel and 420.68 GiB/s for a large device copy. The isolated
half-width path is not yet promoted; its pre-registered acceptance floor is
26.272 Gproduct/s. Using those rates gives:

| phase | useful products | projected floor | projected 80%-roof cap |
|---|---:|---:|---:|
| mixed prefix: full width | 134,209,536 | 7.414891 ms | included below |
| mixed prefix: half width | 67,624,960 | 2.574032 ms | included below |
| mixed prefix total | both rows above | 9.988923 ms | 12.486154 ms |
| expanded prefix: half width | 268,419,072 | 10.216926 ms | 12.771158 ms |
| native low fold: half width | 134,217,728 | 5.108775 ms | 6.385968 ms |
| mixed plus fold | mixed counts above | 15.097698 ms | 18.872122 ms |
| expanded plus fold | 402,636,800 half products | 15.325700 ms | 19.157125 ms |

The mixed path's half-width count includes its 63 halo rows per `x_lo`.
Treating every mixed and fold product as the slower 18.10-Gproduct/s full
operation gives a 23.208027-ms 80%-roof projection, leaving too little
complete-member headroom. A validated half-width path is therefore part of
the 5x architecture, not an optional cosmetic optimization.

Charging the full 3.5-ms producer and 2.5-ms service/host allowances gives a
24.872122-ms mixed envelope (5.269017x) and a 25.157125-ms expanded envelope
(5.209324x). The corrected halo traffic therefore does not falsify the 5x
architecture. Expanded's 12.771158-ms build projection does exceed the initial
12.5-ms command-1 diagnostic; it needs at least 26.842 Gproduct/s at the same
80%-roof rule, or a same-run control that revises that rule, before promotion.

The 80%-roof mixed projection leaves 7.338202 ms under the hard member cap for
producer attribution, two command-service intervals, visibility/readback,
host table construction, both CPU ladders, and adapter work. Initial component
gates are:

| component | maximum |
|---|---:|
| incremental shared-plane production | 3.5 ms |
| command 1 GPU-active | 12.5 ms |
| command 2 GPU-active | 6.4 ms |
| two command-service intervals plus all host work | 2.5 ms |
| complete fairly attributed member | 26.210324 ms |

The component gates sum below the hard cap and are diagnostic; only the
complete paired median decides 5x. If the expanded kernel or a better
half-width result makes the 16.381453-ms 8x cap credible, continue toward it
instead of stopping at 5x.

## Observed standalone result

The registered runtime can either upload the three diagnostic host planes or
attach exact-size buffers from a producer on the same Metal device. It
preallocates all command-private buffers, and `execute` performs no
device-buffer allocation. Consuming `submit`/`join` handles keep every source
and output alive, prevent concurrent scratch reuse, and report submit, overlap,
join, and GPU-active time independently.

A target sweep selected mixed `(build_threads = 64, high_tile = 128)` and
native fold `(fold_threads = 32)`. The expanded build and a column-parallel
fold were both measured and rejected. The latter regressed a focused parity
run to seconds rather than milliseconds.

The focused suite runs 16 Spartan-shift tests, including element-for-element
Metal parity for both prefix strategies and the native fold against an
independent host oracle. The runtime also proves that both commands retain the
same source allocation identities.

The original retained-only harness ran one standalone prefix and fold before
measuring the hybrid. Five fresh-process observations at `T = 2^26` then
produced prepared service walls of:

```text
22.636833 ms, 15.750583 ms, 24.399833 ms, 13.224375 ms, 45.228959 ms
```

The `22.636833-ms` median is `5.789309x`, with four of five samples below the
cap. This remains a useful GPU-residency control, but it is not the one-pass
PIOP boundary.

The corrected harness measures the hybrid before any control dispatch. Five
fresh-process first-use service walls were:

```text
42.074375 ms, 39.641667 ms, 72.069666 ms, 53.753750 ms, 39.713084 ms
```

The median is `42.074375 ms`, only `3.114761x`; no sample clears 5x. Median
prefix and fold active times were `16.318208 ms` and `5.280417 ms`. Adding the
current challenge-dependent preparation gives a `58.775876-ms` median resident
member, or `2.229684x`, still excluding the source producer. A bounded `2^14`
dispatch of the retained entry points did not improve the following target
command: the prefix remained `16.697458 ms` active. Pipeline cold start is
therefore rejected as the explanation.

The gap instead appears when the GPU first consumes the 1.099-GB host-written
source. This is consistent with, but does not prove, a first-touch or
host-to-device visibility cost. The production hypothesis is narrower: a
stage-1 GPU producer writes these exact buffers, later stages retain them, and
Spartan shift consumes them without a CPU-sized upload. The warmed control is
evidence that such residency can matter, not permission to omit producer cost.
The next decisive experiment must use the typed producer attachment and charge
its incremental write/visibility wall. If that one-pass resident boundary does
not clear `26.210324 ms`, redesign or keep this member on CPU.

## Occupancy and tuning

The prefix partial grid has 524,288 independent threads and 8192 target
threadgroups, so launch width is not expected to limit saturation. Its
source-level live set is dominated by four field accumulators, current/next
outer temporaries, two weights, gamma powers, and one wide product. This is a
structural estimate, not a compiler register count. The low-fold grid has 8192
threadgroups and five field accumulators; its 80-byte target scratch is less
than 1% of the 32-KiB legal threadgroup-memory limit.

The first target sweep covered both build variants at widths 64, 128, 256, and
512 and high-tile heights 32, 64, 128, 256, and 512. The fold covered widths 32
through 1024. Before promotion, capture compiler register allocation,
spills/local memory, resident SIMD groups, active cores, achieved half/full
products per second, and achieved bytes per second for the retained geometry.
A phase below 80% of its matched roof is unfinished even when an aggregate
prepared-path median happens to clear 5x.

## CPU fallback and crossover

Use the optimized CPU kernel whenever any resident-plane invariant fails,
allocation admission fails, or the trace is below the frozen crossover. Do
not upload native planes inside the Metal member to force the GPU path.

The provisional sweep points are log 18, 20, 22, 24, 25, and 26. Start with
CPU below log 20 and Metal at log 20 or above only when the shared producer is
already active. Freeze the final crossover from alternating complete-member
measurements, then validate it without adaptive winner picking. At every
scale, compare mixed, expanded, and CPU from the same production boundary.

## Correctness and falsification cases

Before shared wiring, the parity harness must cover:

1. odd and even logs, including the one-variable and two-variable domains;
2. the exact `x_hi * P + x_lo` map and reversed low-challenge point;
3. the successor identity against `EqPlusOnePrefixSuffix`, especially the
   final high row and no-wrap zero;
4. sparse unique values at `high_tile - 1`, `high_tile`, and
   `high_tile + 1`, checked element-for-element at tile heights 64, 128, and
   256 against the direct rank-1 oracle;
5. mixed Q, expanded Q, and direct rank-1 Q equality element-for-element;
6. every prefix and suffix round polynomial, previous-claim check, bind, and
   final transcript state;
7. gamma `0`, `1`, `p - 1`, and seeded nontrivial values;
8. PC values around `2^32` and `u64::MAX`;
9. independent current flag patterns, including cycle zero and final padding;
10. flag word order, 32-row chunk ownership, and unused high bits;
11. all five dense tables and final outputs in canonical order;
12. the InstructionInput `UnexpandedPC` alias check without relying on it for
    the self-contained output;
13. clear and ZK generated-driver paths;
14. Rust/Metal sizes, alignments, slots, zero allocations per command, and
    exact readback bytes; and
15. rejection of a foreign device, incorrect buffer length, duplicate or zero
    allocation identity, and uncertified current flags.

Kill or redesign the path under any of these conditions:

- a duplicate full-domain traversal or upload is required inside the member;
- the producer requires atomic flag updates or a second packing pass;
- prefix build exceeds 12.5 ms active or low fold exceeds 6.4 ms active
  without a same-run control that revises the applicable roof;
- the half-width primitive fails exact parity or its 26.272-Gproduct/s floor,
  and the complete mixed fallback cannot clear the hard cap;
- either main pipeline spills or fails to keep the device broadly active;
- producer attribution plus member wall exceeds 26.210324 ms;
- any round polynomial, output value, opening point, alias, proof, or transcript
  differs from optimized CPU; or
- a third host-dependent full-domain command is needed.

## Integration map

Before PIOP promotion:

1. add a typed resident-plane owner to the shared stage-1 producer;
2. change that producer to 32-row chunk ownership and add the current `IsNoop`
   source without changing witness semantics;
3. replace per-invocation scratch allocation with a reusable workspace and
   explicit challenge-weight updates;
4. add a Metal backend slot that borrows the owner, runs the two commands, and
   delegates both ladders to a small serial host state;
5. add generated-driver and proof-level CPU/Metal parity in both clear and ZK
   modes;
6. extend the existing microbenchmark to charge workspace updates, producer
   attribution, command joins, and the complete member; and
7. run the paired PIOP evaluator before changing the backend cutoff.
