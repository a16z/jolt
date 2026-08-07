# Spartan shift successor design

Status: static successor packet. This directory is deliberately unregistered.
It does not change the kernel registry, source assembly, stage driver,
transcript, or protocol.

The existing `spartan_shift/` implementation established exact GPU parity but
missed the fair boundary. Its first-use resident member measured a
`79.592083 ms` median at `T = 2^26`, and its two kernels used a
`20.319875 ms` median of GPU-active time. The host-written source incurred a
large first-consumption/completion penalty. More importantly, the old design
scans the full trace twice inside stage 3 and cannot fit the `16.381453 ms` 8x
cap even when command service is ideal.

This successor changes the ownership boundary rather than tuning the rejected
launch. Upstream stages retain the low-coordinate partials they already need
to produce Spartan shift's five input openings. Stage 3 starts from those
partials, proves the first half on the host, then performs one residual native
fold. `UnexpandedPC` at the midpoint is borrowed from InstructionInput, whose
member has the same stage-3 challenges and output point. The shift-owned large
work falls from two full-domain scans to one `PC + flags` scan.

## Frozen comparison boundary

The denominator is the complete optimized `SpartanShift` member from
`benchmark-runs/metal-piop-eval/20260806-133709-697013`, revision
`5f520c21e338632aa0bf5936ceb02be6c22fa40f`, Apple M4 Max, 16 Rayon threads,
and `log_T = 26`:

```text
131.051624 ms
131.584500 ms
129.304918 ms
130.343291 ms
134.289502 ms
```

The median is `131,051,624 ns`. The hard 5x cap is `26,210,324 ns`; the 8x
target is `16,381,453 ns`. Both arms include preparation, all 26 member
rounds, terminal bind, and output claims. Host Fiat-Shamir is excluded from
the local member ratio only when excluded from both arms; the full-PIOP ratio
includes it.

The optimized median decomposes as follows:

| component | wall |
|---|---:|
| native row collection and four Q tables | 116.075167 ms |
| prefix rounds 0 through 12 | 0.583040 ms |
| prefix-to-dense transition | 13.739417 ms |
| suffix rounds 13 through 25 | 0.653083 ms |
| finish and outputs | 0.000917 ms |

At log 25, the retained optimized PIOP summary reports `70.920209 ms` for
this member. That is a crossover hint, not a five-sample denominator. The
initial selector therefore keeps CPU below log 25 and requires a fresh
alternating multi-scale campaign before admitting log 25.

The algebra was traced independently through
`src/reference/spartan_shift.rs`, `src/optimized/spartan_shift.rs`, and the
stage-3 verifier relation before inspecting the rejected Metal path. The
physical row constants come from `solinas/instruction_input/mod.rs` and
`solinas/spartan_outer_uniskip/mod.rs`; the MSL accessors are in
`solinas/spartan_outer_common.metal`. These are protocol and layout sources,
not benchmark-derived assumptions.

## Exact relation and orientation

Let `N = 2^n`, with cycle variables bound low-to-high. For row `j`, define

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

`r_outer` is product uni-skip `tau_low`; `r_product` is the stage-2 product
remainder opening point. Both points are big-endian and contain `n`
coordinates. If the shift challenges in binding order are
`c_0, ..., c_(n-1)`, every output opens at

```text
reverse([c_0, ..., c_(n-1)]).
```

The five output values remain, in order:

```text
UnexpandedPC, PC, VirtualInstruction, IsFirstInSequence, IsNoop.
```

The verifier continues to evaluate both `EqPlusOne` publics from their
closed forms. There is no transcript or protocol change.

## Exact partial-carrier identity

Use

```text
prefix_vars = ceil(n / 2)
suffix_vars = floor(n / 2)
P = 2^prefix_vars
H = 2^suffix_vars
j = x_hi * P + x_lo.
```

For a point `r = r_hi || r_lo`, write `e[h] = eq(r_hi, h)`. For any native
column `v`, define two low-coordinate tables:

```text
current_v[l]   = sum_(h=0)^(H-1) e[h] * v(h, l)
successor_v[l] = sum_(h=1)^(H-1) e[h-1] * v(h, l).
```

The second table uses a zero successor beyond `H - 1`; it has no wraparound.
These are the two Q tables in the exact `EqPlusOne` prefix-suffix
decomposition. A source-centric implementation reads each native row once
and contributes it under `e[h]` and, when `h > 0`, under `e[h-1]`. This removes
the old tile halo and makes the boundary row explicit.

The stage-1 outer carrier has eight tables:

```text
current/successor for upc, pc, is_virtual, is_first.
```

The stage-2 product carrier has two tables:

```text
current/successor for 1 - is_noop.
```

After stage 3 draws `gamma`, a small host combine forms the four production
Q tables:

```text
q0 = upc.current
   + gamma * pc.current
   + gamma^2 * virtual.current
   + gamma^3 * first.current

q1 = upc.successor
   + gamma * pc.successor
   + gamma^2 * virtual.successor
   + gamma^3 * first.successor

q2 = gamma^4 * nonnoop.current
q3 = gamma^4 * nonnoop.successor.
```

The four P tables are the canonical `EqPlusOne` prefix tables for
`r_outer` and `r_product`. The host proves rounds 0 through 12 exactly as the
optimized CPU kernel does. Its exact message-and-bind core is
`16P - 24` full-field products.

The independent oracle in this directory does not use this decomposition for
its authority. Since `eq+1(x, y)` means `y = x + 1`, it builds the Boolean
table directly as `eq+1(r, 0) = 0` and `eq+1(r, j) = eq(r, j - 1)` for
`j > 0`, evaluates the dense summand, and binds all seven dense tables.
Carrier equality is a differential test against that path.

## Why upstream partials are a fair seam

The input claim consists of five upstream `Next*` openings. Stage 1 must
already evaluate `NextUnexpandedPC`, `NextPC`, `NextIsVirtual`, and
`NextIsFirstInSequence` at `r_outer`; stage 2 must evaluate `NextIsNoop` at
`r_product`. Their successor tables are precisely the high-coordinate
partials of those opening calculations:

```text
opening(v_next, r)
  = sum_l eq(r_lo, l) * successor_v[l].
```

The current implementation reduces in the transpose orientation and discards
these tables, so they cannot be relabeled as a handoff. The producer must
change. The baseline companion producer emits both orientations and replaces
the five scalar opening paths with a final `P`-element dot. A later fused
producer may reuse source loads from the existing opening pass, but it gets no
performance credit until capture proves the loads were actually removed.

Two accounting views are mandatory:

1. **Gross standalone:** charge all ten partial tables, their native source,
   the compact projection writes, the stage-3 combine, the midpoint fold, and
   the host ladders to shift.
2. **Resident PIOP incremental:** the five successor tables replace upstream
   opening work. Charge the five additional current tables, extra writes and
   dots, the midpoint fold, and all stage-3 work. Report the upstream stage
   wall before and after the replacement.

The complete PIOP wall decides promotion. The incremental view cannot hide a
regression in stages 1 or 2.

## Midpoint alias and residual fold

All three stage-3 members receive the same batch challenges. InstructionInput
contains an `UnexpandedPC` table and ends at the same opening point as shift.
After the thirteenth bind, it can publish its exact `H`-element partially
bound table:

```text
upc_dense[h] = sum_l eq(reverse(c_0..c_12), l) * upc(h, l).
```

The handoff is valid only when its producer id, witness generation, ordered
challenge digest, table length, device id, and allocation identity match. A
scalar final output is not enough; shift needs the entire `H`-element table
for its suffix messages.

One residual Metal command then folds only:

```text
PC, VirtualInstruction, IsFirstInSequence, IsNoop.
```

It reads a resident `u64` PC plane and three packed flag bitplanes. The shader
uses four named field accumulators. The PC multiplication is the promoted
canonical 128-by-64 Solinas operation. Boolean columns select-add the low
weight. `IsNoop` may be computed as `1 - NonNoop` because the low equality
weights sum to one, but the output table remains canonical `IsNoop`.

After the command completes, the host combines the borrowed UPC table and
four residual tables with the two partially bound `EqPlusOne` tables. The
suffix message-and-bind core is exactly `19H - 19` full products. Fiat-Shamir
remains in the generated batch driver.

The self-contained fallback folds all five columns and uses two half-width
products per row. It is a parity/control route, not the primary 8x route.

## Producer layout and lifetime

The preferred stage-1 projection retains:

| plane | representation | bytes at log 26 |
|---|---|---:|
| UPC | `u64[N]` | 536,870,912 |
| PC | `u64[N]` | 536,870,912 |
| virtual, first, noop | three `u32[N/32]` bitplanes | 25,165,824 |

UPC is transient: the preferred route releases it after the outer partial
build, while the self-contained control retains it through the midpoint. PC
and the flags survive through the residual fold.

The stage-1 row already carries PC, virtual, and first. It does not carry the
current `InstructionFlags::IsNoop` bit in a form this consumer can prove.
Witness production must add that bit to a checked producer input and the
stage-1 projection must write it with the other two masks. Inferring noop from
other flags is rejected.

The compact projection must be co-materialized while the existing stage-1 row
producer already owns the source witness values. It may be host- or
GPU-written, but a later projection pass or upload recreates the rejected
first-use boundary. All added stores and any producer-wall regression are
charged even when source loads are shared.

The ten partial tables occupy `10P * 16` bytes (`1,310,720` bytes at log 26).
The four combined Q tables occupy `524,288` bytes. The preferred dense output
occupies another `524,288` bytes. Partial carriers are released after Q
combination; Q is released at the midpoint; the PC/flag planes are released
after the residual fold.

## ABI and command ownership

The proposed metadata header binds:

```text
producer stage
witness generation
row count
prefix and suffix lengths
big-endian point digest
ordered challenge digest (midpoint only)
device registry id
allocation identities and byte lengths
```

Every field cell is canonical little-endian `SolinasFp128`. Point digests are
host metadata; shaders never hash.

The standalone shader sketches define these entries:

| entry | purpose | output columns |
|---|---|---:|
| `solinas_spartan_shift_successor_outer_numeric` | current/successor UPC and PC partials | 4 |
| `solinas_spartan_shift_successor_outer_flags` | current/successor virtual and first partials | 4 |
| `solinas_spartan_shift_successor_product_flags` | current/successor nonnoop partials | 2 |
| `solinas_spartan_shift_successor_reduce_partials` | reduce optional high tiles | 2 or 4 |
| `solinas_spartan_shift_successor_fold_residual` | preferred midpoint PC/three-flag fold | 4 |
| `solinas_spartan_shift_successor_fold_full` | self-contained five-column control | 5 |

The primary integrated route does not require three new waits. Outer partials
finish under stage 1's existing opening completion, product partials finish
under stage 2's existing opening completion, and stage 3 performs one
midpoint command/wait. If integration adds a new upstream completion solely
for a carrier, that wait belongs to the shift numerator.

The stage-3 batch currently calls members sequentially. The midpoint alias
requires a stage hook that lets InstructionInput apply the shared challenge
and publish UPC before shift consumes it, while preserving the fixed member
aggregation order. Reordering transcript absorption is forbidden.

The checked host ABI accepts only 2- or 4-column partial dispatches, SIMD-width
fold threadgroups from 32 through 1024 threads, and exactly 4 or 5 fold output
columns. Dynamic threadgroup allocation is
`output_columns * (threads / 32) * 16` bytes. Any other shape fails before
encoding; the MSL sketches do not carry recovery branches for malformed
dispatches.

## Exact target work

Let a half-width term mean one canonical field coefficient multiplied by one
`u64`. Conditional flag additions are reported separately because no retained
matched add-rate control exists.

At `N = 2^26`, `P = H = 8192`:

| phase | half-width terms | full products | selected field adds (max) |
|---|---:|---:|---:|
| outer eight-table carrier | `4N - 2P = 268,419,072` | 0 | `4N - 2P` |
| product two-table carrier | 0 | 0 | `2N - P` |
| stage-3 Q combine | 0 | `8P = 65,536` | lower order |
| prefix host ladder | 0 | `16P - 24 = 131,048` | lower order |
| preferred midpoint fold | `N = 67,108,864` | 0 | `3N` |
| suffix host ladder | 0 | `19H - 19 = 155,629` | lower order |
| gross primary total | `5N - 2P = 335,527,936` | `352,213` | at most `9N - 3P` |
| incremental shift attribution | `3N = 201,326,592` | `352,213` | measured delta |
| self-contained fold total | `6N - 2P = 402,636,800` | `352,213` | at most `9N - 3P` |

The retained-input kernel traffic model is:

| phase | bytes |
|---|---:|
| outer numeric/flags, high weights, eight outputs | 1,091,698,688 |
| product flag, high weights, two outputs | 8,781,824 |
| ten-to-four Q combine | 1,835,008 |
| preferred midpoint PC/flags, low weights, four outputs | 562,692,096 |
| kernel total | 1,665,007,616 |

The compact UPC/PC/flag projection writes `1,098,907,648` bytes. The resulting
co-materialized logical lower bound is `2,763,915,264` bytes. Its source loads
may be shared with row materialization, but its stores may not be discarded
from either gross or incremental accounting.

The self-contained midpoint adds `537,001,984` kernel bytes and retains the
same UPC projection longer. Its retained-input kernel total is
`2,202,009,600` bytes and its co-materialized logical lower bound is
`3,300,917,248` bytes. These are cache-optimistic logical bytes. A producer
that does not share row materialization must also charge its full source scan.
Captures must report issued bytes for repeated high/low weight reads and
packed-flag transactions.

Those logical totals are not the compulsory traffic of a fresh outer scan.
The actual stage-1 inputs are a 48-byte `InstructionInputRow` and a 112-byte
residual row. At log 26 they occupy `3,221,225,472` and `7,516,192,768` bytes.
A fresh fused outer scan reads `10,737,418,240` bytes; the two standalone
entries read the InstructionInput allocation twice and total
`13,958,643,712` source bytes. Replacing the logical selected source with
those physical layouts gives:

| outer source plan | total bytes | copy-roof floor |
|---|---:|---:|
| compact buffers co-materialized with stage 1 | 2,763,915,264 logical | 6.118895 ms plus measured producer delta |
| fresh fused outer scan | 11,873,943,552 | 26.287135 ms |
| fresh split outer scans | 15,095,169,024 | 33.418446 ms |

The fresh fused traffic floor already exceeds the complete 5x cap before host
work. Therefore source reuse is not an optimization to pursue later; it is an
admission condition. Production co-materializes the compact source while the
upstream producer already owns the witness values, then runs the compact
carrier entries below. The producer's before/after wall is charged.

The host-visible handoff is `4P + 5H = 73,728` fields, or `1,179,648` bytes,
including the borrowed UPC table. Unified memory does not make cache
visibility or host reads free.

## Roofs and falsification bars

Retained M4 Max controls are:

| control | rate |
|---|---:|
| large streaming copy | 451,701,710,520 B/s |
| isolated unsigned half-width chain | 86.592 Gterm/s |
| canonical three-accumulator Q kernel | 33.168 Gterm/s |
| half-width promotion floor | 26.272 Gterm/s |
| register-pressured full product | 18.10 Gproduct/s |

The 86.592-Gterm/s chain is not used as the kernel roof: it has one dependent
accumulator and does not price four live sums. The 33.168-Gterm/s Q result is
the closest retained control. The 26.272-Gterm/s floor remains the
fail-closed rate until a matched four-accumulator control is captured.

| model | compute floor | traffic floor | binding floor |
|---|---:|---:|---:|
| co-materialized compact at 33.168 Gterm/s | 10.116014 ms | 6.118895 ms | 10.116014 ms |
| co-materialized compact at 26.272 Gterm/s | 12.771314 ms | 6.118895 ms | 12.771314 ms |
| fresh fused outer at 33.168 Gterm/s | 10.116014 ms | 26.287135 ms | 26.287135 ms |
| fresh split outer at 33.168 Gterm/s | 10.116014 ms | 33.418446 ms | 33.418446 ms |
| incremental attribution at 33.168 Gterm/s | 6.069905 ms | must be measured in upstream stages | compute |
| self-contained logical lower bound at 33.168 Gterm/s | 12.139316 ms | 7.307737 ms | 12.139316 ms |
| self-contained logical lower bound at 26.272 Gterm/s | 15.325701 ms | 7.307737 ms | 15.325701 ms |

The compact projection's store-only copy-roof floor is `2.432818 ms`. If none
of it is hidden by required row materialization, the matched-rate projection
plus arithmetic floor is `12.548832 ms` before host work; at 26.272 Gterm/s it
is `15.204132 ms`. Thus 8x is credible only near the matched arithmetic rate
with a small measured producer delta. The aggregate roof row above is valid
for the resident PIOP boundary only to the extent that producer work overlaps
or replaces already-required work.

The lower-order `352,213` full products have a `0.019460 ms` GPU-rate floor,
but they execute on the host and must be measured there. Existing host ladders
suggest about 1.2 to 2.5 ms, not zero.

Pre-registered phase bars are:

| phase | target |
|---|---:|
| compact projection upstream wall delta | at most 1.0 ms for 8x admission |
| compact outer-carrier GPU-active | at most 10.2 ms |
| product carrier incremental wall | at most 0.5 ms when fused, 1.0 ms standalone |
| preferred midpoint GPU-active | at most 2.6 ms |
| all host Q/round work | at most 2.5 ms |
| complete fairly attributed member | at most 16.381453 ms for 8x |
| hard fallback gate | at most 26.210324 ms for 5x |

The phase bars do not add because upstream work may replace opening work and
may overlap unrelated stage work. Both gross active time and critical-path
wall must be reported. If 8x remains physically credible after parity, the
search continues past 5x.

Kill or redesign the primary route if any of these occurs:

- the upstream carriers require an additional full-row upload or fresh scan;
- either upstream carrier needs a new host wait that makes the complete wall
  exceed the 8x cap;
- the integrated outer numeric arithmetic sustains less than 26.272 Gterm/s
  after shared source loads are assigned upstream, or spills;
- the midpoint sustains less than 26.272 Gterm/s, spills, or exceeds 2.6 ms;
- the UPC alias cannot publish the exact `H`-element table before shift's
  suffix message without changing transcript order;
- carrier production regresses its upstream stage by more than the shift work
  it removes;
- any point digest, round polynomial, output, transcript, proof byte, or
  verifier result differs from optimized CPU.

## Occupancy and launch sketch

Start the compact outer numeric entry with one thread per `x_lo`, width 128.
At log 26 this is 64 threadgroups and 256 SIMD groups. Each long-lived thread
has four named field accumulators, current/previous weights, two native
scalars, and one half-width helper. The structural live floor is about 40
32-bit words before address and compiler scratch. Outer flags also has four
named field accumulators but no multiplication helper. Product flags has two.

The 64-group launch is acceptable only if capture shows all 40 cores active,
at least two resident SIMD groups per core, and no local-memory traffic. If it
does not, tile the high coordinate by two and then four. Tiling adds partial
writes, reads, and a reduction but no source halo. Separately, the compact
producer must preserve the upstream row materializer's wall closely enough to
fit the complete target.

The midpoint launches one threadgroup per `x_hi`, giving 8192 groups. Start at
32 threads because the old fold sweep favored 32 and because named
accumulators remove the dynamic-array pressure that invalidated the earlier
result. Dynamic threadgroup memory is four fields per SIMD group. Search 32,
64, and 128 only after the compiler artifact is captured.

The recent registers-claim control found canonical per-term reduction 2.17x
faster than a 224-bit deferred array accumulator. These sketches therefore use
canonical half-width products and explicit accumulators. A deferred variant
is retained only as an A/B control if a new compiler artifact shows lower
liveness.

Promotion requires execution width, physical register allocation when
available, resident SIMD groups, spills/local memory, active cores, achieved
bytes/s, achieved half terms/s, and command-buffer gaps for every retained
entry point.

## Hybrid cutoff and experiment order

Use optimized CPU below log 25. At logs 25 and 26, Metal is admissible only
when both upstream carriers and the midpoint UPC alias are present before the
member starts. Missing metadata selects CPU; it must not trigger a late upload.

Run experiments in this order:

1. Small host differential tests: direct dense oracle, carrier identity,
   every round, final five values, and no-wrap boundaries.
2. Small GPU parity for each isolated entry point and both midpoint paths.
3. Compact outer compiler/occupancy/throughput capture, then an upstream
   co-materialization before/after delta screen; never tune a fresh raw-row
   scan. Screen the midpoint separately with its matched accumulator control.
4. One log-26 complete member with full attribution. Stop spending target
   runs if it cannot beat 26.210324 ms.
5. If it clears 5x and the measured components still fit 16.381453 ms, optimize
   to 8x before confirmation.
6. Five fresh alternating log-26 pairs, stratified by first arm, then a frozen
   log-24/25/26 crossover validation.
7. Generated-driver, clear/ZK, deterministic proof-byte, and verifier parity.

## Correctness matrix

The parity suite must cover:

1. odd and even logs, including one- and two-variable domains;
2. `x_hi * P + x_lo` orientation and reversed low challenges;
3. direct `eq+1(r, 0) = 0`, `eq+1(r, j) = eq(r, j - 1)` against both
   current/successor carriers;
4. rows at high tile boundaries and the final no-wrap row;
5. gamma `0`, `1`, `p - 1`, and seeded nontrivial values;
6. PC/UPC around `2^32`, `u64::MAX`, and Solinas correction boundaries;
7. independent virtual, first, and noop masks, including cycle zero and final
   padding;
8. all four Q tables, every prefix and suffix round polynomial, and all five
   terminal values;
9. UPC alias identity and a negative test with one changed prefix challenge;
10. producer generation, point digest, device, allocation, length, and flag
    certification rejection;
11. clear and ZK generated-driver transcripts, proof bytes, and verification;
12. zero execute-time allocation and exact handoff bytes.

## Integration blockers

1. Stage 1 does not publish the eight outer component tables in the required
   orientation.
2. Stage 2 does not publish the two nonnoop component tables.
3. Stage 1 does not co-materialize the compact UPC/PC/flag planes, including a
   certified current noop bit.
4. InstructionInput does not publish an `H`-element midpoint UPC table with a
   shared challenge identity.
5. The generated stage-3 driver has no prepare/apply/publish midpoint hook.
6. The half-width probe still lacks final emitted-code and occupancy evidence,
   although its parity and throughput screen pass.
7. No fair complete-member or full-PIOP alternating evidence exists for this
   successor.

Until these blockers are resolved, `spartan_shift/` remains a rejected
experiment and production stays on `OptimizedSpartanShift`.
