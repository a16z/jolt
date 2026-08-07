# Instruction Read-RAF v3 phase contract

Status: frozen before production shader/runtime work. This contract replaces the
older design-only timing assumptions in `instruction_read_raf.v2.md`; it does
not reinterpret any earlier run as v3 evidence.

## 1. Boundary and fixed evaluator

The primary local member is the complete `InstructionReadRaf` wall span:
prepare, 128 address rounds, 26 cycle rounds, output claims, and finish. The
evaluator is `scripts/metal_piop_eval.py` schema 10 with local kernel
`InstructionReadRaf`. It compares the same exact Fibonacci trace in optimized
CPU and Metal-hybrid PIOP runs. Trace construction, generic witness
construction, and PCS are outside the primary PIOP denominator. Any
Instruction Read-RAF-specific topology or repack that is not co-produced by an
already charged backend witness producer remains inside this member. A second
PIOP-plus-backend-prep metric must expose any co-produced preparation cost.

Fiat--Shamir stays on the host. The member returns one polynomial before each
shared batch challenge. The challenge after address round 127 is the barrier
that permits the address-to-cycle handoff. The final cycle challenge arrives
through `finish_rounds`. The kernel must preserve:

- 128 address challenges in most-significant-to-least-significant order;
- 26 cycle challenges in low-to-high binding order and reversed `r_cycle`
  output order;
- the canonical 40 table flags, four virtual-RA values, and RAF flag;
- both clear and ZK proof transcripts.

The independent relation oracle is
`src/metal/solinas/instruction_read_raf_v3/oracle.rs`. The production CPU
implementation is a comparison target, not the oracle.

The upstream input claim is lookup output plus `gamma * left` plus
`gamma^2 * right`. It does not contain the canonical-address `gamma^3` term.
That term is deliberately present in the address relation, so a RAF row whose
raw address has an all-ones upper half makes the first Boolean sum disagree
with the upstream claim. An oracle fixture must test that rejection; it must
not seed its claim from the constrained relation value.

## 2. Current denominator and required savings

The only current exact log-26 diagnostic is
`benchmark-runs/metal-piop-eval/20260807-134222-116440/result.json`, revision
`c2d777891f0fdccb8e3799ed6ff1259bd3683a29`. It is one CPU-first pair, so it
sets a design denominator but is not promotion evidence.

| component | optimized CPU | Metal hybrid |
| --- | ---: | ---: |
| prepare | 218.584750 ms | 493.598208 ms |
| address rounds | 2157.751920 ms | 299.099669 ms |
| cycle rounds | 1452.772622 ms | 235.107250 ms |
| output + finish | 11.306876 ms | 9.065916 ms |
| complete member | 3840.416168 ms | 1036.871043 ms |

The observed local speedup is 3.703851x. The hard 5x wall cap is

```text
3840.416168 / 5 = 768.083234 ms.
```

The implementation therefore must remove at least 268.787809 ms. Holding the
current cycle and output walls fixed, prepare plus address must be at most
523.910068 ms. The v3 checkpoint is stricter: prepare plus address at most
471.519061 ms, leaving 52.391007 ms of contingency for the cycle path. The
checkpoint is a preregistered engineering bar, not a claim about current
variance.

## 3. Address architecture and ceilings

### Primary: exact-key address atoms

The backend witness producer partitions cycles by the exact key

```text
(table_plus_one, RAF flag, raw u128 lookup index).
```

The raw integer is mandatory. Grouping field-reduced addresses could merge
non-canonical representatives before the load-bearing `gamma^3`
upper-half-all-ones term checks them.

Let `T` be cycles, `U` unique atoms, `M` mass jobs, `S` split atoms, and `P`
temporary mass partials. The topology must satisfy `M = U - S + P`. Phase zero
computes each atom mass from split reduction-equality factors. Phases 1--15
multiply each resident mass by the preceding 256-entry equality table. Each
phase emits six sufficient-statistic lanes per segment; `Suffixes::One`
aliases the mass lane and the remaining suffixes occupy lanes 3--5. The host
constructs and binds the eight quadratic address messages from one
`94 * 256`-field readback per phase.

For the target `T = 2^26`, before six-lane partial traffic:

```text
large-state bytes = 4T + 736U + 32M + 32P
useful full products = T + 15U + R + Q
address output bytes = 94 * 256 * 16 * 2 = 12,320,768
```

`R + Q` is the exact RAF-plus-suffix scalar-product census. Five-word atomic
adds are counted separately from full field products. Split equality tables
have only 262,144 cache-unique bytes but issue `32T` logical lookup bytes;
until counters show where those requests land, the model reports them
separately from DRAM traffic.

Retained controls on the same M4 Max are 451.702 GB/s copy, 16.42 G full field
products/s for the general product control, 18.10 Gproduct/s for a
register-constrained control, and 32.69 Gproduct/s for the multi-accumulator
cycle control. For any observed target census, admission uses

```text
roof_wall = 1.25 * max(
    issued_large_bytes / 451.702e9,
    full_products / matched_product_rate,
    five_word_atomic_ops / measured_atomic_rate
) + host_bind_and_readback_wall + topology_wall.
```

No synthetic `U/T` selects the production path. Before the first target
benchmark the implementation must publish `U`, the cycles-per-atom histogram,
`M/S/P`, jobs per phase, `R + Q`, nonzero accumulator contributions, topology
wall, and topology bytes. If the charged roof exceeds the 471.519061 ms
prepare-plus-address checkpoint, the atom path is killed for this target.

### Fallback: direct grouped rows

The fallback has 82 `(table_plus_one, RAF)` segments and scans each grouped row
once per 8-bit address phase. A standalone CPU repack is forbidden. The
producer must emit the grouped layout or the repack remains fully charged.

Its conservative log-26 bounds are 48.138 GiB issued large-buffer traffic and
1.074--6.309 billion useful products. At retained rates these imply a
114.430 ms traffic floor and a 65.413--384.200 ms product floor, before atomic
and host costs. The exact suffix census chooses the point in that range. The
grouped path is retained as both a fallback and an every-phase parity control
for atom compression.

### Launch and occupancy constraints

There are 16 protocol-required address command barriers, not 128 device
launches. The current exact atom shader plan has 49 dispatches: four in phase
zero and three in each later phase. Dispatches sharing one command buffer are
not each charged the full command-wall latency. The retained command control
is 141 us, so 16 barriers contribute about 2.256 ms before host work.

The admission report must include compiled register count, spills, threadgroup
memory, resident SIMDgroups, and a geometry sweep. Static fp128 temporary
counts are not occupancy evidence. Any large dispatch that spills, or exceeds
1.25 times its matched traffic/product/atomic floor without an explained
counter, fails the phase.

## 4. Cycle architecture and ceiling

The first cycle message derives four raw virtual-RA factors from the 16 address
phase tables. Those factors are reused across the first bind rather than
recomputed. The relation factor remains cycle ordered. A fused bind-and-next-
message command writes the five Product5 tables and accumulates the following
message before the bound values are reread. Later Product5 rounds remain
resident until the calibrated CPU tail cutoff.

At log 26, the current path performs 23T products for the first message, 20T
for the handoff, and 536,190,976 products for nine dense transitions. At the
matched controls the active floor is 150.997 ms. Caching the four raw factors
removes exactly `12T = 805,306,368` recomputation products, giving a corrected
active floor of 106.505 ms. The preregistered active gate is 133.131 ms. The
full cycle wall target is 185 ms until command and host attribution justify a
tighter limit.

The cache costs a `4T * 16`-byte write and read. It is admitted only if the
measured end-to-end cycle wall wins; the product saving alone is not evidence
of a net win.

### First dense transition: factor then sample

The first unregistered transition binds the five factor-major tables from
`N = 2^25` to `M = 2^24` and produces the next Product5 message in the same
command. Keep the persistent destination unweighted. Weight only factor zero's
threadgroup copy by `e_in`; after tile reduction, multiply each of the five
sample columns by `e_out` exactly once. Applying `e_out` in the main kernel
would repeat the same product for every tile.

Two launch shapes remain for the first screen:

| shape | pairs per threadgroup | main threadgroups | dynamic memory |
| --- | ---: | ---: | ---: |
| tile-32 | 32 | 262,144 | 5,120 B |
| tile-64 | 64 | 131,072 | 10,240 B |
| tile-128 | 128 | 65,536 | 20,480 B |

Each 160-thread group assigns one SIMDgroup to each factor. It reads four
source fields per pair, writes two bound fields, and retains both endpoints in
factor-major threadgroup scratch. The five SIMDgroups then independently
accumulate `q(1)`, `q(2)`, `q(3)`, `q(4)`, and `q(infinity)`. A second dispatch
collapses tiles into 8,192 outer partials and applies `e_out`; the existing
Product5 reducer performs `8192 -> 256 -> 8 -> 1`. Main, tile collapse, and
the three reductions use one encoder and command buffer, with dead tile
storage reused as reduction scratch.

The ABI is factor-major source and destination, `e_in`, `e_out`, unweighted
tile partials, and one 32-byte parameter block containing `N`, `M`, both
equality lengths, tile width, tiles per outer coordinate, total tiles, and a
zero reserved word. Admission validates `N = 4 * |e_in| * |e_out|`, `M=N/2`,
tile width 32 or 64, exact divisibility, and every derived count fitting `u32`.
The message field order is fixed as `[q(1), q(2), q(3), q(4), q(infinity)]`.

The exact useful work is 268,476,416 full-field products: 83,886,080 for five
binds, 16,777,216 for the two factor-zero endpoint weights, 167,772,160 for
five four-product samples, and 40,960 outer weights. At 18.10 Gproduct/s the
compute floor is 14.832952 ms. Tile-64 moves 4,048,856,400 logical large-state
bytes, a copy-roof floor of 8.963562 ms, so this transition should be
compute-bound. Tile-128 reduces logical traffic to 4,038,370,640 bytes. Its
persistent peak, including resident source and destination, both equality
tables, and the partial buffers, is 4,032,577,536 bytes before allocator
padding; 20,480 bytes are threadgroup-local.

The first implementation is a probe, not a production registration. It must
match the independent dense transition oracle, the Akita/BigUint Product5
oracle, and the current Metal Product5 path for all five bound tables and
message fields. Prepare once, allocate and upload nothing in the timed region,
and measure the complete five-dispatch command. Promote the better fixed tile
only if median GPU-active time is at most 18.6 ms, there are no spills, and a
capture records registers, resident SIMDgroups, external bytes, and achieved
products per second. If all three shapes miss 18.6 ms, kill this factor-then-sample
candidate before implementing cache-writing successors.

The first target-size quick screen on the M4 Max used the exact `2^25 -> 2^24`
shape, `|e_in|=1,024`, `|e_out|=8,192`, one prepared invocation per candidate,
and no allocation or upload in the timed region:

| kernel | wall interval | active interval | useful throughput at median |
| --- | ---: | ---: | ---: |
| retained Product5 | 11.327--11.335 ms | 11.095--11.097 ms | 24.19 Gproduct/s |
| tile-32 | 10.479--10.538 ms | 10.270--10.296 ms | 26.09 Gproduct/s |
| tile-64 | 10.154--10.209 ms | 9.911--10.001 ms | 27.04 Gproduct/s |
| tile-128 | **9.860--9.889 ms** | **9.671--9.672 ms** | **27.21 Gproduct/s** |

Tile-128 reduced median wall by 12.9% versus the retained transition and by
2.9% versus tile-64. Exact BigUint parity passed for all three shapes. This is
a screening result, not production promotion: it still needs a full Criterion
run, compiled register/private-memory evidence, an adjacent-size holdout, and
complete-member attribution.

## 5. Falsification and promotion

Implementation order is deliberately narrow:

1. Exact target topology/census and atom-versus-grouped roof decision.
2. Address shader with dense/grouped/atom parity after every 8-round phase.
3. Cycle factor reuse and resident Product5 handoff.
4. Final 45-output opening and production owner wiring.

A candidate is rejected on the first failed condition:

- every round polynomial, claim transition, final claim, and all 45 outputs
  differ from the independent oracle or optimized CPU reference;
- clear or ZK proof verification differs;
- producer generation, command serial, allocation identity, or last-owner
  receipt is ambiguous;
- prepare plus address exceeds 471.519061 ms after the census-selected path is
  tuned once;
- cycle GPU-active time exceeds 133.131 ms or cycle wall exceeds 185 ms;
- the complete member exceeds 768.083234 ms on the diagnostic pair;
- a large dispatch spills or misses its matched 80%-of-roof gate.

Promotion requires five alternating exact log-26 pairs with at least 5x
complete-member speedup in both order strata, the full hybrid PIOP gate, an
untuned holdout, and five alternating log-27 transfer pairs. Five times is a
floor: if the measured ceilings show clear additional headroom, the loop keeps
optimizing.
