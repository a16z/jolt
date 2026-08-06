# Instruction claim-reduction Metal contract

This directory contains a registered standalone runtime whose six Metal entry
points compile and execute on the target device. It is not yet selected by the
generated backend or supplied by the shared stage-2 producer. The
implementation is split by responsibility:

- `abi.rs`: native layouts and the host protocol boundary;
- `oracle.rs`: independent shader-intermediate oracles;
- `model.rs`: geometry, storage, useful/issued work, traffic, and gates;
- `shader.metal`: the standalone fused-message kernels;
- `runtime.rs`: resident buffers, round dispatch, exact reductions, and the
  typed GPU-to-CPU tail handoff;
- this file: integration and promotion contract.

The algebra is traced to
`optimized/instruction_claim_reduction.rs`, the symbolic relation, and the
stage-2 verifier. Projected numbers below remain distinct from the observed
standalone evidence recorded later in this document.

## Frozen comparison

The development denominator is the optimized-CPU log-26 artifact at
`benchmark-runs/metal-piop-eval/20260806-133709-697013/result.json`, revision
`5f520c21e338632aa0bf5936ceb02be6c22fa40f`. Its five attributed
`InstructionClaimReduction` samples are:

```text
303.878085, 317.469165, 300.781461, 311.492709, 306.683705 ms
```

The median is 306.683705 ms. Fail-closed wall limits are therefore:

```text
5x: 61.336741 ms
8x: 38.335463 ms
```

The timed member boundary includes native input production attributable to
this choice, allocation, command encoding and completion, every reduction and
readback, host round-polynomial construction, host Fiat-Shamir, final opening
recovery, and derived-equality validation. It excludes witness generation
shared identically by both arms and PCS work after the PIOP. A fresh integrated
binary must rebaseline this historical denominator with five alternating
optimized/Metal pairs; the frozen number is not acceptance evidence.

## Algebra and orientation

For cycle row `j`,

```text
C(j) = lookup_output(j)
     + gamma   * left_lookup_operand(j)
     + gamma^2 * right_lookup_operand(j)
     + gamma^3 * left_instruction_input(j)
     + gamma^4 * right_instruction_input(j).
```

The relation is the degree-two sum

```text
sum_j eq(tau_low, j) * C(j).
```

It binds the cycle variables low-to-high. If the common batch challenges are
`r = [r0, ..., r25]` in round order, each output is opened at
`reverse(r)`, and the verifier checks

```text
eq(reverse(r), tau_low)
    * (o0 + gamma*o1 + gamma^2*o2 + gamma^3*o3 + gamma^4*o4).
```

The orientation is not interchangeable for odd dimensions. For `n=log2(T)`,
`k=floor(n/2)`, and message round `b`, let `h=n-b-1`. The exact Gruen
shape is

```text
E_out.len = 2^min(h, k)
E_in.len  = 2^(h - min(h, k)).
```

The row index is `x_out * E_in.len + x_in`. At the final opening,
`E_out.len=2^k` and `E_in.len=2^(n-k)`.

The device returns the unscaled inner endpoints `q(0), q(2)`. The host owns
the current linear factor `l`, forms `s(0)=l(0)q(0)` and
`s(2)=l(2)q(2)`, and calls
`UnivariatePoly::from_evals_and_hint(previous_claim, [s(0), s(2)])`.
Only the generated batch driver absorbs the combined polynomial and draws the
next challenge. After the last message, `finish_rounds(r25)` binds the final
two resident `C` values once on the host. The resulting `C*` is not the
same scalar as the final member claim `eq(reverse(r),tau_low) * C*`.

## Optimized CPU work at log 26

The model counts mathematical full-field products separately from the CPU
kernel's native-limb accumulator. It also counts native column payload rather
than making a claim about compiler padding in `InstructionOperandRow`.

| Component | Work or visible payload |
|---|---:|
| gamma combination | 268,435,456 useful products; 469,762,048 scalar-limb FMADDs |
| three-point round messages | 201,670,728 useful products |
| state and Gruen binds | 67,108,915 useful products |
| five-output opening walk | 335,585,280 useful products |
| total useful field products | 872,800,379 |
| native collection payload writes | 3.5 GiB |
| combined-table pass | 4.5 GiB |
| message state reads | about 2.0 GiB |
| bind state traffic | about 3.0 GiB |
| output native-row reads | 3.5 GiB |
| visible payload total | 17,716,740,016 bytes, about 16.5 GiB |

The source rows read during extraction and equality-table traffic are omitted
from that byte total, so it is a lower bound. The CPU denominator is measured
wall time; these counts explain it but do not replace it.

## Standalone kernel architecture

The implemented control uses five structure-of-arrays native planes totaling
56 bytes per row. The first command reads each row once, computes `C`, writes
the `T`-element resident state, and emits the first `q(0),q(2)` partials.
Each subsequent device round reads four old state elements, binds two adjacent
pairs, writes two new elements, and computes the next message from those two
values before they leave registers. States ping-pong; no round allocates.

Each message writes two column-major partial arrays. The recursive reducer
uses `m -> ceil(m/32)` until one field remains per column. The final two
state elements are read once and bound on the host.

Output has three modes:

- `Aliased`: ProductRemainder supplies `lookup_output`,
  `left_instruction_input`, and `right_instruction_input`; the device
  scans only the two lookup operands.
- `CoreAndRecover`: scan four unsigned columns and recover the signed input
  from `C*` using `gamma^-4`.
- `AllColumns`: scan all five columns. This is mandatory for `gamma=0` and
  remains the independent parity control.

The alias path also handles `gamma=0`, but its recombination check then pins
only `lookup_output`. Zero-gamma parity must therefore compare the two unique
openings independently and rely on the ordinary PCS opening checks; a passing
combined-claim check alone is insufficient.

The two-column alias path is algebraically valid because the verifier declares
those three ProductRemainder openings as aliases at the identical reversed
point. Integration still must make their values available to this kernel
without changing canonical absorb order.

## Strongest producer and fusion

A second full witness extraction is not an acceptable production architecture.
ProductUniskip and ProductRemainder already retain a 40-byte native row
containing the three aliased operands. This member adds only
`InstructionClaimLookupOperandRow`, a 24-byte companion containing
`left_lookup_operand` and `right_lookup_operand`.

The strongest stage-2 design is one co-materialization command:

1. read the existing 40-byte ProductRemainder row once;
2. read the 24-byte instruction companion once;
3. produce ProductRemainder's two state fields and this member's one `C`
   field;
4. emit both members' first-message partials while all native values are live.

For this member, the incremental compulsory bytes are the 24-byte companion
plus the 16-byte `C` write, or `40T = 2.5 GiB`. The complete fused command
must be measured and charged at the stage level: its total row reads and all
three state writes cannot be hidden by assigning the shared 40-byte read to
the sibling. A non-fused reuse of the 40-byte shared row costs this member
`80T = 5.0 GiB` for row reads plus `C`; it avoids production copies but is
not the preferred traffic shape.

Lifetime is explicit:

- the upstream row producer writes the shared ProductRemainder row and the
  24-byte companion during one authoritative extraction;
- the co-materializer borrows both;
- ProductRemainder retains its row for its eight openings;
- this member retains only the companion for its two unique openings;
- `C` remains resident through all 26 rounds;
- the companion and `C` are released after output claims;
- the three aliased values come from ProductRemainder, never from a second
  scan owned by this member.

Fusing the two final opening scans is secondary. It saves equality loads and a
dispatch but not native streaming bytes, and ten live accumulators may lower
occupancy. Keep separate opening kernels unless a compiled/countered experiment
wins.

The current five-plane shader is the parity and standalone performance
control. The cross-member co-materializer requires a later shared stage-2 edit
and is not implemented in this isolated directory.

## Exact Metal work at log 26

Here `T=67,108,864`, the first message has
`E_in=4096, E_out=8192`, the later-message sum is
`sum E_out=106,495`, and the opening has
`E_in=E_out=8192`.

“Issued products” below are source-level SIMD lane-equivalents. A partially
active 32-lane SIMD group charges all 32 lanes, and the block epilogue's
2–5 active outer products charges one 32-lane product instruction. This is
still not a hardware instruction count for the inlined limb arithmetic.

| Phase | Useful products | Issued product lanes | Compulsory bytes | Logical equality bytes | Reduction bytes |
|---|---:|---:|---:|---:|---:|
| materialize + first message | 335,560,704 | 335,806,464 | 4,831,838,208 | 537,001,984 | 279,072 |
| all 25 device transitions | 134,430,714 | 142,868,320 | 3,221,225,376 | 538,574,816 | 3,627,968 |
| two-column aliased opening | 134,234,112 | 134,479,872 | 1,610,612,736 | 1,073,872,896 | 279,072 |
| four-column standalone opening | 268,468,224 | 268,697,600 | 2,684,354,560 | 1,073,872,896 | 558,144 |

Totals are:

| Architecture | Useful products | Issued product lanes | Cache-optimistic bytes | Shader-logical bytes |
|---|---:|---:|---:|---:|
| alias-aware | 604,225,530 | 613,154,656 | 9,674,022,992 | 11,821,244,288 |
| four-column standalone | 738,459,642 | 747,372,384 | 10,748,306,032 | 12,895,527,328 |

Compulsory bytes count large native/state streaming. Cache-optimistic bytes add
each equality table once, partial writes, recursive reduction traffic, and the
32-byte final-state read. Shader-logical bytes instead count every equality
load expressed by the shader. Hardware transactions may fall between them and
must be captured. The four gamma powers are now in Metal's `constant`
address space; their unique footprint is 64 bytes, but compiled constant-cache
behavior remains unverified.

The reducer's `solinas_simd_sum_32` issues five add/shuffle steps in all 32
lanes. At `E_out=8192`, a two-column reduction performs 16,382 useful
additions but 84,800 issued add lanes. This tail is small in bytes but must be
visible in instruction/counter evidence.

## Roof and fail-closed gates

The retained controls are 451,701,710,520 B/s (420.68 GiB/s),
32.69 G useful products/s for a multi-accumulator kernel, and
24.08 G useful products/s for fused bind/message. The 80%-of-analogous-roof
caps computed from compulsory traffic and useful products are:

| Phase | Active-time cap |
|---|---:|
| materialize + first message | 13.371209 ms |
| all transitions | 8.914139 ms |
| two-column aliased opening | 5.132844 ms |
| four-column standalone opening | 10.265687 ms |

Thus the alias-aware active budget is 27.418192 ms and the standalone budget
is 32.551035 ms. The complete member must independently clear 61.336741 ms.
The 8x cap is 38.335463 ms, leaving 10.917271 ms beyond the alias active
budget or 5.784428 ms beyond the standalone active budget for host work,
submission, synchronization, producer attribution, and readback.

These are gates, not predictions. A phase miss cannot be hidden by a faster
phase. Recompute a roof only with same-run controls and counters. If the first
integrated run makes 8x credible, continue toward 8x rather than stopping at
5x. If production must pay a row split or full upload, that wall remains in
the numerator.

## ABI, source, and dispatch

The standalone byte ABI is little-endian:

| Value | Size / alignment |
|---|---:|
| `InstructionClaimLookupOperandRow` | 24 / 8 bytes |
| each unsigned scalar plane entry | 8 / 8 bytes |
| unsigned or signed 128-bit plane entry | 16 / 8 bytes |
| `Fp128` | 16 / 16 bytes |
| each parameter struct | 16 / 4 bytes |

At log 26, no standalone buffer exceeds 1 GiB. The checked no-reuse layout is
about 5.0015 GiB; phase reuse lowers the normal-path member peak to about
4.5 GiB. Cross-member integration must report the complete stage-2 working
set because ProductRemainder states coexist.

Concatenate standalone sources in this order:

1. the Akita `SOLINAS_OFFSET` definition;
2. `fp128.metal`;
3. `simd_reduce.metal`;
4. `instruction_claim_reduction/shader.metal`.

Register:

| Entry point | Purpose |
|---|---|
| `solinas_instruction_claim_materialize_message` | standalone `C` build and first endpoints |
| `solinas_instruction_claim_bind_message` | resident bind plus next endpoints |
| `solinas_instruction_claim_open_core` | four-column standalone opening |
| `solinas_instruction_claim_open_lookup_operands` | two unique openings |
| `solinas_instruction_claim_open_all` | five-column gamma-zero/parity control |
| `solinas_instruction_claim_reduce` | recursive 2/4/5-column reduction |

Standalone buffer indices are fixed as follows:

```text
materialize:
  0 lookup_output u64                 6 E_in
  1 left_lookup_operand u64           7 E_out
  2 right_lookup_operand u128         8 state A
  3 left_instruction_input u64        9 partials
  4 right_instruction_input i128     10 phase params
  5 [gamma, gamma^2, gamma^3, gamma^4]

transition:
  0 source state                      4 partials
  1 destination state                 5 challenge
  2 E_in                              6 phase params
  3 E_out

core opening:
  0..3 four native columns            6 partials
  4 E_in                              7 opening params
  5 E_out

aliased opening:
  0 left lookup operand               4 partials
  1 right lookup operand              5 opening params
  2 E_in
  3 E_out

all-column opening:
  0..4 five native columns            7 partials
  5 E_in                              8 opening params
  6 E_out

reduction:
  0 column-major input
  1 column-major output
  2 reduction params
```

The two partial allocations alternate through recursive reductions. Source
and destination state must be distinct for every transition. A later shared
co-materializer gets a separate ABI; it must not silently reinterpret these
standalone indices.

The default widths are 128 threads for materialization/openings and 64 for
transitions. Source-level live-word lower bounds remain roughly 32, 32, 24,
32, and 36 for materialize, transition, two-, four-, and five-column
openings. Every reduction threadgroup must also be a nonzero multiple of 32;
the padded global count alone does not establish that invariant. These source
bounds do not establish occupancy.

## Promotion order

Correctness precedes timing:

1. compile with the Akita offset and validate every Rust/MSL size and buffer
   offset;
2. compare native conversion at `0`, modulus boundaries, `u64::MAX`,
   `u128::MAX`, `i128::MIN`, and negative one;
3. compare materialized `C`, each pre-reduction partial column, every
   recursive reduction pass, every bound state, and every round polynomial
   against the independent oracle and optimized CPU kernel;
4. cover even/odd log sizes and challenges `0`, `1`, `p-1`, and seeded
   nontrivial values;
5. compare all five output claims, reversed points, transcript bytes, clear
   and ZK proofs, and final verification;
6. prove allocation identity and no second extraction/upload for the shared
   producer;
7. capture each pipeline's compiler register count, spills/local memory,
   execution width, resident SIMD groups, occupancy limiter, achieved
   bandwidth/product rate, cache behavior, and command/readback wall;
8. run five alternating complete-member log-26 pairs, then a held-out scale
   and workload.

Fail closed to optimized CPU on any parity, capacity, producer-lifetime,
phase-roof, or 5x-wall failure. Record every run and exclusion. Large search
winners require a clean revalidation binary and held-out transfer; one fast
sample is not promotion evidence.

## Standalone runtime evidence

The runtime owns five native operand planes, one resident combined state, two
split-equality buffers, and two reusable reduction buffers. No Metal buffer is
allocated during a round. `InstructionClaimCpuTail` copies the selected small
resident state once, preallocates one scratch vector, and then alternates those
two host vectors without a per-round allocation. The host remains responsible
for scaling the returned endpoints, transcript absorption, and challenge
generation.

Focused `nextest` parity covers every message and resident state for odd-log
and even-log shapes, nonzero-gamma recovery, the gamma-zero five-column path,
the two-column alias path, signed and unsigned conversion boundaries, final
binding, buffer identity, stale-tail rejection, and GPU-to-CPU handoff. All 26
focused tests pass on the Apple M4 Max.

The retained Criterion family is `instruction-claim-reduction`. Its input
planes and split-equality tables are prepared outside the timed service. Each
service includes equality-buffer writes, command encoding, completion,
readback, final binding, the selected CPU tail, and the two-column opening
scan. Challenges are deterministic inputs; actual transcript hashing is not
included. Setup and the standalone five-plane upload are reported separately.
`reset` reuses the same five operand planes, so repeated samples are
same-input resident diagnostics rather than distinct-proof throughput. A
production runner needs producer-owned buffers or an explicit rebind path;
allocating and copying a fresh five-plane sequence outside the timer is not a
valid substitute.

At `T=2^20`, an all-Metal service paid a completion boundary for every tiny
late round. The bounded CPU-tail sweep produced these Criterion wall medians:

| CPU-tail cutoff | Wall |
|---:|---:|
| `2^12` | 5.6383 ms |
| `2^14` | 4.4638 ms |
| `2^16` | 3.4915 ms |
| `2^17` | 4.4143 ms |
| `2^18` | 4.9666 ms |

The retained cutoff is therefore `2^16`. At `T=2^26`, the first isolated run
with that cutoff reported 36.823 ms median wall and 30.339 ms median GPU-active,
or 8.33x against the historical 306.683705 ms CPU denominator. A warmed
one-shot in the same process was 25.179417 ms wall and 22.036126 ms active.
Those numbers establish adequate kernel headroom but are not promotion
evidence: the first consumption of host-written planes was 112.638958 ms, the
CPU denominator was not remeasured in the same binary, and the intended GPU
producer was absent.

A later sustained alternating experiment exposed a second integration issue.
GPU-active time remained concentrated around 22--25 ms, within the analytical
27.418192 ms alias budget, while standalone member wall ranged from about
36 ms to 300 ms. System memory remained 95% free. The `2^16` candidate and
all-Metal control had active medians of 22.732207 ms and 23.167252 ms in the
same allocation; the cutoff is useful at smaller sizes but does not explain
the target-scale wall variance. The observed phase diagnosis is
`KernelActiveWithinAnalyticalCap / StandaloneWallUnstable /
ProducerUnmeasured`; it is not a kernel promotion state.

The next wall experiment must run at the stage boundary. All active PIOP
members for a round need to encode into one command buffer, wait once, build
the batched polynomial, and perform Fiat--Shamir once on the host. Charging a
separate submission/completion boundary to every member is neither the target
architecture nor stable acceptance evidence. That scheduler experiment must
also consume the ProductRemainder row and 24-byte companion from a GPU-resident
producer; a warmed host-upload control cannot substitute for it.

The current convenience API cannot be placed inside that scheduler: each
method privately allocates a command buffer, commits it, waits, timestamps,
and reads the reduction result. Before backend wiring, split it into an
encode-only phase over an externally owned command buffer and a completion /
read phase after the stage wait. That refactor must also define borrowed
producer-buffer lifetimes and distinct-proof rebinding.

## Unverified integration seams

- the ProductRemainder/InstructionClaimReduction co-materializer;
- allocation ownership and full stage-2 peak memory;
- alias value handoff without transcript-order changes;
- parameter offsets, register allocation, and occupancy beyond entry-point
  compilation;
- transcript parity, generated-backend selection, and proof verification;
- direct companion production versus the split-row fallback;
- stage-level command scheduling and complete alternating 5x/8x wall results;
- encode/read separation and producer-owned operand rebinding.

The registered source has a standalone command runtime, Criterion family, and
26 passing focused tests, including target Metal execution for all six entry
points and both sides of the hybrid handoff. No generated-driver integration,
shared producer, stage-level transcript run, or proof run has occurred.
