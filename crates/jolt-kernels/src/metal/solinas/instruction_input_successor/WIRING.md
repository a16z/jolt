# InstructionInput Metal successor

Status: executable but rejected experimental first transition. The module and
MSL fragment are registered with the Solinas source assembler. Sequence
configuration retains `Compact` and `Successor` as explicit A/B controls, with
`Compact` selected by default. Compact preparation neither compiles nor primes
the successor pipelines. Explicit successor preparation compiles them and
primes the materializer, dense-message entry point, and reduction on a 64-row
prefix. The fused experiment consumes the challenge after the first message,
records one command buffer and one wait, and advances the actual sequence from
`Native` to `Dense`. It borrows the production
`InstructionInputRows` allocation and existing dense, equality-weight, and
reduction buffers.

The checked GPU test compares all eight tables and all three descriptors with
independent scalar oracles, checks allocation identity, and confirms that the
production first message remains first. The fused timer charges weight writes,
command and encoder creation, both dispatches, recursive reduction, submission,
completion, timestamps, and three-field validation. It excludes the preceding
native message, row production, data-buffer allocation, pipeline setup, and
full-table readback. The isolated diagnostics remain available in
`BeforeMessage` and do not advance sequence state.

The normal backend selects the compact native-transition entry point. Five
alternating target-size service pairs produced identical full member traces
between configurations but consistently favored compact, so the successor is
not a promotion candidate in its current form. Full integrated proof-byte and
verifier parity remain unmeasured. The duplicate row type in this directory
freezes bytes only; runtime dispatch uses the actual production allocation.

## Exact relation and orientation

For cycle `j`, the optimized PIOP member proves

```text
left(j)  = is_rs1(j) * rs1(j) + is_pc(j) * upc(j)
right(j) = is_rs2(j) * rs2(j) + is_imm(j) * imm(j)
q(j)     = right(j) + gamma * left(j)
s(j)     = eq(r_product, j) * q(j).
```

Its input claim is `right_instruction_input + gamma *
left_instruction_input`. It is a degree-three, `log_T`-round sumcheck.
Binding is low-to-high: challenge `r` binds pair `(2y, 2y+1)` as
`low + r * (high - low)`. After all rounds, the opening point is the reverse
of the member challenges.

The eight tables and output claims have this fixed order:

```text
is_rs1, rs1, is_pc, unexpanded_pc, is_rs2, rs2, is_imm, imm
```

After the first bind, the next message pairs adjacent entries in each new
`N/2` table. For factor pairs `(a0,b0)` and `(a1,b1)`, the device returns

```text
[a0*b0, a1*b1, (a1-a0)*(b1-b0)].
```

After adding the four flag/value products, applying `gamma`, `E_in`, and
`E_out`, the three returned fields are `[q(0), q(1), q_quadratic]`. The host
reconstructs

```text
q(2) = 2*q(1) - q(0) + 2*q_quadratic
q(3) = 3*q(1) - 2*q(0) + 6*q_quadratic,
```

then multiplies these four values by the current linear Gruen factor. Fiat-
Shamir remains on the host. The selector does not intentionally change
`finish_rounds`, the eight outputs, derived `EqProduct`, the three stage-3
aliases, curated output absorption, proof bytes, or verification. Full
proof-byte and verifier parity remain promotion requirements.

## Producer/consumer ABI and lifetime

The successor borrows the existing 48-byte, 16-byte-aligned row:

```text
word 0  rs1
word 1  unexpanded_pc
word 2  effective_rs2
word 3  imm magnitude low
word 4  imm magnitude high
word 5  flags
```

Required flag bits are immediate-positive 18, left-is-rs1 20, left-is-pc 21,
right-is-rs2 22, and right-is-imm 23. The producer already supplies every
field needed by the relation. There is no missing producer field. Two details
are load-bearing:

- `effective_rs2` must already be zero on load rows, as in the production
  producer; the shader must not recover the unmasked witness value.
- the signed immediate is canonical magnitude/sign: no negative zero, positive
  magnitude at most `i128::MAX`, and negative magnitude at most `2^127`.

At `N = 2^26`, the row allocation is 3,221,225,472 bytes. It is produced once
during backend witness preparation, consumed in stage 1, retained through
stage 3, and borrowed by InstructionInput. A zero marginal row cost is valid
only if telemetry proves the same buffer identity at production, stage 1, and
stage 3, with exactly `N` writes and no projection, copy, or repack. If the
producer is no longer required by stage 1, charge InstructionInput the measured
incremental producer cost instead.

The successor reuses the six existing proof-scoped sequence buffers
(6,443,433,984 owned bytes at log 26), including dense A/B, equality weights,
and reduction partials. It adds no persistent or per-round data-buffer
allocation; command-buffer and encoder creation remain in the measured wall
time. Its first dispatch overwrites dense A completely before the second
dispatch reads it. The CPU handoff remains eight `2^16` tables, or 8 MiB.

Exact entry-point bindings are:

| Entry point | Buffers |
|---|---|
| materialize | 0 resident rows, 1 dense A, 2 first challenge, 3 16-byte params |
| dense message | 0 dense A, 1 `E_in`, 2 `E_out`, 3 partials, 4 gamma, 5 16-byte params, threadgroup slot 0 scratch |

Materialize params are `(source_elements, bound_elements, 0, 0)`. Message
params are `(table_elements, E_in length, E_out length, 0)`. Host validation
requires `bound_elements = source_elements/2` and
`2 * E_in length * E_out length = table_elements`; all table addresses must
fit the shader's 32-bit index space.

## Architecture

The current 67 ms fused first transition binds eight native columns, writes
dense state, evaluates the relation, and reduces its message while all state is
live. A prior streaming rewrite cut tens of milliseconds with unchanged
arithmetic and traffic, which is direct evidence of compiler liveness/spill
sensitivity.

The primary successor splits that transition inside one command buffer:

1. **Boolean-specialized materializer.** One thread reads rows `2y` and
   `2y+1`, binds four value columns normally, selects each Boolean result from
   `{0, r, 1-r, 1}`, and writes eight table-major entries. It consumes native
   words one field at a time; no `[Fp128; 8]` array is permitted.
2. **Dense message.** One threadgroup owns one `E_out` coordinate and streams
   its `E_in` coordinates. It reads adjacent entries from the eight tables,
   builds four quadratic factor descriptors, and emits three reduced lanes. It
   does not write another dense table.
3. The existing recursive three-lane reduction finishes the command. There is
   one device-local dependency between dispatches, one command completion, and
   one host wait before the round polynomial is absorbed.

The command topology is therefore

```text
host challenge -> materialize -> device dependency -> dense message
               -> recursive reduction -> one wait -> host Fiat-Shamir.
```

An encoder boundary or explicit buffer barrier may implement the dependency;
a command-buffer completion may not. The full sequence retains 11 protocol
command buffers and 11 round waits.

The charged asynchronous primer exercises the materializer on a fixed 64-row
prefix in the same command as the existing zero-weight native message primer.
A second encoder in that command consumes the 32 materialized rows with zero
weights, runs the dense-message entry point and reduction, and checks a zero
descriptor without changing protocol state. It may overwrite that dense
prefix because the real materializer overwrites the complete active range
before the message reads it.

`shader.metal` is appended after the production InstructionInput fragment
because it deliberately reuses `InstructionInputRow`, native conversions,
Solinas bind arithmetic, and the three-lane reduction. The two isolated phase
experiments are implemented: the materializer compares all eight dense tables
with `materialize_first_bind`, and the dense message compares all three
descriptors with a separate direct relation implementation rather than shader
output fed back as its own oracle.

## Launch and occupancy sketch

Initial geometry is frozen from retained controls rather than searched:

| Phase | Grid at `2^26` | Width | Dynamic TG memory |
|---|---:|---:|---:|
| Native message (unchanged) | 8,192 groups | 256 | reduction scratch |
| Materializer | 33,554,432 threads / 131,072 groups | 256 | 0 B |
| First dense message | 8,192 groups, 2,048 `E_in` values/group | 128 | 192 B |
| Dense ladder (unchanged) | round-dependent | 128 | 192 B |

The dense message has a structural live-state floor of three output lanes,
three left descriptors, three right descriptors, and four current factor
inputs: 13 field values, or 52 32-bit words, before helper and address scratch.
The shader uses a named three-field descriptor rather than a dynamically
indexed coefficient array, but that does not prove physical register use.

`maxTotalThreadsPerThreadgroup` is only a legality limit. Promotion requires a
target-device artifact/capture with execution width, static and dynamic
threadgroup memory, physical register allocation if available, active SIMD
groups, private/thread-memory traffic, spills, and achieved bandwidth. The
materializer target is bandwidth-bound with no spills. The message target is
at least the retained 18.10-Gproduct/s register-pressure control; 32.33
Gproduct/s is stretch headroom. Search threadgroup width only after the capture
identifies an occupancy loss.

## Useful products and compulsory traffic

Let `N` be native rows and `C` the per-table CPU handoff length. Dense source
lengths are `N/2, N/4, ..., 2C`, summing to `N - 2C`. Counts below include core
full-field relation products. They exclude lower-order `E_out` products,
weight maintenance, reductions, signed conversion, command overhead, and
cache-line amplification. Bytes are cache-optimistic compulsory native/dense
traffic; equality weights and partial buffers are additional.

| Phase | Useful products | Compulsory bytes | At `N=2^26, C=2^16` |
|---|---:|---:|---:|
| Native message | `3N` | `48N` | 201,326,592 products; 3,221,225,472 B |
| Specialized materialize | `2N` | `112N` | 134,217,728; 7,516,192,768 B |
| First dense message | `9N/2` | `64N` | 301,989,888; 4,294,967,296 B |
| Dense ladder | `17(N-2C)/2` | `192(N-2C)` | 569,311,232; 12,859,736,064 B |

The complete split therefore performs 1,206,845,440 core products and moves
at least 27,892,121,600 large-state bytes. The current compact plan performs
1,341,063,168 products and 23,597,154,304 bytes. The split trades an extra
`64N` bytes for `2N` fewer products and, more importantly, separates the
compiler live ranges.

Retained M4 Max anchors are 451.701710520 GB/s copy, 32.33 Gproduct/s
message-like, 18.10 Gproduct/s register-pressured, and 16.42 Gproduct/s
conservative. Summing `max(traffic floor, compute floor)` phase by phase gives
complete device-work floors of 61.749, 75.901, and 81.964 ms. Their 80%-roof
caps are 77.186, 94.876, and 102.455 ms. These omit command, reduction, and
tail overhead and are not latency predictions.

For the split first transition alone, materialization has a 16.640 ms traffic
floor and first-message floors of 9.508 ms traffic, 16.685 ms at the register
control, or 18.392 ms at the conservative control. Fail the mechanism before
width tuning if materialization exceeds 20.800 ms, dense message exceeds
20.856 ms at the register target or spills, or combined round 1 exceeds 45 ms.

The executable fused transition clears that falsifier on the retained M4 Max.
At `N = 2^26`, it performs `13N/2 + 3E_out = 436,232,192` benchmark-counted
useful products. A fresh run measured 26.659 ms first wall, 26.874 ms warm
wall, and an outer-call Criterion wall interval of 26.950--27.384 ms with a
27.241 ms median (16.014 Gproduct/s). The paired current native-transition
benchmark in the same binary measured a 31.350 ms wall median; the successor
is 1.151x faster for this phase. The paired preallocated scalar-mirror
transition measured 457.48 ms, so this isolated successor phase is 16.79x
faster than that control. These are development measurements, not
complete-service promotion evidence; unrelated uncommitted static design
modules were present in the benchmark build.

Resident-member A/B reverses that isolated result. A five-pair log-26 run used
one prepared and primed resident sequence, identical protocol tape within each
pair, alternating `Successor -> Compact` and `Compact -> Successor`, and exact
trace equality. Every pair favored compact. Successor wall median was 77.621 ms
versus 74.750 ms compact; GPU-active medians were 68.455 and 65.303 ms. The
paired `compact/successor` wall ratios were 0.963557, 0.965193, 0.968997,
0.963988, and 0.951520. These are uncommitted screening observations rather
than an auditable promotion artifact; they suffice to reject the slower split,
not to promote either arm.

A separate CPU-first Criterion run measured 875.44 ms for the scalar control
mirror and 84.330 ms for the compact Metal hybrid, or 10.38x, including all GPU
rounds, the 8 MiB readback, and CPU tail. This does not establish a 10.38x
speedup over `OptimizedInstructionInputKernel`; that denominator remains the
frozen service measurement below until a fresh production-CPU holdout runs.
The Criterion service route now invokes `OptimizedInstructionInputKernel`
directly and clones its consumed row vector before starting each timed sample.

## Frozen CPU denominator and gates

The comparison boundary is `instruction_input_kernel_service` from
`benchmark-runs/metal-piop-eval/20260806-133709-697013/result.json`, revision
`5f520c21e338632aa0bf5936ceb02be6c22fa40f`. Both arms include prepare, all 26
member rounds, terminal bind, and output claims. The Metal arm also charges
prefetch submission, any primer join delay on the critical path,
commits/waits, coefficient readbacks, 8 MiB tail readback, and CPU tail.
Shared batch Fiat-Shamir is outside both member spans and stays on the host;
the whole-PIOP metric includes it.

```text
CPU ns:   718621795, 866175959, 731548962, 727212419, 719473334
Metal ns: 141558454, 142462799, 154909748, 141181207, 155366626
order:    CPU, Metal, CPU, Metal, CPU first
```

CPU median is 727,212,419 ns. The exact median planning caps are 145,442,483
ns for 5x and 90,901,552 ns for 8x. The current pooled median passes 5x, but
the CPU-first stratum is only 4.722x, so it is rejected.

`instruction_input_kernel_service` is a critical-path member boundary. It
charges prefetch submission and the join delay observed by the member, but
not the full asynchronous primer lifecycle or overlapped storage
initialization outside the span. Shared producer preparation is excluded from
both arms. These gates therefore compare PIOP critical-path latency; they are
not standalone total-work or energy claims.

Promotion uses five fresh alternating pairs and requires all of:

- exact every-round polynomial, challenges, terminal values, `EqProduct`,
  aliases, transcript, deterministic proof bytes, and normal verification;
- current source and binary identities;
- exact resident-row identity, no round allocation, resource admission, no
  spills, and the fixed noise bound;
- median paired speedup at or above the gate in the pooled, CPU-first, and
  Metal-first strata.

Five times is the minimum gate. If phase evidence shows more headroom, the
search continues against the identical 8x gate. Replacing frozen round 1 by
the split's modeled 80%-roof cap projects roughly 110.9--122.0 ms complete
service, enough for 5x but not 8x. Eight times leaves 84.080 ms for GPU/device
work plus command, reduction, and preparation overhead after the median 1.300
ms readback and 5.522 ms CPU tail. The 61.749 ms message-anchor floor makes 8x
physically plausible; the 75.901 ms register floor makes it tight. After the
split passes 5x, capture and optimize the native message, dense ladder,
command/reduction overhead, and cutoff rather than declaring the member
finished. An 8x claim is forbidden until the fresh order-stratified holdout
passes.

## Hybrid cutoff

Keep the trace admission cutoff (currently log 25) distinct from the in-member
CPU handoff `C` (currently log 16). Freeze both for the first architecture
result. The current handoff costs a median 1.300 ms readback plus 5.522 ms CPU
tail.

Moving one more round to Metal is profitable exactly when

```text
G(C) < CPU_round(C) + R(C) - R(C/2),
```

where `G` includes dispatch, reduction, wait, and readback synchronization and
`R` is the eight-table readback. After the primary architecture passes, sweep
`C` in `{2^14, 2^15, 2^16, 2^17}` with preallocated host storage and complete
wall time. Then sweep trace admission at logs 24--27. Do not tune a cutoff on
the same samples used for confirmation.

## Claim-to-code map and ambiguity register

| Invariant | Isolated realization | Required promotion evidence |
|---|---|---|
| 48-byte resident ABI and signed immediate | `abi.rs` plus production parity test | full producer-to-stage-3 identity telemetry |
| low-to-high first bind and table order | `oracle.rs::materialize_first_bind` plus checked GPU test | target-scale randomized parity evidence |
| quadratic relation descriptors | `oracle.rs::dense_message`, independent direct walk, and checked GPU test | target-scale `q(0)..q(3)` parity evidence |
| checked work, roofs, 5x/8x gates | `model.rs` | parser-derived fresh paired evidence |
| compiler-visible candidate | compiled `shader.metal` pipelines | ISA and occupancy capture |
| transcript/output seam | this document | optimized-backend differential proof |

Open integration choices are explicit:

- select encoder boundary versus buffer barrier only after compiling both, but
  neither may add a host wait;
- the half-width Solinas primitive is unpromoted and is not assumed here;
- pipeline limits do not reveal occupancy, so width 256/128 remains provisional
  until capture;
- the 8x route beyond the split is selected from measured phase loss, not from
  another simultaneous shader search.

The runtime already derives live `E_in`/`E_out` lengths and validates
`2 * E_in * E_out == table_elements`; it never hardcodes the log-26 split.
The compact path remains selected. Re-engage the successor only if a new
mechanism removes its measured complete-service loss; any such candidate must
repeat full proof parity, occupancy/spill capture, and the order-stratified
holdout rather than relying on the isolated transition microbenchmark.
