# Instruction claim-reduction successor

This is an unregistered design packet. It does not replace the existing
standalone runtime. The recommended production unit is a paired
`ProductRemainder`/`InstructionClaimReduction` service because the relations
have the same 26-round batch window, the same `tau_low`, the same bind
challenges, and three identical output openings.

## Decision

The instruction kernel is arithmetically admissible as a standalone resident
kernel. Its retained alias-aware active gate is 27.418192 ms against the
strict 38.335463 ms 8x cap. The existing private-command execution is not an
admissible production boundary: fresh first-use and sustained wall times were
unstable, and the current PIOP backend still runs this slot on optimized CPU.

The paired service is the preferred successor. It removes one 40-byte product
row read during materialization, submits one command buffer for both members in
each round, reduces four message columns together, and lets the product member
cache the instruction result before the generated driver calls the instruction
member. This changes no transcript bytes or verifier logic.

At log 26 with a `2^16` CPU tail, the 80%-of-roof device-active gate is
66.287 ms. The latest same-binary CPU pair was 853.711417 ms, giving 170.742283
ms and 106.713927 ms wall caps at 5x and 8x. A 75--95 ms complete paired
service is therefore analytically plausible (8.99--11.38x), but is a model,
not an observation.

## Exact protocol boundary

The host retains:

- both members' input claims and batching coefficients;
- `GruenSplitEqPolynomial` scalar updates and round-polynomial construction;
- the batched round check, transcript absorb, and Fiat-Shamir challenge;
- alias validation, derived `EqSpartan` validation, and canonical opening
  absorb order.

The device owns:

- the 40-byte ProductRemainder row and a 24-byte lookup companion for every
  cycle;
- two resident ProductRemainder state planes and one resident instruction
  combined plane, each ping-ponged through the dense rounds;
- common balanced `E_in` and `E_out` buffers;
- one four-column message reduction and the final opening reductions.

For each round, the device returns four fields in this order:

```text
product q(0), product q_infinity, instruction q(0), instruction q(2)
```

The product host reconstructs its cubic with the existing Gruen helper. The
instruction host multiplies its inner endpoints by the current linear Gruen
factor and constructs the degree-two polynomial with the member's previous
claim. The generated batch driver combines member polynomials, absorbs once,
and draws the common challenge. Fiat-Shamir never moves to Metal.

After the last challenge, the service returns ProductRemainder's eight
openings and the instruction member's two unique lookup-operand openings. The
instruction member obtains `lookup_output`, `left_instruction_input`, and
`right_instruction_input` from the product result. Their alias relation and
opening point are already enforced by the generated verifier.

## Algebra

For row `j`, ProductRemainder materializes `L(j)` and `R(j)` from the three
uni-skip Lagrange coefficients. InstructionClaimReduction materializes

```text
C(j) = lookup_output(j)
     + gamma   * left_lookup_operand(j)
     + gamma^2 * right_lookup_operand(j)
     + gamma^3 * left_instruction_input(j)
     + gamma^4 * right_instruction_input(j).
```

Both members bind low-to-high. In round `b`, `h = log_t - b - 1` equality
bits remain after selecting the round variable. The successor uses the same
balanced factorization for both members:

```text
E_in.len  = 2^floor(h / 2)
E_out.len = 2^ceil(h / 2)
row pair  = x_out * E_in.len + x_in.
```

This is algebraically the same equality MLE as the fixed Gruen split. It also
matches the balanced ProductRemainder work now used by the high-level Metal
adapter and avoids thousands of nearly idle threadgroups after the fixed
outer half stops shrinking.

For a low/high row pair, the four unscaled message columns are

```text
P0   += E_in * L_low * R_low
Pinf += E_in * (L_high - L_low) * (R_high - R_low)
I0   += E_in * C_low
I2   += E_in * (2*C_high - C_low).
```

Each column is then multiplied by `E_out[x_out]` and reduced. Later rounds
first bind two adjacent pairs in each of the three state planes, then apply
the same formula to the two bound values.

## Native ABI and ownership

Keep two buffers rather than a padded 64-byte row:

```text
ProductRemainderRow (40 bytes, align 8)
  left input u64
  right-input magnitude low u64
  right-input magnitude high u64
  lookup output u64
  flags u64

InstructionLookupCompanion (24 bytes, align 8)
  left lookup operand u64
  right lookup operand low u64
  right lookup operand high u64
```

A single padded row is acceptable only with a single ten-column opening scan.
Separate opening scans over a padded row would read 128 bytes per cycle
instead of the compulsory 64 bytes.

The joint materializer's buffer indices are fixed by the sketch:

```text
0 ProductRemainderRow[T]       5 E_out
1 InstructionLookupCompanion[T]
2 product Lagrange weights[3]  6 product state destination
3 gamma powers[gamma..gamma^4] 7 instruction state destination
4 E_in                         8 four-column partials
                               9 ProductInstructionPhaseParams
```

`ProductInstructionPhaseParams` is four little-endian `u32` values
(`source_elements`, `e_in_length`, `e_out_length`, reserved), size and
alignment 16/4. Product state stores the left plane followed by the right
plane. The instruction state is one contiguous field plane. Partials are
column-major in the protocol endpoint order listed above. At 128 threads the
dynamic threadgroup allocation is `4 * (128 / 32) * 16 = 256` bytes.

Later transitions retain each registered shader's state ABI. Bind both to the
same `E_in`, `E_out`, and challenge, and bind the instruction partial buffer
at a byte offset of `2 * E_out.len * 16` into the common four-column partial
allocation. The opening scans similarly write columns 0--7 and 8--9 of one
ten-column allocation. Buffer-offset alignment must be checked against the
device requirement before dispatch.

The proof session should hold one shared `Stage2ProductInstructionService`.
The product witness preparation creates the row owner and allocates the joint
workspace. Instruction prepare supplies `gamma`, its host Gruen state, and the
companion lease. Product prepare occurs first in the generated member order,
but execution begins only after all members are prepared, so the first product
round can require that instruction registration is complete.

The authoritative producer must write both native buffers once. A second
witness traversal, host repack, or upload is charged to Metal wall. If the
backend cannot obtain producer-owned GPU-readable buffers, it must report the
extra bytes and remain a diagnostic candidate. A protocol-inert dummy command
may prime pipelines and page residency before stage 2, but it must reset all
logical state and distinct-proof generation tags.

At log 26, native rows occupy 4.0 GiB. Product state ping-pong capacity is
3.0 GiB and instruction state capacity is 1.5 GiB, for about 8.5 GiB before
small equality/reduction buffers. The instruction member's incremental
resident allocation is its 1.5-GiB companion plus 1.5-GiB state capacity.

## Coordinator state machine

The current `ProveRounds` API calls ProductRemainder immediately before
InstructionClaimReduction in every active round. Use that ordering without
changing the generic driver:

1. Product `prove_round(round, bind)` asks the shared service to execute the
   pair. Round zero dispatches the co-materializer. Later rounds encode the
   two transition kernels into one command buffer. Both write one four-column
   partial buffer, followed by one reducer and one completion wait.
2. The service caches the instruction endpoints with `(generation, round,
   bind)` and returns the product endpoints.
3. Instruction `prove_round` requires the exact cached tag, consumes its two
   endpoints, performs its own round check, and advances its host Gruen state.
4. A second product call before instruction consumption, a mismatched bind, or
   a cross-generation cache hit is an invariant error.

The final `finish_rounds` order is also product then instruction. Product binds
the final three small states once and caches the instruction scalar; the
instruction call checks the same challenge and consumes it.

Output extraction again visits product first. Product encodes the eight-column
product scan and two-column companion scan into one command buffer, reduces
ten columns, waits once, and caches the two instruction values plus the three
aliases. Instruction extraction consumes that cache. `gamma = 0` is valid:
the combined-state check then constrains only lookup output, while the two
unique values remain independently PCS-opened and alias validation still
checks the other three fields.

The initial successor should use separate transition and opening shaders in a
shared command buffer. They have no large compulsory-traffic overlap. A fused
transition shader saves a dispatch and repeated equality loads but risks more
register pressure. A ten-column fused opening saves no native row bytes and is
rejected until counters show no occupancy loss and at least a 3% phase win.

## Work and roof at log 26

The checked model is in `model.rs`. It uses the retained M4 Max controls:

```text
bandwidth                 451.701710520 GB/s
dependent product rate     18.10 Gproduct/s
bind/message product rate  24.08 Gproduct/s
multi-accumulator rate     32.69 Gproduct/s
promotion fraction          80%
```

With `T = 67,108,864`, first `E_out = 8192`, and a `2^16` tail:

| Phase | Useful products | Compulsory bytes | Traffic floor | Compute floor | 80% gate |
| --- | ---: | ---: | ---: | ---: | ---: |
| joint materialize + first messages | 671,121,408 | 7,516,192,768 | 16.640 ms | 20.530 ms | 25.662 ms |
| 10 paired transitions | 402,323,456 | 9,654,239,232 | 21.373 ms | 16.708 ms | 26.716 ms |
| 8+2 opening scans | 335,626,240 | 4,294,967,296 | 9.508 ms | 11.127 ms | 13.908 ms |
| total | 1,409,071,104 | 21,465,399,296 | | | 66.287 ms |

The compute floor is the larger of aggregate issue time and the dependent
ProductRemainder subgraph time. This prevents instruction work from being
declared free merely because it can fill product dependency bubbles.

Separate producer-aware materializers read 152 bytes per row. The joint
materializer reads/writes 112 bytes per row and saves exactly `40T`, or
2,684,354,560 bytes. Transitions save synchronization and reducer work but no
large state traffic. Separate product and companion opening scans already read
the compulsory `40T + 24T`, so shader fusion cannot improve their traffic
floor.

## Occupancy screen

The joint materializer has four field accumulators (16 32-bit words) and up to
six transient field values (24 words) before native scalars, weights, and
indices. Its structural lower bound is therefore about 40 32-bit words, with
a practical source-level envelope around 52--80. Four message columns at 128
threads require only 256 bytes of threadgroup reduction storage.

Promotion requires the compiled facts, not the source estimate:

- execution width 32;
- no thread-local spill traffic;
- register count and resident SIMD groups no worse than the slower standalone
  materializer, unless measured active time still clears 25.662 ms;
- achieved bandwidth/product rates reported for each phase;
- partially active lanes and reducer tail instructions included in the issued
  count.

If joint materialization loses occupancy enough to miss its gate, keep the
paired coordinator but run the existing product and instruction materializers
as two dispatches in one command buffer. This retains one wait and correct
producer ownership while giving up only the 40T read saving.

## Current evidence and denominators

The latest integrated log-26 diagnostic at
`/private/tmp/jolt-product-preinit-f43f58cbf/benchmark-runs/metal-piop-eval/20260807-003949-810360/result.json`
reported:

```text
optimized ProductRemainder              496.128750 ms
Metal ProductRemainder                  152.987749 ms   3.243x
optimized InstructionClaimReduction     357.582667 ms
"Metal" InstructionClaimReduction       314.342583 ms
```

The last instruction number is optimized CPU running in the Metal backend's
unreplaced slot. Its 1.138x ratio is CPU jitter, not a Metal result. The pair's
current ratio is 853.711417 / 467.330332 = 1.827x.

The earlier standalone resident instruction experiment observed 36.823 ms
wall and 30.339 ms active in one fresh-process Criterion interval, but later
same-allocation repetitions had a 114.847 ms upper-median wall with 22.732 ms
active. The exact measured source was not preserved and no producer or
same-run CPU arm was present. Those observations establish active headroom
and a wall/scheduling problem only.

For individual-member gating, retain the stricter historical optimized CPU
median 306.683705 ms:

```text
5x complete-member cap  61.336741 ms
8x complete-member cap  38.335463 ms
alias-aware active gate 27.418192 ms
```

For the fused pair, use fresh same-binary CPU measurements. The current
853.711417 ms pair is a screening denominator only.

## Falsification and promotion

The first implementation experiment answers one question: does paired
command ownership eliminate the wall gap while preserving the active roof?

Minimal surface:

- shared owner and generation-tagged endpoint cache;
- common balanced equality buffers;
- encode-only wrappers around the existing transition/opening pipelines;
- one four-column reducer and one wait per round;
- separate producer-aware materializers first, then the joint materializer as
  an attributable candidate.

Reject or redesign on any of these outcomes:

- any round, state, opening, transcript byte, clear proof, ZK proof, or final
  verification differs from optimized CPU;
- a fresh distinct-proof run performs a second row extraction/upload;
- more than one product/instruction command-buffer completion is paid in a
  round;
- joint materialize active time exceeds 25.662 ms, paired transitions exceed
  26.716 ms, or openings exceed 13.908 ms;
- complete pair wall exceeds the fresh CPU pair divided by five;
- strict instruction incremental wall exceeds 61.336741 ms after charging its
  companion production and all non-shared work;
- the joint shader spills, or occupancy drops without a compensating measured
  phase win.

If pair wall is already below the fresh 8x cap, continue toward the phase
roofs. If it lands between 5x and 8x, inspect producer/page residency and wait
gaps before changing arithmetic. If separate-dispatch co-scheduling clears 8x,
joint transition and ten-column opening fusion remain unnecessary.

Promotion needs five alternating log-26 CPU/pair samples on fresh proof data,
a held-out workload, log 27, exact parity, compiler/counter artifacts, and an
append-only record of discarded candidates. Same-input warmed repetitions are
diagnostic only.

## Could not verify in this design pass

- compiled register allocation, spills, and resident SIMD groups for the
  sketched joint materializer;
- whether the witness producer can emit both native buffers without a Metal-
  only copy at the current proof boundary;
- first-use page residency after a protocol-inert primer;
- complete stage-2 working-set overlap with the other three batch members;
- same-binary paired wall time and strict instruction incremental attribution;
- log-27 capacity and transfer.
