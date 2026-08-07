# Spartan outer successor design

Status: the deferred signed-`Bz` arithmetic probe is exact but rejected on its
predeclared log-26 activity bar. Continue with challenge-collapsed `Az`; keep the
current uni-skip and production remainder kernels. The probe is available only
through the `test-utils` runtime-artifact path and changes no backend slot,
transcript, or protocol.

The latest same-binary log-26 diagnostic measured the complete optimized stage 1
at 3,816.081 ms and Metal at 530.542 ms, or 7.193x. The split is decisive:

| member | optimized CPU | Metal | ratio |
|---|---:|---:|---:|
| Spartan outer uni-skip | 2,722.459 ms | 307.664 ms | 8.849x |
| Outer remainder | 1,093.454 ms | 222.638 ms | 4.911x |
| complete stage 1 | 3,816.081 ms | 530.542 ms | 7.193x |

This was one exact diagnostic pair at revision `9de144572`; it is not promotion
evidence. The older five-pair isolated remainder result was 881.996 ms CPU and
219.487 ms Metal, or 4.015x. A fresh alternating evaluator must freeze the final
denominator before implementation promotion.

## Exact boundary

Stage 1 remains in this order:

1. The host draws `tau`.
2. Metal uni-skip reads the resident 48-byte compact row and 112-byte residual row
   for each cycle and returns nine extended-node values.
3. The host builds and absorbs the degree-27 polynomial and draws the stream
   challenge.
4. Metal materializes the remainder state, emits each message, and binds the large
   prefix. The host constructs every Gruen polynomial, performs Fiat-Shamir, and
   finishes the measured CPU tail.
5. Metal evaluates all 35 canonical outer openings. The host validates derived
   tables, checks the final relation, records the output claims, and absorbs them
   in the existing order.

The protocol-visible values are unchanged: nine uni-skip nodes, 27 remainder round
polynomials, 27 challenges, the final claim, 35 openings in
`SPARTAN_OUTER_R1CS_INPUTS` order, all derived values, and the transcript digest.
The device never hashes. No command may cross a Fiat-Shamir dependency.

The resident producer remains backend witness preparation. It writes the compact
and residual allocations once and records row count, allocation identities, and
Metal registry identity. The timed stage uploads no full-domain row plane and
allocates no round buffer. Capacity failure may select optimized CPU before the
first protocol command; failure after a command advances state is terminal.

The CPU denominator is the complete `OptimizedOuterUniskip` or
`OptimizedOuterRemainder` trait member, including preparation, every round, the
terminal bind, output claims, and validation. A local Metal numerator includes all
command waits, mapped-buffer visibility, CPU-tail work, host polynomial assembly,
and resource handoff at the same boundary. Backend witness preparation is reported
separately for both arms.

## Exact optimized algorithm

The optimized uni-skip evaluates 19 constraint rows as signed integers. For each
cycle it extends the two centered row groups to nine out-of-domain nodes, forms 18
wide integer products, and accumulates them under the factored `E_out * E_in`
equality weights. Only the nine totals become fields on the wire.

After the stream challenge, the remainder folds the first ten and second nine rows
into `Az` and `Bz`. It stores both `Bz` stream values and computes the first
`q(0), q(infinity)` endpoints. The stream bind reconstructs both `Az` values from
flags, binds `Az/Bz`, and writes interleaved state. Later rounds fuse bind, state
write, and the next message. A final equality-weighted trace scan computes the 35
openings. This is the right algorithmic boundary; the successor changes arithmetic
and work ownership inside it.

Uni-skip and remainder materialization cannot be fused under the current protocol.
The stream challenge depends on the completed uni-skip polynomial and host
Fiat-Shamir. Retaining a 10 GiB row plane does not retain its contents in cache.
A protocol change that draws the challenge earlier would alter soundness and is
rejected.

## Floors at `T = 2^26`

The retained M4 Max controls are 420.68 GiB/s for a large copy, 24.08 G useful
full-field products/s for fused bind/message work, 16.42 G field products/s for the
uni-skip control, and 958 G independent `u32` multiply-adds/s.

### Uni-skip

The retained two-pass shader reads 24 logical GiB. It performs 1,207,959,552 field
products and about 35.03 billion `u32` multiply-adds. The independent floors are:

| bound | floor |
|---|---:|
| traffic | 57.05 ms |
| field products | 73.57 ms |
| integer multiply-adds | 36.57 ms |

Field and integer work are interleaved and have long wide-integer dependency chains;
adding their floors gives a conservative 110.14 ms serial issue estimate. Neither
estimate prices register-limited occupancy. The retained 307.7-ms implementation
already clears the 340.3-ms 8x bar, and the prior dispatch search found only 2.06%
from a third full-occupancy pass. Do not spend an implementation cycle on another
uni-skip schedule until an ISA/Instruments capture identifies spills, registers per
thread, active SIMD groups, and a lever with its added work priced.

### Outer remainder

| phase | logical traffic | useful products | binding floor |
|---|---:|---:|---:|
| materialize and first message | 12.000 GiB | 23T = 1,543,503,872 | 64.10 ms compute |
| stream bind plus dense prefix | 10.494 GiB | 536,608,768 | 24.95 ms traffic |
| 35 scalar openings | about 10.009 GiB | 17T = 1,141,137,408 | 47.38 ms compute |

The machine balance at the 24.08-Gproduct/s control is 0.0533 product/byte.
Materialization is 0.1198 product/byte and openings are 0.1062 product/byte, so both
are compute-bound in this model. The prefix is about 0.0476 product/byte and is
traffic-bound. The sum is a 136.43-ms bottomed-out floor; 80% efficiency is
170.54 ms.

The standalone remainder's 8x cap is 136.682 ms, only 0.25 ms above the sum of
three optimistic independent floors. Host Fiat-Shamir, command service, and the CPU
tail make that unattainable without reducing work. Standalone 8x is rejected under
the unchanged algorithm. The useful stretch target is different: with the retained
307.664-ms uni-skip, a 169.13-ms remainder makes the complete stage 8x. That is an
honest 6.47x remainder and agrees with the 80%-roof envelope.

## Successor schedule

### 1. Deferred signed `Bz` dots

The current materializer reduces every signed 192-bit row value to a field and then
performs a reduced field multiplication for each of the 19 dot terms. Per cycle it
therefore executes 19 signed-value reductions and 19 product reductions before the
two final `Bz` values exist.

The successor multiplies each canonical four-limb Lagrange weight directly by the
signed row magnitude, accumulates the ten or nine products in one signed wide dot,
and reduces once for each stream. The CPU kernel already uses this algebra through
`SignedProductAccumulator`; the Metal path must match it rather than invent a new
formula.

The maximum row magnitude is below `2^130`. Ten products with a canonical
128-bit weight fit below `2^262`, so a signed ten-limb accumulator has explicit
headroom. Use one two's-complement accumulator, not separate positive and negative
arrays. Inspect the sign, take the magnitude, fold all high limbs with the active
Solinas offset, and apply the sign in the field. The field oracle must exercise
positive and negative maxima and mixed cancellation.

This lever also has a cost. A direct four-by-five-limb term has 20 `u32` products,
versus 16 for the current canonical field product, adding 76 limb products per
cycle. It removes 36 canonical reductions per cycle but raises the structural live
set by a ten-limb accumulator and product carry. The first probe is a resident
materialization microbenchmark with compiler and occupancy capture. Reject it if
generalized reduction, spills, or lower occupancy keep materialize GPU-active above
75 ms.

Root ran the compiled parent and candidate through the same resident buffers,
threadgroup width, reduction dispatch, allocation lifetime, and alternating order.
The candidate matched every `Bz` state cell at log 8 and both message endpoints at
every scale. It did not meet the activity bar:

| log T | parent active | deferred active | active speedup |
|---:|---:|---:|---:|
| 20 | 1.646 ms | 1.526 ms | 1.079x |
| 24 | 25.033 ms | 23.184 ms | 1.080x |
| 26 | 99.386 ms | 92.555 ms | 1.074x |

The log-26 candidate is 17.555 ms above the 75-ms kill threshold. Retain the
shader and evaluator as a rejected-parent artifact; do not compose this wider
accumulator into the next probe unless another lever removes enough work to price
its register cost again.

### 2. Challenge-collapsed `Az` for the stream bind

Round zero still needs both `Az` stream values. After the stream challenge is known,
the next message does not. Expand the two small-scalar folds into their affine flag
coefficients on the host and combine the coefficients with the stream challenge.
Each cycle then computes the already-bound `Az` with conditional field additions;
it does not compute `Az(0)`, compute `Az(1)`, and perform a full-field bind.

`oracle.rs` compares this affine expansion against the 10/9-row definition. An
optional resident `u64[T]` flag plane makes the first bind read 512 MiB of flags
instead of a 48-byte-stride 3 GiB compact span. It is a protocol-neutral producer
sidecar and must be charged in backend witness preparation. Do not add it unless a
cache or counter capture shows the compact stride is charged and the complete
prefix improves. The earlier generic affine-`Az` rewrite was neutral; this candidate
is narrower because it removes the stream bind itself.

Dense ping-pong rounds remain unchanged. Their 15.00-ms GPU-active sum is already
near the 14.25-ms dense-state traffic floor. Two-round fusion is impossible because
the second challenge depends on the first message.

### 3. Low-coordinate opening partials

The current opening kernel gives one `x_out` block to a threadgroup and keeps an
array sized for nine column accumulators in every lane. The successor transposes
ownership:

```text
partial[column][x_in]
    = sum_x_out E_out[x_out] * value[column][x_out, x_in]

opening[column]
    = sum_x_in E_in[x_in] * partial[column][x_in].
```

Use 32 adjacent `x_in` lanes per tile and four high-coordinate shards. At log 26
that is 1,024 threadgroups. Each high step stages 32 packed rows (5,120 bytes) and
32 weights (512 bytes); eight SIMD groups cover the 35 columns. A lane holds at
most five field accumulators rather than the source's fixed nine. Four shards write
17.5 MiB of partials, and a small second kernel reduces shards and `E_in`.

The compulsory row scan remains 10 GiB and the scalar-opening product count remains
17T. The candidate is not credited with savings from lower registers until the
compiler and Instruments establish the live register count, spills, resident
threadgroups, and achieved issue rate. The exact screen is at most 55 ms GPU-active
for scalar openings and at most 59.2 ms for the first conservative checkpoint. The
previous one-thread-per-row and fixed-five-accumulator candidates are rejected
parents; this coordinate-transposed partial schedule is a different work owner.

### 4. Optional Spartan Shift carrier

The partial layout is also the exact high-coordinate carrier requested by
`spartan_shift_successor`. Retain current partials for `UnexpandedPC`, `PC`,
`VirtualInstruction`, and `IsFirstInSequence`. In portfolio mode also accumulate

```text
successor[column][x_in]
    = sum_(x_out=1) E_out[x_out - 1] * value[column][x_out, x_in].
```

There is no wraparound. The sequential high loop reuses each loaded row and the
previous weight, so it adds no row traffic. It does add two full products per cycle
for the two numeric columns: 134,217,728 products, with a 5.57-ms issue floor. The
boolean successors use conditional additions. Price that work in stage 1 even if it
saves a much larger stage-3 scan.

Scalar-only and carrier modes rank separately. Scalar mode must satisfy the local
outer gate. Carrier mode may be up to 7 ms slower only when the same PIOP run shows
the expected downstream scan disappeared and total PIOP wall improved. The carrier
is an internal typed owner; it changes no opening, claim, transcript, or verifier
formula.

## Pre-registered bars

| boundary | 5x cap | 8x cap | current | successor decision |
|---|---:|---:|---:|---|
| uni-skip | 544.492 ms | 340.307 ms | 307.664 ms | retain |
| outer remainder | 218.691 ms | 136.682 ms | 222.638 ms | require <=169.0 ms; reject standalone 8x |
| complete stage 1 | 763.216 ms | 477.010 ms | 530.542 ms | require <=477.0 ms |

The 169-ms remainder envelope is 75 ms materialize, 32 ms stream/prefix, 55 ms
openings, and 7 ms for command service, CPU tail, validation, and ownership. The
stretch envelope is 70 + 29 + 50 + 6 = 155 ms, giving a 7.05x remainder and 8.24x
stage. These are wall-time bars, not sums of selectively reported GPU-active spans.

Immediate kill rules:

- deferred `Bz` is rejected above 75 ms GPU-active or on any spill/occupancy loss
  that makes the 169-ms envelope impossible;
- collapsed `Az` is rejected unless the complete first-bind/prefix wall is at most
  32 ms; a sidecar must recover its charged producer cost in the complete PIOP;
- scalar opening partials are rejected above 59.2 ms at the first checkpoint and
  must reach 55 ms for the stage-8 envelope;
- carrier mode is rejected if its incremental stage-1 wall exceeds 7 ms or the
  downstream owner performs the old native scan anyway;
- any mismatch in a round polynomial, challenge, claim, opening, derived value,
  transcript digest, buffer identity, or release count rejects the candidate.

## Evaluator and cutover

Start with three isolated probes, not a full production build:

1. ~~Compare the current and deferred signed `Bz` dot.~~ Rejected at 92.555 ms
   GPU-active versus the 75-ms bar despite exact parity and a 1.074x matched-parent
   gain.
2. Compare current stream bind with challenge-collapsed `Az`, first without a flag
   sidecar. Add the sidecar only after the stride charge is observed.
3. Compare current opening ownership with scalar low-coordinate partials. Only an
   exact scalar winner advances to carrier mode.

Then run the exact production stage boundary on five alternating log-26 pairs. Both
proofs must verify. Revalidate the winner in a fresh process, use an untouched
five-pair holdout, and transfer to log 27. Report scalar outer, carrier outer,
complete stage 1, PIOP, and PIOP plus backend witness preparation.

Keep the current log-18 trace cutoff and `2^16` CPU-tail cutoff as starting points.
After a target-scale winner, measure trace cutovers at logs 20 through 27 and tail
cutoffs `2^14` through `2^18`. CPU remains selected below the largest measured
crossover. A cutover result may not revise the log-26 evaluator or the 5x/8x bars.

## Implement decision

Implement the deferred signed-dot probe first. It is the only candidate here that
reduces the dominant materialization instruction count rather than rearranging it.
If it clears 75 ms, implement challenge-collapsed `Az` and scalar low-coordinate
opening partials behind isolated entrypoints. Do not replace the retained uni-skip.

The unchanged-protocol standalone remainder cannot honestly target 8x, but the
complete Spartan outer stage can: a 169-ms remainder reaches 8x with the already
retained uni-skip. The optional partial carrier is worth pursuing separately because
it can remove a later full-domain Spartan Shift scan, but it cannot subsidize a
standalone outer miss.

Not verified in this static packet: actual register counts, spills, active SIMD
groups, generalized-reducer code generation, shared-cache traffic, sidecar producer
cost, exact Metal parity, cutoff, log-27 transfer, or any fresh multi-pair ratio.
