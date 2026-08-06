# Registers value-evaluation Metal backend contract

This directory is an isolated continuation design for the high-level
`RegistersValEvaluation` member. It does not register a backend, source
fragment, benchmark, or test. The existing low-level implementation in
`solinas/registers_val/` is a read-only input to this design and already owns
the four required entry points:

```text
solinas_registers_val_first_message_factorized
solinas_registers_val_native_transition
solinas_registers_val_dense_transition
solinas_registers_val_reduce
```

No duplicate shader is added here. The checked plan and dense scalar oracle
make the integration boundary and the one analytically justified alternative
explicit without consuming GPU test time.

The production relation was traced through:

- `crates/jolt-claims/src/protocols/jolt/relations/registers/val_evaluation.rs`;
- `crates/jolt-verifier/src/stages/stage5/registers_val_evaluation.rs`;
- `crates/jolt-verifier/src/stages/stage5/outputs.rs`;
- `crates/jolt-prover/src/stages/stage5.rs`;
- `crates/jolt-sumcheck/src/prover.rs`;
- `crates/jolt-kernels/src/optimized/registers_val_evaluation.rs`; and
- `crates/jolt-kernels/src/metal/solinas/registers_val/`.

## Frozen denominator and acceptance bar

The fixed development denominator is the complete optimized-CPU member seam
in the accepted Fibonacci log-26 artifact
`benchmark-runs/metal-piop-eval/20260806-133709-697013`, revision
`5f520c21e338632aa0bf5936ceb02be6c22fa40f`, with 16 Rayon threads. The five
samples are:

```text
334.968579 ms
339.681917 ms
336.744752 ms
350.060129 ms
337.038126 ms
```

The authoritative median is `337.038126 ms`, so the exact 5x condition is

```text
5 * metal_member_ns <= 337_038_126
```

and the displayed cap is `67.4076252 ms`. The 8x stretch condition is
`8 * metal_member_ns <= 337_038_126`, displayed as `42.12976575 ms`.

The median CPU component values are `0.163000 ms` prepare,
`336.874626 ms` over 26 active `prove_round` calls, `0.000167 ms` finish, and
`0.000333 ms` output extraction. The complete-member samples, not a sum of
component medians, define acceptance.

The frozen attribution excludes the generic batch driver's host
Fiat--Shamir span. Exactly one such span encloses each of this member's 26
active rounds; its median total is about `0.05 ms`. A production evaluator
must either charge those 26 spans to both arms or exclude them from both. It
must not count all stage-5 Fiat--Shamir spans: this member is active only in a
suffix of the much longer stage batch.

The artifact's current `metal` arm has a `318.612627-ms` median for this seam,
but it still selects the optimized CPU implementation. It is a CPU-control
sample, not Metal evidence.

## Exact relation, point, and consumers

Let `T = 2^log_t`, let `r_address` be the first seven coordinates of the
stage-4 `RegistersVal` opening point, and let `r_cycle` be its remaining
`log_t` coordinates. For cycle `j`,

```text
wa(j) = 0                              if the cycle has no rd write
      = eq(r_address, rd_index(j))     otherwise

S(j) = LT(j, r_cycle) * rd_inc(j) * wa(j).
```

The input claim is the upstream `RegistersVal` value. The relation has degree
three and binds the cycle variables low-to-high. It draws no relation-local
challenge. Stage 5 draws the instruction gamma and then the RAM gamma before
the batch starts.

The member is tail-aligned by the generic `ConcreteSumcheck` default. In the
frozen Akita configuration, `InstructionReadRaf` has 128 address variables
plus 26 cycle variables, so the stage batch has 154 rounds and this member is
active at batch offsets `128..154`. Its member-local point is the final 26
batch challenges. This long inactive prefix is a useful scheduling window,
not part of the member's algebra.

If the member-local challenges in bind order are
`c_0, ..., c_(log_t-1)`, both outputs are opened at

```text
[r_address || reverse([c_0, ..., c_(log_t-1)])].
```

The observable output order is exactly:

```text
rd_inc, rd_wa.
```

The kernel's fully bound LT scalar must equal

```text
LtPolynomial::evaluate(reverse(challenges), r_cycle),
```

and the terminal relation is that scalar times `rd_inc * rd_wa`.
`rd_wa` feeds the stage-6a bytecode read-RAF address phase. In the base
protocol, `rd_inc` feeds stage-6b `IncClaimReduction`; in Akita it is one of
the four fused-inc consumer claims in the lattice bytecode read-RAF address
phase. The point and value therefore both remain live after stage 5.

The exact clear-output path is the generated stage driver: `prove_batch`
delivers the last pending bind through `finish_rounds`,
`RegistersValEvaluation::derive_opening_points` constructs the shared point,
`validate_derived_tables` checks the resident LT scalar, and the adapter's
`output_claims` returns `RegistersValEvaluationOutputClaims { rd_inc, rd_wa }`.
The driver then validates the output shape and expected final claim before the
recorder absorbs stage-5 openings. In `Stage5Sumchecks::opening_values`, these
two values are the last member and follow the instruction read-RAF and RAM-RA
outputs; their within-member order is `rd_inc` then `rd_wa`. Verification
reconstructs the same points, verifies the batch, and calls
`append_output_claims` in that order before building `Stage5ClearOutput`.

## Host transcript contract

The low-level device returns the three evaluations at `t = 0, 2, 3`. The host
must call the same hinted interpolation path as the CPU kernel: `s(1)` is
`previous_claim - s(0)`, and those four values define the cubic. The generic
batch driver alone:

1. multiplies the member polynomial by its batch coefficient;
2. combines it with the other active members;
3. verifies `s(0) + s(1)` against the running batch claim;
4. records or commits the combined polynomial;
5. absorbs it and draws the challenge; and
6. returns that challenge as the next call's pending bind.

The Metal adapter must not hash, draw, absorb, or evaluate an independent
transcript. `finish_rounds` consumes the final pending challenge exactly once.

## Existing low-level coverage

The current Criterion family measures these isolated operations with all
buffers already allocated and populated:

| Operation, log 26 | Diagnostic median |
| --- | ---: |
| CPU first message | 260.986160 ms |
| Metal first message, GPU active, 32 threads | 11.621965 ms |
| Metal first message, resident wall, 32 threads | 16.559892 ms |
| CPU next message only, excluding bind persistence | 74.857302 ms |
| Metal native transition, GPU active, 32 threads | 8.814095 ms |
| Metal native transition, resident wall, 32 threads | 10.009982 ms |
| Nine dense transitions, sum of active medians, 64 threads | 11.700760 ms |
| Nine dense transitions, sum of wall medians, 32 threads | 15.013537 ms |
| Nine dense transitions, sum of wall medians, 64 threads | 19.341287 ms |

These are unversioned local diagnostics recorded on 2026-08-06. Sums of
independently sampled medians are not complete-path measurements, and the
wall discrepancy between widths is evidence that a paired sequence benchmark
is still needed.

The microbench covers the first message, the native first bind, and dense
ping-pong transitions down to a `2^16` state. It does not cover:

- pipeline preparation, input conversion, allocation, or upload;
- the complete 154-round stage scheduling window;
- asynchronous first-message lifecycle and overlap;
- all 26 host Fiat--Shamir interactions;
- production state export and the CPU tail;
- the last split-LT boundary;
- terminal output extraction and derived-LT validation;
- producer ownership or resident-buffer identity; or
- proof and transcript parity through downstream consumers.

Those omissions are why the primitive ratios cannot be reported as the
member speedup.

## Selected resident sequence

The first implementation should reuse the factorized low-level sequence
without changing a message byte:

1. Borrow resident `rd_inc` and `rd_index` planes. Build only the 128-entry
   address equality table and the three split-LT tables on the host. Allocate
   both dense arenas and reduction scratch once.
2. Submit the first-message command during `prepare`. It has no pending
   challenge, so it can run during the member's 128 inactive batch rounds.
   The first active `prove_round` joins it and returns `[s(0), s(2), s(3)]`.
3. After host challenge `c_0`, run the native transition. It binds raw
   `rd_inc` and sparse `rd_index`/address equality into a resident
   32-byte `{inc, wa}` row and computes message one from those register
   values without rereading the new row.
4. For each retained dense round, bind the two resident fields into the
   opposite arena and compute the next message before they are evicted from
   registers. The host binds the small LT-low table and writes only its active
   prefix between commands.
5. Export exactly the active `{inc, wa}` prefix once, construct the optimized
   CPU tail at the same bound state, and continue through the terminal bind.
6. Return `inc[0]` and `wa[0]`; independently validate the final split-LT
   scalar against the verifier's `LtCycle` derivation.

At the provisional `2^16` handoff, device messages are member-local rounds
`0..=10`: one first message, one native transition, and nine dense
transitions. The CPU handles messages `11..=25` and the terminal bind. There
are 11 challenge-serialized device commands and waits. The first can overlap
the inactive batch prefix; the other ten cannot be submitted before their
host challenges exist.

The current split representation may run until the resident state has
`2 * high_blocks` rows. At log 26 this is `2^14`: the LT-low table then has
two entries, and the next challenge crosses into the high split. A CPU
handoff at or before that point is mandatory. Treating `lt_hi` as fixed after
that challenge is incorrect.

## Producer ownership and fair attribution

The optimized CPU lifecycle supplies the two native inputs asymmetrically:

- stage-4 `RegistersReadWriteChecking` parks the `rd` indices in
  `SharedRdIndices`; and
- stage 5 materializes `RdInc` from `OwnedRows` on its first active round.

A Metal producer should mirror that lifetime with a typed session owner over
two cycle-ordered planes:

| Plane | ABI | Log-26 bytes |
| --- | ---: | ---: |
| `rd_inc` | canonical `Fp128`, 16-byte stride | 1,073,741,824 |
| `rd_index` | `u8`, `0xff` means no write | 67,108,864 |

The owner records row count, byte lengths, device-registry ID, allocation
identities, cycle order, field canonicality, sentinel convention, and the
fact that every non-sentinel index is below 128. The member borrows those
planes through the native transition. They may be released or returned to
the session only after that command completes.

Starting the timer with free populated inputs is not a valid complete-member
comparison. Either the Metal service interval includes the incremental
production of both planes, or the evaluator applies one fixed producer
attribution to both implementations and checks whole-PIOP wall as the final
arbiter. Asynchronous production and first-message overlap may improve PIOP
wall, but their full lifecycle remains visible in service accounting.

The existing `prepare_registers_val_first_message` converts an
`AkitaField` slice, builds tables, allocates, and copies into private invocation
ownership before the timed command. Production integration needs a borrowed
buffer constructor. A resident-only timer started after that convenience API
would exclude the largest setup costs; invoking it for every member would
instead charge the conversion and copy while forfeiting upstream residency.

The current sequence-owned storage at log 26 is:

| Storage | Bytes |
| --- | ---: |
| Dense arena A | 1,073,741,824 |
| Dense arena B | 536,870,912 |
| Two three-lane partial arenas | 786,432 |
| LT-low, LT-high, EQ-high | 393,216 |
| Address equality table | 2,048 |
| Total sequence-owned | 1,611,794,432 |
| Borrowed inputs | 1,140,850,688 |
| Peak modeled resident set | 2,752,645,120 |
| `2^16` CPU-tail export | 2,097,152 |

This excludes command objects and tiny parameter buffers. The largest
individual allocation is 1 GiB. At log 27 it is 2 GiB; at log 28 it is 4 GiB
and exceeds the retained machine's 2-GiB maximum-buffer control. Log-28
support therefore requires sharded input and dense arenas with explicit
dispatch offsets. It must fall back rather than relying on aggregate working
set alone.

## Factorized work and roof

Let `N` be the initial cycles, `H` the high-split length, `C` the CPU-tail
state, and `D = log2(N/C) - 1` the number of dense transitions. The existing
factorized implementation has the following cache-optimistic dominant work:

| Phase | Useful full products | Compulsory row bytes |
| --- | ---: | ---: |
| First message | `3N + 6H` | `17N` |
| Native transition | `2.5N + 6H` | `33N` |
| Dense ladder | `2.5(N - 2C) + 6HD` | `48(N - 2C)` |

“Useful” includes relation products and binding products. Bytes count unique
large-row traffic. The 128-entry address table, split factors, partial
reductions, cache-line overfetch, and command traffic are reported separately.
The model assumes the address and split tables remain cached; counters must
verify that assumption.

At `N = 2^26`, `H = 2^13`, `C = 2^16`, and `D = 9`:

| Phase | Products | Bytes | Intensity |
| --- | ---: | ---: | ---: |
| First message | 201,375,744 | 1,140,850,688 | 0.176514 product/B |
| Native transition | 167,821,312 | 2,214,592,512 | 0.075780 product/B |
| Dense ladder | 167,886,848 | 3,214,934,016 | 0.052221 product/B |
| Total | 537,083,904 | 6,570,377,216 | 0.081744 product/B |

The matched retained M4 Max controls are `451,701,710,520 B/s`
(`420.68 GiB/s`) streaming copy and `18.10 Gproduct/s` for the
six-accumulator Solinas shape. Applying the slower of the arithmetic and
traffic floors to each phase gives:

| Phase | Projected roof floor | Projected 80%-roof cap |
| --- | ---: | ---: |
| First message | 11.125732 ms arithmetic | 13.907165 ms |
| Native transition | 9.271896 ms arithmetic | 11.589870 ms |
| Dense ladder | 9.275516 ms arithmetic | 11.594396 ms |
| Device prefix | 29.673144 ms | 37.091430 ms |

This leaves `30.316195 ms` below the hard 5x cap for the charged producer,
host table work, 11 submission/wait boundaries, state export, CPU tail,
output checks, and transcript work. That is enough analytical headroom to
integrate the existing primitives before changing their algebra. It is not a
speedup claim.

## One justified stretch alternative

The factorized shader keeps three `A = sum(inc*wa)` accumulators and three
`B = sum(inc*wa*lt_lo)` accumulators, then applies `lt_hi*A + eq_hi*B` once
per high block. Six live field accumulators are its likely occupancy limiter.

A genuinely different alternative forms

```text
lt = lt_hi[high] + eq_hi[high] * lt_lo[low]
```

for every sample and accumulates the complete triple product directly. This
uses three accumulators and permits a flattened pair grid, but adds one full
product per sample. Its work is:

| Phase | Useful products | Row bytes |
| --- | ---: | ---: |
| First message | `4.5N` | `17N` |
| Native transition | `3.25N` | `33N` |
| Dense ladder | `3.25(N - 2C)` | `48(N - 2C)` |

Using the unmatched `32.33-Gproduct/s` best relevant low-pressure control,
the target's projected 80%-roof caps are `11.676071`, `8.432718`, and
`8.896729 ms`, or `29.005517 ms` total. Direct LT is projected to beat the
factorized form only if the actual low-pressure/high-pressure rate ratio is
greater than `1.5` for the first message and `1.3` for transitions. The
retained controls satisfy those inequalities, but this exact shader has not
been measured.

Therefore the direct-LT primitive is the only shader experiment justified by
this analysis. Implement it after the resident factorized path establishes a
complete baseline, or earlier only if an Instruments capture confirms
register-limited residency or spills. It loses immediately if the measured
rate ratio misses those thresholds, if extra factor loads defeat the cache
model, or if complete wall does not improve. Do not add variants that merely
duplicate the existing factorization.

## Crossover

The provisional CPU-tail cutoff is `2^16`. The frozen optimized traces give
these diagnostic CPU suffix medians when the preceding message is already
available:

| Exported state | First CPU message | CPU suffix median | Export bytes |
| ---: | ---: | ---: | ---: |
| `2^17` | 10 | 4.673835 ms | 4 MiB |
| `2^16` | 11 | 3.808875 ms | 2 MiB |
| `2^15` | 12 | 3.173795 ms | 1 MiB |
| `2^14` | 13 | 2.409292 ms | 512 KiB |

The incremental CPU savings below `2^16` are comparable to one small GPU
command, and `2^14` is the split-LT hard stop. The production choice must come
from alternating complete-sequence measurements over cutoffs
`{2^14, 2^15, 2^16, 2^17, 2^18}`. Freeze the winner before validation; do not
pick per proof or sum independently sampled round medians.

## Occupancy and latency gates

The existing factorized entry points dispatch `H = 8192` high blocks at the
target. That is ample global work for 40 GPU cores in the long rounds. Dynamic
threadgroup storage is only 96 bytes at 32 threads and 192 bytes at 64 threads,
so it is not the expected residency limit.

The source-level live set includes six field accumulators (24 32-bit words),
three arrays of three sample fields inside the product helper, deltas,
intermediate products, and loop state. Compiler lifetime reuse may reduce that
set; source inspection cannot establish the actual register allocation. Late
dense rounds also waste lanes because every high block has fewer low pairs
while the dispatch remains one group per high block.

Promotion requires an Instruments capture for the first message, native
transition, and representative early/late dense transitions reporting:

- execution width and legal threadgroup maximum;
- compiled registers per thread and register-limited resident SIMD groups;
- static and dynamic threadgroup memory;
- zero spills and zero unexpected local-memory traffic;
- active cores/SIMD groups and lane utilization;
- achieved useful products/s and compulsory bytes/s; and
- cache/DRAM evidence for the small-table assumption.

Sweep legal widths from 32 upward, but retain a width only when complete-path
wall improves in both benchmark orders. A phase below 80% of its matched roof
requires an occupancy, dependency, or bandwidth explanation before more
algebra variants are tried.

## Integration steps

1. Add a typed resident input owner and a low-level constructor that borrows
   its buffers. Validate identity, shape, device, ABI, and allocation limits.
2. Add production state export to `RegistersValSequence`; the existing
   `read_current_dense_state` is test-only and allocates a new vector.
   Production export must fill preallocated host storage and report exactly
   `32C` bytes.
3. Implement the high-level `PrepareKernel` adapter with an asynchronous,
   protocol-inert first-message submission. Its state machine must distinguish
   submitted, joined, native, dense, CPU-tail, and finished states.
4. Keep a host split-LT mirror. At handoff, build the optimized dense CPU tail
   from the exact resident state; deliver the last device message's pending
   challenge on the first CPU call.
5. Emit outer spans for `prepare`, every `prove_round`, `finish_rounds`, and
   `output_claims`, plus inner Metal spans for storage, first-message submit
   and join, native transition, dense rounds, readback, and CPU tail.
6. Add a parser that requires 26 outer rounds, selects the 26 enclosing
   Fiat--Shamir spans, checks the configured device/CPU topology, validates
   zero round allocations, input identities, command completion, and exact
   readback, and computes complete service and member walls separately.
7. Run the cutoff sweep once, freeze the winner, then run five alternating
   optimized/Metal log-26 pairs from one stable binary. Whole-PIOP proof bytes,
   verifier acceptance, transcript state, and every member-local message must
   match.
8. If factorized wall clears 5x and the direct-LT capture still projects a
   material stretch gain, run that single controlled variant. Otherwise move
   to the next member.

## Parity and falsification gates

Correctness must cover:

1. odd and even trace logs, including the smallest CPU-only shapes;
2. `rd = 0xff`, every boundary index `0` and `127`, and rejection of `128`;
3. zero, positive, and negative increments, including maximal `u64`
   differences after canonical field conversion;
4. address and cycle challenges `0`, `1`, `p - 1`, and seeded values;
5. all four values `s(0), s(1), s(2), s(3)` for every round, with the device
   returning exactly the `0,2,3` lanes;
6. every native and dense resident row against the independent dense oracle,
   in both ping-pong directions;
7. handoffs at every candidate cutoff, including the `2^14` split boundary;
8. the pending challenge being applied once, and only once, after handoff;
9. final `rd_inc`, `rd_wa`, and bound LT scalars;
10. the exact `[address || reverse(challenges)]` output point;
11. canonical stage-5 output absorption order and downstream Akita consumers;
12. clear proof bytes, verifier result, and final transcript state; and
13. CPU fallback for wrong device, stale shape, failed allocation admission,
    and log-28 unsharded buffers.

The design is falsified for promotion by any parity mismatch, transcript
movement into the device, an uncharged producer/upload, a round allocation,
more or fewer than `32C` export bytes, a split-boundary overrun, buffer identity
drift, spills, or failure of the paired complete member to clear the exact 5x
condition. A primitive active-time win cannot override any of those failures.
