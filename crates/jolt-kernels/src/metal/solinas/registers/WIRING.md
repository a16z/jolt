# Register-family Metal implementation packet

The register backend will use one challenge-independent, certified event SoA
produced with the stage-1 outer-row extraction. Register claim reduction,
register read/write checking, and the later value-evaluation member borrow that
owner; none may rescan the trace or upload a private copy on the production
path. Fiat--Shamir remains on the host, and the optimized CPU implementation
remains authoritative until each shadow below passes.

This packet covers register claim reduction and register read/write checking.
It does not include `RegistersValEvaluation` in the speedup denominator, change
the protocol, register a backend, or preserve the older 40-byte-per-cycle row
plus 16-byte-per-cycle canonical increment design.

## Frozen evidence and budgets

**Measured production evidence.** The denominator artifact is
`benchmark-runs/metal-piop-eval/20260806-133709-697013/result.json`, revision
`5f520c21e338632aa0bf5936ceb02be6c22fa40f`, Fibonacci at `log_T = 26`, M4
Max, 16 Rayon threads, five alternating pairs.

| Boundary | Optimized CPU median | 5x cap | 6x cap | 7x cap | 8x cap |
|---|---:|---:|---:|---:|---:|
| Register read/write | 934.665875 ms | 186.933175 ms | 155.777646 ms | 133.523696 ms | 116.833234 ms |
| Register claim reduction | 99.905582 ms | 19.981116 ms | 16.650930 ms | 14.272226 ms | 12.488198 ms |
| Paired family | 1,033.465165 ms | 206.693033 ms | 172.244194 ms | 147.637881 ms | 129.183146 ms |

The paired median is the median of pairwise sums, not the sum of member
medians. Complete numerators include incremental production, allocations,
commands, GPU-active time, waits, host rounds and hashing, readback, CPU tails,
output evaluation, and validation.

**Retained same-machine controls.** Achievable copy bandwidth is
`451,701,710,520 B/s`; matched unsigned half-width throughput is
`33.168 Gterm/s`; full-field product throughput is `18.10 Gproduct/s`. The
isolated claim Q slice measured `6.092375 ms` active and `6.701125 ms` resident
wall for `3T = 201,326,592` half-width terms. These are microbench controls,
not complete-member results. The current Metal prefix's local active medians,
`15.825661 ms` for round 0 and `57.225186 ms` for round 1, are also retained
measurements but are not part of the frozen production artifact.

The minimum promotion bar is 5x for each complete member. A larger modeled
gain is not optional headroom: pursue 7x for the family if a topology-native
round 1 can meet the gate below.

## Owner ABI, certification, and lifecycle

Blocks contain 256 cycles and have 128 register columns. The selected baseline
ABI is CSR-256:

```text
start_values:  u64[block_count * 128]

rs1_offsets:   u32[block_count * 128 + 1]
rs2_offsets:   u32[block_count * 128 + 1]
rd_offsets:    u32[block_count * 128 + 1]

rs1_positions: u8[rs1_events]
rs2_positions: u8[rs2_events]
rd_positions:  u8[rd_events]
rd_post_values:u64[rd_events]
```

At `log_T = 26`, the current analytical fixture has 262,144 blocks,
33,554,432 block/register columns, 59,652,323 rs1 events, 55,924,053 rs2
events, and 50,331,648 rd events. Its CSR owner is 1,239,649,860 bytes. These
counts are exact for that fixture, not yet a production Fibonacci census.

`RegistersOwnerCertificate` records ABI version, `log_T`, cycle range, block
geometry, witness generation, device registry id, source kind and identity,
owner id, every buffer allocation identity, event counts, and a digest of the
ordered offsets and carried-state checks. Production validates indices,
monotone and in-bounds offsets, per-block counts, every read value, every rd
pre-value, and `rd_inc = rd_post - rd_pre` before publication. A source or
capacity failure selects CPU before round 0; failure after a Metal round has
run aborts the proof.

The same producer emits capped `RdIncrementActivity`, sorted by cycle and
storing each nonzero signed `i128` difference. Its certificate names the
owner id, generation, cycle range, and overflow state. If the cap is exceeded,
discard the sparse activity and retain the owner as authoritative. Building
the offsets, state-flow certificate, and activity may postprocess events, but
must not re-extract `T` witness rows.

Lifecycle:

1. Extend the owned/random-access stage-1 fill to extract an outer row and
   `RegisterCycleRow` from the same source window. A retained
   `SpartanOuterRow` lacks register indices; if the original row source cannot
   supply the composite extraction, reject Metal before stage 1 or explicitly
   charge the fallback register collection pass.
2. Park the certified owner and activity in `ProofSession`. Stage 1 also parks
   a point-bound `RegistersClaimComponents` carrier naming the outer producer,
   compact/residual input allocations, remainder-point digest, `P`, `H`, and
   its three component allocations.
3. Stage 3 consumes the component carrier and a one-shot InstructionInput
   alias handoff. Stage 4 borrows the owner. Stage 5 consumes the owner or
   activity instead of recollecting rd indices or increments.
4. Release each allocation after its last stage-5 consumer. Copying the owner
   into a member-private arena is an invariant violation.

## Algebra, topology, and transcript boundary

Register claim reduction uses

```text
C(j) = rd(j) + gamma*rs1(j) + gamma^2*rs2(j)
s(j) = eq(product_tau_low, j) * C(j).
```

It has `log_T` low-to-high cycle rounds and degree two. For challenges
`r_0..r_25`, all outputs open at `reverse(r)`, and the verifier checks

```text
eq(reverse(r), product_tau_low)
    * (rd* + gamma*rs1* + gamma^2*rs2*).
```

At `log_T = 26`, use `P = H = 8192`, `j = x_hi*P + x_lo`, and

```text
Q_v[x_lo] = sum_x_hi eq(tau_hi, x_hi) * v(x_hi || x_lo)
q          = Q_rd + gamma*Q_rs1 + gamma^2*Q_rs2.

v_dense[x_hi] = sum_x_lo eq(reverse(r_prefix), x_lo)
                          * v(x_hi || x_lo).
```

Stage-3 declaration order is Shift, InstructionInput, then
RegistersClaimReduction. On round `k > 0`, InstructionInput applies
`r_(k-1)` before claim reduction sees it. It may therefore publish rs1/rs2
dense tables after the final prefix bind for one same-round consumption. The
handoff must bind producer/table ids, remaining length, round count,
generation, and ordered-prefix digest; it is neither reusable nor allowed to
carry a different challenge.

Register read/write checking uses, for register `k` and cycle `j`,

```text
ra(k,j) = gamma*rs1_ra(k,j) + gamma^2*rs2_ra(k,j)

s(k,j) = eq(r_cycle,j)
       * (ra(k,j)*val(k,j)
          + rd_wa(k,j)*(val(k,j) + rd_inc(j))).
```

It runs 26 low-to-high cycle rounds followed by seven low-to-high address
rounds. Cycle messages have degree three. Sparse replay returns quadratic
inner endpoints `[q(0), q(infinity)]`; the host Gruen helper constructs the
canonical cubic. The opening point is
`reverse(address_challenges) || reverse(cycle_challenges)`, and output order
is `registers_val, rs1_ra, rs2_ra, rd_wa, rd_inc`.

The host draws each gamma before member preparation, validates and absorbs
every combined round polynomial, then draws the bind challenge. Stage 4 draws
the register gamma and RAM gamma on the host before its batch. No command may
hash, draw, or cross a challenge dependency. The CPU address tail starts only
after cycle round 25 is absorbed and `c_25` is known.

## Log-26 roofs and projections

Every roof below uses measured achievable rates. `80%-roof` means
`max(traffic floor, compute floor) / 0.80`; it is a preregistered active-time
cap, not a measurement.

The CSR producer has a 2.744399-ms traffic floor and 3.430499-ms 80%-roof. A
packed alternative with one 64-bit base/count descriptor per block/register
column has an ideal size of 1,105,432,120 bytes, a 2.447261-ms floor, and a
3.059077-ms cap. Its variable blob can make rd-post loads unaligned, so its
ideal byte count is only a candidate model.

For claim reduction, fused stage-1 Q performs `3T` half-width terms. Its
incremental partial roundtrip and component output are 192 MiB plus 384 KiB;
the measured compute floor is 6.069905 ms. The sparse rd midpoint performs
50,331,648 half-width terms and moves 587,464,708 bytes: 1.517476-ms compute
versus 1.300559-ms traffic. The gross GPU floor is therefore 7.587381 ms and
its 80%-roof is 9.484226 ms. Charging the full CSR producer cap and the prior
2.371689-ms fixed-work reserve to this member gives a **projected**, not
measured, 15.286414 ms or 6.54x.

The RW phase counts are exact for the analytical topology fixture:

| Phase | Full products | Half-width products | Compulsory bytes | Binding floor | 80%-roof |
|---|---:|---:|---:|---:|---:|
| Bridge rounds 2--4 | 73,449,472 | 413,961,160 | 9,463,922,688 | 20.951709 ms traffic | 26.189637 ms |
| Raw rounds 5--8 | 10,157,296 | 506,365,721 | 6,769,024,016 | 19.835147 ms compute | 24.793934 ms |
| Dense rounds 9--25 | 135,085,048 | 0 | 4,847,746,576 | 10.732186 ms traffic | 13.415233 ms |
| Output scan | 262,144 | 0 | 395,816,832 | 0.876280 ms traffic | 1.095350 ms |

The model conservatively serializes full and half-width issue. Retaining the
measured 73.050847-ms prefix and an 8.756582-ms host/tail reserve gives:

| Complete RW projection | Latency | Speedup |
|---|---:|---:|
| Phase floors + 2.744399-ms producer + 1.188552-ms bound-inc handoff | 138.135702 ms | 6.77x |
| 80%-roofs + 3.430499-ms producer + 1.485690-ms handoff | 152.217772 ms | 6.14x |

These projections assume a topology-native prefix can avoid a resident dense
canonical increment plane without exceeding the retained prefix times. That
prefix mechanism is unresolved and must be tested early.

Using the optimistic RW projection plus the claim GPU floor gives a projected
family latency of 145.723083 ms, or 7.09x. A conservative planning budget of
152.217772 ms for RW plus the claim's 8x cap gives 164.705970 ms, or 6.28x.
Holding all other terms fixed, 7x requires round 1 at or below 40.157 ms; 8x
requires 21.702 ms. These are design gates, not achieved results.

## First shadows and integration order

1. **Claim-component carrier.** In the existing stage-1 outer opening scan,
   emit a typed diagnostic carrier alongside the authoritative 35 openings.
   Reconstruct the rd/rs1/rs2 scalar openings from its component tables and
   compare exactly. Do not change selector, protocol, or opening authority.
2. **Certified owner and activity.** Produce CSR-256 and
   `RdIncrementActivity` during the composite outer/register extraction while
   CPU register members remain authoritative. Record incremental extraction,
   postprocessing, allocation, validation, and first-touch wall time. Verify
   that no second trace scan occurred.
3. **Round-8 junction.** Against the certified owner and increment state bound
   through `c_7`, replay round 8, emit `[q(0), q(infinity)]`, and materialize
   the width-256 `(val, ra, wa)` state in one timed command. Exact work is
   868,480 full products, 141,656,012 half-width products, and 2,869,819,780
   compulsory bytes; the traffic floor is 6.353352 ms and the active cap is
   7.941690 ms.

Before filling in the remaining RW sequence, run a topology-native round-1
shadow from the same owner. Stop pursuing 7x if its exact roof or median
cannot reach 40.157 ms. If the round-8 junction passes, implement replay
backward through rounds 5--7, then rounds 2--4, dense rounds 9--25, the output
scan, and finally the high-level adapters.

Concrete seams are:

- `optimized/spartan_outer.rs`: `RowsStore`, `RowsAccess`, and
  `prepare_metal_spartan_outer_rows` for the one-pass composite producer.
- `metal/solinas/outer_remainder/{shader.metal,sequence.rs}` and
  `metal/spartan_outer.rs` for component production and carrier parking.
- `metal/instruction_input.rs::MetalInstructionInputKernel::prove_round` and
  `optimized/instruction_input.rs` for the one-shot rs1/rs2 handoff.
- New high-level `metal/registers_claim_reduction.rs` and
  `metal/registers_read_write.rs`; shared low-level code belongs under this
  `solinas/registers/` boundary.
- `optimized/registers_read_write.rs::{collect_register_entries,
  collect_register_entries_par}` is the charged fallback producer seam.
  Replace or bridge `SharedRdIndices` with the certified owner.
- `metal/registers_val_evaluation.rs::prepare` becomes a downstream borrower;
  it must not upload indices or increments again.
- `metal/backend.rs::with_metal_compute`, `metal/mod.rs`, Solinas source, and
  kernel registry wiring wait until the corresponding complete shadow passes.

## Parity, promotion, and kill gates

The claim oracle independently computes dense Q components and midpoint folds;
the RW oracle evaluates the full relation over a small field and recomputes raw
prefix replay from the multilinear definition. Neither may call the optimized
sparse implementation. Cover odd/even logs, empty and overlapping accesses,
all register indices, `gamma = 0, 1, random`, values around `2^32` and
`u64::MAX`, and signed increment boundaries.

Compare every round polynomial, running claim, dense junction cell, output and
order, opening point, terminal equality, alias identity, transcript bytes, and
proof bytes in clear and ZK modes. Then run five alternating matched log-26
pairs from one binary and source tree plus one untuned log-27 transfer; every
proof must verify. Initial routing is CPU below log 25 and the complete
Metal/CPU hybrid at or above log 25, frozen only after an alternating scale
sweep. Metal never falls back after round 0.

Promotion requires each complete member to clear its 5x cap and each phase to
meet its 80%-roof cap with an emitted-code/Instruments artifact showing no
spill/local memory, register allocation, resident SIMD groups, all cores active
outside tails, achieved operation rate, issued traffic, command gaps, and peak
resident bytes. The packed owner promotes over CSR only with exact parity, no
occupancy loss, and at least 8% lower active and wall medians in both run
orders. Otherwise retain CSR.

Kill or redesign if either layout misses the 7.941690-ms round-8 cap, the
complete RW projection cannot fit 186.933175 ms, producer/readback/wait time is
omitted, a component changes the paired outer wall by more than 2%, a handoff
cannot fail closed on identity, or any parity surface differs. Do not rescue an
over-cap kernel by tuning threadgroup width without revising the exact roof.

## Open uncertainties

- Regenerate the event census from the same Fibonacci witness and binary as
  the denominator; current RW counts use a deterministic analytical fixture.
- Determine the topology-native round-0/1 algorithm and whether the bound-inc
  handoff can be produced without a dense canonical increment allocation.
- Measure `RdIncrementActivity` density and choose its cap; sparsity is not an
  assumption.
- Confirm composite extraction is available for every production witness
  source. The retained outer-row representation alone is insufficient.
- Measure packed-blob issued traffic and alignment behavior; ideal bytes do
  not establish a win.
- Re-freeze copy, half-width, full-product, prefix, producer, and host-tail
  controls from the eventual stable binary. The 8.756582-ms host reserve is a
  planning allowance, not a measured decomposition.
- Confirm the stage-1 component schedule does not reduce outer-opening
  occupancy and expose the current InstructionInput midpoint tables with a
  safe one-shot lifetime.
- Measure aggregate residency through stage 5. The rejected 1.98x complete
  value-evaluation diagnostic is a downstream warning, not part of this
  packet's family speedup claim.
