# Register value/claim family v2

## Decision and fixed target

Treat `RegistersClaimReduction` and `RegistersValEvaluation` as one
shared-owner family. Build and certify one challenge-independent register CSR,
retain its Metal allocations through stage 5, and charge its incremental
producer once in the family numerator. Claim reduction consumes a stage-1
partial-Q carrier plus the owner's rd event plane. Value evaluation consumes
the owner's rd state-flow view directly; it must not construct or upload dense
`rd_inc[T]` or `rd_index[T]` inputs.

The planning artifact is
`benchmark-runs/metal-piop-eval/20260807-103715-208977/result.json`: revision
`2ed9ce265f00ca06120a7d4a46fb979ee07919b8`, binary
`0f110f55e59f6b2e89b2087e00566e24c677905f51782d0d412b4bbd3e2fd03d`,
Fibonacci at `log_T = 26`, 16 Rayon threads, on the retained M4 machine. It is
a dirty-worktree, one-order diagnostic, not acceptance evidence. The following
member times are fixed inputs from that trace analysis:

| Boundary | Optimized CPU | Current Metal arm | Current ratio | 5x cap |
|---|---:|---:|---:|---:|
| Registers value evaluation | 348.740124 ms | 338.587751 ms | 1.02998x | 69.748025 ms |
| Registers claim reduction | 102.402293 ms | 99.064708 ms | 1.03369x | 20.480459 ms |
| Planning sum | 451.142417 ms | 437.652459 ms | 1.03082x | 90.228483 ms |

The current Metal-arm numbers do not establish execution of either successor.
The acceptance harness must measure pairwise family sums; it must not replace
that distribution with a sum of independently sampled medians. The hard bar is
5x for each member and the family. The selected roofs make 7x family speedup
credible, so the pursuit target is 7x (`64.448917 ms`), not merely 5x. An 8x
family result would require `56.392802 ms` and currently has no credible
fixed-work margin.

## Exact relations and transcript boundary

For claim reduction, with `tau = product_uniskip_tau_low`,

```text
C(j) = rd_write_value(j) + gamma*rs1_value(j) + gamma^2*rs2_value(j)
S(j) = eq(tau, j) * C(j).
```

It has 26 degree-two, low-to-high cycle rounds. If the round challenges are
`r_0..r_25`, all outputs open at `reverse(r)`, and the terminal expression is

```text
eq(reverse(r), tau) * (rd* + gamma*rs1* + gamma^2*rs2*).
```

Stage 3 declares Shift, InstructionInput, then RegistersClaimReduction. The
host draws `gamma` before member preparation. On round `k > 0`,
InstructionInput applies `r_(k-1)` before claim reduction sees it. It can
therefore publish its rs1/rs2 midpoint tables for one same-round consumption.
Every combined round polynomial is checked and absorbed on the host; no device
command may cross a challenge dependency.

For value evaluation, split the upstream register point into seven address
coordinates and 26 cycle coordinates:

```text
wa(j) = 0                                  if cycle j has no rd write
      = eq(r_address, rd_index(j))         otherwise
S(j)  = LT(j, r_cycle) * rd_inc(j) * wa(j).
```

This member has 26 degree-three, low-to-high cycle rounds and no local
challenge. Both outputs open at
`r_address || reverse(c_0..c_25)`, in the order `rd_inc, rd_wa`. Its fully
bound LT value must equal
`LtPolynomial::evaluate(reverse(c), r_cycle)`. In the current stage-5 shape it
is tail-aligned behind 128 inactive batch rounds. The first message may execute
during that interval; every later transition remains serialized by a host
Fiat--Shamir challenge.

## Shared owner and residency

The authoritative owner is `CertifiedRegisterOwner`. Its CSR-256 has 256-cycle
blocks and 128 register columns:

```text
start_values:   u64[blocks * 128]
{rs1,rs2,rd}_offsets: u32[blocks * 128 + 1]
{rs1,rs2,rd}_positions: u8[event_count]
rd_post_values: u64[rd_events]
```

The non-authoritative log-26 census retained for design has 262,144 blocks,
59,652,323 rs1 events, 55,924,053 rs2 events, and 50,331,648 rd events. The
full logical owner is 1,239,649,860 bytes. These event counts must be replaced
by a census from the admitted Fibonacci witness before any performance claim.

The fields have distinct consumers:

| Owner field | Consumer and eliminated work |
|---|---|
| `rd_offsets`, `rd_positions`, `rd_post_values` | Claim midpoint: sparse `rd_write_value` fold without a `T`-row value scan. |
| `start_values` plus the three rd fields | Value first message/transition: reconstruct every signed increment and destination register without `oracle_table(RdInc)`, `SharedRdIndices`, or a trace walk. |
| rs1/rs2 offsets and positions | Stage-4 register read/write; release their device allocations after stage 4. |
| `state_flow()` | Fail-closed proof that reads, rd pre-values, block starts, and final carried state came from one row history. |
| `rd_increment_activity()` | Exact nonzero census and independent increment oracle when complete; overflow does not invalidate CSR. |

`RdIncrementActivity` is not the selected shader input. Its entries contain
cycle and signed increment, but not the destination register required to form
`wa`. Pairing it with a new register side plane would be a separate measured
layout experiment. The CSR route works for both complete and overflow activity
states.

The present Rust owner is backend-neutral: it has host `Vec`s but no
proof-session generation, source identity, Metal registry identity, or allocation
identities. Production must attach those identities, fill final owner buffers
during the existing owned-random-access extraction, and forbid a temporary
`Vec<RegisterOwnerRow>` or second `T`-row scan. After stage 4, retain only the
855,638,020-byte rd view. After the value first transition, release that view.
Copying any owner plane into member-private storage is an invariant violation.

The existing value sequence owns 1,611,794,432 bytes of dense arenas, split-LT
tables, and reduction scratch. Its owner-native peak is therefore
2,467,432,452 bytes after rs1/rs2 release. Whole-PIOP admission must include
all other live stage buffers; this family-local number is not an aggregate
working-set claim.

## Selected algorithms

At `log_T = 26`, use the balanced claim split `P = H = 8192`, with
`j = x_hi*P + x_lo`:

```text
Q_v[x_lo] = sum_x_hi eq(tau_hi, x_hi) * v(x_hi || x_lo)
q          = Q_rd + gamma*Q_rs1 + gamma^2*Q_rs2

rd_dense[x_hi] = sum_x_lo eq(reverse(r_prefix), x_lo)
                          * rd_write_value(x_hi || x_lo).
```

Stage 1 produces the three canonical Q component tables while it owns the
outer rows, then uses `3P` full products to recover its three scalar openings.
Stage 3 combines the components with `2P` products after `gamma` is known and
runs the prefix over `p` and `q`. At the midpoint, one Metal command folds only
owner rd events. The same-round InstructionInput handoff supplies immutable
rs1/rs2 tables with producer, table, length, generation, round count, and
ordered-prefix digest. The host forms `C_dense` with `2H` products, runs the
suffix, returns the alias scalars, and recovers
`rd* = C* - gamma*rs1* - gamma^2*rs2*`. This remains valid at `gamma = 0`.

For value evaluation, use one threadgroup per CSR block. Register lanes replay
their rd columns from `start_values`, place `(increment, address_eq)` at the
recorded 256 local positions, and cycle lanes evaluate the three canonical
message samples. Message zero reads the owner without materializing a dense
input. After `c_0`, the same topology reconstructs the block again, binds each
adjacent pair into a resident `{inc, wa}` row, and computes message one before
eviction. Subsequent rounds reuse the existing factorized dense ping-pong
ladder and split LT, then export one `2^16` state for the optimized CPU suffix.
This first slice changes input topology, not round polynomials.

Recomputing from sparse events for several rounds is rejected: after one bind,
`inc` and `wa` are separate weighted sums, so their product contains cross-event
terms within each bound bucket. Repeated event scans would either be wrong or
cost one near-dense scan per Fiat--Shamir round.

## Log-26 work and ceilings

All projections use the retained measured controls: 451,701,710,520 B/s copy,
33.168 G half-width terms/s, and 18.10 G full-field products/s. An 80%-roof is
`max(traffic floor, compute floor) / 0.80`; it is a falsification cap, not a
measurement.

The full owner write has a 2.744399-ms traffic floor and 3.430499-ms 80%-roof.
The claim component producer performs `3T = 201,326,592` half-width terms. Its
retained measured floor is 6.069905 ms. The sparse rd midpoint performs
50,331,648 half-width terms and moves exactly 587,464,708 logical bytes:
rd offsets, positions, post-values, prefix weights, and output. Its compute
floor is 1.517476 ms versus a 1.300559-ms traffic floor.

| Claim GPU work | Floor | 80%-roof |
|---|---:|---:|
| Stage-1 Q components | 6.069905 ms | 7.587381 ms |
| Owner rd midpoint | 1.517476 ms | 1.896845 ms |
| Combined | 7.587381 ms | 9.484226 ms |

The host also moves a two-table alias snapshot of `2H` fields and performs the
`2P` q combine, `2H` dense combine, and the exact
`4P + 8H - 12` prefix/suffix message-and-bind core. Equality generation,
command waits, validation, and transcript service stay in the complete
numerator.

For value evaluation, define `N = 2^26`, `C = 2^16`, and nine dense
transitions. The rd-owner view is exactly

```text
B_rd = 8*(blocks*128) + 4*(blocks*128 + 1)
     + rd_events + 8*rd_events
     = 855,638,020 bytes.
```

| Value phase | Useful full products | Large-state logical bytes | Binding floor | 80%-roof |
|---|---:|---:|---:|---:|
| Owner first message | `3N + 6H` = 201,375,744 | `B_rd` | 11.125732 ms compute | 13.907165 ms |
| Owner first transition | `2.5N + 6H` = 167,821,312 | `B_rd + 16N` = 1,929,379,844 | 9.271896 ms compute | 11.589870 ms |
| Dense ladder | `2.5(N-2C) + 54H` = 167,886,848 | `48(N-2C)` = 3,214,934,016 | 9.275516 ms compute | 11.594396 ms |
| Device prefix | 537,083,904 | 5,999,951,880 | 29.673144 ms | 37.091430 ms |

The displayed byte counts exclude cached address/split tables and reduction
scratch; capture must report their issued traffic separately. The final-row
byte count is cumulative logical traffic, not simultaneous residency.

Using the claim 80%-roof, value prefix 80%-roof, full owner cap once, the
retained 2.371689-ms claim fixed reserve, 3.808875-ms value CPU suffix, and
0.005804-ms export cap gives 56.192523 ms before value host/command service.
That leaves 8.256394 ms under the provisional 7x family cap and only
0.200279 ms under 8x. Seven times is therefore the justified pursuit target.

## Three implementation slices

1. **Resident owner shadow.** Attach session/source/device identities to the
   owner, publish final Metal buffers from the existing extraction, record the
   admitted event census, and implement per-plane release. CPU kernels remain
   authoritative. Reject the slice if it performs a second row scan, creates a
   `T`-row staging vector, changes outer-row wall by more than 2%, or exceeds
   the 3.430499-ms incremental producer cap.
2. **Claim successor.** Add the stage-1 component carrier, owner-native sparse
   rd fold, and one-shot InstructionInput alias handoff. Compare every round,
   alias, output, transcript byte, and terminal expression against an
   independent dense oracle. The Q and midpoint active caps are 7.587381 ms
   and 1.896845 ms; the complete hard cap is 20.480459 ms. Pursue 6x
   (`17.067049 ms`) and continue toward 7x (`14.628899 ms`) whenever measured
   fixed work leaves a credible path.
3. **Value successor and family promotion.** Add CSR-block first-message and
   first-transition entry points, then reuse the retained dense ladder and CPU
   tail. Require the three phase caps in the table, a complete value median at
   or below 69.748025 ms, and continued work toward the credible 7x member cap
   of 49.820018 ms. Finally measure the shared-owner family: hard cap
   90.228483 ms, pursuit cap 64.448917 ms. Do not double-charge the owner, omit
   it, or hide serialized waits behind the asynchronous first message.

Promotion requires five alternating log-26 pairs from one stable binary and
source tree, verified clear and ZK proofs, plus one untuned log-27 transfer.
Capture must show register allocation, resident SIMD groups, no spills/local
memory, active cores, issued bytes, useful operation rate, command gaps,
readback, and peak aggregate residency. A phase below 80% of its matched roof
is unfinished even if the complete member clears 5x.

## Open measurements

- Re-census rs1, rs2, rd, and nonzero rd increments on the exact accepted
  Fibonacci witness; the current counts are analytical.
- Measure the block-CSR access pattern. The arithmetic roofs assume offsets
  and small equality tables cache and that register-column replay does not
  destroy coalescing.
- Freeze a production attribution rule for the stage-1 Q carrier. Its work is
  charged once to the family even when overlapped with outer openings.
- Measure the owner fill as part of the existing extraction. The current Rust
  API proves semantics but does not prove a zero-copy Metal lifecycle.
- Re-freeze the value CPU suffix and all fixed reserves from the stable
  successor binary before acceptance.
