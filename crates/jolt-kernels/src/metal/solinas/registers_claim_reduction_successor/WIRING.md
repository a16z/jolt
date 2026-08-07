# Registers claim-reduction successor

Decision: implement the stage-1 opening fusion and midpoint InstructionInput
alias route. Do not build a fresh register SoA or ship the current isolated Q
slice as the production boundary. This packet is unregistered and changes no
runtime source, protocol, or transcript.

## Frozen evidence and bars

The five-pair log-26 artifact is
`benchmark-runs/metal-piop-eval/20260806-133709-697013/result.json`, revision
`5f520c21e338632aa0bf5936ceb02be6c22fa40f`, M4 Max, 16 Rayon threads. The
optimized member samples are:

```text
98.799290, 101.613748, 102.546459, 99.905582, 97.849458 ms
```

The median is `99.905582 ms`; the complete-member 5x and 8x caps are
`19.981116 ms` and `12.488198 ms`. Count every register-specific producer,
command, wait, host round, output, and validation at that boundary.

Two fused controls prevent favorable attribution:

| fixed paired boundary | optimized median | 5x cap | 8x cap |
|---|---:|---:|---:|
| OuterRemainder + RegistersClaimReduction | 1,015.295537 ms | 203.059107 ms | 126.911942 ms |
| InstructionInput service + RegistersClaimReduction | 827.118001 ms | 165.423600 ms | 103.389750 ms |

The first pair is the producer truth; the second is the midpoint-alias truth.
The old artifact's Metal OuterRemainder alone is `215.291623 ms`, so that old
pair cannot reach 5x even with a zero-cost register member. This is an upstream
OuterRemainder limit, not permission to omit its time. Re-freeze both pair
denominators from the eventual same binary.

The existing isolated Q slice is useful evidence, not a complete kernel. At
log 26 it computes `3T = 201,326,592` unsigned half-width terms in
`6.092375 ms` active and `6.701125 ms` resident wall; Criterion reports
`33.168 Gterm/s`. It excludes a production row owner, the midpoint fold, host
rounds, and aliases.

## Exact relation and host boundary

For row `j`,

```text
C(j) = rd(j) + gamma*rs1(j) + gamma^2*rs2(j)
s(j) = eq(product_tau_low, j) * C(j).
```

The relation has degree two and binds cycle variables low-to-high. If its
round challenges are `r_0 .. r_(n-1)`, all three output openings are at
`reverse(r)`. The verifier checks

```text
eq(reverse(r), product_tau_low)
    * (rd* + gamma*rs1* + gamma^2*rs2*).
```

`gamma` is drawn on the host before stage-3 preparation. The batch driver
checks and absorbs every combined round polynomial and draws each bind
challenge on the host. A device command may compute a member's message and
bind using an already drawn challenge, but it may not hash, draw, or cross the
next challenge dependency.

The declaration order is Shift, InstructionInput, RegistersClaimReduction.
On round `k > 0`, InstructionInput therefore sees and applies `r_(k-1)` before
RegistersClaimReduction sees the same challenge. This ordering permits a
one-shot alias handoff; it does not permit a different challenge or transcript.

## Geometry

For `n = log2(T)`:

```text
prefix_vars = ceil(n/2), P = 2^prefix_vars
suffix_vars = floor(n/2), H = 2^suffix_vars
tau = tau_hi || tau_lo
j = x_hi*P + x_lo
```

The prefix decomposition is

```text
p[x_lo]   = eq(tau_lo)[x_lo]
Q_v[x_lo] = sum_x_hi eq(tau_hi)[x_hi] * v(x_hi || x_lo)
q          = Q_rd + gamma*Q_rs1 + gamma^2*Q_rs2.
```

At the midpoint, after the first `prefix_vars` stage-3 challenges,

```text
v_dense[x_hi] = sum_x_lo eq(reverse(r_prefix))[x_lo]
                          * v(x_hi || x_lo).
```

The checked odd/even scalar oracle is `oracle.rs`; `model.rs` freezes the roof
arithmetic independently.

## Selected architecture

### Stage 1: transpose inside the existing opening scan

OuterRemainder already stages every 48-byte compact row and 112-byte residual
row in threadgroup memory to compute 35 openings at `product_tau_low`. Its
current orientation first contracts `x_lo`, which produces one scalar partial
per `x_hi`; those partials cannot be relabeled as `Q_v`.

Change only the opening kernel's schedule:

1. Use `B = 256` long-lived threadgroups. Each owns `H/B = 32` `x_hi`
   coordinates and visits `x_lo` in the existing 32-row tiles. Adjacent lanes
   still stage adjacent rows, so row traffic remains coalesced.
2. Keep the existing scalar-opening accumulators for the other 32 columns.
   For columns rd, rs1, and rs2, SIMD groups 0, 1, and 2 instead retain one
   Q accumulator per lane across the 32 owned `x_hi` values, weighted by
   `eq(tau_hi)[x_hi]` with the promoted unsigned half-width primitive.
3. Write `3*B*P` field partials. A second dispatch in the same command buffer
   reduces them to the three `P`-element component tables and derives the
   three omitted stage-1 scalar openings by dotting with `eq(tau_lo)`.
4. While each row is already staged, write one canonical `u64 rd[T]` plane.
   This is the minimum arbitrary-point state needed later. Release the 112-byte
   residual row after stage 1 as today; retain the 48-byte InstructionInput row,
   the rd plane, and the three component tables.

The carrier must record producer `OuterRemainder`, device registry id,
generation, compact/residual allocation identities, the raw stage-1 remainder
point digest, `P`, `H`, and each output allocation identity. Stage 3 recomputes
`product_tau_low` from the typed stage-1 output and rejects any mismatch.

This schedule replaces, rather than supplements, the three current scalar
opening columns. It removes `3T + 3H` full products and adds `3T` matched
half-width terms, `3P(B-1)` field additions, and `3P` lower-order opening-dot
products. It incurs no second full-domain row read.

### Stage 3 prefix and midpoint

After `gamma` is drawn, the host maps the 384-KiB component carrier and forms
`q` with `2P` products. Prefix messages and binds run on host over only `p`
and `q`; their total is lower order (`about 4P` products).

InstructionInput and this member receive a shared, one-shot handoff object at
prepare. At the last prefix bind, InstructionInput publishes copies of dense
tables 1 and 5 (rs1 and rs2) after applying that challenge. The handoff carries
member/table ids, remaining length `H`, round count, generation, and the ordered
prefix-challenge digest. RegistersClaimReduction consumes it later in the same
round and fails closed on any mismatch.

At that boundary, one Metal command folds the retained native rd plane under
`eq(reverse(r_prefix))`, producing `rd_dense[H]`. The host forms

```text
C_dense = rd_dense + gamma*rs1_dense + gamma^2*rs2_dense
```

and runs the suffix over `eq(tau_hi)` scaled by the bound prefix, `C_dense`, and
the two alias tables. Terminal outputs are the two alias scalars and

```text
rd* = C* - gamma*rs1* - gamma^2*rs2*.
```

This is exact for `gamma = 0` and requires no inversion. `DirectLinear`, which
folds all three native columns at the midpoint, remains the fail-closed parity
fallback but is not the performance route.

## Log-26 floors

Retained same-machine controls are `451,701,710,520 B/s`, the matched Q-slice
rate `33.168 G unsigned terms/s`, and `18.10 G full products/s`. The last rate
prices displaced work only; it is not added to the selected new-work roof.

With `T = 2^26`, `P = H = 8192`, and `B = 256`:

| phase | exact new work | traffic | binding floor |
|---|---:|---:|---:|
| fused stage-1 Q | `3T` half terms | 192 MiB partial roundtrip + 384 KiB components + 512 MiB rd write | 6.069905 ms compute |
| midpoint rd fold | `T` half terms | 512 MiB rd + 128 KiB weights + 128 KiB output | 2.023302 ms compute |
| gross GPU | `4T` half terms | 704.6 MiB | 8.093207 ms |

At 80% of the matched roof the GPU cap is `10.116509 ms`, leaving
`2.371689 ms` under the 8x member cap for command service, the mapped component
combine, host rounds, alias validation, and outputs. Eightfold is credible but
not guaranteed; do not stop at 5x if the measured fixed work fits this budget.

The displaced stage-1 work has an `11.124374 ms` arithmetic floor. The model
therefore predicts the fused producer should not slow OuterRemainder, but only
the paired trace may claim that improvement. Do not subtract the modeled saving
from the standalone numerator; report gross register-specific work and the
actual fused pair independently.

The machine-balance flip for the fused Q phase is about
`33.168e9 / 451.702e9 = 0.07343 term/byte`. Its new-work intensity is about
`201.327M / 738.591M = 0.2726 term/byte`, so arithmetic binds. The midpoint is
`0.1249 term/byte` and also arithmetic-bound.

## Occupancy and command gates

The Q lane adds one long-lived four-limb accumulator to each of three SIMD
groups, not three seven-limb accumulators to every thread. The existing opening
shader already retains up to five field sums per thread. Compiler allocation,
not source counting, decides occupancy.

At `B = 256`, the M4 Max gets 6.4 threadgroups per GPU core; each group performs
32 balanced `x_hi` iterations. Screen only `B = 128, 256, 512`. Promotion needs
an emitted-code/Instruments capture showing register allocation, resident SIMD
groups, no spills/local memory, all 40 cores active outside tails, achieved
half-term rate, partial bandwidth, and command gaps. Reject a block count that
drops below 80% of the matched 33.168-Gterm/s rate even if total wall happens to
clear 5x.

Stage 1 adds dispatches to an existing command buffer and no new wait. Stage 3
adds exactly one midpoint command/wait. A separate Q command over compact and
residual global rows is not the default: its selected-word logical traffic is
already 40T bytes, while cache-line issuance can approach the full 160T row
footprint, whose copy floor alone is 23.77 ms. A fresh three-plane SoA is also
rejected because producing and retaining 1.5 GiB is outside the measured slice.

## Minimal implementation order and falsifiers

1. Add the typed stage-1 carrier and scalar parity for its component tables,
   reconstructed stage-1 openings, and rd plane. No backend selection yet.
2. Add the fused opening shader behind one fixed `B = 256` diagnostic. Compare
   all 35 openings and the carrier against the independent dense oracle.
3. Add the one-shot InstructionInput midpoint handoff and the rd fold. Exercise
   wrong producer, table id, round, length, point digest, generation, and reuse.
4. Add the complete host prefix/suffix kernel and exact optimized-CPU lockstep
   at odd/even logs, `gamma = 0/1/random`, values around `2^32`, and `u64::MAX`.
5. Freeze the cheapest exact evaluator, then run five alternating log-26 pairs,
   one untuned log-27 transfer, proof verification, clear and ZK modes, and both
   fused paired boundaries above.

Reject or redesign if any of these occurs:

- complete member median exceeds `19.981116 ms`;
- fused stage-1 Q or midpoint sustains less than 80% of its matched roof;
- the fused opening makes the paired OuterRemainder wall worse by more than 2%;
- the midpoint adds more than one command/wait or cannot fail closed on alias
  identity;
- any producer/upload/readback time is omitted from both the standalone and
  paired accounting; or
- exact round polynomials, output aliases, derived EqSpartan, transcript, or
  proof bytes differ from optimized CPU.

The model and oracle are unregistered so they can be checked in isolation
before any shared backend wiring changes.
