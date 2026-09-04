# PERF-5 rank 9 — virtual T2 operands

Date: 2026-09-03. Base: `80ac4bc7d`. Scope: design only; no timed run.

## Verdict

**NO-GO for direct `AffineForm` substitution.** The fast path saves an estimated
**2.63 s** at the measured commitment rate, but it is not sound: a stream
`AffineForm` combines committed-column evaluations at the same stage-A point,
while every T2 operand reads `Z` at a different, row-dependent source. A sound
link needs another source-row sumcheck and a two-point reduction, or it must
keep the operand commitments. Neither is the proposed four-factor, one-point
protocol.

The literal same-row model also misses the 2.0 s gate at the 0.30 us/point MSM
floor: **1.25 s** net before any measured member-construction saving. Do not
implement rank 9 as currently described.

## 1. What the 44 columns do now

For slot `s` and Boolean table row `x`, phase 2a commits

```text
X_s(x) = kappa_s(x) * Z_xi(src_x_s(x))
Y_s(x) = y_sign_s(x) * Z_xi(src_y_s(x))

Z_xi(v) = L_0(v) + xi*L_1(v) + xi^2*L_2(v)
```

`L_0` and `L_1` each use six little-endian 16-bit chunks; `L_2` uses four.
The full form therefore has **16 weights, partitioned 6/6/4**. It is not an
18-weight form, and full operands do not vary among 6/12/18 chunks in this
code. `X` applies `kappa` to every weight; `Y` applies `y_sign`. `xi` orders
96-bit limbs low to high.

| consumer | count | stage-A polynomial | stage-A degree | exported factors, before -> after |
|---|---:|---|---:|---:|
| limb products | 22 | `eq_tau * (1-free) * X_s * Y_s` | 4 | `3 -> 3` |
| selected-operand fingerprint | 3 kinds | `eq_tau * H * is_kind * sum_s fp^s Y_s`; slot counts GT/G1/G2 = 22/2/4 | 4 | `2 -> 2` |
| copied `X` | 22 appearances | `eq_tau * copy_root^s * X_s` | 2 | part of one `1 -> 1` affine term |
| non-looked-up `Y` | 22 appearances | `eq_tau * mask_s * copy_root^(22+s) * Y_s` | 3 | part of the same `1 -> 1` affine term |
| phase-2c `H` construction | selected rows | denominator uses the same 22/2/4 `Y_s` fingerprint | n/a | n/a |

The term export has 44 operand weight entries in the limb terms, 28 in the
three read-fingerprint forms, and 44 in the copy form: **116 operand-column
entries across 48 affine factors**. The whole T2 export remains 177 terms and
425 factors; the widest unrelated range/digit terms have four factors.

There is **no** per-operand term `Z_xi = sum_j weight_j * chunk_j`. One global
randomized copy identity binds all `X`, all non-looked-up `Y`, and `F_pos/F_neg`
to source-row chunks:

```text
sum_x eq(tau,x) * sum_i copy_root^i C_i(x)
  = sum_v CopyKernel(tau,v) * Z_xi(v).
```

Selected `Y` values are instead bound by the LogUp fingerprint. `F_pos/F_neg`
are computed from source `Z_xi`, committed in phase 2b, then included in the
copy identity. Algebraic virtualization would remove the operand pieces of
the copy identity, not the two fingerprint-copy terms.

## 2. Why the proposed virtual form is not a stream `AffineForm`

At the final stage-A point `r`, the required value is

```text
X_s_tilde(r)
  = sum_x eq(r,x) * kappa_s(x) * Z_xi(src_s(x))
  = sum_v K_s(r,v) * Z_xi(v).
```

The proposed form computes

```text
sum_j w_j(xi) * chunk_j_tilde(r),
```

which is `Z_xi_tilde(r)`, not the source-mapped functional above. The two are
equal only for the identity source map. T2 uses shifts, tables, restrictions,
and non-bijective maps.

`stream::AffineForm { constant, weights: Vec<(ColumnId, Fr)> }` has no source
point or kernel. `term_reduction_with_weights_observed` turns it into one
linear functional of columns opened at `r`; it cannot express
`sum_v K_s(r,v) Z(v)`.

Keeping uncommitted dense `X/Y` arrays in `RowMatrix` does not fix this. They
are chosen after `tau`, `copy_root`, and `fp_root`; the prover can adapt them
to the single copy/fingerprint equations. Honest witness construction is not
a verifier binding.

### Sound repair

After term compression supplies its four factor evaluations and fresh mixing
weights, split the reduced claim into committed-column and virtual-operand
parts. Prove the virtual part with

```text
sum_s u_s * O_s_tilde(r_A)
  = sum_v [sum_s u_s K_s(r_A,v)] * Z_xi(v)
```

over a fresh source point `r_W`, then reduce the chunk openings at `r_A` and
`r_W` to one HyperKZG point. This preserves four term-factor evaluations, but
adds two 18-round degree-2 reductions. The earlier M2 design instead sent all
`6 * 22 = 132` operand-limb evaluations; at this row count that costs 4,224 B
plus about 2,304 B of sumcheck data.

The compressed repair can avoid the 132 scalars, but not the extra source-row
proof and two-point reduction. It is a new protocol, not a local T2 column
change.

## 3. Cost model

### Stage A: direct local substitution, for cost only

`RowSumcheck` evaluates three full summands per pair in round 0 and four in
later rounds; its separate leading-coefficient calculation does not read
operands. Caching the 44 virtual forms once per summand gives

```text
round 0:  44 * 3 * 2^17       = 17,301,504 form evaluations
rounds 1..17: 44 * 4*(2^17-1) = 23,068,496 form evaluations
total:                           40,370,000 form evaluations
```

At the actual 16 entries per form, the generic evaluator performs
**645,920,000 Fr multiplications**; 18 padded entries would perform
726,660,000. Lane 3's T2 point gives a coarse throughput of
`425*5*(2^18-1)/1.910 s = 291.7 M` factor-work multiplications/s, hence
**+2.21 s** for 16 entries and a **2.1-2.5 s** planning range. This estimate
describes the algebraically wrong same-row form; it is not an implementation
estimate for the sound repair.

### Term stage and stage B

If the same-row substitution were valid:

- Limb and lookup terms retain their factor counts. The copy term drops its
  44 operand entries but remains for `F_pos/F_neg`.
- The 72 remaining operand appearances expand from one entry each to 16:
  `72*16 - 116 = +1,036` affine weight entries versus today.
- `TermStageProver::new` therefore adds about **1,036 Fr multiplications** once.
- Verifier term reduction adds the same **1,036 Fr multiplications** before
  any savings from deleting operand copy-kernel evaluation.
- The number of distinct committed columns in each reduced functional does
  not grow: all false local forms reuse the 16 `Z` chunk IDs. Claimed T2
  columns fall `149 -> 105`.
- The full assembly stays inside the same padded column power, so stage B
  remains **10 rounds** at `k=32` and `k=16`.
- `max_factors = 4`; the proof still sends **4 factor evaluations = 128 B**.

The last three properties do not establish soundness; they expose why the
bad substitution looks cheap.

### Commitment and member construction

There are `44 * 2^18 = 11,534,336` active full-field points:

| full-Fr rate | phase-2a time removed |
|---:|---:|
| measured 0.42 us/point | **4.844 s** |
| campaign floor 0.30 us/point | **3.460 s** |

`operand_columns` and the 44 private row polynomials are still required by a
sound post-stage linker, so its member-construction saving is **0 s**. The
invalid local version could remove up to the whole measured T2 finish/member
bucket (**0.493 s upper bound**), but no sub-timer attributes that bucket to
operands; use zero in the decision.

| model | 0.42 rate | 0.30 floor |
|---|---:|---:|
| direct local form, no member credit | **2.63 s** | **1.25 s** |
| direct local form, possible member credit | `2.63..3.12 s` | `1.25..1.74 s` |
| sound path that retains commitments | **0 s** | **0 s** |

The M2-style sound linker was estimated at about 0.10-0.14 s prover time at
`2^18`, so its time saving would exceed 3 s. Its proof and verifier costs are
different enough that it needs a separate lever and gate.

## 4. Fiat-Shamir, degree, and binding

With the rejected direct substitution, phase 2a would contain only the 22
range helpers and range inverse: **23 alpha-dependent columns and no
xi-dependent column**. `F_pos/F_neg` remain xi-dependent phase-2b columns.

Local T2 commitment groups become:

| packing | current `[1b,2a,2b,2c+VK]` | proposed | commitment bytes |
|---:|---:|---:|---:|
| `k=32` | `[3,3,1,2]` | `[3,1,1,2]` | **-64 B** |
| `k=16` | `[5,5,1,2]` | `[5,2,1,2]` | **-96 B** |

The `k=16` result is not -128 B: `ceil(67/16) - ceil(23/16) = 3` groups.
Phase padding is local, so 44 deleted columns remove 48 padded slots there.

Challenge counts remain **`[39, 23, 1, 3, 232]`**. Order remains

```text
phase 1a -> theta -> phase 1b -> xi,alpha/link challenges
 -> phase 2a -> fp_root -> phase 2b -> beta,fp_combine,copy_root
 -> phase 2c -> stage A.
```

For committed chunk values, the limb identity is a polynomial in `xi` of
degree at most **4**:

- each operand has degree 2, so `X_s Y_s` has degree 4;
- `k(xi) q(xi)` has degree 4;
- `(2^96-xi) C(xi)` has degree 4;
- the exact-sign term has degree 2.

A false identity at sampled `xi` passes with probability at most
**`4 / |Fr|`, about `2^-251.6`**. A genuine algebraic substitution has the
same xi argument as today and removes the operand-copy batching error.
The direct stream substitution is weaker for a separate reason: it does not
bind the source-mapped operand MLE at `r_A`.

Today's committed columns supply two bindings that disappear:

1. `X` and non-selected `Y` are fixed before `copy_root/tau`, then checked by
   the copy identity against source chunks.
2. Selected `Y` tuples are fixed before `fp_root`; the LogUp read fingerprint
   and phase-2c `H` denominator cannot be selected after that root.

Operands have no direct range check. Their source `Z` chunks do. The table
fingerprints remain committed and copy-bound, so their copy terms cannot be
deleted.

## 5. Bytes and gas

For the unsound no-new-stage shape, commitment removal alone changes the N4
model by:

| packing | removed groups | payload | ecMul | Keccak | commitment-only gas |
|---:|---:|---:|---:|---:|---:|
| `k=32` | 2 | -64 B | -2 | -2 | **-17,648** |
| `k=16` | 3 | -96 B | -3 | -3 | **-26,472** |

Each group saves 7,700 ecMul gas, 100 transcript gas, and 64 gas-charged bytes
(32 proof bytes plus 32 precompile-input bytes) at 16 gas/byte.

The false 16-chunk forms add 1,036 observed verifier multiplications
(**+20,720 gas**) before deleting operand copy-kernel work. Thus the rank-9
`-15k` estimate is not established: the partial total is **+3,072 gas** at
`k=32` and **-5,752 gas** at `k=16`, with the removed structured-kernel cost
still unmeasured. There is no final gas delta for a sound four-factor design
until the new source-row proof is specified.

The known M2 repair changes payload by at least **+6,464 B at `k=32`** and
**+6,432 B at `k=16`** (`132*32 + 2*18*2*32` minus commitments), before proof
framing. It therefore does not have the advertised byte saving.

## 6. Implementation specification for a future sound lane

Do not start by deleting `Col::X/Y`. Prove the replacement link first.

1. **Typed virtual references.** In `limb_table/terms.rs` and
   `stream/types.rs`, add a distinct operand-functional reference; do not
   encode it as a same-point `ColumnId`. Make term reduction return committed
   column weights plus virtual operand weights.
2. **Source-row link.** In `limb_table/wiring.rs`, `row_sumcheck.rs`, and
   `stream/term_stage.rs`, prove the mixed operand functional at `r_A` against
   the 16 source `Z` chunk columns. Add the second-point reduction in
   `stream/protocol.rs`; fix its transcript position after term-factor mixing
   and before the final opening.
3. **Private operands.** In `columns.rs`, `stream.rs`, and `lookup.rs`, retain
   `operand_columns` as stage-A/LogUp private data until the linker is green.
   Separate committed-column positions from private `RowMatrix` fields.
4. **Delete commitments.** In `relation.rs` and `export.rs`, remove `X/Y`
   from the committed ranges, remove only their copy-kernel contributions,
   retain fingerprint copying, and shift later `Col` IDs. In `wrap/key.rs`,
   update phase groups and physical IDs; keep all challenge counts and order.
5. **Then remove storage if measured.** A lazy row view may replace the 44
   dense private polynomials only after its stage-A cost beats materializing
   and binding them. The performance gate decides; no speculative callback
   layer.

### Tests to change

- `stream_builder_phase_slices_match_declared_geometry`: pin
  `[18,6,1,3]`, `[5,2,1,2]`, and `[3,1,1,2]` at `k=4/16/32`.
- `stream_exporter_terms_match_the_members`: committed claims `149 -> 105`,
  177 terms, maximum four factors, four final factor evaluations, and the new
  virtual-link final claim.
- `tampered_witnesses_are_rejected`: replace direct `Col::X + 3` and
  `Col::Y + 3` mutations with source-chunk mutations. One copied source and
  one selected lookup source must reject through the virtual link.
- Replace `selected_operand_collision_for_a_guessed_fingerprint_root_is_rejected`
  with a collision over three source chunk rows; it must reject after the real
  `fp_root` is drawn.
- Add a verifier-path negative that changes a private operand while leaving
  every committed chunk unchanged and forces the stage-A round checks. The
  source-row link, not an unrelated row constraint, must reject it.

### Acceptance gate

1. Real `fibonacci_2_18_blake3.bin` prove/verify at `k=32` and `k=16`.
2. Every existing proof and witness tamper plus the two virtual-link negatives.
3. Exact payload, bincode, statement bytes, ecMul/ecAdd, Fr mul, inversions,
   Keccak, and N4 gas.
4. Idle-machine phase timing at the same 10-thread settings; compare phase
   2a, T2 finish/member, T2 stage A, added link/reduction, and honest wall.
5. Accept only with a formal source-map identity, no lost commit-before-root
   binding, and at least **2.0 s** honest-wall saving at the selected MSM rate.

Effort: **32-48 engineer-hours** for the sound compressed linker, stream
integration, adversarial tests, and two idle gates. The direct deletion is
roughly 8-12 hours but must not ship.
