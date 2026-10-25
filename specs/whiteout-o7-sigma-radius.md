# O7: lowering the Σ collision radius

| Field | Value |
|---|---|
| Ticket | O7 (`specs/whiteout-optimization-campaign.md`, P2) |
| Status | analysis; no code changed |
| Created | 2026-10-25 |
| Pins | jolt `whiteout/dev` at `b3b29be54`; Akita `~/akita-whiteout` at `299567ec6`; paper `akita-paper/sections/whiteout/` as read on 2026-10-25 |
| Inputs | `whiteout-sigma.md` §5, §5.1; `whiteout-paper-deltas.md` D27, D29; `whiteout-v6-costs.md`; Akita `whiteout-akita.md` §5.5.2; `akita_params::sis::sigma_binding` (CSV `sigma_binding_infinity_width.csv`); Akita's main ℓ∞ table `generated_sis_table/q128.rs` |

## Decision

**Implement one point, P9 (D = 9, α = 2^13), on top of O6. Gate it on one
generator run. Drop every other lever.**

- **The only lever that matters is α, through the digit count D.** The
  radius is `R_Σ = 8κ₁·B(D)`. B(D) is set by the Gaussian width, and the
  width scales linearly with α. One digit buys 4 bits of radius and costs 16×
  in α, so the per-attempt rejection rate rises about 16×. At d = 64, one bit
  of radius is worth about 0.56 rank.
- **κ₁, the difference set and the extraction argument together are worth at
  most about 0.4 bits (§4, §5).** That moves no rank in any measured cell.
  The 15/16 diameter tightening (Lemma O7.1) is free but small, at 0.09 bits.
  Adopt it only if the generator shows that it flips a rank.
- **P9 beats O6 at no ring dimension. It compounds with O6 at every one**
  (§6). At d = 256 it takes the residual from rank 5 to rank 4. Residual work
  falls from 0.27 to 0.19 of today's, about 30% on top of O6. u_a falls from
  1,280 to 1,024 coefficients, and the z digits in e fall by 10%. In exchange,
  a proof needs a second Σ attempt once in about 595 proofs instead of once
  in 9,400 (2^−9.2 instead of 2^−13.2 per attempt).
- **Gate.** No checked-in cell covers P9's rank 4 at d = 256 (and rank 16 at
  d = 64). Interpolation puts it inside the margin (§3.3). The Σ-binding
  generator must be run at `R_Σ(9)`. If it gives rank 5 at d = 256 for any
  residual row, P9 gains nothing under O6 at d = 256, and this note's
  recommendation becomes *don't implement*, unless O6 stops at d = 128, where
  P9 still takes rank 10 to 8.
- **P9 reopens §5.1's review decision.** That review (Q. Dao) chose D = 10 over
  D = 9 when D = 9 was worth 6 points of verifier scans. At the Σ-binding
  ranks D = 9 is worth 24–30% of the residual. Rejection rate is a
  predictability property, not a security one, so the call is the
  conductor's. This note does not make it silently.
- **P8 and P7 are for the owner, only if the almost-never-rejection
  requirement is relaxed.** At d = 256, P8 gives the same rank 4 as P9, with
  existing certified cells, at 2.7% rejection. P7 is the paper's own
  operating point: rank 3 over R_256 at radius 2^35.8, residual work 0.12 at
  d = 256, and 35% rejection.

Every security statement survives each point with the same theorem; only
the numbers move (§7). No point needs a new assumption. The optional
diameter form needs one small lemma, written in §7.1.

## 1. What sets R_Σ

`R_Σ = 4Γ_Δ·B(D) = 8κ₁·B(D)`, where:

- `κ₁ = 53`;
- `B(D) = 8(16^D − 1)/15` is the negative end of the D-digit balanced
  radix-16 box `𝓑_cert`;
- D is the largest digit count of any Gaussian segment
  (`jolt_zk::whiteout::binding_collision_radius`, capped by
  `BINDING_RESPONSE_DIGITS`).

D comes from the width (`params.rs`, `SigmaParams::new_with_alpha`). It is the
fewest digits with

```
⌊L(D)²/k²⌋ ≥ 1 + ⌈log₂(2·2^20·N)⌉,   k = ⌈√(2 ln2 · Γ²α² · Σ_b γ_b²)⌉ + 1,
L(D) = 7(16^D − 1)/15,
```

with Γ = 18 and N the number of Gaussian coefficients. At the jolt-zk
headline table, k = 58,642,511,897 ≈ 2^35.77 at α = 2^17, and the digit rule
needs ⌊L²/k²⌋ ≥ 45. D = 10 gives 76.

So R_Σ ∝ κ₁ · L(D) with L(D) ≳ √45·k ∝ Γ·α·√Σγ². The radius moves with:

| knob | effect on log₂ R_Σ | what it trades against |
|---|---|---|
| α | +1 bit per doubling, quantized to 4-bit digits | per-attempt rejection `r ≈ t/α`, t = √(2·137·ln 2) = 13.78 |
| κ₁ | log₂(κ₁′/53) | challenge-space size, so the knowledge error 2/\|𝓒\| (§4) |
| Γ | none at a fixed D; it moves α_max(D) instead | rejection, as α does |
| Σγ² | as Γ: it moves α_max(D) | the instance size (fixed) |
| the extraction constant 4·B | §5 | the proof of Thm `wo:thm:knowledge` |

**Scaling α with D.** α = 2^{17−4(10−D)} keeps ⌊L(D)²/k²⌋ = 76 exactly, as
today. Every point below therefore keeps today's headroom in instance size,
about 1.7× the headline's N before a digit is added. The rejection rate
depends only on α, not on N:
`r = 1 − (p_𝓑 − ε)/M` with `ln M = t/α + 1/(2α²)`.

## 2. The points

| point | α | D | log₂ R_Σ (paper form / diameter form, §7.1) | r per attempt | second attempt once in | T_att | E[attempts] | sampler ℓ | rel. mean Σ sampling time | z digits per Gaussian coeff. |
|---|---|---|---|---|---|---|---|---|---|---|
| **P10 (today)** | 2^17 | 10 | 47.82 / 47.73 | 2^−13.20 | 9,400 proofs | 10 | 1.0001 | 77 | 1.00 | 10 |
| **P9** | 2^13 | 9 | 43.82 / 43.73 | 2^−9.22 | 595 | 14 | 1.0017 | 69 | 0.93 | 9 |
| **P8** | 2^9 | 8 | 39.82 / 39.73 | 2^−5.23 | 38 | 25 | 1.027 | 61 | 0.88 | 8 |
| **P7** | 2^5 | 7 | 35.82 / 35.73 | 2^−1.51 (0.35) | 2.9 | 85 | 1.54 | 53 | 1.20 | 7 |

How the columns were computed:

- **r and T_att.** `p_𝓑 ≥ 1 − 2^−20` and ε = 2^−137. T_att is the smallest T
  with `r^T ≤ 2^−128`. The code's `max_attempts` computes it in 128-bit
  fixed point with every rounding against acceptance, so it may come out one
  higher.
- **Sampler cost.** ℓ = ⌈log₂ 27k²⌉, and the per-sample cost is about
  `200 + 6ℓ` ns (`whiteout-sigma.md` §5.1, A16). Relative mean time is
  E[attempts] × (200 + 6ℓ)/662.
- **Bytes and the whole-proof rerun.** Σ bytes are unchanged at 160 B
  (t_D plus the nonce). The whole-proof rerun after T_att rejections stays at
  probability ≤ 2^−128 at every point.

**Is B(D) the right bound?** It is the right bound for extraction. Tightening
it needs a stronger range proof, and that buys at most half a bit.

- **Why B_cert and not the rejection bound.** The extractor sees only what
  the verifier checks, and the response proof certifies digit alphabets,
  i.e. 𝓑_cert. The rejection box |z| ≤ L(D) is a rule of the honest prover;
  a cheating prover is not bound by it.
- **The most a tighter certificate could buy.** Certifying |z| ≤ L_need =
  √45·k would need a top digit with a non-power-of-two alphabet (about
  [−5, 5] instead of [−8, 7]) in Akita's range check. At α = α_max(D),
  L_need = √(45/76)·L(D) = 0.77·L(D), so the certified diameter would fall
  from 15u to 10.8u (u = (16^D − 1)/15). That is 0.48 bits: about a quarter
  rank at d = 64, nothing at d = 256. Not worth a new alphabet in the range
  check.
- **Finer control of α needs no new alphabet.** A fractional digit would
  make the radius continuous in α. Because the rank is quantized anyway, the
  digit-aligned points above already sit at the useful corners.

## 3. Rank against log₂ R_Σ (Q1)

### 3.1 Source and model

All ranks come from Akita's ADPS16 quantum ℓ∞ tables over q128: estimator
`akita-infinity-width-v4`, profile `local-minimum+proven-pruned-certification`,
128-bit target, which means BKZ block size β ≥ 484.

- **The Σ-binding CSV** has one radius only, 2^47.82, at d ∈ {64, 128, 256,
  512}.
- **Akita's main ℓ∞ table** (`q128.rs`) has the same model and generator at
  honest bounds up to 2^40.73 at d = 64 and 2^40.67 at d = 128 and 256.
- **Certification rule.** A cell at bound b ≥ R certifies a width at R,
  because SIS hardness is monotone in the bound. The certified ranks below use
  the smallest table bound ≥ R.

**The model of the residual.** It reproduces every checked-in W5e row
exactly. Ranks are 18/19, 19/19 and 19/19, with [B|B′] widths 5,794, 7,906 and
13,378, and A_p is rank 18 at width 898. It also reproduces Akita's d = 256
estimate of [B|B′] widths 1,601, 2,081 and 3,521. The pieces:

- A_a's input is fixed: 1,024, 4,096 or 8,192 R_64 columns.
- t̂_a has `32·blocks·rank_A` R_d columns, with 9, 12 or 21 blocks.
- r_a is `⌈(n_B·128 + 264)/(4d)⌉` R_d columns, where n_B = rank_B·d.
- [B|B′]'s rank is the fixed point of its own width.
- A_p's width is 320 + r_p.

### 3.2 Rank of the widest matrices at each certified cell

Ranks below are (A_a 1,024 / A_a 8,192 / [B|B′] 5,794 / [B|B′] 13,378 / A_p
610 / A_p 2,500), with widths in R_64 columns scaled to R_d.

| log₂ bound (cell) | d = 64 | d = 128 | d = 256 |
|---|---|---|---|
| 31.67 | 9 / 10 / 10 / 10 / 9 / 9 | 5 / 5 / 5 / 5 / 5 / 5 | 3 (all) |
| 35.67–35.73 | 11 / 12 / 12 / 12 / 11 / 11 | 6 / 6 / 6 / 6 / 6 / 6 | 3 (all) |
| 37.67 | 12 / 13 / 13 / 13 / 12 / 12 | 6 / 7 / 7 / 7 / 6 / 6 | 3 / 4 / 4 / 4 / 3 / 3 |
| 39.67–39.73 | 13 / 14 / 14 / 14 / 13 / 14 | 7 (all) (cell 39.95) | 4 (all) (cell 39.95) |
| 40.67–40.73 | 14 / 15 / 14 / 15 / 14 / 14 | 7 / 8 / 7 / 8 / 7 / 7 | 4 (all) |
| 47.82 (Σ table) | 18 / 19 / 19 / 19 / 18 / 19 | 9 / 10 / 10 / 10 / 9 / 10 | 5 (all) |

**Slope.** Hardness tracks n = rank·d, nearly independent of d. For
example, n = 1,024 gives the same max width at d = 64, 128 and 256 to within
1%. The required n grows by about 36 rows per bit of radius, i.e. about
0.56 rank per bit at d = 64, 0.28 at d = 128 and 0.14 at d = 256. One digit
(4 bits) is therefore about 2.2 / 1.1 / 0.56 ranks. Because the residual
widths sit right at the cliff of the width curve (at d = 64 and 2^47.82,
rank 18 holds up to 2,463 columns and rank 19 up to 26,544), every
quantization step lands as a whole rank or not at all.

### 3.3 The uncovered radius: P9

R_Σ(9) = 2^43.82 lies between the 2^40.73 cell and the 2^47.82 cell. Log-width
interpolation gives the following, bracketed by the chord and by the slope
of the two cells below:

| n | max width at 2^43.82 (R_64 columns) | needed (widest row) |
|---|---|---|
| 896 (d = 64 rank 14) | 55–817 | no |
| 960 (d = 64 rank 15) | 891–3,796 | A_a 1,024 borderline; A_p ≈ 770–802 yes |
| 1,024 (d = 64 rank 16; d = 128 rank 8; d = 256 rank 4) | 11,579–16,526 (13,314–18,800 at the diameter radius) | [B\|B′] 11,266 (36:1 at rank 16) |

So P9 is *estimated* at the following ranks:

- d = 64: rank 16 for A_a and [B|B′], and A_p at 14 or 15;
- d = 128: rank 8;
- d = 256: rank 4.

The margin at n = 1,024 is 3% at the pessimistic end. These ranks are not
certified.

**Generator run needed.** Run `examples/sigma_binding_width_table.rs` at
R_Σ(9), and at the diameter form 4κ₁(16^9 − 1) = 14,568,529,067,820, for
`--dims 64,128,256`.

- **Mechanics.** The example takes its radius from
  `jolt_zk::whiteout::binding_collision_radius()` through Akita's path patch to
  the jolt `whiteout-dev` worktree, and it writes the checked-in CSV. The run
  therefore needs a scratch Akita worktree whose `[patch]` points at a
  scratch jolt worktree with `BINDING_RESPONSE_DIGITS = 9`. Alternatively, add
  a `--radius` flag to the example, which the implementation needs anyway to
  regenerate the table.
- **Cost.** One release build of `akita-sis-estimator` with `zk,parallel`,
  then one search of at most 20 ranks per dimension with ADPS16 early stops.
  The W5e commit (`384bde87d`) records no wall time, so the search time is
  unknown.
- **Why it was not run here.** The machine was at load average 29 with test
  bars running, so no build was started.

P8 and P7 need no new cells, because existing ones certify them:

- **P8**: 2^40.67 at d = 64, and 2^39.95 at d = 128 and 256.
- **P7**: 2^37.67 at d = 64, 2^36.95 at d = 128 and 2^36.52 at d = 256.

## 4. Lever κ₁ (Q3): dead

- **Smallest ℓ₁ at equal entropy.** The family needs 2^128.06 elements after
  the operator-norm filter. Today's filter keeps 2^−2.09 of the unfiltered
  2^130.15, so assume about 2^130.2 unfiltered.
  - Over all of Z^64, `#{v : ‖v‖₁ ≤ κ} = Σ_j C(64,j)·2^j·C(κ,j)` first passes
    2^130.2 at κ = 45 (2^131.5; κ = 44 gives 2^129.9). No family of that size
    has max ℓ₁ below 45: at most log₂(53/45) = 0.24 bits.
  - The best fixed-composition family over {±1, ±2, ±3} has ℓ₁ = 49
    (27 × ±1, 8 × ±2, 2 × ±3; 2^130.4): 0.11 bits.
  - The best over {±1, ±2} has ℓ₁ = 52: 0.03 bits.
  - The larger coefficients also raise ‖c‖₂ and with it Γ, which lowers
    α_max(D) and so raises rejection.
- **A smaller family is not an option.** 2/|𝓒| is a knowledge-error term,
  and D27 already puts the response proof at 2^−119.7. A family of 2^121
  would buy κ ≈ 41, or 0.37 bits, while weakening the c round to the level
  of the weakest component.
- **The difference set.** Γ_Δ = 2κ₁ is attained: c′ = −c is in the family,
  and ‖c − (−c)‖₁ = 106. For a fixed c and a uniform c′ from the unfiltered
  family, the exact distribution of ‖c − c′‖₁ has mean 76.5, and
  Pr[‖c − c′‖₁ = 106] = 2^−26.5, Pr[≥ 100] = 2^−15.3, Pr[≥ 90] = 2^−5.8
  (computed by exact enumeration over the composition classes).
- **Restricted extraction does not help.** An extractor that only uses
  challenge triples with pairwise ℓ₁ ≤ L (generalized special soundness)
  needs the adversary's accepting set to contain such a triple. A set with
  every pair at distance > L can have size up to about
  Pr[distance > L]·|𝓒|, which is ≥ 2^−26.5·2^128 even at L = 104. The
  knowledge error would then become about 2^−23, not 2^−127, to gain
  log₂(106/104) = 0.03 bits.
- **Invertibility is not the binding constraint.** Differences are units by
  [LS18, Cor. 1.2], since ‖c − c′‖∞ ≤ 4 < q^{1/4}/2. The norm factor Γ_Δ
  enters only through the size of Δ, never through invertibility.

## 5. Lever on the extraction argument (Q4): 0.09 bits of slack

The radius `4Γ_Δ·B_M` comes from two steps (`05_security.tex:186-190`,
`02_preliminaries.tex:124-136`):

1. **Three-challenge extraction.** Δ = (c₁ − c₂)(z₃ − z₁) + (c₁ − c₃)(z₁ − z₂),
   bounded by Γ_Δ·2B + Γ_Δ·2B.
2. **Uniqueness of weak openings across ρ-leaves.** ρ′(ρs̄) − ρ(ρ′s̄′), at
   2Γ_Δβ with β = 2B.

What can and cannot be tightened:

- **Box diameter instead of twice its magnitude.** 𝓑_cert is
  [−8u, 7u], so |z_i − z_j| ≤ 15u = 16^D − 1, not 2B = 16u. Both steps then
  give 2Γ_Δ(16^D − 1): a factor of 15/16, 0.093 bits. Lemma O7.1 (§7.1) states
  it. This is the only slack in the bound itself.
- **Centering does not help.** Rewriting step 1 as
  Δ = (c₁ − c₂)z₃ + (c₂ − c₃)z₁ + (c₃ − c₁)z₂, whose coefficients sum to zero,
  lets each z be centered at the box's midpoint. That gives
  3Γ_Δ·(16^D − 1)/2, 0.42 bits below step 2. But step 2 needs
  2Γ_Δ(16^D − 1) regardless, because each z appears there with its own full
  coefficient, so the binding radius is the larger of the two. The centered
  step 1 changes nothing.
- **Choosing the best pair of each triple.** Picking, from each triple, the
  pair with the smallest ‖c_i − c_j‖₁ bounds that ℓ₁ by at most about
  (2·Σ_j(max − min))/3 ≤ 100 against 106. That is 0.08 bits, and it needs a
  new argument.
- **The norm choice is right.** The Euclidean alternative has Γ_Δ = 36 but
  multiplies the bound by √(input width). For A_a at 65,536 coefficients the
  ℓ₂ radius is about 2^53.3, already past Akita's ℓ₂ table (§5.5.2).

Verdict: 4Γ_Δ B is tight for this proof up to 15/16.

## 6. Parameter points: ranks, sizes, residual work (Q5)

**Units.**

- **Residual work.** The NTT-domain cost of the residual's `N·z` products, in
  R_64-product units: `(rank_A·W_A + rank_B·W_B)·d/64`. The same quantity ×64
  is the residual setup in field elements, and it reproduces Akita's 26.2M
  and ≈7.1M.
- **Normalization.** Work is normalized to today's value for the same row.
  The three rows (dense 14:1, 31:1 + 14:1, 36:1) agree within ±0.05, and the
  table gives the 36:1 headline.
- **u_a.** u_a is n = rank_B·d coefficients. Raw it is 16 B per coefficient;
  compressed (O4) it is ⌈n/512⌉ chunks of 128 B (W5c's split chain, as V6
  inferred it).
- **t_p.** t_p is rank_p·64 coefficients, in ⌈·/512⌉ chunks.

| point | d | ranks A_a / [B\|B′] (36:1) | certified? | n | u_a raw → compressed | A_p rank (d = 64), t_p chunks | residual work | residual setup (36:1) |
|---|---|---|---|---|---|---|---|---|
| P10 today | 64 | 19 / 19 | yes | 1,216 | 19,456 B → 384 B | 18, 3 | 1.00 | 26.2M |
| P10 + O6 | 128 | 10 / 10 | yes | 1,280 | 20,480 B → 384 B | 18, 3 | 0.54 | 14.1M |
| P10 + O6 | 256 | 5 / 5 | yes | 1,280 | 20,480 B → 384 B | 18, 3 | 0.27 | 7.1M |
| **P9** | 64 | 16 / 16 | est. | 1,024 | 16,384 B → 256 B | 14–15, 2 | 0.76 (0.72–0.77 across rows) | 19.9M |
| **P9 + O6** | 128 | 8 / 8 | est. | 1,024 | 16,384 B → 256 B | 14–15, 2 | 0.38 | 10.0M |
| **P9 + O6** | 256 | 4 / 4 | est. | 1,024 | 16,384 B → 256 B | 14–15, 2 | **0.19** | 5.0M |
| P8 | 64 | 15 / 15 (14 / 14 on the smaller rows) | yes, upper | 896–960 | 14,336–15,360 B → 256 B | 14, 2 | 0.60–0.69 | 18.1M |
| P8 + O6 | 128 | 7 / 7 | yes | 896 | 14,336 B → 256 B | 14, 2 | 0.31 | 8.1M |
| P8 + O6 | 256 | 4 / 4 | yes | 1,024 | 16,384 B → 256 B | 14, 2 | 0.19 | 5.0M |
| P7 | 64 | 13 / 13 | yes, upper | 832 | 13,312 B → 256 B | 12, 2 | 0.49–0.55 | 14.4M |
| P7 + O6 | 128 | 6 / 6 | yes | 768 | 12,288 B → 256 B | 12, 2 | 0.24 | 6.3M |
| P7 + O6 | 256 | 3 / 3 | yes | 768 | 12,288 B → 256 B | 12, 2 | 0.12 | 3.1M |

"yes, upper" means the certified cell lies above R_Σ, so the true rank may be
one lower. A_p stays at d = 64 under O6, because O6 moves only the residual.

**Reading the table.**

- **O7 alone never beats O6.** The best O7 point alone (P9, 0.76) is
  worse than O6 at d = 128 (0.54) and far worse than O6 at d = 256 (0.27).
- **O7 compounds with O6 at every d.** At d = 256, P9 takes the residual
  from 0.27 to 0.19, a 1.4× cut, with rejection still 2^−9.2. At d = 256, P8
  buys no more than P9, because rank 3 at d = 256 needs a radius of about
  2^37 or below. P7 reaches rank 3 (0.12, another 1.6×).
- **Bytes.**
  - **Before O4**, P9 cuts raw u_a by 3,072 B (19,456 → 16,384), which is
    10% of today's +28.8 KB Δ.
  - **After O4**, u_a drops from 3 to 2 chunks (−128 B). Every point also
    takes t_p from 3 chunks to 2 (−128 B), half of O5's pad target.
  - **O5's one-chunk t_p is impossible at these radii.** It needs
    rank_p·64 ≤ 512, i.e. rank 8 at d = 64, which the tables give only near
    2^28.7 (about D = 5).
- **e and the response proof.**
  - **How much e shrinks.** The z digits are most of e: 87.8M of 90.1M cells
    at the jolt-zk headline (`whiteout-sigma.md` §5). P9 removes one tenth of
    the Gaussian z digits, P8 two tenths and P7 three tenths. Today's Jolt
    rows have 13.1M, 20.0M and 39.5M live digits at ν = 24, 25 and 26 (V6).
  - **ν drops when e crosses a power of two.** ν steps down when e crosses
    2^{ν−1}. If the z digits are about 90% of e, P8 brings fib 2^16 below
    2^24 and fib 2^20 below 2^25, while P9 crosses neither. Each step down
    halves the response prover's domain, the dominant prover RSS term
    (5 GiB at fib 2^20).
  - **This is an estimate.** V6 does not log e's z share.

**Estimated verifier share.** O0 has not measured the residual's share f of
the Whiteout verifier Δ. If the residual were all of it (f = 1, the upper
bound that Akita's W5e harness suggests), the fib 2^20 Δ of +240 ms would
become:

| point | Δ at f = 1 |
|---|---|
| O6 d = 256 | +65 ms |
| P9 + O6 d = 256 | +46 ms |
| P7 + O6 d = 256 | +29 ms |
| P9 alone (d = 64) | +182 ms |

In general Δ′ = Δ·(1 − f·(1 − work)). V6's 1-thread verify equals its
16-thread verify, and O3's parallelization multiplies all of these.

## 7. Security deltas (Q6)

Each point changes three numbers and no theorem: `BINDING_RESPONSE_DIGITS`,
`ALPHA`, and the matrix ranks that follow from them.

### 7.1 Lemma O7.1: binding at the box diameter (optional)

**Lemma O7.1.** For each matrix M that the Σ-protocol opens, let δ_M be the
largest coordinatewise diameter `max 𝓐_t − min 𝓐_t` of the certified
alphabets 𝓐_t on M's input coordinates t. Write "the commitments bind" for
`MSIS(M, 2Γ_Δ·δ_M)` being hard for each M. Then Lemma `wo:lem:sigma-extract`
holds with "norm at most 2B_M" replaced by "norm at most δ_M", and
Thm `wo:thm:knowledge` holds with radius 2Γ_Δ·δ_M in place of 4Γ_Δ B_M.

*Proof sketch.*

1. For z_i, z_j ∈ 𝓑_cert, every coordinate of z_i − z_j lies in
   [min 𝓐_t − max 𝓐_t, max 𝓐_t − min 𝓐_t], so ‖z_i − z_j‖∞ ≤ δ_M on M's
   input. (Norms are of centered integer representatives. Every bound below
   is under q/2, so integer and R_q arithmetic agree.)
2. In the proof of `wo:lem:sigma-extract`, Δ = (c₁ − c₂)(z₃ − z₁) +
   (c₁ − c₃)(z₁ − z₂). Since ‖ρv‖∞ ≤ ‖ρ‖₁‖v‖∞ ≤ Γ_Δ‖v‖∞ for every challenge
   difference ρ, ‖Δ‖∞ ≤ 2Γ_Δδ_M. The rest of that proof is unchanged, and
   (c₁ − c₂)s̄ = z₁ − z₂ has norm ≤ δ_M.
3. `wo:lem:weak-unique` at β = δ_M gives a kernel vector of norm ≤ 2Γ_Δδ_M,
   and the knowledge proof uses it unchanged.

∎

**Its value here.** For D balanced radix-16 digits, δ = 16^D − 1 = (15/8)·B(D),
so 2Γ_Δδ = (15/16)·4Γ_ΔB(D). Public segments have fewer digits and a smaller
δ, so δ_M is set by the Gaussian digits.

**Recommendation.** Adopt the lemma only if the generator shows a rank change
at 2^43.73 that 2^43.82 does not give. Otherwise keep the paper's form. A
second radius formula is not worth 0.09 bits.

### 7.2 Binding and knowledge soundness

**What changes.** Only the radius in "the commitments bind"
(`05_security.tex:33`), and with it the ranks of A_a, [B_a|B′_a] and A_p.

| point | radius | ranks |
|---|---|---|
| P9 | 15,539,764,339,008 ≈ 2^43.82 | as in §6, each to be certified at 128 bits by `catalog_security --check` (Akita's `sis_occurrences` already adds both residual occurrences at R_Σ) and by jolt's A_p test |
| P8 | 2^39.82 | — |
| P7 | 2^35.82 | — |

**What does not change.**

- **The theorem's statement and error.** Thm `wo:thm:knowledge` keeps
  `ε_int = ε_{≤k} + ℓ/q + 2/|𝓒| + ε_cmp`, with ℓ = 14 and |𝓒| = 2^128.06.
  None of these depends on α or D.
- **ε_cmp.** D27's ledger depends on D only through ν. ν can only fall, and
  each step removes (1 + 1 + 3)/q from the τ_pad, binary and fused rows: at
  the headline, 310/q → 305/q. The total stays ≈ 2^−119.7.
- **The other matrices.** Lemma `wo:lem:randomized-base`, the matrices before
  the cut, Akita's matrices in the response proof (priced by Akita at their
  own bounds) and A_D (digit-difference bound) are untouched. P9 only
  shrinks e.
- **The public digits.** The public coordinates' D_pub = 3 still covers
  κ₁·8 = 424 ≤ L(3) = 1,911.

### 7.3 Completeness (Thm `wo:thm:completeness`)

**The hypothesis.** `‖cs‖_σ ≤ 1/α` holds at every α, because the width is
derived from α and checked exactly (`SigmaParams::new_with_alpha`).

**The bound.**

| point | M | per-attempt failure ≤ 1 − (p_𝓑 − ε)/M | T_att | all-rejected bound | E[attempts] ≤ M/(p_𝓑 − ε) |
|---|---|---|---|---|---|
| P9 | e^{13.78/8192 + 1/(2·8192²)} = 1.001683 | 2^−9.22 | 14 | 2^−129.0 | 1.0017 |
| P8 | 1.02728 | 2^−5.23 | 25 | 2^−130.9 | 1.027 |
| P7 | 1.5397 | 0.3505 = 2^−1.51 | 85 | 2^−128.5 | 1.540 |

### 7.4 Zero knowledge (Thm `wo:thm:zk`)

The hypotheses are unchanged: ‖cs‖_σ ≤ 1/α, which the exact check enforces;
a fresh mask per attempt; a secret nonce per attempt; and randomness entropy
≥ image + 2λ + 8 bits. r_a and r_p are re-sized to the new ranks by
`required_randomness_columns` and the A_p fixed point.

Each term, with Q_tot ≤ 2^64 and about 2^22.5 Gaussian samples per attempt at
the headline:

| term | P10 (today) | P9 | P8 | P7 |
|---|---|---|---|---|
| T_att·ε/M | 2^−133.7 | 2^−133.2 | 2^−132.4 | 2^−131.2 |
| (T_att·Q_tot + T_att²)/2^256 | 2^−188.7 | 2^−188.2 | 2^−187.4 | 2^−185.6 |
| δ_LHL (14 slices) | 2^−129.2 | 2^−129.2 | 2^−129.2 | 2^−129.2 |
| sampler precision, 2^−183.3 per sample × samples × T_att | ≤ 2^−157 | ≤ 2^−157 | ≤ 2^−156 | ≤ 2^−154 |
| PRG, n_ch ≤ 405·T_att | t·2^−244 | t·2^−243.5 | t·2^−242.7 | t·2^−241 |
| **statistical total** | **2^−129.1** | **2^−129.1** | **2^−129.0** | **2^−128.9** |

The paper's α = 16 point had 2^−128.8. Every point stays above 128 bits.

Constant time (§2.8), H2's nonces and the bounded loop are unchanged. With
T_att larger, more rejected attempts are possible, but §2.8 already covers
every attempt.

### 7.5 Rejection-arithmetic and sampler preconditions

- **Rejection arithmetic.** `check_rejection_arithmetic`'s numerator
  `Σ(2LV + V²)·len < 2^126` falls with L, so every point passes. The width
  `k ≥ 1` and the box are below 2^62.
- **The sampler.** Its width range (≤ 2^58) holds, and it ran at α = 16 in W4.

## 8. If P9 is accepted

Scope: one ticket in each repository, plus a paper delta.

1. **Generator gate (§3.3).** Run at R_Σ(9) for d ∈ {64, 128, 256}, plus the
   diameter form. Promote only if rank 4 holds at d = 256 for every residual
   row. Otherwise stop, or apply P9 only at d = 64 or 128.
2. **jolt-zk.** Set `ALPHA = 1 << 13` and `BINDING_RESPONSE_DIGITS = 9`;
   `BOX_FAILURE_INVERSE` is unchanged.
   - **Headline test.** `tests/whiteout_params.rs` re-pins it: D = 9, the
     width, and ⌊L²/k²⌋ = 76.
   - **The table guard catches a stale radius.**
     `cells_match_the_certified_csv_and_the_current_radius` in Akita fails
     until the table is regenerated.
3. **Akita.**
   - Regenerate `sigma_binding_infinity_width.csv` and `generated.rs`, which
     gives a new digest; add a `--radius` flag to the example or keep reading
     jolt-zk.
   - Re-pin the planner's checked-in ZK rows to the new ranks, the
     admission tests, and §5.5.2's table.
   - The clear catalogs are unchanged (C0): the Σ table has its own digest.
4. **jolt.**
   - **A_p.** Re-pin `whiteout_ranks_sit_at_the_tables_minimum` and
     `whiteout_pad_rank_steps_at_the_sigma_binding_brackets`; the brackets
     move.
   - **Records.** Re-bless the census and the FS inventory, since e and the
     ranks change.
   - **Checks.** `whiteout_e2e` and `catalog_security --check` must hold
     ≥ 128 bits at every Σ occurrence.
5. **Paper delta (D33).** It updates D29's 19 × R_64 and §5.1's table, and
   records the review decision that P9 revisits.
6. **Measurement.** Use the O0 benchmark on all three workloads, and record
   the retry rate, which should be 0 in ≈ 15 runs, and Σ time.

## 9. Not verified

- **P9's ranks are interpolated, not certified.** They need the generator
  run (§3.3). The margin at n = 1,024 is 3% at the pessimistic end.
- **The residual model is not the planner.** It is mine, checked against the
  checked-in W5e rows and Akita's d = 256 estimate. It does not run the
  planner, which may pick other shapes (offload, block count) at new ranks.
- **Verifier time is not measured.** Verifier shares are upper bounds
  (f = 1) pending O0; the R_64-product work model is Akita §5.5.2's, not a
  measurement.
- **T_att and the sampler numbers come from formulas.** T_att is computed
  in floating point. The code's fixed-point `max_attempts` may give one more
  attempt. Sampler ℓ and per-sample costs use §5.1's formula, not new runs.
- **e's z share and the ν steps are estimates.** V6 does not log the z share
  for Jolt rows.
- **The headline numbers come from jolt-zk's headline table.** α_max and
  the 76/45 headroom are taken from it (rank-5 R_256 era, ΣN = 6,597,888),
  not from today's rank-19 Jolt rows. Because the rejection rate depends only
  on α, only the instance-size headroom depends on this.
- **The difference-norm distribution uses the unfiltered family.** The
  operator-norm filter keeps 2^−2.09 of it and could shift the tail.
