# W4-T2 — limb table (non-native BN254 Fq arithmetic for the Dory deferred check)

Lane journal. Code: `crates/jolt-wrapper/src/limb_table/`, tests `crates/jolt-wrapper/tests/limb_table_*.rs`
(+ `tests/common/mod.rs`). Checkpoint commits: `eea2a0d27` (fixed layout), `a6e09751d` (relation v2 + terms export,
this entry).

## What the table proves

The Dory deferred check `e(A1, E2_fin)·e(H1, B2)·e(A3, H2)·e(A4, Γ2_0) = Σ_k s_k·X_k` (`dory-offload-study.md` §1.4,
regrouped so that every scalar is a named `DoryScalar` wire or the constant one; `dory.rs::FlattenedCheck`), as
`2^18` rows of `z ≡ Σ_s κ_s·x_s·y_s (mod q)` over BN254 `Fq` with the operands being other rows. Rows are grouped in
16-row cells; GT elements use rows `0..12` of a cell, four-row G1 ops ride in rows `12..16`, eight-row G2 ops pack two
per cell, Miller steps use 32 rows. The Straus schedule (radix-16 signed digits, 64 windows, tables `X^1..X^8` /
`(j−8)·P + Z0`) is laid out so that every operand relation is a bit-field kernel of the row index (`layout.rs`).

## Protocol (post 00:42 steer: terms, zero-round wiring, phase-separated columns)

Committed columns (`relation::col`, packing order = index order; `export::columns()` carries the phase per column):

| phase | columns | count | col range |
|---|---|---|---|
| 1 | `z` chunks 16, `k' = k + 2^267` chunks 17, carry chunks 4×7 (`c_i + 2^111`) | 61 | 0..61 |
| 1 | digit bits `zero, neg, e0, e1, e2`, digit value `D`, lookup multiplicities `m_pos, m_neg`, range multiplicity | 9 | 61..70 |
| 2 (after `ξ, α, β, γ, φ, fp, …`) | operands `X_0..X_21` (= `κ·Z_ξ(src)`), `Y_0..Y_21` (= `±Z_ξ(src)`) | 44 | 70..114 |
| 2 | LogUp range helpers (22 groups of 3 over the 61 chunks + 5 digit bits), range inverse table `1/(α−x)` | 23 | 114..137 |
| 2 | lookup read helper `h`, table helpers `g_pos, g_neg`, fingerprints `f_pos, f_neg` | 5 | 137..142 |
| VK | `pin`, `pin_limb_0..2`, `free` | 5 | 142..147 |

Phase 1 = 70 columns, phase 2 = 72, VK = 5 (`col::PHASE1_END`, `col::PHASE2_END`, `col::VK_END`). The prover also
carries 11 public columns (`eq_tau`, copy kernel, `sel/is_gt/is_g1/is_g2`, `S0`, `coord`, constancy, `small`, `id`)
that the verifier recomputes in closed form (`PublicEvals`).

`Z_ξ(v) = Σ_a ξ^a·limb_a(v)` (96-bit limbs recomposed from the chunks, affine in the chunk columns).

Row member (`relation.rs`, 18 rounds, degree 5, one member, input 0), per row `x` with `eq(τ,x)`:
- limb identity on non-free rows: `Σ_s X_s·Y_s − z(ξ) − k(ξ)·q(ξ) − (B − ξ)·C(ξ) = 0` (`B = 2^96`, `C` the carry
  polynomial of degree 3; exact over the integers by the chunk range checks: `Σ|κ| ≤ 2^7` (measured max 102) keeps
  every coefficient below `2^201` and every carry below `2^105`);
- pins `pin·(z(ξ) − pin(ξ)) = 0`; LogUp range groups `h_g·Π(α − c_i) = 1`; digit booleanity, `(1−neg)e0e1e2 = 0`,
  `D = sel·(1−zero)(1−2neg)(1+e)`;
- operand lookup, reading side: `h·(β + key + φ·F) = sel` with `F = Σ_{s<n} fp^s·Y_s` (`n = 22` GT, `2` G1, `4` G2),
  GT key `(1−zero)(S0 + 16e) + zero·(one_row + c) + 2^18·neg`, EC key `S0 + stride·D` (`16` G1 cells, `8` G2 half cells);
  table side `g_±·(β + row + [2^18] + φ·f_±) = m_±`; `f_pos/f_neg` are the ±-signed slot fingerprints of every table
  entry row (copied from the entry's coordinates by public kernels; the `f_neg` sign is GT conjugation, EC rows never
  read `f_neg`);
- `Σ_x`-only identities (no `eq`): range LogUp `Σ h_g·e_{2,g} = Σ mult·inv`, lookup LogUp `Σ h = Σ g_pos + Σ g_neg`,
  copies `Σ_x eq(τ,x)·Σ_i β_i·C_i(x) = Σ_v B(v)·Z_ξ(v)` with `B(v) = Σ_i β_i·K_i(τ, v)` (every fixed operand and the
  fingerprint columns; looked-up `Y_s` masked out by `1 − is_gt − is_g1·[s<2] − is_g2·[s<4]`), digit constancy
  `Σ_x W(x)·Σ_b β'_b·bit_b(x) = 0` with `W(x) = eq(τ,x)cst(x) − eq(τ,x+1)cst(x+1)`.

Digit-link member (`digit_link.rs`, 18 rounds, degree 2): `Σ_x ω(x)·D(x)` with `ω = ρ^{kd}/mult(kd)·16^{63−w}` on
each op's first slotted row; input claim `Σ_{kd<K} ρ^{kd}·s_kd + ρ^K` (`K` named wires in the published order,
`ρ^K` for the constant-one bases). W5 compares against R's `Σ_s ρ^s·scalar_s`. Its final relation is the single
linear term `ω̃(r)·D(r)` (`digit_link::link_term`).

Verifier (`lookup::public_evals`, `wiring::copy_kernel_eval`, `verifier::Evaluator`): every public multilinear at
the stage point in closed form — `eq(τ,r)`, kernels `Σ_i β_i K_i(τ,r)` (one memoized evaluation per distinct row
field group; eq tables per block `[0,6) [6,12) [12,18)` of the row index shared by every field; kernel values summed
per weight index and multiplied once), `sel/is_*` (family indicators), `S0` and `c` (field moments), constancy
kernels, `small`, `id`; then `relation::terms()` — the whole final relation as
`Term { coefficient, factors: Vec<AffineForm> }` over the claimed column evaluations (`terms.rs`).

### Terms export (W5 interface)

`terms.rs`: `ColumnId(u32)` (index into `export::columns()`, i.e. `relation::col`), `AffineForm { constant, weights }`,
`Term { coefficient, factors }` — the same shape as W5's `stream::types` (`ColumnId { group, slot }` there; the
adapter maps my column index to the packed `(group, slot)` through the export list). `RowRelation::terms(&PublicEvals)`
returns **131 terms, max degree 4** (fibonacci profile and every profile: the count depends only on the relation):
1 linear term (all linear constraints incl. `eq·` limb-identity linear part, pins, booleanity linear parts, LogUp
sums), 22 operand products `X_s·Y_s`, 22 range groups × (1 + 3) factors-as-affine-forms `γ + x_i`, digit
booleanity/value products, lookup read/table products, inverse-table term. `eq(τ,r)` and every public evaluation are
folded into the coefficients. Every term's factors are affine in the *claimed* columns only (virtual operand limbs
are affine forms over the chunk columns), so the batched stage-A final claim equals `Σ_t coeff_t·Π_j L_{t,j}(v)` —
tested against the prover's native row-relation evaluation (`limb_table_e2e`).

`export.rs`: `columns()` (147 committed/VK columns with `Phase::{One, Two, Vk}`), `members()` (row 18 rounds degree
5; digit link 18 rounds degree 2), `ClaimedColumns::assemble(...)` with `phase_one()/phase_two()/vk()` slices.

## Digit-base order (for W6-RT / W5)

`kd` = index of the wire in the order passed to `schedule::build` (the adapter passes `DoryLinks.scalars` order and
checks it equals the set `FlattenedCheck::wires()`), `kd = K` for the constant one. A wire used by several MSMs
(e.g. `Alpha(j)` in `C+`, `E1+`, `E2+`) has `mult(kd)` digit sets; `ω` divides by it. Window `w` holds digit index
`63 − w` (weight `16^{63−w}`), digits are centered radix-16 (`d = j − 8 ∈ [−8, 7]`) with `Σ_i 16^i d_i = s` exactly.
Digit columns: `zero, neg, e0, e1, e2` (`d = (1−zero)(1−2·neg)(1+e)`, `e = e0 + 2e1 + 4e2`), constant across an op's
slotted rows (constancy identity), `D` committed as the product.

## Numbers (fibonacci profile σ = 11, N = 42; `random_values`)

| item | value |
|---|---|
| rows used | 189,586 of 262,144 (families 186,502; inputs/constants 3,084) |
| GT online ops | 9,216 `gt_mult` + 192 `gt_sq` + 64 `gt_mult0` + 63 `gt_sq0` (+ 1,015 table ops) |
| digit ops (`DigitOp`) | 14,208 (GT 9,280; G1 2,624; G2 2,304) |
| max slots / max Σ\|κ\| | 22 / 102 |
| pins / input rows | 12,668 / 1,526 |
| fixed copy pieces (kernels) | 2,490 (+ fingerprint kernels), 117 families |
| terms / max degree | 131 / 4 |
| verifier `Fr` mults (`VerifierObserver::fr_mul`) | **9,511** = public evals 8,884 + digit link 627 (budget 10,000) |
| largest kernel family | `gt_table` ≈ 800 (44 pieces), then `gt_mult` ≈ 450, `g2_table_up` ≈ 300 |

Verifier cost anatomy (before/after this checkpoint): 13,526 → 9,511. Eq tables 5,232 → 801 (block-aligned
pieces instead of per-field tables), per-map weight multiplications 2,610 → 88 (bucket per weight index), digit-link
`Σ_w eq(r_w,w)16^{63−w}` in closed form `16^{63}·Π_i(1 − r_i + r_i·16^{−2^i})` (1,123 → 627). Remaining: 2,492
distinct `eq(τ_u,c)·eq(r_v,t)` pairs, 1,592 wide-field piece products, 712 field-group products.

σ = 8, n = 5 real opening (`limb_table_e2e`, synthetic Dory opening through the adapter): layout + witness 0.7 s,
columns 0.4 s, row member 2.8 s (debug build, 2^18 rows × 158 columns), verifier 8,047 + 368 `Fr` mults.

## Tests (all green at `a6e09751d`, `cargo nextest run -p jolt-wrapper --test limb_table_*`)

- `limb_table_program`: program reproduces the deferred check bit for bit on a real opening; fibonacci profile fits
  2^18 rows; `evaluator_matches_kernel_mles` (2,610 kernels: Evaluator = factored MLE = Σ edges, grouped evaluation =
  Σ kernels); `kernels_match_program_rows` (every copy-kernel edge ↔ program slot, fingerprint edges ↔ table reads,
  no duplicate edges, attributed by family); `verifier_arithmetic_within_budget_at_fibonacci_profile` (≤ 10k).
- `limb_table_miller`: Miller cells vs arkworks step by step.
- `limb_table_e2e`: every constraint vanishes on the honest witness; both members driven with random challenges,
  round checks, verifier closed forms == prover public columns at `r`, `Σ_t coeff·Π L(v) == final claim`, digit-link
  input `Σ ρ^k s_k + ρ^K`, `ω̃(r)` closed form; tamper suite (flipped chunk, wrong digit bit, broken copy, replaced
  looked-up operand) rejected by the term check.

## Bugs fixed this checkpoint (for reviewers)

- Families whose row set was only implied by a `Table` elem factor (`glue`, `frobenius`, `ml_psi*`,
  `ml_dbl_after_add`, `mg_sq_after_*`, `mg_ell_dbl*`, `ma_ell*`, `fe_sq`, `fe_mul_*`) leaked their constant/`ONE`
  operand kernels onto every other cell: `place()` now restricts scattered families to the placed cells
  (`Factor::weight(CELL, mask)`), and the Miller/final-exponentiation domains carry explicit step/slot restricts.
- `TableRead.conjugated` was computed from the absolute coordinate (`≥ 6`), flagging G1/G2 `x, y` (coords 14, 15) as
  conjugated while the kernel used the template coordinate; the fingerprint map's conjugated weight is now the one
  owner of the sign.

## Item 4 (steer): in-table G2 subgroup and compressed-point sign checks — costed, not implemented

- Subgroup membership of the 3σ+1 = 34 committed G2 points (fibonacci profile): the one-x-multiplication test
  `ψ²(P) + ψ([6x+3]P) + [6x+1]P = 0` (equivalent to arkworks' `[6x²]P = ψ(P)` on `E'(Fq2)` since
  `gcd(3x²+3x+2, h2·r) = 1`, checked numerically). Cost per point: NAF(x) has 63 digits / 24 nonzero → ≈ 98
  half-cells (dbl + add ops) + ψ/ψ² (2 cells) ≈ 49 cells; 34 points ≈ 1,666 cells ≈ 26.7k rows. With the dyadic
  packing the layout uses (`p: 6 bits`, `t: 7 bits`) it needs 64 cells/point = 2,176 cells; free full cells ≈
  1,660, fragmented (largest contiguous block 640 cells) → **does not fit at 2^18 without layout compaction**
  (or 3-NAF ladders sharing the doubling chain across points). Uncovered today: all 34 proof G2 points (`E2`
  chains, `B2` halves); the VK G2 constants are trusted.
- Compressed-point sign flags: `y > −y` in `Fq` (G1) / lexicographic `(c1, c0)` in `Fq2` (G2) via 16-bit-chunk
  comparison rows: ≈ 6 chunk-comparison rows + 1 borrow row per Fq coordinate → G1 points 4 (A1, A3, A4 + H1
  pinned) ≈ 28 rows, G2 34 points × 2 coords ≈ 476 rows, ≈ 32k rows including the LogUp range helpers on the
  comparison chunks — fits in the spare rows. Not implemented; the compressed-flag columns would join phase 1.

## Open for the parent

- `tests/perf1_profile.rs` (W5, `a346cf32e`) imports the removed `limb_table::wiring::Wiring` and the old
  `RowSumcheck::new`/`Slot` shape; `cargo clippy -p jolt-wrapper --all-targets` fails on it. Not mine to edit; the
  replacements are `RowSumcheck::new(&relation, &columns)` over `col::WIDTH` columns (see `limb_table_e2e::matrix`)
  and `Slot { x, y, kappa, y_sign }`.
- The fibonacci 2^18 fixture and the 2^22 synthetic opening still run only the native-check path
  (`program_reproduces_the_deferred_check_on_a_real_opening`); the member e2e runs at σ = 8, n = 5 (same code path,
  fixed 2^18 rows — the row count does not depend on σ).
