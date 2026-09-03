# W4-T2 — limb table (non-native BN254 Fq arithmetic for the Dory deferred check)

Lane journal. Code: `crates/jolt-wrapper/src/limb_table/`, tests `crates/jolt-wrapper/tests/limb_table_*.rs`
(+ `tests/common/mod.rs`). Checkpoint commits: `eea2a0d27` (fixed layout), `a6e09751d` (relation v2 + terms export),
`93f166b80` (style pass), `ffd8cfb2a` (milestone 2 + review #1 fixes), then this checkpoint (decision (A) offsets,
ψ-chains, sign flags, stream adapter, execution-derived budget, σ = 11 fixture, module split).

## What the table proves

The Dory deferred check `e(A1, E2_fin)·e(H1, B2)·e(A3, H2)·e(A4, Γ2_0) = Σ_k s_k·X_k` (`dory-offload-study.md` §1.4,
regrouped so that every scalar is a named `DoryScalar` wire or the constant one; `dory.rs::FlattenedCheck`), as
`2^18` rows of `z ≡ Σ_s κ_s·x_s·y_s (mod q)` over BN254 `Fq` with the operands being other rows. Rows are grouped in
16-row cells; GT elements use rows `0..12` of a cell, four-row G1 ops ride in rows `12..16`, eight-row G2 ops pack two
per cell, Miller steps use 32 rows. The Straus schedule (radix-16 signed digits, 64 windows, tables `X^1..X^8` /
`(j−8)·P + Z0`) is laid out so that every operand relation is a bit-field kernel of the row index (`layout.rs`).

Beyond the pairing equation the table pins, for every proof-derived input: canonical integer encoding of each
byte-linked `Fq` coordinate (review B3), on-curve membership of every G1/G2 point, norm one `x·conj(x) = 1` of every
raw GT input (B4), G2 subgroup membership of the two pairing inputs `E2_fin`, `B2` (ψ-chains), and the compressed-point
sign flag of every G1/G2 proof point (arkworks canonical `y > −y`).

## Decision (A): transcript-derived Straus offsets (`schedule/ec/mod.rs` module doc has the full argument)

Every Straus chain over a proof point needs a non-degenerate accumulator start `R` and table offset `Z0`. Both are
now derived from one wrapper Fiat–Shamir challenge `θ` drawn after phase 1a (no W4-R wire): per group one
fixed-base Straus chain computes `R = [θ]G` with constant offsets `R'' = G`, `Z'' = 9G` (`FIXED_TABLE_OFFSET`),
`Z0 = φ(R)` by the GLV endomorphism (`G1Config::ENDO_COEFFS`, `endomorphism_affine`; G2 alike), and every main chain
carries one extra base `Operand::Constant(−K)` with `Wire::Offset` whose digits are θ's, `K = offset_correction(g,
g_endo, n')` = `−(16^64·R'' + n'(16^64−1)/15·Z'')` so the θ-dependent correction is a fixed-base multiple of θ.

Soundness of the `+`-only (unguarded) adds in the θ chains: an exceptional add in window `w` of a chain over `n`
bases means `θ·(16^{w+1} + λ(nw + k ∓ 1))·G` equals a fixed point — a single root in `θ`; a doubling of the identity
means `16^w + λnw ≡ 0 (mod r)`; both are swept for `n ≤ 64`, `w < 64` by the unit test `offsets_are_nondegenerate`.
The R-chain itself runs in the integer regime: its multipliers stay in `(16^{w+1}, 2.07·16^{w+1}) < r` for `w ≤ 62`
(`fixed_base_multipliers_stay_below_the_modulus`). θ's digits are prover-committed and bound by the digit link with
verifier-known right-hand side (`ρ^{K+1}·θ`, below). Negatives: `point_crafted_for_fixed_offsets_is_rejected`
(a point `P = −(d−8)^{-1}·G` crafted against the old constant offsets no longer degenerates the chain — the crafted
add is now guarded by θ), `twist_point_outside_g2_is_rejected` (order-10069 / twist-torsion pairing input).

Add sites (B5) — enumerated in the `ec` module doc: (1) θ chains: exceptional ⟹ one root in θ; (2) main chains:
`acc + T[d]` with `acc = 16·acc_prev + …` starting from `R = [θ]G`: exceptional ⟹ `θ`-linear relation, one root;
(3) table `(j−8)·P + Z0` with `Z0 = φ([θ]G)`: exceptional ⟹ `(j−8)·P = ±φ([θ]G)`, one root per `j`; (4) ψ-chain
adds are the only ones over prover-chosen operands without θ — they use guarded 10-row adds
(`g2_add_guarded`: the row set is complete for `P = ±Q` and `P = O`), see below; (5) Miller-loop adds use the
constant `Q` bases and `E2_fin`/`B2` after subgroup membership — degenerate only if `[x]P = ±Q`, excluded by `P` and
`Q` being independent random points of a prime-order group (a fixed-`Q` collision has probability `2/r` over the
proof's `P`).

## G2 subgroup checks (`schedule/ec/psi.rs`, `Cells::PSI_CHAIN = 15872 + 256·chain + local`)

For each of `E2_fin`, `B2`: `ψ²(P) + ψ([6x+3]P) + [6x+1]P = 0` (equivalent to `[6x²]P = ψ(P)`, checked numerically in
`psi_identity_holds_on_g2_only`). NAF of `6x+1` (top digit `2^64`, `naf(u128)`): doublings `2^i·P` at half cells
`start+i`, guarded adds from the midpoint cell on (reading the previous partial sum from the previous cell's first
half and the power through `Factor::table(PSI_CL, PSI_LH, pairs) + same(CH)`), final `+2P` gives `B = [6x+3]P`; tail
cells: `t0` = `ψ²(P)` (rows 0–3, `g2_psi(false, false)`) and `ψ(B)` (rows 4–7), `t1` = guarded sum `S`, `t2` pins
`S + A = 0` (`g2_negation_pins`). 256 cells per chain (4,096 rows), 2 chains = 8,192 rows. Negative: an order-10069
twist point (outside `G2`) as pairing input is rejected by the pins.

## Sign flags (`schedule/ec/sign.rs`, `Cells::G1_SIGN = 15168 + b` row 0, `Cells::G2_SIGN = 14912 + b` rows 0–5)

`Source::Exact` rows (k = 0 enforced by `exact·k(ξ) = 0`, VK column `exact`) and `Source::Sign{of}` rows: exact
`Σκxy + (1−flag)·2^256 = z` with the committed `flag` column (phase 1b, boolean-checked); the limb identity gains
`+ exact·(1−flag)·2^64·ξ²` (`flag_xi`). G1: one sign row `y − (q+1)/2`. G2 (Fq2 lexicographic `(c1, c0)`): rows
`inv = InverseOrZero(y1)`, `z = y1·inv`, exact pins `y1(1−z) = 0`, `z(z−1) = 0`, exact `v = z·y1 + (1−z)·y0`, sign row
of `v`. `Layout.sign_rows: Vec<(InputElement, RowId)>` is the T1 link list. Flag semantics = arkworks canonical
`y > −y` (`sign_flags_match_arkworks_and_flips_are_rejected`). Uncovered points: only the VK constants `H1, H2, Γ1_0,
Γ2_0` (trusted); GT inputs are absorbed raw (no compression). 34 G2 + 4 G1 points → 38 cells (608 rows).

## Protocol (phases, columns, members)

Committed columns (`relation::Col`, packing order = index order; `export::phases()` carries `challenges_before`):

| phase | after | columns | count | col range |
|---|---|---|---|---|
| 1b | `θ` (after 1a) | `z`/`k'`/carry chunks (16-bit) 61, digit bits `zero, neg, e0, e1, e2`, digit value `D`, `m_pos, m_neg`, range mult, sign flag | 71 | 0..71 |
| 2a | `ξ, α` | operands `X_0..X_21`, `Y_0..Y_21`, range helpers 22, range inverse | 67 | 71..138 |
| 2b | `fp_root` | fingerprints `f_pos, f_neg` | 2 | 138..140 |
| 2c | `β, fp_combine, copy_root` | lookup read `h`, table helpers `g_pos, g_neg` | 3 | 140..143 |
| VK | — | `pin`, `pin_limb_0..2`, `free`, `exact` | 6 | 143..149 |

Stage A challenges: `τ` (18), `γ, λ, λ_lookup, constancy_root`. `Col::CLAIMED = 149`. Commit-before-challenge is
tested by `phases_commit_values_before_their_challenges` and the collision negative
`selected_operand_collision_for_a_guessed_fingerprint_root_is_rejected` (B2).

Row member (`relation.rs`, 18 rounds, degree 5, input 0) and digit-link member (`digit_link.rs`, 18 rounds, degree
2) are unchanged in shape from `a6e09751d` (see the git history of this file for the identity list); the digit-link
input claim is now `Σ_{kd<K} ρ^{kd}·s_kd + ρ^K + ρ^{K+1}·θ` (`Wire::One = K`, `Wire::Offset = K+1`,
`Layout.digit_bases = K + 2`; `stream::link_input_claim`).

## Stream adapter (W5 interface, `limb_table/stream.rs`, pattern `hash_table/adapter.rs`)

- `PHASE_CHALLENGES = [2, 1, 3, LOG_ROWS + 4]` (1b→2a: `ξ, α`; 2a→2b: `fp_root`; 2b→2c: `β, fp_combine, copy_root`;
  stage A: `τ`×18, `γ, λ, λ_lookup, constancy_root`); `T2Challenges { theta, row, rho }::from_challenges(theta,
  phase_slice, rho)` — `θ` is the wrapper challenge after phase 1a, `ρ` is R's link challenge.
- `commitment_phases(packing)`, `prover_group_count`, `vk_group_range`; `LimbTableKey::new(layout, packing, setup)`
  commits the VK columns once (`pinned_commitments(group_offset)`).
- `StreamColumns::new(&ClaimedColumns, &Columns, &Layout, packing, group_offset)` → `{columns, ids, group_count,
  vk_groups}`: phase column lists in packing order with kinds (`Column::{U16, Bit, Fr}`), members with degree/offset.
- `Members::new(relation, matrix, layout, digit_values, rho)` → `{rows: RowSumcheck, link: LinkMember}`.
- `StreamTermExporter { layout, challenge_offset, theta_offset, rho_offset, columns, row_member, link_member }`
  implements `TermExporter`: relation via `RowRelation::new_with(observer)`, `public_evals`, `omega_eval`,
  `terms_with(observer)`, batching coefficients applied, link term appended. Digit link pairs with W6-RT's
  `DoryScalarLink` (R scalar cells `ρ^s`-weighted) through `link_input_claim(r_link_claim, ρ, θ, K)`.
- Verified against the members: `stream_exporter_terms_match_the_members` (175 terms incl. the link term).

## Verifier budget (execution-derived, review MAJOR)

`Evaluator<'o, O: TermObserver + ?Sized>` counts every `Fr` multiplication of the real verifier path (relation
construction incl. challenge powers, public evals, `terms_with`, link) —
`verifier_arithmetic_within_budget_at_fibonacci_profile` asserts `≤ 10,000` on the σ = 11, n = 42 layout through
`StreamTermExporter`: **9,930 Fr mults, 175 terms, max degree 4**. Levers that got there from 11,271: unrestricted
`same` factors as per-bit products; ranged shift/identity groups through an 8-state automaton (`shift_automaton`,
≤ 17 mults per bit) when `3·values > 17·width`; `field_sum` factored by 6-bit block; no linear folding in
`terms_with` (the fold cost more than it saved).

## Numbers (fibonacci profile σ = 11, n = 42, real 2^22 opening, `limb_table_e2e`)

| item | value |
|---|---|
| rows used | 199,783 of 262,144 |
| θ-offset chains | G1: 64 windows in the G1 lane + one 16-cell table (≈ 1,280 rows); G2: `Cells::G2_OFFSET_*` region 10048..10240 = 192 cells (3,072 rows) |
| ψ-chains | 2 × 256 cells (8,192 rows) |
| canonicality | 0 extra rows: 4 extra 16-bit chunk columns (`CANON_CHUNKS`) on the 1,526 data rows |
| sign flags | 38 cells (608 rows); GT norm-one 110 cells (1,760 rows) |
| terms / max degree | 174 row + 1 link / 4 |
| verifier `Fr` mults | 9,930 execution-derived (stream exporter); e2e path 9,007 + 541 link |
| layout + witness / columns / row member (debug) | 6.6 s / — / 2.8 s |
| σ = 8, n = 5 fixture | 174 terms, 8,614 + 365 `Fr` mults |

### PERF-4 — committed column widths and density (σ = 11, n = 42 witness, 2^18 rows)

| phase | column | cols | max bits (signed) | nonzero fraction |
|---|---|---|---|---|
| 1b | chunk | 61 | 16 | 0.34–1.00 |
| 1b | digit_bit | 5 | 1 | 0.09–0.23 |
| 1b | digit_value `D` | 1 | 4 signed (`[−8, 7]`) | 0.43 |
| 1b | lookup_mult_pos / neg | 2 | 11 / 4 | 0.065 / 0.051 |
| 1b | range_mult | 1 | 23 | 0.25 |
| 1b | sign_flag | 1 | 1 | 0.0005 |
| 2a | operand_x / operand_y | 44 | 254 (full `Fr`) | x 0.08–0.70, y 0.43–0.69 |
| 2a | range_helper | 22 | 254 | 1.00 |
| 2a | range_inverse | 1 | 254 | 0.25 |
| 2b | fingerprint_pos / neg | 2 | 254 | 0.076 |
| 2c | lookup_read / table_pos / table_neg | 3 | 254 | 0.52 / 0.065 / 0.051 |
| VK | pin, free, exact | 3 | 1 | 0.063 / 0.012 / 0.0006 |
| VK | pin_limb | 3 | 96 | 0.006 |

Dense full-width `Fr` columns: **72** (2a 67, 2b 2, 2c 3), of which `range_helper` (22) is fully dense and the rest
8–70 % nonzero. ≤ 16-bit candidates: 61 chunk + 5 digit-bit + `D` (offset by 8) + `m_neg` + sign flag + the three
VK bits = 72 columns; `m_pos` (11 bits) fits too; `range_mult` needs 23 bits (multiplicity of chunk value 0 over
61·2^18 lookups) — split it into two 16-bit halves if a 16-bit commitment tier is wanted.

## Tests (all green at this checkpoint; `cargo nextest run -p jolt-wrapper --lib --test limb_table_e2e --test limb_table_program --test limb_table_miller`, 39 tests, 60 s)

- `limb_table_program`: program reproduces the deferred check bit for bit on a real opening (`NativeCheck` as the
  intermediate-value oracle); fibonacci profile fits 2^18 rows; evaluator/kernel/program cross-checks;
  `verifier_arithmetic_within_budget_at_fibonacci_profile` (execution-derived ≤ 10k).
- `limb_table_miller`: Miller cells vs arkworks step by step.
- `limb_table_e2e`: every constraint vanishes on the honest witness; both members at σ = 8 and at the real σ = 11,
  n = 42 opening (`members_verify_and_tampers_are_rejected_at_the_fibonacci_profile`); verifier closed forms ==
  prover public columns at `r`; `Σ_t coeff·Π L(v) == final claim`; digit-link input `Σ ρ^k s_k + ρ^K + ρ^{K+1}θ`;
  tamper suite (chunk flip, chunk past the range table, wrong digit bit, broken copy, replaced looked-up operand,
  `x+q`/`x+2q` aliases, out-of-range multiplicity forgery, non-norm-one GT input, twist point, crafted point, sign-flag
  flips, fingerprint-root collision) rejected by the term check via the test-side `Cheating` prover wrapper (round
  checks forced, so rejection is the verifier's final relation); `stream_exporter_terms_match_the_members`;
  independent oracle `production_verifier_and_pins_agree_on_tampered_proofs` — the production `DoryScheme::verify`
  and the table's pins both reject a proof with a replaced G1 (`vmv.e1`) or GT (`d1_left`) message.
- Fixture (`tests/common`): `synthetic_opening(num_vars, n, seed)` runs the production Dory verifier on the honest
  proof and exposes it (`ProductionVerifier::accepts`) for tampered proofs. Evaluations are `u64`-valued: full-width
  row scalars send every row MSM of a 2^22 commitment through the arkworks fork's `msm_bigint_wnaf`, which builds a
  fresh 2-thread rayon pool per chunk (`variable_base/mod.rs:856`) — at 2^11 rows that spawns thousands of short-lived
  pools and dies with `EAGAIN` (8 MiB stacks) or a stack overflow (default stacks) on macOS. Worth a look upstream.

## Module map (all ≤ 1,000 lines; `schedule/` split this checkpoint)

`schedule/mod.rs` (Cells, Layout, Builder core, `build`) · `gt.rs` (GT leaves, tables, norm-one, online) · `miller.rs`
(ate schedule, Miller loop) · `final_exp.rs` · `ec/mod.rs` (θ-offset argument, `Chain`/`Lane`/templates, math tests) ·
`ec/g1.rs`, `ec/g2.rs` (Straus lanes, tables, on-curve) · `ec/psi.rs` (subgroup chains) · `ec/sign.rs`. `relation.rs`
is 1,082 lines (soft blocker; `RowSumcheck` is the natural next cut).

## Open for the parent

- `tests/perf1_profile.rs` (W5) still imports the removed `limb_table::wiring::Wiring` and the old
  `RowSumcheck::new`/`Slot` shape; `cargo clippy -p jolt-wrapper --all-targets` fails on it. Not mine to edit; the
  replacements are `RowSumcheck::new(&relation, &columns)` over the export columns and `Slot { x, y, kappa, y_sign }`.
- `tests/relation_fixture.rs` carries another lane's uncommitted change that breaks `--features prover-fixtures`.
- The `Cargo.lock` diff in the worktree is not mine (left unstaged).
- The arkworks-fork MSM pool storm above: any dense full-width polynomial committed through `commit_rows_dense` at
  2^22 hits it on macOS; production polynomials are small-valued or one-hot, which is why nobody sees it.

## Fix #2 (review #2, `b44217e65`)

**Blocker — θ-digit correction base.** Kept the correction base `−K` (scalar `θ`) inside each main chain — the
alternative, one fixed-base start chain `R_n = θ·G_n` per chain so that no `θ` digit enters a main chain, was
implemented and measured: it costs six extra single-base chains (their tables, doublings and selected families) and
**+1,800 verifier `Fr` mults** (copy kernels 8.5k → 10.2k), i.e. 11,818 at σ = 11 / N = 42, so it was reverted.
Instead every correction-base add is guarded: `ops::g1_add_guard` / `g2_add_guard` (3 / 6 rows: `t = λ² − 2x1 − x3
= x2 − x1` from the add's own rows, `inv = t⁻¹`, pin `inv·t = 1`) at slot `n + 4` of every main chain (64 guards per
chain, +128 verifier mults, rows used 201,319 at N = 42). The `ec` module doc now states exactly what the code
enforces: an add with `x_entry = x_acc` is either `entry = acc` (vacuous slope pin — the only exploitable case) or
`entry = −acc` (the pin reads `0 = −2y1`: no witness); the correction adds admit neither; for a proof-base add or a
doubling the accumulator is `θ·A·G − 16·k_K·P_w·G + H` with `P_w` the `θ`-digit prefix consumed so far, and with
`θ = 16^{64−w}·P_w + S_w` the exceptional equation is `(A ∓ λ)·S_w + ((A ∓ λ)·16^{64−w} − 16k_K)·P_w = c`: at most one
`S_w` per prefix and one prefix per suffix, so ≤ 2^129 bad `θ` per site (2^{−125}) provided both coefficients are
nonzero. `offsets_are_nondegenerate` sweeps them for every `(w, k, n ≤ 64)` on both groups — the previous sweep used a
wrong (linearly accumulated) offset count; the corrected `A_{w,k} = 16^{w+1} + λ(n(16^{w+1} − 16)/15 + k)` exposes the
single zero: the last add's `acc = −entry`, the zero MSM, θ-independent by construction and witness-free.
`Program::evaluate` no longer fails on a zero slope denominator (slope 0, the pinned slope row then fails), so every
exceptional case is a verifier-path rejection: `zero_msm_output_is_rejected_for_every_offset_challenge` (`A1 = FinalE1
+ d·Γ1_0 = 0`, `θ = 1, 2`), `exact_small_torsion_pairing_inputs_are_rejected` (orders 10,069 and 5,864,401), both from
the reviewer's patch.

**Major — `commitment_phases` and the VK groups.** The last phase's `group_count` now includes the pinned VK groups
(their physical position), so the phase list owns the block's whole group geometry: packing 4 → `[18, 17, 1, 3]` (39
groups), 16 → `[5, 5, 1, 2]` (13), 32 → `[3, 3, 1, 2]` (9); `prover_group_count` and `vk_group_range` are unchanged.
Tests: `stream::tests::phases_cover_every_group` (4/16/32) and the reviewer's e2e test, plus the exporter e2e
asserting `vk_groups.end == offset + Σ phases`. **W5 API note:** only this semantic change; `StreamTermExporter`,
`StreamColumns`, `Members`, `T2Challenges`, `link_input_claim` are unchanged.

**Major — budget at the real profile.** `verifier_arithmetic_within_budget_at_fibonacci_profile` runs σ = 11, N = 42
(the reviewer's 10,019). Trims: `RowRelation::batched_terms` scales the five quantities every term is linear in
(`eq(τ,r)`, `λ`, `λ_lookup`, copy kernel, constancy kernel) instead of 175 coefficients (−170); `public_and_omega_evals`
evaluates the digit link on the public-evaluation evaluator (shared eq tables); `Evaluator::group_into` memoizes each
family's cell product and sums the maps of one bucket before multiplying. The `ρ` powers and `1/mult` scalings of the
digit-link weights are now observed too (`rho_weights_with`, +197 — they were uncounted). **Exact count: 9,986 Fr
mults** (175 terms, degree 4), 14 below the cap. Remaining lever: the assembly computes the same `ρ` powers for R's
scalar link; an exporter that receives them would save 175.

**Minor.** `RowSumcheck`, `eq_tau_column` moved to `row_sumcheck.rs` (re-exported from `relation`; relation.rs 923
lines).

**Consumed scalars of `link_input_claim` (for W5 / R, σ = 11, N = 42, K = 173).** `FlattenedCheck::wires()` order —
first occurrence over the GT bases, then the G1 chains (acc, A3, A1, A4), then the G2 chains (acc, B2):
`CommitmentWeight(0..N)`, `D2Init`, then for `j = 0..σ`: `Alpha(j), AlphaInv(j), UAlpha(j), U(j), VAlphaInv(j), V(j),
Delta1R(σ−1−j), Delta2R(σ−1−j)`; then `Chi(0..σ)`, `Ht`, `Beta(0..σ)`, `GammaInv`, `PairingG1ZeroScalar`, `D`,
`DSquared`, `DInv`, `BetaInv(0..σ)`, `Evaluation`, `Gamma`, `PairingG2ZeroScalar`; digit base `K` is the constant one
(`ρ^K`), `K + 1` the offset challenge (`ρ^{K+1}·θ`). Not consumed (R must not publish them): `Chi(σ)`, `S1Acc`,
`S2Acc`.

## Fix #3 (07:00 API delta + review #3, `cbd75fffc` base)

### Point order (W5 blocker `Stream(StageLink)`)

- `RowSumcheck` and `LinkMember` bind the most significant row bit first: round `i` pairs row `j`
  with `j + rows/2` and writes the bound row at `j` (in place; the row member's scratch matrix is
  gone). The stage point the members return is big-endian, `EqPolynomial::evals(point)` /
  `PackedColumns::column_evaluations(point)` order. `Challenges::tau` stays big-endian (bound first).
- The kernels/evaluator keep little-endian points internally: `StreamTermExporter` reverses
  `TermContext::row_point` itself. Tests pass `little_endian(&point)` to `public_evals`/`omega_eval`.
- Regression (permanent, in `stream_exporter_terms_match_the_members`): both members driven jointly,
  `commit_packed(stream.columns, 4)` at `2^20`, and `column_evaluations(&point)[physical(local)] ==
  claims()[local]` for all 149 claimed columns plus the link's digit final.

### Staged export (the production path; one owner of every column)

```text
let mut b = StreamBuilder::new(&layout, &columns /* Columns::generate */, packing);
b.phase_1b()                                  -> &[Column]  // chunks u16, digit bits, D, m_pos/m_neg u32, range_mult u32, sign_flag
b.phase_2a(xi, alpha)                         -> &[Column]  // X_s, Y_s, range helpers, range inverse
b.phase_2b(fp_root)                           -> &[Column]  // f_pos, f_neg
b.phase_2c(beta, fp_combine, copy_root)       -> &[Column]  // h, g_pos, g_neg, then the VK suffix (pin, pin_limb×3, free, exact)
let w = b.finish(tau /* Vec<Fr>, 18 */, gamma, lambda, lambda_lookup, constancy_root, group_offset);
// StreamWitness { relation: RowRelation, matrix: Vec<Vec<Fr>> /* Col::WIDTH: claimed in Col order, then public */, stream: StreamColumns { columns, ids, group_count, vk_groups } }
let members = Members::new(&w.relation, &w.matrix, &layout, &w.matrix[Col::D], rho);
StreamTermExporter { layout, challenge_offset, theta_offset, rho_offset, columns: &w.stream.ids, row_member, link_member }
```

- Every phase slice is padded to whole groups; phase 2c's slice includes the six verifier-key
  columns (`commitment_phases` counts them with it — unchanged: 39 / 13 / 9 groups at k = 4 / 16 / 32).
  Phases panic when called out of order; nothing is recomputed and no future challenge is used:
  the lookup multiplicities now live in `PublicColumns::{m_pos, m_neg}` (challenge-free), the range
  multiplicities in `Columns::range_multiplicities(digit_bits)`, and only `range_helpers(alpha, ..)`,
  `fingerprint_columns(reads, z_xi, fp_pow)` and `LookupColumns::new(public, y, f_pos, f_neg,
  fp_pow, beta, fp_combine)` take challenges.
- `row_challenges(&[Fr]) -> Challenges` is the single owner of the per-phase challenge order
  (`T2Challenges::from_challenges` and `finish` both use it).
- Removed: `ClaimedColumns` (+ `assemble`), `StreamColumns::new`, `Columns::logup_columns`,
  `AffineForm::scale` (unobserved; `-form` via `Neg`, constants via `AffineForm::scaled`).
- Unchanged: `Col::CLAIMED = 149`, T = 175 terms, d = 4, rows 201,319 / 262,144.

### Review #3 blocker — one digit-link equation per chain occurrence

- Every `(chain, base)` occurrence has its own index `DigitOp::link` (`Layout::link_occurrences`
  of them; `SelectedFamily::digit_base` maps `k → link`), and the link weighs it `ρ^link`:
  `ω(x) = ρ^{link(k(x))}·16^{63−w(x)}` on each op's first slotted row; the verifier's `ω̃(r)` keeps
  the product form per family (`eq(r_c, first_c) · Σ_k eq(r_k, k)·ρ^{link(k)} · 16^{63}·Π(1 − r_i +
  r_i·16^{−2^i})`, the window factor memoized per lane). No multiplicity division anywhere.
- **R-side contract (W5, `DoryScalarLink` / `DoryScalarTermExporter`):** publish
  `Σ_{k<K} W_k(ρ)·s_k` over the `K = layout.digit_bases − 2` named wires in `check.wires()` order
  (173 at σ = 11 / N = 42) with `W = link_weights(&layout, ρ)` (`link_weights_with(.., mul)` for
  the observed verifier: `W_kd(ρ) = Σ_{occurrences of kd} ρ^link`, one `ρ` power per occurrence —
  `layout.link_occurrences` = 230 at the fibonacci profile, i.e. 229 multiplications, R's budget).
  T2 adds the constant-one and offset bases: `link_input_claim(r_claim, ρ, θ, &layout)` =
  `r_claim + W[K] + W[K+1]·θ` (`_with(.., mul)` observed). `link_input_claim` no longer takes
  `named_wires`; the layout carries the occurrence structure (profile-fixed).
- Soundness: `Σ_occ ρ^{link}·(recoding_occ − s_{kd(occ)}) = 0` is a degree-`< 230` identity in `ρ`,
  so each occurrence's recoding equals its scalar individually (the θ-prefix premise of the EC
  module doc holds per chain again). Permanent verifier-path negative
  `shared_scalar_recoded_differently_per_chain_is_rejected`: the reviewer's `±1` shift of one
  window digit in two offset chains (honest claim == `link_input_claim(..)`, forged `≠`).

### Review #3 major — every verifier multiplication observed

- Previously unobserved and now routed through the evaluator/observer: `eq(τ, r)`, `small`, `id`,
  the families' coordinate/`S0` products, the `ω̃` per-field product and `16^63` scaling, the
  free-field moments, `AffineForm::scale(−1)`/`scale(−2)` (now negation / constant weights).
- Trims (all exact): `eq(τ, r)` as `Π(1 − t − r + 2tr)` (2 per bit), `small` one product, `id` and
  field moments recombined by doublings (additions, not multiplications), all-but-one products by
  prefix/suffix (`3m − 4` per family), the digit link's window factor memoized per lane.
- Measured at σ = 11 / N = 42 (`verifier_arithmetic_within_budget_at_fibonacci_profile`, which
  also asserts the component sum equals the exporter's count): **9,875 Fr** = relation 162 +
  public evaluations and `ω̃` 9,573 + terms 139 + link batching 1. Cap stays 10,000 (margin 125).
  `link_input_claim_with` (229 powers) is the stream's derivation, outside this count as before.

### Review #3 minor

- `offsets_are_nondegenerate` checks `A_w·16^{64−w} − k_K` for the doublings (the correction prefix
  is not yet doubled); the EC module doc states both formulas. Still passes for both groups, all `n, w`.

### Not in this fix

- R publishing 176 vs 173 scalars (W5). The reviewer's `equal_point_exception_is_accepted_by_the_
  unguarded_add` scratch test is not landed: it documents the accepted `entry = acc` case of the
  unguarded proof-base add, which the per-occurrence binding makes θ-dependent again per the
  module-doc argument; the guards stay on the correction-base adds only.

## Fix #4 (review #4, `8ec8ea0f0` base)

### Blocker — unique recoding per occurrence (window check)

- Every signed radix-16 string `t` with `Σ 16^i·d_i ≡ s (mod r)` was a valid recoding of `s`; the
  link only proved the congruence. Now each occurrence `o` also proves `V_hi(o) ∈ 0..=WINDOW_BOUND`
  for `V_hi = Σ_{i=48}^{63} 16^{i−48}·d_i` (its top 16 digits), `WINDOW_BOUND = R_HI − 2`,
  `R_HI = r >> 192 = 0x30644e72e131a029` (`digits.rs`). With `|Σ_{i<48} 16^i·d_i| ≤ L = 8·(16^48 −
  1)/15 < 2^192·8/15`, an admitted `t` lies in `[−L, (R_HI − 2)·2^192 + L]`, an interval of length
  `< (R_HI − 2 + 16/15)·2^192 < R_HI·2^192 ≤ r`: at most one admitted recoding per residue class,
  so the digits of every occurrence are a function of its scalar (zero and small scalars included:
  `V_hi = 0` is admitted). **Completeness:** the canonical recoding of `s < r` has `V_hi ∈ {s_hi,
  s_hi + 1}`; it is rejected only when `s > (R_HI − 2)·2^192 + 2^192·7/15`, fewer than `3/R_HI ≈
  2^−60` of the scalars per occurrence (no witness; the prover's saturated window row fails the
  link). One rule for every scalar — no classification of 125-bit challenges needed.
- **Rows:** `Cells::WINDOW = 9616` (a free, `256`-row-aligned block; `WINDOW_ROW_BASE = 153,856`):
  row `o` is a `Source::Window` free row (limb identity off, canonicality witness as for inputs)
  holding `V_hi(o) + 2^64·(WINDOW_BOUND − V_hi(o))`, i.e. chunks `0..4` = `V(o)` and `4..8` =
  `V'(o)`, both range-checked by the existing u16 LogUp. Rows `M..256` (no occurrence) hold
  `V_hi = 0`. `used_rows` 201,319 → **201,575**; no new column (`Col::CLAIMED` 149).
- **Member:** the digit link's summand is `ω·D + κ·V + κ'·V'` (degree 2, `LOG_ROWS` rounds):
  `ω(x) = ρ^o·16^{63−w}` (+ `ρ^{M+o}·16^{15−w}` on windows `w < 16`), `κ(row_o) = ρ^o·(ρ^{M+256} −
  ρ^M)`, `κ'(row_o) = ρ^{M+256+o}`, `V/V' = Σ_{j<4} 2^{16j}·chunk_{j}/chunk_{4+j}`. The batched
  identity `Σ_o ρ^o·(t_o − s_{b(o)}) + Σ_o ρ^{M+o}·(V_hi(o) − V(o)) + Σ_{o<256} ρ^{M+256+o}·(V(o) +
  V'(o) − WINDOW_BOUND) = 0` has distinct `ρ` powers (`< M + 512`), so each equation holds on its
  own (`(M + 512)/r`). Verifier: `ω̃` per family reuses the `Σ_k eq·ρ^{link}` sum (`W64 +
  ρ^M·W16`, `W16` = high-2-bits-zero × geometric over 4 bits, memoized per lane); `κ̃, κ̃'` =
  `eq(r_{8..18}, 601)·Σ_{o<256} ρ^o·eq(r_{0..8}, o)` (one geometric) × constants;
  `Σ_{o<256} ρ^o = Π_{i<8}(1 + ρ^{2^i})`. Terms: `link_terms(&LinkEvals) = [ω̃·D, κ̃·V, κ̃'·V']`
  → **T = 177**, `d = 4` unchanged.
- **R-side contract unchanged:** R still publishes `Σ_{k<173} link_weights(&layout, ρ)[k]·s_k`
  (`W_k = Σ_{o of k} ρ^o`, occurrence exponents `[0, M)`; the window exponents `[M, M + 512)` touch
  no scalar). T2's `link_input_claim(r_claim, ρ, θ, &layout)` (same signature) now adds the window
  constant `WINDOW_BOUND·ρ^{M+256}·Σ_{o<256} ρ^o` besides `W_K + W_{K+1}·θ`.
- **EC argument:** `H` and the correction prefixes `P_w` are functions of the pre-`θ` transcript and
  of `θ` respectively (no digit is chosen after `θ`), so the per-site `2^129/r` bound and its union
  over the 4,928 unguarded proof-base adds (`≈ 2^−112`) follow; module doc updated.
- **Negatives (permanent, `limb_table_e2e`):** `modulus_alias_recodings_are_rejected` — the
  constant-one occurrence recoded as `1 + r` and as `1 − r`: with honest window rows the link's
  claim differs from the verifier's `link_input_claim` (a); with window rows matching the alias
  (`V = V_hi(alias)` as one field element) the link accepts but the row member's range LogUp
  rejects (b). `shared_scalar_recoded_differently_per_chain_is_rejected` kept.

### Major — fixed powers and inversions out of the verifier path

- `Fr::pow2` loops and `16⁻¹` moved to process-wide constants: `Constants::get()`
  (`LazyLock`, plus `pow_64`), `lookup::SIXTEEN` (`16^k`, `k < 64`, and `16⁻¹`). The exporter's
  derivation now performs no exponentiation and **no inversion** (`fr_inv` = 0; `TermObserver`
  has only `fr_mul`, so nothing is left uncounted). `Fr::from_u64/from_i64` constructors of small
  constants are not counted (constructors, not arithmetic).
- Measured at σ = 11 / N = 42: **9,973 Fr** = relation 162 + public evaluations and link weights
  9,669 + terms 139 + link batching 3 (component sum asserted; cap stays 10,000, margin 27).

### Minors

- `StreamBuilder::end(phase)` asserts in release builds that the phase emitted exactly
  `phases()[phase].columns` (the range `commitment_phases` publishes); the reviewer's slice test
  `stream_builder_phase_slices_match_declared_geometry` is permanent (`[3, 3, 1, 2]` at k = 32).
- `FlattenedCheck::wire_multiplicity` deleted; `digit_link.rs` documents the occurrence-weight
  equations (no averaging).

### API delta for W5 (`StreamBuilder`/`Members::new`/exporter unchanged)

- `LinkMember::new(&layout, rho, digit_values, &matrix[Col::CHUNKS..Col::CHUNKS + 8])`;
  `LinkMember::final_values() -> LinkFinals { digit, v, v_prime, evals: LinkEvals }`.
- `lookup::{link_evals, public_and_link_evals} -> LinkEvals { omega, kappa, kappa_prime }`
  (replace `omega_eval`/`public_and_omega_evals`); `digit_link::link_terms(&LinkEvals)` (replaces
  `link_term`); `lookup::LinkPowers` (occurrence powers, window bases, `base_weights`,
  `window_constant`); `link_weights(_with)` unchanged.
- `program::Source::Window(Fq)`, `schedule::{Cells::WINDOW, WINDOW_ROW_BASE}`,
  `digits::{WINDOW_TOP_DIGITS, R_HI, WINDOW_BOUND, WINDOW_ROWS, window_value, window_row_value}`.
