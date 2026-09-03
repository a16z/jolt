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
