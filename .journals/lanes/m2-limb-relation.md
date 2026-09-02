# Lane M2 — limb-relation microbench (non-native Fq arithmetic as a HyperKZG-committed table + one sumcheck)

Date 2026-09-02 · tree 8305611c4 (on b051a7576) · Apple M4 mini, 10 Rayon threads (4P+6E), release, quiet machine (other lanes' builds finished; ~87% idle during the sweep).
Code: `crates/jolt-crypto/src/ec/bn254/msm.rs` (`g1_msm_small`, ships), `crates/jolt-limb-bench/` (harness, not production).
Commits: `e8e1964f0` feat(jolt-crypto): small-scalar G1 MSM + `HyperKZGProverSetup::g1_powers`; `8305611c4` test(jolt-limb-bench): harness.
Reproduce: `cargo run --release -p jolt-limb-bench -- <log2 rows> <t> [commit-operands] [tamper]`; `cargo bench -p jolt-crypto --bench crypto -- g1_msm_small`.

## 0. Verdict (numbers first)

- **Layout:** w = 96-bit limbs (3 per Fq value), 16-bit chunks, **54 chunk columns/row** (z 16 + k 17 + 3 carries × 7) with operands virtual; **+32 chunk columns per operand pair** if operands are re-committed. Sumcheck **degree 3** (eq · x·y), one opening.
- **Range check is the blocker.** The only sound option that keeps the proof small (LogUp with one committed inverse column per chunk column) costs **75 ms per column at 2^17 = 4.08 s for 54 columns** — 10× everything else combined. Alternatives trade it for proof bytes: LogUp-GKR ≈ +27 KB, bit columns (M6 kernel) ≈ +55 KB. No option meets both "< 1 s" and "single-digit KB" (§5).
- **Everything else at 2^17 rows, t = 24:** u16 chunk commits 341 ms (6.3 ms/column), sumcheck 1,027 ms (≈ 472 + 21.4·t ms; ≈ 2× headroom with known levers), RLC + HyperKZG open 262 ms, verifier 6 ms. Proof 15.4 KB, of which 64 B per committed column (claim **and** commitment — the brief's 32 B/column model undercounts 2×) and 32 B per operand-limb claim (6t = 144 → 4.6 KB).
- **Small-scalar MSM (ships):** u16 columns 9.8× faster than the full-width path at 2^17 (8.5 vs 83 ms), 10.1× at 2^20 (59 vs 594 ms); u8 19–20×. Per witness bit, u16 chunks (≈ 4 ns/bit) tie M6's grouped bit-column kernel (77.8 ms / 163 columns / 2^17 ≈ 3.6 ns/bit) and beat it 11× per column when the bit kernel runs one column at a time.
- Tamper tests: flipping a z-chunk bit (CRT) and moving 2^16 between two chunks of one limb (value preserved, chunk out of range) are both rejected at the final sumcheck check.

## 1. Layout decision

Row = one Fq output coefficient `z ≡ Σ_{i<t} x_i·y_i (mod q)`, `t ≤ 24`. Integer identity `Σ x_i y_i = k·q + z` with `k < 24·2^512/q < 2^263` (17 chunks, 272 bits), `z < 2^256` (16 chunks; not canonical — canonicalise once at the final GT equality), operands `x_i, y_i < 2^256` (outputs of earlier rows, their chunks range-checked there).

Limb width: products of w-bit limbs summed over `t·(c+1) ≤ 72` terms must stay below r ≈ 2^253.5 as integers, so `2w + 7 < 253` → `w ≤ 123`, and w must be a multiple of 16 for chunk recomposition to be linear → `w ∈ {64, 80, 96, 112}`. Number of carry equations L = ⌈m/w⌉ with m ≥ 265 (§2): w=64 → 5 carries × (64+8 bits → 5 chunks) = 25 chunks; w=80 → 4 × 6 = 24; **w=96 → 3 × 7 = 21**; w=112 → 3 × 8 = 24. w = 96 minimises range-checked chunks: `54 = 16 + 17 + 21`. (Two 128-bit limbs would give L = 3 as well but a single 128×128 product overflows r.)

Per row, committed 16-bit chunks: z (6,6,4 per limb), k (6,6,5), carries C_c = c_c + 2^111 (7 each, c = 0,1,2). Public constants: q limbs, 2^96, 2^192, 2^111. Operand limbs `x_{i,a}` (3 per operand) are virtual polynomials — supplied by the wiring sumcheck in the real design; the bench evaluates them as dense uncommitted polys (their 6t evaluations at the final point are part of the proof either way, §4). With `commit-operands` the bench also commits 16 chunks per operand and adds 6t linear consistency terms (limb − Σ 2^{16j} chunk).

Relation (all linear combinations of chunks are free): with `X_i = Σ_a 2^{96a} x_{i,a}` etc.
```
native:  Σ_i X_i·Y_i − K·q − Z = 0                                   (mod r, degree 2)
P_0 = Σ_i x_{i0}y_{i0} − k_0q_0 − z_0                = 2^96·c_0
P_1 = Σ_i (x_{i0}y_{i1} + x_{i1}y_{i0}) − k_0q_1 − k_1q_0 − z_1  + c_0 = 2^96·c_1
P_2 = Σ_i (x_{i0}y_{i2} + x_{i1}y_{i1} + x_{i2}y_{i0}) − k_0q_2 − k_1q_1 − k_2q_0 − z_2 + c_1 = 2^96·c_2
range:   h_j·(α − chunk_j) = 1  for every chunk column j (LogUp),  Σ_rows Σ_j h_j = Σ_v m_v/(α − v)
```
One sumcheck over the 17 row variables: `Σ_row eq(τ,row)·[γ^0·native + γ^{1..3}·carry_c + γ^{4+j}·(h_j(α−chunk_j) − 1)] + λ·(Σ_j h_j(row) − m(row)·inv(row)) = 0`, where m (2^16 multiplicities, u32, padded) is committed and `inv(v) = 1/(α−v)` is public (verifier: 2^16 batched inversions + a 2^17 eq table, 6 ms). Degree 3 (eq × product). Committed columns: 54 chunks (u16 MSM) + 54 inverses (full-width) + m; one RLC opening at the sumcheck point (bound low-to-high; HyperKZG's point is the reverse).

## 2. CRT soundness (why mod r and mod 2^288 suffice)

All chunks range-checked to 16 bits ⇒ integer bounds `x_{i,a}, y_{i,b}, k_0, k_1, z_0, z_1 < 2^96`, `k_2 < 2^80`, `z_2 < 2^64`, `C_c < 2^112` ⇒ `|c_c| < 2^111`. Every carry equation is a field identity between integers of magnitude `< 72·2^192 + 3·2^192 + 2^96 + 2^207 + 2^111 < 2^208 ≪ r/2`, so it holds over ℤ. Chaining the three: `T_low := P_0 + 2^96 P_1 + 2^192 P_2 = 2^288·c_2 ≡ 0 (mod 2^288)`. The full integer `T := Σ x_i y_i − kq − z = T_low + 2^288·(P_3 + 2^96 P_4)` (P_3, P_4 the unconstrained high positions), hence `T ≡ 0 (mod 2^288)`. The native identity gives `T ≡ 0 (mod r)`; gcd(r, 2^288) = 1 ⇒ `T ≡ 0 (mod r·2^288)`. Bound: `|T| ≤ max(Σ x_i y_i, kq + z) < max(24·2^512, 2^272·2^254 + 2^256) < 2^526.1`, and `r·2^288 > 2^541.5` ⇒ **T = 0**, i.e. `Σ x_i y_i = kq + z` over ℤ and `z ≡ Σ x_i y_i (mod q)`. Margin 2^15; the m ≥ 265 requirement (k as 17 chunks: kq < 2^526 needs `2^m > 2^526/r = 2^272.5`… with k range-checked to 263 bits it drops to 2^264.5) is why L = 3 at w = 96 and why no w ≤ 123 gives L = 2. Honest carries are bounded forward (`|c_c| < 2^103`), so the 2^111 offset never underflows.

## 3. Measurements (quiet machine, 10 threads; ms; RSS = max resident set)

Relation harness, 54 chunk + 54 inverse + 1 multiplicity column, operands virtual:

| rows | t | generate | commit chunks (u16) | per col | commit inverses | per col | sumcheck | RLC+open | prover total | verify | proof B | RSS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2^16 | 24 | 505 | 172 | 3.18 | 1,984 | 36.7 | 493 | 134 | **2,785** | 8.0 | 15,168 | 1.53 GB |
| 2^17 | 24 | 1,062 | 341 | 6.32 | 4,077 | 75.5 | 1,027 | 262 | **5,710** | 6.0 | 15,392 | 3.12 GB |
| 2^18 | 6 | 618 | 622 | 11.5 | 7,612 | 141 | 1,214 | 509 | **9,960** | 6.3 | 12,160 | 3.45 GB |
| 2^16 | 6 | 161 | 170 | 3.15 | 2,023 | 37.5 | 300 | 136 | 2,631 | 6.8 | 11,712 | 0.83 GB |
| 2^16 | 12 | 279 | 171 | 3.16 | 2,017 | 37.4 | 374 | 134 | 2,698 | 5.4 | 12,864 | 1.14 GB |
| 2^16, operands committed | 1 | 59 | 272 (86 cols) | 3.16 | 3,368 (86 cols) | 39.2 | 372 | 161 | 4,175 | 6.7 | 14,848 | 0.88 GB |

2^18 at t = 24 was not run (projected RSS ≈ 6.2 GB on the shared 16 GB box); commit/open phases are t-independent, so the 2^18 row gives them exactly. Generation (BigUint witness synthesis, random Fq operands) is not prover work here — M1's real witness generator replaces it.

Laws (per 2^17 rows unless noted; every phase doubles with rows, 1.9–2.0× measured):
- u16 chunk column commit: **6.3 ms/column** (47 ns/point when 54 columns run in parallel; 8.5 ms standalone) → 54 columns 341 ms.
- full-width column commit: **75 ms/column** (0.55 µs/point) → each LogUp inverse column costs 12 u16 columns.
- sumcheck: **≈ 472 + 21.4·t ms** (from 2^16: 236 + 10.7·t; t-independent part = 54 recompositions + 108 range mults + 3 × 255 extrapolations per pair) → 8 µs/row at t = 24, 163 ns/row/product. Levers not yet applied, ≈ 2× together: Gruen split-eq (one fewer evaluation point), unreduced `fmadd` accumulation of the 168 products per evaluation, P-core-only pool (the balanced fold runs at ≈ 7 P-core-equivalents on 4P+6E).
- RLC (109 columns) + HyperKZG open: **262 ms** (134 / 509 at 2^16 / 2^18) — lane G's open law (170/336) plus ≈ 30 ms RLC and combine.
- proof bytes = 17 rounds × 3 coeffs × 32 + **64 B per committed column** (32 B claim + 32 B commitment) + 32 B per operand-limb claim (6t) + HyperKZG (2^17: 16 + 1 G1 + 51 Fr = 2,176 B). 109 columns + t = 24: 1,632 + 6,976 + 4,608 + 2,176 = 15,392 ✓. Without the 55 range-check columns: 11.9 KB; with t = 12 instead of 24: −2.3 KB but 2× rows.

Small-scalar MSM (criterion, `g1_msm_small`, bases `[i]G`, uniform scalars; bit column = one 50%-ones column through M6's `g1_bit_columns_msm`):

| points | full-width | u16 | u8 | bit column (M6 kernel, 1 col) |
|---|---|---|---|---|
| 2^17 | 83.3 ms | **8.49 ms (9.8×)** | 4.37 ms (19×) | 6.07 ms |
| 2^20 | 594 ms | **58.7 ms (10.1×)** | 29.6 ms (20×) | 49.5 ms |

Cost ≈ one L1-resident bucket add per scalar byte: 4 ns per witness bit at 10 threads for both u8 and u16. Committing a 16-bit chunk as 16 bit columns through the single-column bit kernel costs 97 ms vs 8.5 ms; through M6's grouped mode (0.48 ms/column at 163 columns) ≈ 7.6 ms — commit parity, but 16× the columns in the proof (64 B each).

## 4. Extrapolation for the orchestrator

Component cost at R rows (R = 2^17 baseline, scale linearly), C_u16 chunk columns, C_fw full-width columns, t products/row:
```
prover_ms ≈ 6.3·C_u16 + 75·C_fw + (472 + 21.4·t) [sumcheck, ÷2 with levers] + 262 [RLC+open, shared with the other tables]
proof_B   ≈ 1,632 + 64·(C_u16 + C_fw) + 32·6t + 2,176
```
With the LogUp layout (C_u16 = 54, C_fw = 55): 5.7 s / 15.4 KB at 2^17; 2.9 s / 15.2 KB at 2^16. Range checks removed (C_fw = 0): 1.6 s (≈ 1.1 s with sumcheck levers) / 11.9 KB.

**Wiring (not implemented, estimate).** Operand limbs `X_{i,a}(ρ) = Σ_{ρ'} W_i(ρ,ρ')·Z_a(ρ')` with W_i public 0/1 copy matrices (one 1 per row). After the row sumcheck ends at r, the prover sends the 6t = 144 limb evaluations `X_{i,a}(r)` (4.6 KB — inherent to t; halving t halves them but doubles rows) and proves them with ONE linking sumcheck: `Σ_{i,a} γ_{i,a} X_{i,a}(r) = Σ_{ρ'} Σ_a [Σ_i γ_{i,a} W̃_i(r,ρ')]·Z_a(ρ')`, 17 rounds, degree 2, prover O((6t+3)·R) ≈ 19M field ops ≈ 40–60 ms at 2^17, no new columns. Its point r' ≠ r, so one more 17-round degree-2 sumcheck folds the z-chunk claims at r and r' into one point (≈ 10 ms) before the single opening. Proof +2 × 17 × 2 × 32 ≈ 2.2 KB. Verifier evaluates the public `Σ_i γ_{i,a} W̃_i(r, r')` natively: O(6t·R) ≈ 19M mults ≈ 0.3 s single-thread, outside the prover clock. Operands from proof inputs (the committed GT limbs) enter the same matrices.

## 5. Range check: the decision and why the budget is missed

Options for 54 × 2^17 ≈ 7.1M 16-bit range checks under HyperKZG + sumcheck (prover ms at 2^17 / proof bytes added):

| option | prover | proof | notes |
|---|---|---|---|
| **A. LogUp, one committed inverse per chunk column (implemented)** | **+4,080** | +3.6 KB (55 cols × 64 B) | degree 3, folds into the row sumcheck; measured |
| A′. LogUp helper per s chunks, degree s+2 | +4,080/s | +3.6/s KB + 17·(s−1)·32 B | s=2: 2.0 s, +2.3 KB; s=6: 0.68 s, +4.3 KB rounds |
| B. bit columns (M6 grouped kernel), booleanity only | ≈ +420 | **+55 KB** (864 columns) | no inverses, no m column, degree 2 |
| C. LogUp-GKR fraction tree (2^23 leaves) | ≈ +300 (est.) | **+27 KB** (23 layered sumchecks, O(log² N)) | not implemented |
| D. 2^20 range table (20-bit chunks → 44/row) | A: +3,300 | +2.9 KB | verifier 2^20 inversions ≈ 25 ms; same shape |

Full-width commits run at 0.55 µs/point; a committed inverse costs exactly one such point per chunk, so A is bounded below by 7.1M × 0.55 µs ≈ 3.9 s at 2^17 regardless of layout tweaks — 10× the 0.4 s component budget by itself. The remaining phases (0.34 + 1.0→0.5 + 0.26 s) already exceed 0.4 s. **Budget verdict: missed by ≈ 14× with A, ≈ 3–4× without any range check.**

Next technique (ranked): (1) GPU (Metal) full-width MSM for the inverse columns — the user's Metal lanes put Dory tier-1 on the M-series GPU; even a 3–4× win leaves ≈ 1 s. (2) Move the 2^16 range table to Dory for the inverse columns — no gain (Dory tier-1 is the same G1 MSM). (3) Accept option C's proof size (27 KB) for a ≈ 2 s prover, or A′ with s = 2–3 (2.0–1.4 s, +0.3–0.6 KB rounds). (4) Reduce chunks per row below 54 only by shrinking k's range check (k < 2^263 exactly: 16 chunks + a 7-bit chunk range-checked as chunk·2^9 — saves nothing at 16-bit granularity) or by re-deriving the operation set (M1) so fewer rows exist. There is no known HyperKZG-compatible range argument with both cheap commits and O(log N) proof; the relation itself (§1–2) is sound and cheap.
