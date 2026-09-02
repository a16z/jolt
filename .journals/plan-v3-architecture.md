# Plan v3 architecture — size/gas-ranked re-budget (lane M5, planner, read-only)

Date 2026-09-02 19:38–20:20 · tree c229590c4 (wrap/spartan-hyperkzg) · objective per USER DECISION 19:34 (`plan.md` tail):
rank designs by (1) proof bytes → (2) EVM verifier gas → (3) prover seconds (2.5 s fine, more if it buys KB).
Inputs (do not re-derive): `plan-relation.md` §2/§4/§5/§8 (R1CS side, public IO, witness replay), `lanes/dory-offload-study.md` §1–§3
(Dory closed form, 9σ+N+4 GT bases), `lanes/m1-gt-multiexp-counts.md` (exact rows), `lanes/m2-limb-relation.md` §6 (limb relation
measured), `lanes/m3-blake3-transcript.md` (C3, hash relation, microbench), `lanes/m6-bit-commit-kernel.md` (bit commits), `lanes/m4-blake3-gadget.md`
(Blake3 R1CS 15,536/continuation block). "exact" = counted from code/lanes; "est." marked. Profile fibonacci L=20 (K=16, σ=12, N=41 [42 real]);
L=18 (σ=11) in parentheses where it differs.

## 0. Verdict

| rank | design | proof bytes (wire) | EVM calldata | est. gas | prover s | verifier | setup |
|---|---|---:|---:|---:|---:|---|---|
| **1** | **D4** two layers, Groth16 final (layer 1 = sumcheck-offload wrapper, k=8 packing; layer 2 = Groth16 over Blake3-FS verification of layer 1, HyperKZG pairing deferred on-chain) | **≈1.2 KB** (128 Groth16 + 672 deferred G1 + 256 public Fr + Jolt IO) | ≈2.0 KB | **≈560k** (≈450k with an O(1)-proof KZG multilinear PCS, lever L6) | ≈11–14 s (layer 1 7.7 + Groth16 3.35M·g µs, g unmeasured → lane N2) | EVM or native | Groth16 circuit CRS per profile + universal 2^21 SRS |
| 2 | D2 single layer + KZG-committed round polys, k=16 | **8.8 KB** | — (native verifier only) | n/a on-chain (see D5) | 10.4 s (k=16) / 7.7 s (k=8, 9.2 KB) | native ≈0.4 s (O(rows) wiring) | universal 2^22 SRS |
| 3 | D3 two layers, Spartan+HyperKZG final | 8.3 KB | — | n/a (needs SPARK: +≈3 KB, ≥1M gas est.) | ≈23 s | native ≈0.15 s | universal 2^22 SRS |
| 4 | D1 single layer, size-optimized, k=16 / 8 / 1 | 11.3 / 11.7 / 17.7 KB | — | n/a | 10.4 / 7.7 / 5.3 s | native | universal |
| 5 | D0 current measured shape (lanes M2/M3, 2^17 tables) | 26 KB (≈30 KB for the full statement) | — | n/a | 2.5 s (≈5.3 s full statement) | native | universal |
| 6 | D5 D1 verified directly on-chain | ≈17 KB (D1 + committed wiring/SPARK) | 17 KB | **≈3M** (public-matrix MLEs ≈48k Fr ops) | ≈9 s | EVM | universal |

- Only D4 reaches "a couple of KB". Every single-layer design is floored by its sumcheck round polynomials: 3 stages × 18–28 rounds ×
  (deg 2–5 coefficients) × 32 B ≈ 7.0 KB (4.4 KB with KZG-committed rounds) — no column trick touches that term.
- D4's gas is 55% the deferred HyperKZG check (≈24 ecMul + 2 pairings ≈ 261k), 41% Groth16 (≈230k), 4% calldata/hashes. Layer-1 proof
  size is irrelevant in D4 (it is layer-2 witness); what matters is bytes *hashed* in-circuit (2.9M constraints for 12 KB) and the number of
  deferred G1 points (ℓ+1 = 21).
- Two facts the earlier lanes did not budget, both now in every number below: (a) with GT elements and commitments committed (not sent),
  their transcript absorption is in-table: C_tot = C3 + 290 (42 commitments) + 636 (Dory VMV/rounds/final, σ=12) = **1,980** compressions
  (1,893 @L=18), hash rows 116·C_tot = 229.7k → **2^18** (M3 measured the hidden segment only: 1,054 → 2^17); (b) M1's limb rows are 205k
  (σ=11) / 230.6k (σ=12) → **2^18**, not M2's 122k fused estimate. Both tables at 2^18 double every per-row cost of M2/M3.

## 1. Statement (all designs share layer 1)

Layer 1 proves: "the Jolt verifier (stages 1–8 incl. Dory's deferred check, `dory-offload-study.md` §1.4) accepts proof π for preprocessing digest
d_pp, program IO io, with transcript profile Blake3-streaming (`crates/jolt-transcript/src/blake3.rs`)".

| class | content | where |
|---|---|---|
| public input (verifier-computed) | `state_pre` = Blake3 state after the native preamble absorb (`verifier.rs:580-650`), `val_io`, `init_eval`, `stage_value[0..5]` (plan-relation §5), Dory verifier-key GT constants χ[k], Δ1R, Δ2R, HT as limb rows' public operands | x column of the R1CS + public operands of the limb table |
| public output (transmitted) | `r_ram_addr` (K), `r_bytecode_addr` (12) → 28 Fr = 896 B | proof |
| committed witness | 42 commitment GTs (as limb operand rows **and** as hash message bytes, 42×416 B), round polys/claims of stages 1–7 (R1CS witness W, mirrored as hash bytes), Dory proof: 6σ+2 GT, 3σ+2 G1, 3σ+1 G2 (limb rows + hash bytes), all GT/G1/G2 intermediates, hash-table bits, limb chunks, LogUp helpers | HyperKZG polys |
| decision: commitments | **committed, absorption proven in-table** (+290 compressions = 33.6k hash rows, +1,320 operand rows). Sending them = 42×128 = 5,376 B (lane J) — alone exceeds the D4 total | — |
| decision: final exponentiation | **in-table** (3,288 rows, M1). Native FE would need the pre-FE Fq12 sent (384 B) or exposed as 12 Fq public outputs (384 B) | — |
| decision: Dory challenges | derived in-table over the committed GT bytes (636 compressions). The "expose `state_rlc`, derive natively" variant needs the Dory proof public (12 KB) | — |

Linking (all as extra public-matrix terms of the wiring sumcheck, no new committed rows): L1 absorbed Fr in W ↔ hash message bits; L2 R1CS
challenge wires ↔ squeeze output words (125/128-bit decode is linear in bits); L3 `state_pre` (public) → first-compression key words,
transcript never exits the table again (Dory challenges are consumed by the limb table's public scalar arithmetic in the R1CS: 143 scalars,
≈600 constraints); L4 GT/G1/G2 limb chunks ↔ hash message bits (16 bits per chunk, linear); L5 Dory Fr scalars (R1CS) ↔ limb rows' public
scalar inputs (Straus digit selectors, §2 W).

## 2. Layer-1 component catalog (common to D0–D5; measured unit costs, 10 threads)

Unit costs (measured): HyperKZG open 2^17 0.17 s · 2^18 0.34 · 2^19 0.67 · 2^20 1.35 · 2^21 ≈2.7 · 2^22 ≈5.4 (est. ×2/doubling); bit-column commit
3.6 ns/bit (M6, ≥16 columns shared); u16 column 6.3 ms per 2^17 column (54 in parallel, M2) → 12.6 ms @2^18; full-width column 72 ms per 2^17
→ 144 ms @2^18; limb sumcheck 0.95 s @2^17 (54+9 cols, s=6, t=12, split-eq+fmadd) / 0.88 s (s=3, deg 5); hash sumcheck 0.35 s @2^17 × 163 cols
deg 3; Spartan 2.4 µs/constraint incl. its PCS share; G1 32 B compressed (64 B on EVM), Fr 32 B.

| id | component | rows (L=20) | committed columns | degree | prover s @2^18 | notes / anchors |
|---|---|---:|---|---:|---:|---|
| R | Spartan R1CS: stage algebra 9,895 (plan-relation §2) + Dory Fr scalars ≈600 + Straus digit bits 143×254 = 36.3k booleans + heads/IO ≈100 | m ≈ 47k → 2^16 | W (1 poly, 2^16) | outer 3 / inner 2 | 0.11 | `jolt-blindfold/src/prove.rs:805-1025` fork, plan-relation §7 |
| T1 | Blake3 transcript table (M3 §B): row = half-G step; 116 rows/compression; C_tot 1,980 (1,893) | 229.7k → 2^18 (88%) | 163 bits (131 + 32-col message table, lever L2) | 3 | commit 0.135 (37.4M bits) · sumcheck 0.70 | `crates/jolt-r1cs/src/gadgets/blake3.rs` (round structure), `jolt-transcript/src/blake3.rs` (profile) |
| T2 | non-native limb table (M2 §1–2: 96-bit limbs, 16-bit chunks, CRT mod r·2^288): row = one Fq output coefficient, t ≤ 24 products | Pippenger (M1) 205k/230.6k → 2^18; Straus fixed-wiring (§W) ≈259k → 2^18 (99%) or 2^19 | 54 u16 chunks + LogUp helpers (s=3: 18 fw, deg 5 · s=6: 9 fw, deg 8) + 1 multiplicity | 5 / 8 | chunks 0.68 · helpers 2.6 (s=3) / 1.3 (s=6) · sumcheck 1.8 / 1.9 | GT mul 12 rows, cyc-sqr 12, G1 add 11–15, G2 add 22–30, Miller 14,380, FE 3,288 (M1) |
| T2-ops | operand rows: 42 commitments + 74 proof GT + 38 G1 + 37 G2 as 16-bit chunk rows (range-checked once) | 1,320 + ~500 | (inside T2) | — | — | operands of §1 statement |
| W | wiring/linking sumcheck S2 (M3 §B, M2 §4): `Σ_u Σ_j ρ_j M̃_j(r,u)·col_src(j)(u)`; + L1–L5 | 2^18 (no rows added) | none | 2–3 | ≈0.12 | verifier cost = §W |
| S3 | multi-point → one-point reduction (`Σ_u (eq(r,u)+β·eq(u*,u)+…)·col(u)`) | 2^18 + log(P·k) | none | 2 | ≈0.05 | ends at the single HyperKZG point |
| O | ONE HyperKZG opening of the RLC (eq-weighted, §3) of all P polys, size k·2^18 | — | — | — | k=1 0.34 · k=8 2.7 · k=16 5.4 | `jolt-hyperkzg/src/scheme.rs:128-158`; proof (ℓ+1) G1 + 3ℓ Fr |

Layer-1 prover total (L=20): **k=1 5.3 s · k=8 7.7 s · k=16 10.4 s** (s=6; s=3 +1.0 s). Split @k=8: opening 35%, T2 helpers 17%, T2 sumcheck
25%, T1 sumcheck 9%, T2 chunk commits 9%, rest 5%. M2/M3's "2.5 s" was the 2^17 shapes (C3-only hash rows, 122k fused limb rows).

### W — wiring choice (decides recursion feasibility)
- (a) **Declared directed edges, public sparse matrices evaluated natively** — verifier O(#edges) ≈ 2^18 × 66 wired inputs ≈ 17M Fr mults
  ≈ 0.3 s native, 0 proof bytes. Fine for D1/D2 (native verifier). **Fatal for D3/D4/D5**: 17M constraints in a layer-2 circuit, ≈850M gas on-chain.
- (b) **Structural wiring** (required for D3/D4/D5): T1's 8 wiring shapes (same-G, round permutation, message schedule, chaining) are
  fixed per compression → MLE = eq(compression) × P̃(pos,pos') with P̃ a 128×128 public matrix → O(128²) ≈ 16k mults, or sparse ≈2k; intra-op
  wiring in T2 (12 coefficient rows of an op read the operand op's 12 rows) = row shift by a constant → `EqPlusOnePolynomial::evals`-style kernels
  (`jolt-poly/src/eq_plus_one.rs:71`), O(log). The instance-specific part is the multi-exp schedule: Pippenger's bucket wiring depends on the
  public challenge digits → O(rows). Fix: **Straus w=4 with fixed wiring + public digit selectors** (Dory-assist's exponentiate-and-multiply
  with `digit(s,x)` selector, `/tmp/dory-assist-protocol.md` §GT Semantics): per base k precompute B_k^j, j<16 (15 GT mults, fixed wiring);
  64 steps × (144 selected mults + 4 cyclotomic sqr); selected operand = Σ_j δ(k,step,j)·B_k^j with δ = [digit(k,step)=j] a public 0/1 MLE of
  144×64×16 entries, evaluated as Σ_{k,step} eq(r,(k,step))·eq(r_j, digit(k,step)) → **9.2k terms ≈ 20k mults** (verifier), digits are R1CS
  bits (row R). Cost: GT ops 11.6k (vs Pippenger 6.3k) → +65k rows; same for G1/G2 Straus (≈100k rows, ≈ M1's Pippenger 112k). Alternative with
  fewer rows: keep Pippenger and prove the bucket wiring by offline memory checking over 37 windows × 128 buckets with the 5.3k-entry public
  digit table as addresses (Twist/Shout-style, new argument, +2–3 committed columns, +1 stage).
  **Decision:** (a) for D1/D2; (b)+Straus for D3/D4/D5 (rows 205k → ≈259k; if it does not fit 2^18 after the T2-ops rows, 2^19: opening ×2).

### Row/column shape (brief's reshape question)
Opening cost ∝ (total committed entries)/(#polys) = E/P; proof ∝ P (32 B commitment) + claims + rounds. Reshaping rows↔columns changes
neither E nor the algebra: T1 as (G-step,bit) rows = 56·32·C_tot = 3.5M rows × 14 committed columns → 2^22 (same E = 49M as 163 × 2^18 ≈ 43M,
+14% for per-bit carries); T2 as (Fq output, limb) rows = 3× rows × 19 chunk + 7 helper columns → 2^20 (E unchanged, helper *entries* ×1.1).
Reshaping only lowers P at the same E/P as column packing does, with a harder relation (bit-level ripple carries, rotation as row shifts).
**Decision: keep the M2/M3 row shapes; pack k columns per polynomial** (P = ⌈163/k⌉ + ⌈73/k⌉ + 1: k=1 → 237, k=8 → 32, k=16 → 17; the packed
poly's extra log k variables are the column index; opening size k·2^18).
