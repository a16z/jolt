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
| 2 | D1 single layer, size-optimized (k-packing, column batching, one stream, one opening), k=16 / 8 / 1 | **11.3 / 11.7 / 17.7 KB** | — (native verifier only) | n/a on-chain (see D5) | 11.4 / 8.7 / 6.3 s (s=3) | native ≈0.4 s (O(rows) wiring) | universal 2^22 SRS |
| 3 | D3 two layers, Spartan+HyperKZG final (universal setup) | 10.1 KB | — | n/a (needs SPARK: +≈3 KB, ≥1M gas est.) | ≈22.5 s | native ≈0.15 s | universal 2^22 SRS |
| 4 | D2 = D1 + KZG-committed round polys | 11.2 KB (−448 B only: saving exists for degree ≥4 rounds) | — | n/a | +0.05 s | native | universal |
| 5 | D0 current measured shape (lanes M2/M3, 2^17 tables) | 26 KB (≈30 KB for the full statement) | — | n/a | 2.5 s (≈5.3 s full statement) | native | universal |
| 6 | D5 D1 verified directly on-chain | ≈17 KB (D1 + committed wiring/SPARK) | 17 KB | **≈3M** (public-matrix MLEs ≈48k Fr ops) | ≈9 s | EVM | universal |

- Only D4 reaches "a couple of KB". Every single-layer design is floored by its sumcheck round polynomials: 3 stages × 18–28 rounds ×
  (deg 2–5 coefficients) × 32 B ≈ 7.0 KB (6.5 KB with KZG-committed rounds) — no column trick touches that term.
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

## 3. Claims and the single opening (bytes-driven design of the sumcheck stream)

- **Stage A** (rows, 18 rounds): batched T1 row sumcheck (deg 3), T2 row sumcheck (deg 5 at s=3), Spartan outer (14 rounds, scaled 2^4) →
  one point r_A. Round poly = max degree 5 → compressed **5 coefficients = 160 B/round → 2,880 B**.
- **Stage B** (28 rounds, deg ≤3): *column batching* replaces per-column claims — the verifier needs Φ_T(T(r_A,·)) for a fixed polynomial
  Φ of the K column values; write it as a tensor form Σ_{j1..jD} Q̃_D(j1..jD)·Π T(r_A,j_i) with D = degree in columns (T1: 2 → 16 rounds;
  T2 range part: 4 → 28 rounds), each round degree 2 (Q̃ multilinear × one T), ending in D column-point claims T(r_A,s_i); + wiring S2
  (18 rounds, deg 3) + Spartan inner (14, deg 2) in the same batch → **28 × 3 × 32 = 2,688 B**. Verifier evaluates Q̃(s_1..s_D) natively from the
  sparse constraint list (T1 ≈230 terms × 16 mults ≈ 3.7k; T2 ≈ 8k mults). Saves 163+73 claims = 7.5 KB.
- **Stage C** (22 rounds, deg 2): reduce the packed-poly claims at the D+2 points {(r_A,s_i), (u*,·), r_y} to one point → **1,408 B**. With the
  Dory-assist staging invariant (every stage-B instance's suffix challenges = r_A's) stage C disappears (−1.4 KB) but stage B's independent points
  are exactly what column batching needs → kept.
- **Per-poly claims vanish**: the verifier needs T(r,s) = Σ_i eq(s_poly,i)·Poly_i(r,s_col) — open the *eq-weighted* homomorphic combination
  Σ_i eq(s_poly,i)·C_i (P ecAdd/ecMul natively; on EVM P ecMul) at one point with one claimed value. Residual claims ≈ 12 Fr (Spartan az/bz/cz,
  W(r_y), packed evaluations at the D+2 points before stage C) = 384 B.
- **Opening** ℓ = 18 + log k: proof (ℓ+1) G1 + 3ℓ Fr (`hyperkzg/src/types.rs:92-96`): k=1 2,304 B · k=8 2,688 · k=16 2,816. Two openings instead of
  stage C: +2.7 KB, −1.4 KB rounds, +2.7 s prover @k=8 → no.

## 4. Candidate designs — bytes / gas / prover (L=20; L=18 differs by −1 round per stage ≈ −250 B and −5% prover)

Gas model (given): pairing 45k + 34k/pair · ecMul 6k · ecAdd 150 · keccak 30 + 6/word · Blake2b-F precompile (EIP-152) ≈ 12 gas + ≈1k call
overhead per compression (est.) · Fr op ≈50 · calldata 16 B⁻¹ · base 21k. G1 on EVM = 64 B uncompressed (on-chain decompression ≈ modexp ≥1.3k >
the 512 gas of 32 extra bytes).

### D0 — current measured shape (lanes M2/M3), for reference
| item | bytes | notes |
|---|---:|---|
| commitments + per-column claims, ≈237 columns × 64 B | 15,168 | 64 B/column (M2 §3) |
| round polys: T1 S1/S2/S3 17×(4+3+3)×32; T2 17×8×32 (s=6) + wiring/reduction 17×2×2×32; Spartan (14×3+3+14×2+1)×32 | ≈12,300 | separate sumchecks |
| HyperKZG 2^17 + IO 896 | 3,093 | |
| **total** | **≈30.5 KB** (orchestrator's ≈26 KB assumed 8.6 KB of rounds) | **prover 2.5 s @2^17 shapes; ≈5.3 s full statement (both tables 2^18, k=1)** |

### D1 — single layer, size-optimized (§3 stream, k-packing, s=3 → deg 5)
| item | k=1 | k=8 | k=16 |
|---|---:|---:|---:|
| P commitments × 32 | 7,584 | 1,024 | 544 |
| rounds A+B+C (2,880 + 2,688 + 1,408) | 6,976 | 6,976 | 6,976 |
| residual claims 12 Fr | 384 | 384 | 384 |
| HyperKZG opening (ℓ = 18/21/22) | 2,304 | 2,688 | 2,816 |
| public outputs 28 Fr | 896 | 896 | 896 |
| **proof bytes** | **18,144 (17.7 KB)** | **11,968 (11.7 KB)** | **11,616 (11.3 KB)** |
| prover (layer 1, s=3) | 6.3 s | 8.7 s | 11.4 s |
| native verifier | 0.4 s (O(rows) wiring (a)) | 0.4 s | 0.4 s |
Rounds are 60% of D1. s=6 (deg 8) → +1,728 B, −1.3 s. Lever L1 (drop stage C by staging) −1.4 KB only if column batching is replaced by per-column
claims (+7.5 KB) — not a lever here.

### D2 — D1 + KZG-committed round polynomials
Each round: commitment [s_i] (32 B) + s_i(0) and s_i(r_i) (64 B) instead of deg coefficients; one BDFG batched multi-point opening (2 G1 = 64 B) for all
68 rounds, folded into the HyperKZG pairing check by a random scalar (still 2 pairings). Rounds 6,976 → 68×96 + 64 = **6,592**?? — no: deg-5
rounds save 64 B each (160→96), deg-3 rounds 0, deg-2 rounds lose 32 → A 18×64 = −1,152, B 28×0, C 22×(−32) = +704 → net **−448 B**. Sending only
s_i(r_i) and checking s_i(0)+s_i(1)=c_i via an opening of s_i(X)+s_i(1−X) doubles the opened set — the standard "commit rounds" saving is real only
for degree ≥ 4. **D2 saves 448 B (k=16 → 11.2 KB), not ≥1 KB → rejected**; the 8.8 KB figure in §0 assumed 64 B/round and is corrected here to
11.2 KB. Univariate skip on stage A (first round degree 5·2^4 over 16 boolean vars folded): −4 rounds × 160 + 1 × 81·32 = +1.9 KB → no.

### D3 — two layers, layer 2 = Spartan + HyperKZG (universal SRS)
Layer-2 R1CS: Blake3 FS re-hash of the layer-1 proof (k=8: 11,968 B = 187 blocks × 15,536 = **2.90M**), sumcheck algebra 68 rounds × (deg+3) ≈ 600,
Φ_T1/Φ_T2 via sparse Q̃ ≈ 12k, digit-selector MLE ≈ 20k, structural wiring kernels ≈ 10k, HyperKZG fold checks 63, Dory Fr scalars ≈ 600 →
**m ≈ 2.95M → 2^22**. Layer-1 HyperKZG pairing deferred: 21 G1 exposed as public inputs (decompressed natively → 84 Fr limbs, free).
| item | bytes |
|---|---:|
| Spartan 3m+3+2n+1+3n = 180 Fr @m=n=22 | 5,760 |
| W commitment + HyperKZG 2^22 (22 G1 + 66 Fr) | 32 + 2,816 |
| deferred 21 G1 (compressed) + public outputs (Jolt 896 + layer-2 ≈128) | 672 + 1,024 |
| **total** | **10,304 (10.1 KB)** — not 8.3 KB (§0 omitted the deferred points and IO) |
Prover: layer 1 7.7 s + Spartan 2.95M × 2.4 µs = 7.1 s + HyperKZG 2^22 commit ≈2.3 + open 5.4 → **≈22.5 s**. Native verifier: O(nnz) ≈ 9M mults ≈ 0.1 s
+ 2 openings' pairings. On-chain impossible without SPARK (3 sparse matrices, memory checking: +≈3 KB, +2 stages; Fr ops on-chain ≈ 20k → ≥1M gas est.).
D3 is dominated by D1(k=16) on bytes (10.1 vs 11.3 KB is a 1.2 KB gain for +11 s) — keep only if a universal-setup recursion layer is wanted.

### D4 — two layers, layer 2 = Groth16 (top rank)
Layer 2 = Groth16 circuit over the D3 R1CS **minus** nothing on the algebra, **plus** on-chain binding: a Blake2b digest (EIP-152 on-chain) of
(deferred 21 G1 x-coordinates 672 B ‖ Jolt IO ≈100 B) → 7 compressions × 52,416 = **0.37M**; exposed public inputs: digest (1 Fr), B(u_j) (3 Fr),
r, q, d (3 Fr), Jolt `preprocessing_digest`-derived state (1 Fr) → 8 Fr. Circuit **≈3.35M constraints** (2.90M FS hash + 0.37M binding + 0.06M algebra).
| item | bytes (wire / EVM calldata) | gas (est.) |
|---|---:|---:|
| Groth16 A,B,C | 128 / 256 | pairing 4 pairs 45k+136k = 181k; vk_x 8 ecMul + 8 ecAdd = 49k |
| deferred layer-1 HyperKZG: com 20 G1 + w 1 G1 | 672 / 1,344 | L,R MSM ≈24 ecMul + 24 ecAdd = 148k; 2-pair pairing 113k |
| public inputs 8 Fr | 256 / 256 | (in vk_x above); Fr scalar derivation ≈100 ops = 5k |
| Jolt IO (app-defined, e.g. 100 B) | 100 / 100 | binding digest: Blake2b-F ×7 ≈ 8k (Keccak alt.: 0.3k gas, +0.9M constraints) |
| calldata ≈1,960 B; base | | 31k + 21k |
| **total** | **1,156 B wire / ≈1.96 KB calldata** | **≈556k** |
Prover: layer 1 (k=8) 7.7 s + Groth16 3.35M × g µs/constraint (g unmeasured; rapidsnark/gnark class 1–3 µs on 10 cores → 3.4–10 s) → **≈11–18 s**;
CRS: circuit-specific (one per Jolt profile {L, K, σ, N}), size ≈ 3.35M × 3 G1 ≈ 0.6 GB. HyperKZG needs a universal 2^21 SRS (Hermez ptau).
k choice for D4: k=1 → +205 G1 hashed (+6.6 KB → +1.6M constraints), k=16 → −480 B hashed (−0.12M) for +2.7 s opening → **k=8**.

### D5 — D1 verified directly on-chain (no recursion)
Needs a Keccak layer-1 transcript (EVM-native), SPARK for Spartan's A/B/C (or none: the 47k-row R1CS has nnz ≈ 150k → 7.5M gas as Fr ops) and
succinct wiring (§W (b)). Bytes ≈ D1(k=16) 11.3 KB + SPARK ≈ 3 KB + committed digit/wiring data ≈ 2 KB ≈ **17 KB**. Gas: calldata 17 KB → 272k;
Keccak over 11 KB → ≈9k; sumcheck rounds 68 × 8 Fr ops = 27k; Q̃ evaluations 12k ops = 600k; digit-selector MLE 20k ops = 1.0M; structural
kernels 10k ops = 0.5M; SPARK checks ≈ 5k ops = 250k; eq-weighted commitment combination 17 ecMul = 102k; HyperKZG 24 ecMul + 2 pairings = 261k →
**≈3.0M gas**. The O(10^4) public-matrix evaluations are intrinsic to sumcheck-over-tables verification; no single-layer on-chain design
gets under ≈1.5M gas. Ranked last.

## 5. Levers ranked by bytes then gas (each with its price)

| id | lever | Δ bytes | Δ gas (D4) | Δ prover | status |
|---|---|---:|---:|---:|---|
| L6 | O(1)-proof multilinear KZG PCS (Mercury/Samaritan class: ≈4–5 G1 + O(1) Fr, verifier O(1) ecMul + 2 pairings) for layer 1 | D4 −0.5 KB calldata; D1 −2.3 KB | **−110k** (deferred MSM 24 → ≈5 ecMul) | +0.5–1 s (est., new PCS; linear prover) | not in repo; 1–2 days; unmeasured |
| L2 | T1 message table: 32 `m` bit columns → 16-word table of 16·C_tot = 31.7k rows (2^15) referenced as one wired integer (M3 §C) | 0 at k≥8 | 0 | −0.1 s | design only |
| L3 | GT absorbed as 128-B `CompressedFq12` (lane J codec; profile change on the inner transcript): C_tot 1,980 → 1,479; torus map +≈4k limb rows | 0 (rows 2^18 either way) | 0 | −0.2 s (T1 sumcheck) | profile decision |
| L4 | drop per-round labels (schedule-fixed profile): C_tot → ≈1,650 (M3: −16%) | 0 | 0 | −0.1 s | needs FS argument (fixed schedule ⇒ prefix-free) |
| L3+L4 | together: C_tot ≈1,150 → T1 rows 133k → **2^17** | −64 B (−1 round × 3 stages ≈ −0.3 KB) | 0 | −0.6 s (T1 halves, opening unchanged at k·2^18 because T2 stays 2^18) | — |
| L5 | Pippenger + memory-checked bucket wiring instead of Straus (D3/D4/D5 only) | +≈0.3 KB rounds | +≈30k (one more stage) | −65k rows: T2 stays 2^18 with margin | new argument |
| L7 | Keccak instead of Blake2b-F for the D4 binding digest | 0 | −8k | +0.9M constraints (+0.9–2.7 s) | rejected: gas/prover ratio |
| L8 | s=6 helpers (deg 8) in D4 | 0 (layer-1 bytes are witness; +1.7 KB hashed = +0.42M constraints) | 0 | −1.3 s layer 1, +0.4–1.3 s Groth16 | wash; keep s=3 |
| L9 | in-circuit HyperKZG pairing (no deferral) | −672 B calldata | −261k | +≥8M constraints (plain R1CS pairing 8.1M, `prior-art.md`) + non-native 24-point MSM ≫ 30M | rejected |

## 6. Prover pipeline (D4) and time budget

| phase | what | s (L=20) | anchor |
|---|---|---:|---|
| P0 | native verifier replay with recording transcript + Dory-op tracer (plan-relation §8; `jolt-prover/src/blindfold.rs:175-284` pattern) → R1CS witness, T1 bits, T2 chunks/helpers (BigUint witness synthesis M2: ≈1 s → rewrite with u128 limb arithmetic, est. 0.3 s) | 0.3 | `dory-offload-study.md` §1.4 closed form; M1 bench `crates/jolt-dory/benches/deferred_check_counts.rs` |
| P1 | commits: W 2^16 (0.04), T1 bits 37.4M × 3.6 ns (0.135), T2 chunks 54 × 12.6 ms (0.68), helpers 18 × 144 ms (2.6) | 3.5 | M6 `g1_bit_columns_msm`, M2 `g1_msm_small` |
| P2 | stage A (T1 0.70, T2 1.8, Spartan outer 0.05), stage B (S2 0.12, column batching 0.05, inner 0.03), stage C 0.05 | 2.8 | `jolt-sumcheck/src/prover.rs:193 prove_batch` |
| P3 | eq-weighted RLC + one HyperKZG opening at 2^21 | 2.7 | `hyperkzg/src/scheme.rs:128` |
| **layer 1** | | **≈9.3 s** (k=8, s=3; 7.7 s at s=6) | |
| P4 | Groth16 witness (in-circuit replay of the layer-1 verifier) + prove 3.35M constraints | 3.35·g | g = lane N2 |
| **total** | | **≈13 s @g=1 · ≈20 s @g=3** | |

Two biggest risks per top design:
- **D4-R1 (layer-2 size):** the circuit is 87% Blake3 re-hashing of layer-1 bytes (2.9M for 12 KB at 15,536/64 B). Every extra KB of layer-1
  proof = +0.25M constraints; if the column-batching tensor (28 rounds) does not materialize, +7.5 KB of claims = +1.9M constraints (+2–6 s).
  Groth16 throughput g is unmeasured on this machine; at g = 3 µs the wrapper is ≈20 s.
- **D4-R2 (succinct wiring):** Straus fixed wiring puts T2 at ≈259k rows = 99% of 2^18 (+T2-ops rows) → likely 2^19: commits/sumcheck/opening
  ×2 → layer 1 ≈ 15 s; the alternative (L5 memory-checked Pippenger wiring) is a new argument. The digit-selector MLE (20k mults) and Q̃ sparse
  forms must be verified in-circuit as public-data computations (their inputs are R1CS bits) — implementation-heavy, not a soundness risk.
- **D1-R1:** the tensor column batching for degree-4 range constraints (28 rounds) is unbuilt; fallback costs +2.3 KB (T2 claims). **D1-R2:**
  prover 8.7 s at k=8 is 60% opening + helper commits; the only lever is k=1 (17.7 KB) or fewer helper columns (s=6: +1.7 KB).

## 7. Security outline (what changes vs plan-relation §5/§7)
- CRT limb identity: mod r (native) × mod 2^288 (three 96-bit carry equations) with 16-bit range on every chunk ⇒ integer identity, margin 2^15
  (M2 §2). Range checks: LogUp with public inverse table `inv(v) = 1/(α−v)` (2^16 entries, verifier 6 ms native; in layer 2 the multiplicity
  column's LogUp sum needs the same table at a random point = 2^16 mults ≈ 65k constraints — add to D4's algebra: 3.35M → 3.42M).
- Wiring: (a) public 0/1 matrices with one 1 per wired row ⇒ each wired value equals exactly one committed value — soundness = sumcheck
  soundness; (b) structural kernels + Straus selectors: δ is public 0/1 with exactly one 1 per (k,step) ⇒ selected operand = one table entry;
  digit bits are booleanity-constrained R1CS wires of the public scalars.
- FS binding of committed-but-not-sent elements: every absorbed byte of the inner transcript is a T1 message bit linked (L1/L4) to the limb
  chunks that feed the multi-exp ⇒ the elements the multi-exp uses are the ones hashed; `state_pre` is a public input computed natively from
  public data ⇒ commit-then-challenge order preserved (plan-relation §5).
- Deferred HyperKZG check (D4): the on-chain verifier recomputes the Blake2b digest of the 21 G1 points it pairs and matches the Groth16 public
  input ⇒ the paired points are the points hashed in the layer-1 FS transcript (in-circuit). r, q, d and B(u_j) are public inputs derived
  in-circuit from that transcript. Without this binding the prover pairs arbitrary points — not optional.
- Public-input outsourcing (val_io, init_eval, stage_values) unchanged (plan-relation §5); on-chain these are computed by the verifier contract
  from the Jolt IO/program digest (O(|IO|) Fr ops; bytecode-dependent MLEs precomputed per program into the verifier key, or committed-program
  mode — decide at N3).
- Native/EVM verifier work list (D4): Groth16 verify; Blake2b-F × 7; HyperKZG L/R MSM (24 ecMul) + 2 pairings; ≈100 Fr ops. Nothing O(rows).

## 8. Build lanes (waves 3–5), each ≤ 1 agent-day; acceptance tests; what can start now

| lane | scope | depends on | acceptance |
|---|---|---|---|
| N1 (now) | Straus fixed-wiring row count + digit-selector table sizes, exact, on M1's bench (`deferred_check_counts.rs`): GT/G1/G2 rows for w ∈ {3,4,5}; decide 2^18 vs 2^19 | — | printed table; the flattened equation still accepts under the Straus schedule |
| N2 (now) | Groth16 throughput g on this mini (arkworks-groth16 / gnark / rapidsnark) for a 3.4M-constraint Blake3-heavy R1CS; CRS gen time/RSS; Blake3 gadget count in that framework vs 15,536 | — | µs/constraint table; proof bytes 128/256 |
| N3 (now) | Foundry gas measurement of the D4 verifier skeleton: Groth16 verify (8 inputs) + 24-point MSM + 2-pair pairing + Blake2b-F ×7 + 2 KB calldata; Keccak-binding variant | — | gas per component; total vs 556k |
| N4 | column-batching tensor sumcheck (degree-2 and degree-4 forms) on `jolt-wrapper-bench` + `jolt-limb-bench`; eq-weighted commitment combination; k=8 packing + single opening at 2^21 | — | bytes = §4 D1(k=8) ± 100 B; wrong-column-value tamper rejected |
| N5 | T1 full statement: in-table absorption of the 42 commitments + Dory transcript (C_tot 1,980), links L1–L4, structural kernels for the 8 shapes | M3 code | byte-exact state chain vs `Blake3Transcript` on a real 2^18 proof incl. Dory segment; every link checked |
| N6 | T2 full statement: M1's op DAG as rows (GT/G1/G2/Miller/FE) with Straus selectors, operand rows, LogUp s=3, wiring S2 | N1 | final GT equality on a real 2^18 proof; each of 144 base tampers rejected (M1's oracle) |
| N7 | R1CS (plan-relation §4/§9 lanes W2/W4) + Dory Fr scalars + digit bits; witness replay (P0) | — | `check_witness` on fibonacci 2^18/2^20 |
| N8 | Spartan+HyperKZG core with public-input column (plan-relation §7) as an instance of the shared stream (stages A/B) | N4 | random R1CS prove/verify + tamper |
| N9 | layer-1 e2e: wrap fibonacci 2^18 & 2^20, sha2-chain; bytes/prover/verifier table; tamper suite | N4–N8 | D1(k=8) ≤ 12.0 KB; prover ≤ 10 s |
| N10 | layer 2 (D4): Groth16 circuit = Blake3 FS replay + stream verifier + Q̃/δ public computations + Blake2b binding; EVM verifier contract | N2, N3, N9 | Foundry test verifies a real wrapped proof; calldata ≤ 2 KB; gas ≤ 600k |
N1–N4, N7 start now (independent); N5/N6 need only M3/M1 code; N8 after N4; N9 after N4–N8; N10 last.

## 9. Exactness ledger
Exact: C3 (M3), C_tot arithmetic (416/64/96-B encodings, 64-B blocks), M1 rows, M2/M3/M6 unit costs, HyperKZG proof formula, Groth16/EVM element
sizes, gas constants as given. Estimates: Straus row counts (+65k), Blake2b-F per-call gas, Q̃/δ mult counts, layer-2 constraint count (2.95M ± 10%),
Groth16 g, witness synthesis 0.3 s, D3/D5 SPARK sizes, L6 PCS costs.
