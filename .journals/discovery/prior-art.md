# Prior art — Blindfold, in-circuit pairing costs, deferral, Spartan+HyperKZG sizing, Jolt recursion history

Lane D of wrap/spartan-hyperkzg (card 135). Worktree @ origin/main 756bddce3. Written 2026-09-02.
Read-only lane: no cargo; sources are papers, repos, blog posts, and this repo's git history.
Numbers marked **(est.)** are my derivations; everything else is quoted from the cited source.

## 1. Blindfold (Vega, eprint 2025/2094; Jolt's `jolt-blindfold`)

Paper: Kaviani & Setty, "Vega: Low-Latency Zero-Knowledge Proofs over Existing Credentials", IEEE S&P 2026,
<https://eprint.iacr.org/2025/2094> (BlindFold = §2.1.2 "NovaBlindFold" + §6 "Succinct NovaBlindFold").
Reference implementations: <https://github.com/microsoft/vega-prover> (Spartan2 + Hyrax-PCS + bellpepper), Jolt
`crates/jolt-blindfold` + `crates/jolt-prover/src/blindfold.rs` (book page `book/src/how/blindfold.md`,
<https://jolt.a16zcrypto.com/how/blindfold.html>), Jolt Atlas (<https://arxiv.org/abs/2602.17452>, §3).
Explainer: <https://blindfoldzkp.github.io/>.

### What is inside the verifier R1CS

Per the paper (§6.1.1) the circuit is "a split-committed R1CS structure that encodes the checks of the verifier V on the
prover's plain-text messages and the public-coin challenges". Concretely (Jolt book + Jolt Atlas §3.2):

| Encoded (in R1CS) | Constraints |
|---|---|
| Sumcheck round identity `2c0 + c1 + … + cd = claim` (coefficient form) | 1 per round |
| Chaining `g_j(r_j) = claim_{j+1}` via Horner with auxiliary variables | `d_s` per round (Jolt Atlas: "d_s+1 constraints per round") |
| Uni-skip first rounds: `Σ_k PowerSum[k]·c_k = claim` | 1 per uni-skip round |
| Input-claim binding: initial claim = sum-of-products over prior-stage openings × challenges (`InputClaimConstraint`) | O(#terms) per stage |
| Output binding: `final_claim = Σ_j α_j·y_j` | 1 per stage |
| PCS evaluation binding: `y_com = v·G0 + ρ·H` consistency between folded evaluation/blinding and the Dory ZK evaluation commitment | O(1) per opening |

Size: "The number of constraints in the split-committed R1CS is only a few hundred constraints even when the original
constraint system has a million or more constraints" (Vega §6.1.1, for Spartan); Jolt Atlas §3.2: total
`O(Σ_s n_s·d_s) = O(N·d_max)`, N = total sumcheck rounds — for Jolt that is thousands, not millions, of constraints
**(est.)**.

### What is outside the verifier R1CS

- **Fiat–Shamir / transcript hashing.** Challenges, initial claims and batching coefficients are "baked" into the R1CS
  matrix coefficients (`BakedPublicInputs`); both parties recompute the transcript natively and build identical
  A, B, C. There is no hash in the circuit.
- **The PCS opening proof.** "the polynomial commitment scheme proves evaluations in ZK outside this circuit, so the
  verifier-checks circuit stays O(log m)" (Vega §2.1.2). The circuit consumes only the *committed* evaluation
  (`Cy` / Jolt's `y_com`); the Dory opening argument (GT folding, pairings, Σ-proofs in `dory-pcs` ZK mode) is
  verified natively by the verifier.
- **Sparse-matrix evaluations.** Ã, B̃, C̃ evaluations are public and computed by the verifier (Vega §6.1.1: "the
  split-committed R1CS can simply take as public input purported evaluations of polynomials encoding R1CS matrices").
- **Round-polynomial commitments** are inputs (Pedersen row commitments over BN254 G1 → Hyrax grid), not re-verified.

### Mechanism and proof-size figures

Prover sends Pedersen commitments to round polynomials; the verifier R1CS witness is `Z = [u, coefficient rows…,
non-coefficient rows]`; Nova-fold with a random satisfying relaxed instance (one-time pad); Spartan outer (log m,
degree 3) + inner (log |W|, degree 2) sumchecks; Hyrax-style row openings of W(r_y), E(r_x) against folded row
commitments (Jolt book "Protocol overview" phases 1–6). Vega: witness commitment = Hyrax-PCS (Pedersen rows);
Jolt: same shape (`jolt-blindfold` "Hyrax grid layout").

Proof size: Jolt ZK "increased by only 3KB, with essentially no increase in prover time" (a16z 2026-03-03,
<https://a16zcrypto.com/posts/article/zkvm-jolt-zero-knowledge/>); Vega proofs 83–108 kB (dominated by Hyrax-PCS
√n openings and the credential circuit, not by BlindFold — §8). Jolt Atlas gives no byte figures.

### Consequence for a size-reducing wrapper

BlindFold is instance-specific: its R1CS depends on the challenges, so the verifier must receive every round
commitment (one G1 per round instead of d+1 field elements) and re-derive the whole transcript. It is a ZK layer,
not a compression layer — Jolt's proof stays ~50 KB (+3 KB). The "bake challenges into the matrices" trick is
therefore unavailable to us: a wrapper that hides the inner proof must **replay the Fiat–Shamir transcript
in-circuit** (as `alberto/prover-stack/08-wrapper-verifier`'s `specs/wrapper-protocol.md` does, Poseidon over Fr),
and it must verify the PCS opening *somewhere* — BlindFold simply leaves it native. Reusable pieces: the
`InputClaimConstraint`/`OutputClaimConstraint` sum-of-products vocabulary, Horner chaining layout, the sparse
`VerifierR1CSBuilder`, the Spartan outer/inner prover (`crates/jolt-blindfold`), and the Hyrax row-commitment code.

## 2. In-circuit BN254 pairing / Dory verification costs (R1CS over Fr, non-native Fq)

Unit **C** = cost of one emulated Fq multiplication (including range checks). Fq12 tower costs are multiplication
counts from El Housni, "Pairings in Rank-1 Constraint Systems" (eprint 2022/1162, Table): full Fq12 mul = 54C,
sparse (line) mul = 39C, line mul with dedicated representation = 30C; Granger–Scott cyclotomic squaring ≈ 18C
**(est.: 3 Fq4 squarings × 2 Fq2 mul × 3 Fq mul)**; Fq2 mul = 3C (Karatsuba), Fq6 mul = 18C.

| Operation | Constraints | Source / assumptions |
|---|---|---|
| Fq mul, **plain R1CS** (limbs + bit-decomposition range checks, no lookups) | ≈ 1,000 **(est.)** | xJsnark/OWWB20 limb technique; 4–5 limbs; q, r and carry limbs range-checked bit-by-bit ≈ 1,000 booleanity constraints per mul. Cross-check: circom-pairing-style BN254 pairing = 8.1–8.3 M vs gnark-with-lookups 1.39 M → ≈ 6× ratio. |
| Fq mul, **with lookup range checks** (gnark Groth16 uses commitment-based LogUp range tables) | ≈ 100–150 **(est.)** | derived: gnark Miller loop 706,406 constraints ≈ 6,000 Fq-mul-equivalents (64 doubling steps × ~90C + adds). |
| Fq12 (GT) mul | 54C ≈ 6 K (lookups) / ≈ 54 K (plain) | 2022/1162 Table; gnark v0.12 "direct Fp12 + Eval()" makes one Fq12 mul a single non-native reduction (PLONK pairing 3.53 M → 2.03 M scs, PR #1339) |
| GT exponentiation by a 254-bit scalar (4-bit fixed window: 254 cyclotomic sq + ~64 mul + 16-entry table) | ≈ 8,800C ≈ **1.0 M** (lookups) / **≈ 8–9 M** (plain) **(est.)** | no published single number; torus (T2, Fq6) arithmetic "cost is almost divided by 3" for the final-exp part (El Housni ZK Hackathon notes, <https://github.com/yelhousni/ZKHackathon-zkCircuits>) |
| Full BN254 pairing, gnark **R1CS/Groth16** (with lookup range checks), 2023 | **1,393,318** (Miller 706,406 + final exp 686,912; safe final exp 699,729) | <https://hackmd.io/@yelhousni/emulated-pairing>, <https://hackmd.io/@ivokub/SyJRV7ye2> (ACNS 2023) |
| Multi-pairing size 2 / 4 / 9, gnark R1CS | **1,872,448 / 2,812,614 / 5,163,662** (≈ 480 K marginal per extra pairing) | same |
| Pairing, gnark **PLONK** v0.12 (direct Fp12 + Eval, Jan 2025) | e(a,b): 2,030,447 scs; e(a,b)·e(c,d)==1: 2,239,682 scs; ECPAIR-4: 4,181,633 scs | <https://github.com/Consensys/gnark/pull/1339> |
| Pairing, **circom / plain R1CS**, Jacobian, 5×51-bit or 6×43-bit limbs, bit-decomposition range checks | **8,333,384 / 8,108,878** non-linear (+ 0.48–0.56 M linear) | <https://hackmd.io/@Wimet/ry7z1Xj-2> (circom-pairing techniques; yi-sun/circom-pairing itself targets BLS12-381 at ~2^24–2^25 constraints) |
| Final exponentiation replaced by residue check (prover supplies c with x·c^r = y; merged into Miller loop) | saves ≈ 687 K of the 1.39 M | Novakovic–Eagen, "On Proving Pairings", eprint 2024/640; gnark `AssertFinalExponentiationIsOne` |
| G2 scalar mul (254-bit, affine, Fq2) | ≈ 15 K C ≈ 1.7 M (lookups) **(est.)** | ≈ 254 × 20 Fq2 mul; G1 scalar mul ≈ 1/3 of that |
| arkworks `r1cs-std` `EmulatedFpVar` (formerly `NonNativeFieldVar`) | no BN254-specific published count; ≈ 1000× native for 256-bit emulation over BLS12-381 Fr (constraint-optimised mode) | eprint 2022/1079 Table 1; measure with `cs.num_constraints()` |
| halo2-lib (Axiom) BN254 pairing, PLONKish with lookup range checks (3×88-bit limbs) | benchmark table not retrieved (README fetch 404) — order 10^6 advice cells at k≈19–20 **(unverified)** | <https://github.com/axiom-crypto/halo2-lib> `halo2-ecc` |
| Nova/CycleFold deferral | 0 pairings in-circuit; one scalar mul on the other curve ≈ 1,000–1,500 gates | see §3(a) |

Which numbers assume lookups: every gnark and halo2-lib figure does (gnark Groth16 via its commitment extension +
log-derivative range checks; halo2 via lookup tables). Jolt-style Spartan over plain R1CS has no lookup argument
unless we add one (Lasso/Shout-style range check would itself be a sumcheck family). Plain-R1CS reference points are
the circom numbers (≈ 6× gnark) and the ≈ 1,000-constraint Fq mul.

### Applying this to the Dory verifier (dory-pcs 0.4.0 `DoryVerifierState::process_round`)

Per reduce round the verifier does 10 GT exps (β·D2, β⁻¹·D1, α·C+, α⁻¹·C−, and 3+3 for D1', D2'), ~11 GT mults,
3 G1 scalar muls (E1'), 3 G2 scalar muls (E2') — matching the paper's "10 exponentiations in GT" per Dory-Reduce.
Rounds = log₂ of the longer matrix side (σ), ≈ 12–15 for Jolt-scale batched openings **(est.; lane B measures)**.
Final check: 4 Miller loops + 1 final exponentiation, plus ~4 GT exps and a few G1/G2 scalings.

Collapsed as in the paper (§3.3 "Deferring V computation": one GT multi-exp of size 9m+9, G1/G2 multi-exps of size
4m, one multi-pairing), with Straus shared squarings (w=4) over ≈ 140 GT bases: 254 sq × 18C + 140×64 mul × 54C +
tables 140×15 × 54C ≈ **620 K C** for GT, plus ≈ 170 K C for the G1/G2 multi-exps, plus ≈ 12 K C for the 4-way
multi-pairing → **≈ 0.8 M C ≈ 90 M constraints with lookup-style range checks, ≈ 0.8 B plain R1CS (est.)**.
For scale: a whole 4-pairing is 2.8 M. The GT multi-exp, not the pairing, is the cost centre.

## 3. Deferral / cycle alternatives

### (a) CycleFold / Nova 2-cycle (BN254 ↔ Grumpkin)

- CycleFold (Kothapalli–Setty, eprint 2023/1192, <https://eprint.iacr.org/2023/1192>): the second curve "merely
  represent[s] a single scalar multiplication (≈1,000–1,500 multiplication gates)"; Nova's original 2-cycle needs
  "approximately 10,000 multiplication gates on both curves"; on E1 the CycleFold verifier circuit stays ≈10,000
  gates. Commitments on both curves are Pedersen.
- What the Grumpkin side leaves behind: Pedersen on a non-pairing curve ⇒ "the only option is to use an IPA-based
  evaluation argument" (MicroNova §1) — O(log N) proof but an O(N) verifier MSM ("at least 1,000 group scalar
  multiplications from the on-chain verifier … no precompiles for … Grumpkin, so this is infeasible"); Hyrax instead
  gives O(√N) proof size. MicroNova (Zhao–Setty–Cui–Zaverucha, eprint 2024/2099,
  <https://eprint.iacr.org/2024/2099>) escapes this by proving the Grumpkin polynomial-evaluation *inside a BN254
  MicroSpartan circuit* ("DelegatedSpartan", a fixed ≈1.7 M-constraint R1CS with matrix commitments n1=8, n2=256):
  recursion overhead 74,352 constraints on BN254 + 2,037 on Grumpkin (vs Nova 9,949 + 10,502); compressed proof
  **≈ 11 KB** (+0.2 KB per doubling of N past 2^21), verifier ≈ 14 ms, Solidity verifier **≈ 2.2 M gas**. Nova's
  own compressed proofs (Spartan + IPA) are "≈8–9 KB" (Nova paper) with O(N) verification.
- Jolt's own use of the cycle (a16z "recursion paper" design, see §5): all Fq-heavy Dory verifier work (GT exp/mul,
  G1/G2 scalar mul, Miller loop) is proven natively over Fq = Grumpkin's scalar field with **Hyrax over Grumpkin**;
  final exponentiation + pairing equality stay native. Reported: extended verifier 171–198 M RV64 cycles vs
  1.4–1.9 B for the plain verifier (`recursion_references.md` on the 08 branch). That design optimises in-guest
  cycles, not bytes: a Hyrax opening is O(√n) group elements (dense witness of 2^20–2^24 Fq → 2^10–2^12 × 32 B =
  32–128 KB **(est.)**), so on its own it does not reach single-digit KB; a MicroNova-style re-wrap (prove the
  Grumpkin opening inside the BN254 wrapper) or an IPA (log-size, slow verifier) would be needed.

### (b) Deferred pairing check as public output

The circuit performs the GT folding and exposes the final pairing-equation inputs as public outputs; the native
verifier does the multi-pairing. With dory-pcs 0.4.0 the final check is one 4-way multi-pairing (4 Miller loops +
1 final exponentiation): `e(E1_final + d·Γ10, E2_final + d⁻¹·Γ20) · e(H1, …) · e(…, H2) · e(d²·E1_init, Γ20) =
C + (s1·s2)·HT + χ0 + d·D2 + d⁻¹·D1 + d²·D2_init`. Public outputs needed: C, D1, D2, D2_init ∈ GT (4 × 12 Fq),
E1_final, E1_init ∈ G1, E2_final ∈ G2, scalars s1, s2, d, γ — ≈ 60 Fq ≈ **1.9 KB** of extra public input
**(est.)** (or a hash of them, at the cost of one Poseidon over ~180 Fr limbs). Saves the in-circuit pairing
(1.4–2.8 M gnark-equivalent constraints) but not the 9m+9 GT multi-exp, which stays in-circuit (§2: ≈ 90 M /
0.8 B). Same pattern is what Nova/MicroNova call "deferred" checks and what HyperKZG on-chain verifiers do (all
scalar muls in G1, "a single pairing … Ethereum's pre-compiled contracts", MicroNova §6).

Variant worth flagging: if the inner Jolt proof used **HyperKZG** instead of Dory, the whole PCS verification is
"ℓ+7 scalar multiplications [in G1] and two pairings" (MicroNova §6) and *all of it* can be deferred as public
output (the circuit only has to bind the commitments into the transcript). The Fr-circuit then contains no Fq
arithmetic at all. Cost: Jolt chose Dory for O(√N) URS and pay-per-bit commitments of very long one-hot polynomials
(`book/src/how/dory.md`); HyperKZG needs an SRS of the polynomial length (2^28+ for K^{1/c}·T) and a slower prover.

### (c) Dory-specific: can the verifier do O(1) pairings with prover-supplied helpers?

Dory paper (Lee, eprint 2020/1274, TCC 2021, <https://eprint.iacr.org/2020/1274>):
- Abstract: "Verifier work is dominated by an O(log n) multi-exponentiation in the target group and O(1) pairings";
  proofs are "6 log n target group elements, 1 element of each source group and 3 scalars".
- §3.3 "Concrete costs": "Naively, in each invocation of Dory-Reduce V computes 10 exponentiations in GT"; "Deferring V
  Computation" collapses all rounds so that V's group work is "a multi-exponentiation in GT of size 9m + 9, two
  exponentiations in GT, and one pairing". Extended (§4.4, with G1/G2 messages): GT multi-exp 9m+9, G1 and G2
  multi-exps of size 4m, "3 additional pairings … Whilst naively there are 5 pairings, 2 … can be combined".
- §6.2 Batching ℓ polynomials: messages "(6m + 3ℓ + 5)|GT| + (3m + 2ℓ + 2)(|G2| + |G1|) + 8|F|"; V does "an
  exponentiation in GT of size 9m + 3ℓ + 6, exponentiations in G1 and G2 of size 3m + 2ℓ + 2, and a multi-pairing
  of size 4" plus 2ℓm field ops. Batching amortises *across instances* ("the cost of evaluating each additional
  polynomial commitment is reduced to O(1) group operations and O(log n) additional operations in F") — Jolt already
  opens one batched claim, so there is nothing left to amortise.
- The 9m+O(1) GT multi-exponentiation is intrinsic: the proof's GT elements (D1L, D1R, D2L, D2R, C±) are inner
  pairing products of prover vectors whose G1/G2 pre-images are not succinct, so a scalar cannot be pushed from GT
  into a source group by the verifier. No published follow-up removes it without a SNARK: "On Proving Pairings"
  (2024/640) only replaces the final exponentiation; a16z/dory's own verifier optimisations
  (<https://github.com/a16z/dory/releases>: "5 individual pairings → size-3/4 multi-pairing", VMV check batched at the
  d² slot) leave "O(log n) GT exps and 1 multi-pairing" (`dory-pcs` crate docs). The a16z recursion design (§5) is
  exactly the SNARK route: prove GT exps as Fq-native sumcheck relations (quotient technique
  `a(X)·b(X) = c(X) + Q(X)·p(X)` on the 4-variable hypercube view of Fq12).

Bottom line for (c): the verifier's GT work can be *reorganised* (one multi-exp, Straus/Pippenger sharing, all
pairings in one product) but not *removed*; whoever verifies Dory pays ≈ (9·σ+9) GT exponentiations — natively
(~10 ms) or inside whatever proof system wraps it.

## 4. Spartan + HyperKZG as the outer proof

### Proof-size formula (non-ZK Spartan, no SPARK; Nova `snark.rs` / Jolt `jolt-blindfold` shape)

For m constraints, witness n, ℓ = log₂ n (assume m = n):
- Outer sumcheck: log m rounds, degree 3 → 3 Fr/round with the "omit the linear coefficient" compression Jolt uses
  (4 if uncompressed); then Az(rx), Bz(rx), Cz(rx): 3 Fr.
- Inner sumcheck: log n rounds, degree 2 → 2 Fr/round (3 uncompressed); then W(ry): 1 Fr.
- Witness commitment: 1 G1. (If E/relaxed: +1 G1 and one more opening.)
- HyperKZG evaluation argument (microsoft/Nova `src/provider/hyperkzg.rs`):
  `struct EvaluationArgument { com: Vec<G1> /* ℓ−1 */, w: [G1; 3], v: Vec<[F; 3]> /* ℓ */ }` — (ℓ−1)+3 G1 and 3ℓ Fr;
  Gemini-style split (§2.4.2 of eprint 2022/420) applied to the evaluation form, opened at {r, −r, r²} and batched
  with powers of q; verifier does one aggregated pairing check `e(L, H) == e(R, τH)` (MicroNova §6: "ℓ + 7 scalar
  multiplications and two pairings"). Arecibo adds a Shplonk-style single-point batching variant.

Bytes, compressed G1 = 32 B, Fr = 32 B **(computed from the formula above)**:

| m = n | outer (3ℓ) | claims | inner (2ℓ) | W(ry)+com | HyperKZG (ℓ+2 G1 + 3ℓ F) | **total** | uncompressed G1 (64 B) | + uncompressed round polys |
|---|---|---|---|---|---|---|---|---|
| 2^20 | 1,920 | 96 | 1,280 | 64 | 2,624 | **5,984 B ≈ 5.8 KiB** | +736 → 6.7 KiB | +1,280 → 7.1 KiB |
| 2^22 | 2,112 | 96 | 1,408 | 64 | 2,880 | **6,560 B ≈ 6.4 KiB** | +800 → 7.2 KiB | +1,408 → 7.8 KiB |
| 2^24 | 2,304 | 96 | 1,536 | 64 | 3,136 | **7,136 B ≈ 7.0 KiB** | +864 → 7.8 KiB | +1,536 → 8.5 KiB |
| 2^27 | 2,592 | 96 | 1,728 | 64 | 3,520 | **8,000 B ≈ 7.8 KiB** | | |

Cross-checks from the literature (Table 1 of eprint 2025/908, <https://eprint.iacr.org/2025/908>, n = 2^20):
MicroSpartan (BN254, preprocessed Spartan + HyperKZG) 5 KB; HyperPlonk + HyperKZG 6.1 KB; Gemini 18 KB;
Spartan + KZG 234 KB (naive). MicroNova's whole compressed proof ≈ 11 KB. The 3ℓ Fr of HyperKZG's `v` (1.9–2.3 KB)
is the biggest single line; constant-size multilinear PCSs cut it: Samaritan PCS 7 G1 + 1 F = 368 B (eprint
2025/419), Mercury 8 G1 + 6 F (eprint 2025/385), Vela 2 G1 + 4 F (2026/1438) — all KZG-SRS based, AGM-only
soundness for Mercury/Samaritan (Chopin 2026/480 fixes that). HybridSpartan (2025/908) reaches 1.8 KB at 2^30.
Nova already ships Mercury next to HyperKZG (`nova-snark` README).

### SRS

HyperKZG `CommitmentKey` = n G1 powers of τ + one G2 element τ·H (Nova); no G2 powers needed. BN254 sources:
- Perpetual Powers of Tau (PSE/Semaphore, 2^28, 71+ participants; <https://github.com/privacy-ethereum/perpetualpowersoftau>;
  ~97 GB challenge files). Hermez-prepared truncations as snarkjs `.ptau` (54 contributions + beacon):
  `https://storage.googleapis.com/zkevm/ptau/powersOfTau28_hez_final_{24,25,26,…}.ptau` (snarkjs README table with
  blake2b hashes) — 2^24 is a ~1 GB download; also Groth16-oriented so contains G2 powers we can drop.
- Aztec Ignition (2019–20, 176 participants, 100.8 M G1 points ≈ 1.2 × 2^26, 2 G2 points;
  `aztec-ignition.s3.eu-west-2.amazonaws.com/MAIN IGNITION/sealed/transcriptNN.dat`; tooling
  `Consensys/gnark-ignition-verifier`, `alxiong/ark-srs`; trimmed conversions at `han0110/halo2-kzg-srs`).
- Ethereum KZG ceremony (EIP-4844) is BLS12-381 with 4096 powers — unusable.
A 2^24–2^27 witness fits either ceremony; loading/converting to arkworks form is a one-off script.

### Verifier cost and the SPARK question

Spartan's verifier must evaluate Ã, B̃, C̃ at (rx, ry): O(nnz) field ops (Spartan2 README: "the Spark optimization
is not implemented, so verifier work is proportional to the number of non-zero R1CS entries"). For a wrapper circuit
of 10^6–10^8 constraints that is 10^7–10^9 field ops (~0.1–10 s native; not on-chain). Options:
1. SPARK / computation commitment (Spartan §7; MicroSpartan/eprint 2024/2099 §5 is the current form: preprocess
   commitments to row, col, val_A/B/C; prover adds Lr, Lc + two lookup checks; Lasso-style memory checking).
   Proof grows by a handful of G1 + ~3 more short sumchecks (MicroSpartan total 5 KB at 2^20 — i.e. still ~5–8 KB);
   verifier O(log n) + one more HyperKZG batch. Prover +30–50% **(est.)**.
2. Make the wrapper circuit *uniform* (a repeated round-block, Jolt's own trick) so Ã(rx, ry) has a closed form —
   large engineering, fragile across verifier changes.
3. Accept O(nnz) native verification (fine for "small proof + offline verifier", not for EVM).

On-chain reference for the pairing side: MicroNova's full Solidity verifier (HyperKZG + MicroSpartan + deferred
Grumpkin check) ≈ 2.2 M gas; Jolt book's own pre-Dory estimate for a HyperKZG Jolt verifier was ~2 M gas and
"a couple of dozen KBs" (<https://jolt.a16zcrypto.com/future/on-chain-verifier.html>).
