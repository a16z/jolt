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
