# Plan — Jolt-verifier relation for the Spartan+HyperKZG wrapper (Phase 1, card 135)

Planner output, 2026-09-02 16:10–16:45, read-only. Tree: origin/main 756bddce3 + lane commits. Binding inputs: `.journals/plan.md` v2 (16:15): transcript = production `LegacyBlake2bTranscript` byte-identical in-circuit (Blake2b compression cost = parameter **B** constraints, other lane); Dory verified in-wrapper by an offloaded sub-circuit (cost = parameter **D**, other lane) whose interface is {41 RLC coefficients, opening point, batched claim, 2σ+2 Dory challenges}; budget m, n ≤ 2^18 = 262,144, wrapper prover < 1 s. Profile: fibonacci/sha2, L = log_T ∈ {18, 20}, K = 16 (fib K = 13 in one column), bytecode log_k = 12, log_k_chunk = 4, N = 41 commitments, σ = ⌈(L+4)/2⌉ = 11/12. "exact" = counted from code; "est." marked.

## 0. Verdict

**NO-GO on the 2^18 budget with the Blake2b transcript — for every B, including B = 0.**

| L | compressions C (stages 1–7 + RLC γ) | + Dory-challenge derivation | Fr bit-decompositions (exact) | stage algebra m_A | total m |
|---|---|---|---|---|---|
| 18 | 2,129 | 502 (σ=11) | 1,208 × 254 = 306,832 | 9,701 | 2,631·B + 316,533 + D |
| 20 | 2,209 | 546 (σ=12) | 1,262 × 254 = 320,548 | 9,895 | 2,755·B + 330,443 + D |

- The 254-bit decompositions of the 1,208–1,262 absorbed Fr alone (306–321k booleans, one R1CS row each — no cheaper booleanity exists in plain R1CS) exceed 262,144. The transcript never fits: required B ≤ (262,144 − 330,443)/2,755 < 0.
- With B ≥ 10k (certain): 2,755·10k = 27.6M constraints ≈ 2^24.7 → ≥ 105× the budget; at the pre-estimate B = 45k: 124M ≈ 2^26.9 → 470×. Measured cost model (`throughput.md`/`hyperkzg-perf.md`: ≈2.4 µs/constraint at 2^20, PCS-dominated) → wrapper prover ≈ 65 s (B=10k) … ≈ 5 min (B=45k, est.); at B=45k the SRS (2^27 G1 affine ≈ 8.6 GB) + matrices (nnz ≈ 2–3·m ≈ 300M entries ≈ 12 GB) do not fit 16 GiB.
- Even hashing only the squeezes would not fit: 398 squeezes × B ≥ 4M. Any Blake2b-in-circuit design is ≥ 2^22 (16× over) and that floor ignores the decompositions.
- Everything not transcript-related fits easily: stage algebra 9.7–9.9k (of which lookup tables ≈6k est.), witness ≈10.8k.
- Only design under 2^18 found: field-native sponge (rejected by the user 16:15): duplex Poseidon t=8/rate 7, labels kept, 402/417 permutations × 384 = 154k/160k (L=18/20) + algebra 9.9k → leaves ≤ 92k for D. Recorded for the decision table only; not designed here.
- Decision rule: keep Blake2b ⇒ accept m ≈ 2^25–2^27 (minutes, SPARK needed for the verifier) or send round polys in the clear (no size win); keep < 1 s ⇒ change the inner transcript to an algebraic hash and re-run this plan (§§4–9 hold unchanged either way).

## 1. Transcript accounting (Blake2b-256, exact from code)

Mechanics (`crates/jolt-transcript/src/digest.rs`): `hasher()` (:91-96) = `Blake2b256(state[32] ‖ 0^28 ‖ n_rounds_BE[4] ‖ payload)`; `append_bytes` (:173-176) hashes that and sets `state`; a squeeze (`challenge`/`challenge_scalar`, :178-188 → `challenge_bytes32` :115-120) hashes `state ‖ round_word` (64 B) and keeps the first 16 of the 32 output bytes. Blake2b has no padding block: compressions = max(1, ⌈bytes/128⌉). Hence: Fr append (32 B BE, `legacy.rs:120-129`) = 96 B → **1**; `Label`/`LabelWithCount`/`U64Word` (32 B, `legacy.rs:149-194`) → **1**; empty append (stage-4 separator, `stage4/ram_val_check.rs:556-564`) = 64 B → **1**; every squeeze → **1** (16 B used; `challenge_vector(n)` = n squeezes; `challenge_scalar_powers(n)` = 1 squeeze). Dory adapter (`crates/jolt-dory/src/transcript.rs:27-76`): `append_serde` = `LabelWithCount(dory_serde, len)` + bytes → GT 384 B = 1 + ⌈448/128⌉ = **5**, G1 32 B = **2**, G2 64 B = **2**.

Per-call building blocks: compressed round with d wire coefficients (`jolt-sumcheck/src/verifier.rs:110-129`) = `LabelWithCount(sumcheck_poly,d)` + d Fr + 1 squeeze = **d+2**; uni-skip round (`round_proof.rs:80-86`, `uniskip.rs:137-171`) = label + (deg+1) Fr + squeeze + `append_labeled(opening_claim)` 2 = **deg+5**; batch head with m members (`jolt-verifier-derive/src/lib.rs:566-604`, `jolt-sumcheck/src/lib.rs:114-120`) = m × `append_labeled(sumcheck_claim)` (2 each) + m `challenge_scalar` = **3m**; opening claims (`relations.rs:39-51`) = **2 per absorbed claim** (aliases skipped).

Sequence and counts per stage (label appends / Fr appends / squeezes / compressions; L=18,K=16 | L=20,K=16 | fib L=18,K=13):

| Stage | Absorb/squeeze sequence (FS order) | labels | Fr | squeezes | compressions |
|---|---|---|---|---|---|
| 1 | `tau` = L+2 squeezes; uni-skip 1+28 Fr, 1 sq, claim 2; head(1); (1+L) rounds d=3; 35 claims | 57 / 59 / 57 | 122 / 128 / 122 | 41 / 45 / 41 | 220 / 232 / 220 |
| 2 | `tau_high` 1 sq; product uni-skip 1+7 Fr, 1 sq, claim 2; `gamma_rw`, `gamma_icr`, K raw `challenge()` = 2+K sq; head(5); (L+K) rounds d=3; 15 claims | 56 / 58 / 53 | 130 / 136 / 121 | 59 / 61 / 53 | 245 / 255 / 227 |
| 3 | 3 sq; head(3); L rounds d=3; 13 claims | 34 / 36 / 34 | 70 / 76 / 70 | 24 / 26 / 24 | 128 / 138 / 128 |
| 4 | 1 sq; `LabelWithCount(ram_val_check_gamma,0)` + empty append; 1 sq; head(2); (L+7) rounds d=3; 7 claims | 36 / 38 / 36 | 84 / 90 / 84 | 29 / 31 / 29 | 149 / 159 / 149 |
| 5 | 2 sq; head(3); 128 rounds d=2; L rounds d=10; 66 claims | 215 / 217 / 215 | 505 / 525 / 505 | 151 / 153 / 151 | 871 / 895 / 871 |
| 6a | 7 sq (6 bytecode γ, booleanity γ; reference address is stage-5 data, no draw — `stage6a/booleanity.rs:83-104`); head(2); 8 rounds d=2 + 4 rounds d=3; 2 claims | 16 | 32 | 21 | 69 |
| 6b | 2 sq; head(6); L rounds d=5; 80 claims | 104 / 106 / 104 | 176 / 186 / 176 | 26 / 28 / 26 | 306 / 320 / 306 |
| 7 | 1 sq; head(1); 4 rounds d=2; 39 claims | 44 | 48 | 6 | 98 |
| 8 RLC | `LabelWithCount(rlc_claims,41)` + 41 Fr; 1 sq (γ) (`stage8/verify.rs:207-215`) | 1 | 41 | 1 | 43 |
| **Σ 1–7+RLC** | formulas: labels 6L+K+439; Fr **674+3K+27L**; squeezes 7L+2K+200; **C₁₋₈ = 1,313 + 6K + 40L** | 563 / 575 / 560 | 1,208 / 1,262 / 1,199 | 358 / 372 / 352 | **2,129 / 2,209 / 2,111** |
| 8 Dory challenges | `vmv_c`,`vmv_d2` GT, `vmv_e1` G1 (12); per round k<σ: 4 GT + G1 + G2, sq β, 2 GT + 2 G1 + 2 G2, sq α (44); sq γ, `final_e1`,`final_e2`, sq d (6) (`dory-pcs-0.4.2/src/evaluation_proof.rs:360-363, 429-443, 446, 470-473`) | — | 0 Fr; 29,408 / 32,000 proof bytes | 24 / 26 | **C_D = 44σ+18 = 502 / 546** |

Wire-coefficient check: Fr appends = round coefficients (351+3K+27L = 885/939, matches lane C's measured 876 stored Fr at fib 2^18 + uni-skip padding) + 23 input claims + 259 opening claims + 41 RLC claims.

In-circuit costs beyond B per compression (exact):
- Absorbed Fr → 32 B BE bytes: the bits are the primary witness (254 booleans `b(b−1)=0`; top 2 bits of the word are constants 0), the Fr value is the LC Σ 2^i b_i (free). No canonical (< p) check: a non-canonical encoding is just a different prover message hashed by the oracle — FS soundness counts hash queries, not encodings; completeness uses the canonical bytes. **254 × (674+3K+27L) = 306,832 / 320,548 / 304,546.**
- Labels, counters, zero padding: constants → 0 (they only cost the compression).
- Challenge extraction: `from_challenge_bytes` (`jolt-field/src/bn254/mod.rs:172-183`) takes the 16 LE bytes as u128, masks the top 3 bits, and places the 125-bit value v in Montgomery limbs [0,0,lo,hi] ⇒ field value x = v·2^{-128} mod p — a fixed linear map of the digest bits; `from_scalar_challenge_bytes` (:189-193) = BE u128 < 2^128 ⇒ x = Σ 2^i b_i. **0 constraints** if the gadget exposes digest bits (any Blake2b gadget does — its 64-bit adds/xors are bitwise); the pre-estimate "~256 booleans per challenge" applies only if the gadget outputs packed words (then 128 per challenge: 358–372 × 128 ≈ 46–48k).
- Dory challenge derivation: the 29.4–32.0 KB of absorbed Dory-proof bytes must be bits: **235,264 / 256,000 booleans if the Dory proof is hidden witness** (alone ≈ 2^18); 0 if the Dory proof is a public input (then it is transmitted — the offload lane's call). Cheaper and equally sound variant: expose the 32-B state after the RLC-γ squeeze as 4 public u64 words; the native verifier continues the identical Blake2b chain over the (public) Dory proof and feeds the 2σ+2 challenges to the sub-circuit as public inputs — saves C_D·B and the 256k booleans.

Native cost model that the count replaces: 1 chained compression per 32-B item is intrinsic to `DigestTranscript` (state re-hashed on every append); a transcript that packed 4 Fr per block would still need ≥ 398 squeeze compressions + 1,262 decompositions.

## 2. Stage algebra (exact by gadget formula; tables est.)

Gadget costs (multiplication constraints; linear constraints folded into LCs): eq(n) both-witness = 2n (`u_i = x_i·y_i`, `acc·(1−x_i−y_i+2u_i)`); eq against constants = n−1; LT(n) = 3n (`lt.rs:125-134`); EqPlusOne(n) = 5n via the O(n) prefix/suffix recursion (native `eq_plus_one.rs:37-57` is O(n²); same function); Lagrange over N centered nodes at a witness = 3N (N−1 product chain + N inverse hints `inv·(r−x_i)=1` + N products; `lagrange.rs:20-65`); `centered_lagrange_kernel(N)` = 7N (`lagrange.rs:92-104`); Horner degree d = d.

| Stage | Pieces (file anchors) | L=18 | L=20 |
|---|---|---|---|
| 1 | uni-skip power-sum (linear) + Horner 27; remainder output = TauKernel·Az·Bz in factored form: Lagrange(10) 30 + 19 row weights (`jolt.rs:141-170`) + 35+35 weight·opening products (`AzWeight(i)` are LCs of the 19 weights — the 1,296-term expansion is native only) + kernel(10) 70 + eq(L+1) + 2 (`jolt.rs:224-262, 299-321`) | 256 | 260 |
| 2 | product uni-skip Lagrange(3) 9 + 3 + Horner 6; product remainder Lagrange(3) 9 + kernel(3) 21 + eq(L) + 8 (`stage2/product_remainder.rs:120-170`); RamRW eq(L)+3; InstrCR eq(L)+10; RamRaf `UnmapAddress` linear +1; RamOutputCheck eq(K) + IoMask 2K (`mle.rs:89-111`, constant bounds) + 2; ValIo public input | 244 | 256 |
| 3 | 2 × EqPlusOne(L) + 12; eq(L)+6; eq(L)+4 (`stage3/*`) | 274 | 302 |
| 4 | eq(L)+5; LT(L)+4; `InitEval` public input (`stage4/verify.rs:224-252`) | 99 | 109 |
| 5 | eq(L); 54 lookup-table MLEs at the 128-var witness point with shared `x_i·y_i` (`tables/mod.rs:8-125, 151-156`; linear tables 0, eq/lt chains 63–128, shift/rot tables 64–128, `Pext` 128) **≈6,000 est. (4–8k)**; 54×2 eq·T·flag; RAF 6; Π ra 8; RamRaCR 3·eq(L)+3; RegsVal LT(L)+2 | 6,325 | 6,347 |
| 6a | 88 β-powers + 88 products + 7 (`geometry/bytecode.rs:85-104`); booleanity 0 | 183 | 183 |
| 6b | bytecode 5·eq(L) + eq-const(12) + 2 + 8 (StageValue(s) public inputs); booleanity eq(4+L) + 39 squares + 39 γ-powers + 39; RamHB eq(L)+2; RamRaV eq(L)+3; InstrRaV eq(L)+39; IncCR 4·eq(L)+6 | 664 | 716 |
| 7 | 117 γ-powers + 40 products; output 8 + ≈15 distinct 4-var eqs (est.) + 78 (`stage7/hamming_weight_claim_reduction.rs:297-331`) | 363 | 363 |
| 8 RLC | 40 γ-powers + 41 claim·coefficient + 3 embedding scales (`committed_openings.rs:87-107`) | 84 | 84 |
| heads | 23 coeff·claim + 23 coeff·expected_output; 2^k scalings linear; alias checks linear | 46 | 46 |
| rounds | Horner mults = wire coefficients 351+3K+27L; + 1 running-claim equality per round (6L+K+154 rounds; uni-skip rounds included) | 1,163 | 1,229 |
| **m_A** | | **9,701** | **9,895** |

Witness variables for this part: 2 × wire coefficients (coefficient + Horner accumulator) + 1 per other mult + 266 wire claims ≈ **10,574 / 10,810**; challenges are LCs of digest bits (no variables). Public-IO wiring adds ≈ 40 rows.

## 3. Totals versus the budget

m = m_A + C₁₋₈·B + 254·Fr + [C_D·B (+ 256k if the Dory proof is hidden)] + D; n ≈ m (each mult introduces one variable; Blake2b gadgets are ≈1 variable per constraint); nnz ≈ 2–3·m (est.; Blake2b add/xor rows have 2–3 nonzeros; Poseidon-style dense rows do not occur).

| design | L | m (B=10k) | m (B=45k) | 2^18? | wrapper prover (est., 2.4 µs/constraint) |
|---|---|---|---|---|---|
| Blake2b, Dory challenges in-circuit, Dory proof public | 18 | 26.6M + D | 118.7M + D | no (101–453×) | 64 s / 4.7 min |
| Blake2b, Dory challenges in-circuit, Dory proof public | 20 | 27.9M + D | 124.3M + D | no | 67 s / 5 min |
| Blake2b, Dory challenges native from exposed state | 20 | 22.4M + D | 99.7M + D | no | 54 s / 4 min |
| decompositions only (B = 0, hypothetical) | 20 | 330k + D | — | no (1.26×) | — |
| field-native sponge (rejected 16:15; for the record) | 20 | ≈170k + D | — | yes if D ≤ 92k | ≈0.6 s |

Verifier side (no SPARK): Spartan's O(nnz) A/B/C evaluation at 60–370M nnz ≈ 1–8 s natively — a second consequence of Blake2b-in-circuit; SPARK (or committing to the fixed matrices once) becomes mandatory in that regime.

## 4. Relation construction strategy — decision: (a) declarative walk + one gadget per derived id; (b) rejected

- **(a)** Emit the circuit from the existing symbolic layer, not from `stages/zk/blindfold` (which bakes challenges/publics as matrix constants, `zk/blindfold/mod.rs:296-315` `map_jolt_expr`, and consumes ZK-only `CommittedSumcheckConsistency`). Inputs: per stage the member list with `SymbolicSumcheck::{rounds, degree, domain, input_expression, output_expression}` (`jolt-claims/src/symbolic.rs`), `ConcreteSumcheck::{instance_point_offset, aliased_output_openings, wire_output_openings}` (`jolt-verifier/src/stages/relations.rs:171-226`), the batch fold rule `Σ coeff·claim·2^(max−rounds)` / `Σ coeff·expected` (`jolt-verifier-derive/src/lib.rs:566-604, 857-946`), and the FS schedule of §1. Lowering: `Source::Opening` → witness var; `Source::Challenge` → LC of digest bits (sponge output); `Source::Derived(id)` → gadget selected by `JoltDerivedId` variant (`jolt-claims/src/protocols/jolt/ids.rs:466-495`; the 24 `*Public` enums list ≈30 leaf kinds: eq/LT/EqPlusOne/Lagrange/kernel/RLC-weights/table-MLEs/UnmapAddress/IoMask and the 3 outsourced values). `jolt_r1cs::lowering::{ClaimSourceTable, lower_claim_expr}` (`lowering.rs:51-147, 178+`) already accepts `SourceValue::LinearCombination` for any source, so `Expr` lowering with variable challenges needs no new type. Round constraints: fork `jolt-sumcheck/src/r1cs.rs:283-330` to take the challenge as an LC (Horner `acc·r` products instead of `polynomial_eval_lc` constants) — ≈40 lines; keep the layout/validation code (:95-281).
- **(b)** Making `ConcreteSumcheck` generic over a `Var` type: every stage impl computes deriveds with `F: JoltField` methods (`try_eq_mle`, `LtPolynomial::evaluate`, `centered_lagrange_evals` with `inverse()`, `LookupTableKind::evaluate_mle<F, C>` requiring `F: JoltField + FieldOps<C>`), over 23 impl files + 8 stage drivers + the derive macros hardwired to `F` (`jolt-verifier-derive`). A symbolic `F` would have to implement `JoltField` (inverse, random, serialization, `Accumulator`) — hundreds of lines of trait surface, touching every relation file. `specs/symbolic-sumcheck.md:30-31` names the symbolic layer, not a generic-`F` verifier, as "the natural input to … the wrapping pipeline". Rejected.
- **Drift risk and its guards.** Duplicated logic = the derived-id gadget table (native `derive_input_term`/`derive_output_term`, `relations.rs:247-275`, vs. the circuit gadget). Guards: (1) per-id parity test — random points/challenges, gadget witness value == native `derive_*_term`; (2) the witness generator is the native verifier replay (§8), and a debug-mode `assert_eq!` compares every gadget output with the replay's value while assigning; (3) e2e: a natively accepted proof is accepted by the circuit (fibonacci @2^18/2^20, sha2-chain) and tampered proofs are rejected by both. Schedule drift (transcript order) is caught by (2): the replay's recorded absorb/squeeze log must equal the circuit's schedule (assert on lengths and labels).
