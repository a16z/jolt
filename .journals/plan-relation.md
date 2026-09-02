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

## 5. Public IO layout (z = (1, x, w); x is verifier-known, only prover-produced outputs travel in the proof)

Natively absorbed before the circuit starts (`verifier.rs:275-310` `validate_and_seed_transcript`): `T::new(b"Jolt")`, preamble (`absorb_preamble` :580-650: `preprocessing_digest`, `max_input_size`, `max_output_size`, `heap_size`, `inputs`, `outputs`, `panic`, `ram_K`, `trace_length`, `entry_address`, 4 rw-phase counts, `log_k_chunk`, `lookups_ra_virtual_log_k_chunk`, `dory_layout` — each `Label`+payload), then the 41 commitments (`absorb_transcript_commitments` :760-800: `rd_inc, ram_inc, instruction_ra×32, ram_ra×4, bytecode_ra×3`, each `LabelWithCount(commitment,384)` + 384 B) and optional advice/committed-program commitments. The resulting 32-B digest state (`Transcript::state()`, `digest.rs:190-192`) — 1 + 15·2 + 41·5 ≈ 236 native compressions — is public input `x.state_in`.

| public input (verifier-computed, not transmitted) | Fr | public output (transmitted) | Fr |
|---|---|---|---|
| `state_in`: 4 × u64 words of the stage-1 entry digest (decomposed to 256 booleans in-circuit for the first compression) | 4 | `r_ram_addr`: stage-2 batch point, last K coordinates (`RamOutputCheck`/`RamValCheck` address; `ram_read_write_point()[..log_k]`) | K |
| `val_io = ValIo(r_ram_addr)` (`stage2/ram_output_check.rs:50-86`; hi-scale × IO MLE) | 1 | `r_bytecode_addr`: stage-6a address point (12) | 12 |
| `init_eval = Val_init(r_ram_addr)` (`stage4/verify.rs:224-252`) | 1 | `state_rlc`: 4 × u64 digest words after the RLC-γ squeeze (only in the "Dory challenges native" variant) | 0 / 4 |
| `stage_value[s] = fold_stage_values(r_bytecode_addr)[s]`, s<5 (`stage6b/bytecode_read_raf.rs:152-184`) | 5 | Dory interface {41 RLC coefficients γ^i·scale_i, point (L+4), joint claim, 2σ+2 challenges}: internal wires when the Dory sub-circuit lives in the same R1CS; public outputs (41+L+4+1+2σ+2 = 92/96 Fr) if it is a separate proof | 0 / 92–96 |
| Dory proof bytes, iff the offload design keeps them public (32,000 B → 256,000 boolean public entries; else hidden witness + 256k booleans) | 0 / 256k | | |

Transmitted public outputs: K+12 = 28 Fr (896 B) in the same-R1CS variant; ≈ 3.7–3.9 KB if the Dory interface is exported. IO bytes therefore ≈ 0.9–3.9 KB on top of Spartan (3m+2n+4 Fr) + HyperKZG (n G1 + 3n Fr, `hyperkzg/src/types.rs:92-96` after 6634e39f2: `com` = ℓ−1 G1, `w` 1 G1, `v` 3ℓ Fr) — at m=n=2^18: 94 Fr + 19 G1 = 3,616 B; at 2^27 (Blake2b regime): 139 Fr + 28 G1 = 5,344 B.

Soundness of outsourcing: the three O(program) MLEs (bytecode fold O(code_size), initial-RAM MLE O(image), IO MLE O(|IO|)) are functions of public data evaluated at a circuit-derived Fiat–Shamir point. The circuit binds the point as a public output; the native verifier evaluates the public polynomial there and supplies the value as a public input; both enter the same public vector x that the Spartan verifier checks, so the wrapper statement is exactly "verifier accepts with these values at these points" — identical to inlining the evaluation. The prover cannot steer the point (it is a hash output inside the circuit) and cannot alter the value (verifier-computed). Memory layout constants (`io_mask_start/end`, `lowest_address`, entry index) are circuit constants → part of the profile digest (§7).

## 6. Challenges

- Width is governed by the transcript type only: `Transcript::Challenge = F` with the 16-byte squeeze in `DigestTranscript::challenge/challenge_scalar` (`digest.rs:178-188`) and the decoders `from_challenge_bytes` (125-bit, high-limb Montgomery placement ⇒ x = v·2^{-128}) / `from_scalar_challenge_bytes` (128-bit BE) in `jolt-field/src/bn254/mod.rs:172-193`. `OptimizedChallenge` (`jolt-transcript/src/prover.rs:54-75`) belongs to the unused split-trait surface; `ChallengeOps<F>`/`FieldOps<C>` in the lookup tables are generic over the scalar type but instantiated at `F` on every production path (`stage5/instruction_read_raf.rs:190`, `zk/blindfold/stage5.rs:63`).
- Under the Blake2b decision the wrapped profile keeps the production 125/128-bit challenges unchanged — no Jolt prover change, no slowdown. In-circuit conversion is linear in the digest bits (§1) → 0 constraints (≤128 booleans per challenge = 46–48k total only if the Blake2b gadget hands out packed words). The earlier full-Fr requirement (plan v1 §4) is void.
- Code paths that assume the 125-bit form: `from_challenge_bytes` limb placement (must be reproduced bit-exactly by the circuit's linear map, incl. the 3 masked bits), `draw_spartan_outer_tau`/`draw_spartan_product_tau_high` using `challenge()` vs member draws using `challenge_scalar()` (`uniskip.rs:93-118`, `stage2/ram_output_check.rs:114-129`) — the circuit must apply the matching decoder per squeeze. The prover's `mul_u128` fast path (`arkworks fork ff/.../montgomery_backend.rs:1247`) is untouched.

## 7. Spartan + HyperKZG binding (fork of jolt-blindfold, plain R1CS, public-input column)

- Fork, do not parametrize: `prove.rs:805-907` outer (degree-3 `eq(τ,x)·(Az·Bz − Cz)`; drop `u`, `E`, cross-terms, `RelaxedInstance::fold`), `prove.rs:913-986` inner (degree-2 `L_w(y)·W(y)`), `prove.rs:995-1025` `linear_form_project_columns` (O(nnz) single pass — reuse also on the verifier for the public columns), `prove.rs:1052-1069` `abc_at_point`; verifier `verify.rs:129-212` outer check, `verify.rs:419-505` inner, `verify.rs:527-554` `public_contributions`/`compute_l_w_at_ry` generalized from column 0 to the range `[0, 1+|x|)`; matrix evaluators `jolt-r1cs/src/constraint.rs:169-333`. ≈350 lines forked; Hyrax row openings replaced by one `HyperKZGScheme::open/verify` (`hyperkzg/src/scheme.rs:128-158, 168-242`).
- Public inputs: `R1csBuilder` columns `1..=|x|` are x (allocated first), the rest is w; the committed multilinear is w padded to 2^n. Inner claim `α_a·Az(rx) + α_b·Bz(rx) + α_c·Cz(rx)`; the verifier subtracts `Σ_{j∈x} M_j(rx)·x_j` (one O(nnz) pass over the x-columns) and checks `inner.value == L_w(rx,ry)·W(ry)` with `W(ry)` proven by HyperKZG. `alloc_public_scalar`-style constant pinning (branch 08) is not used — the key must be instance-independent.
- Verifier key: `WrapperProfile { L, K, bytecode_log_k, log_k_chunk, N, σ, rw_config, one_hot_config, memory_layout constants, program-image/bytecode/IO sizes }` → deterministic `build_relation(profile) -> ConstraintMatrices` (same builder call on both sides); `vk_digest = Blake2b256(bincode(profile) ‖ dims)` absorbed into the wrapper transcript (the matrices themselves are never hashed per proof — branch 08's O(nnz) absorb was the defect). Verifier cost = rebuild O(m) + `linear_form_bilinear_eval` O(nnz): with m_A + Poseidon-size circuits ≈ 0.6M nnz ≈ 5 ms; in the Blake2b regime nnz ≈ 60–370M ≈ 1–8 s (est.) → SPARK required.
- Proof struct: `WrapperProof { public_outputs: Vec<Fr>, witness_commitment: G1, outer: CompressedSumcheckProof (m rounds × 3 Fr), az, bz, cz: Fr, inner: CompressedSumcheckProof (n rounds × 2 Fr), w_eval: Fr, opening: HyperKZGProof { com: ℓ−1 G1, w: G1, v: [Vec<Fr>;3] } }`; bytes = 32·(3m + 3 + 2n + 1 + 3n) + 32·(1 + n) + 32·|outputs| + bincode prefixes (≈15). Wrapper transcript (outer FS) = `Blake2bTranscript` seeded with `vk_digest`, absorbs `public_outputs`, `witness_commitment`, then τ, outer rounds, `(az,bz,cz)`, `α_{a,b,c}`, inner rounds, `w_eval`, HyperKZG.

## 8. Witness generation (zero duplicated verifier logic)

Replay the native verifier exactly as `crates/jolt-prover/src/blindfold.rs:175-284` does (`validate_and_seed_transcript` → `stage1::verify … stage8::verify`), with two instruments: (1) a `RecordingTranscript<LegacyBlake2bTranscript>` wrapper (implements `Transcript`, forwards every call, logs `(kind ∈ {label, fr, bytes, squeeze}, payload bytes, 32-B state after)` — the log is the Blake2b gadget witness and doubles as the schedule assertion of §4); (2) the clear stage outputs already expose every scalar the circuit needs: wire claims/round coefficients (proof), batching coefficients and member challenges (`StageNBatchChallenges`, `Stage1Challenges`), opening points (`StageNBatchOutputPoints`), derived values (`derive_*_term`, called through `expected_output`/`input_claim`). Assignment order = constraint emission order, produced by running the same `build_relation` closure in "assign" mode (`R1csBuilder::alloc(value)` vs `alloc_unknown`, `builder.rs:183-191`) with a `WitnessSource` that reads the replay log; gadget internals (Horner accumulators, eq chains, inverse hints, S-box intermediates) are computed inside the gadget from their inputs. No verifier formula is re-implemented; the only computed-twice values are the derived-id gadgets (guarded per §4). Debug builds assert gadget outputs against the replay values.

## 9. Module layout, lanes, tests

`crates/jolt-wrapper` (≈4.5k lines est., excluding the two external lanes):
- `profile.rs` (≈120) `WrapperProfile`, `vk_digest`; `relation/mod.rs` (≈200) `build_relation` two-mode driver (emit/assign), stage iteration over the 8 stages; `relation/schedule.rs` (≈250) the FS schedule of §1 as data (labels, counts, decoders) shared by circuit and replay assertion; `relation/sumcheck.rs` (≈150) variable-challenge rounds + uni-skip (fork of `jolt-sumcheck/src/r1cs.rs`); `relation/expr.rs` (≈120) `Expr` lowering with LC sources via `jolt_r1cs::lowering`; `relation/derived.rs` (≈700) gadgets keyed by `JoltDerivedId`: eq, eq-const, LT, EqPlusOne, Lagrange/kernel, stage-1 weights, IoMask, RLC weights, 54 table MLEs (`tables.rs` ≈500 of it, shared products); `relation/public_io.rs` (≈150); `transcript/blake2b.rs` (external lane: compression gadget, parameter B) + `transcript/digest_chain.rs` (≈200) `state‖pad‖counter‖payload` framing, squeeze decoders; `dory/` (external lane, parameter D) consuming the §5 interface; `spartan/{prove,verify}.rs` (≈600 forked); `hyperkzg_binding.rs` (≈80); `proof.rs` (≈120); `witness/{replay,recording_transcript}.rs` (≈350); `wrap.rs` + `verify_wrapped.rs` (≈200); tests (≈900).

Lanes (dependency order; each with its acceptance test):
1. **W1 Spartan+HyperKZG core** — `spartan/*`, `hyperkzg_binding`, `proof`, `profile`. Test: random satisfiable R1CS with |x|=50, m=n=2^16: prove/verify, tamper (flip a round coefficient, wrong x, wrong opening) rejected; serialized size == formula. Independent of the relation.
2. **W2 Gadgets + schedule** — `relation/{sumcheck,expr,derived,tables,schedule}`. Test: per-id parity vs native `derive_*_term` on random inputs; table gadgets vs `LookupTableKind::evaluate_mle` on 200 random points each; constraint-count snapshot table (asserts the §2 numbers ±0 once implemented). Depends on nothing (uses `R1csBuilder` + `check_witness`).
3. **W3 Transcript chain** — `transcript/digest_chain.rs` over the Blake2b gadget (parameter B lane): test byte-exact equality of the in-circuit state sequence with `LegacyBlake2bTranscript` on the recorded schedule (all 3 profiles), and challenge decoding equality (`from_challenge_bytes`, `from_scalar_challenge_bytes`). Depends on the B lane.
4. **W4 Relation + witness** — `relation/mod.rs`, `public_io`, `witness/*`, `wrap.rs`. Test: `check_witness` passes for fibonacci @2^18 (fixtures from `crates/jolt-verifier/tests/support`), schedule assertion, "native re-execution equals witness" (every derived value and every transcript state). Depends on W2, W3.
5. **W5 e2e + gates** — `verify_wrapped`, Dory sub-circuit integration (parameter D lane), size/time table. Tests: wrap+verify fibonacci @2^18 and @2^20, sha2-chain @2^18; tamper: flipped round coefficient (Jolt proof) → wrapper prover fails `check_witness` / verifier rejects when forced; wrong public input value (`init_eval`+1) rejected; wrong Dory proof rejected; table of {Jolt proof bytes, wrapped bytes, wrapper prover ms, verifier ms, m, n, nnz}. Depends on W1, W4, D lane.

## 10. Parameters and open items for the orchestrator

- **B** (Blake2b compression constraints): enters as C·B with C = 2,631 (L=18) / 2,755 (L=20) incl. Dory-challenge derivation, or 2,129 / 2,209 with native derivation from the exposed state. Decompositions 306,832 / 320,548 are B-independent. Any B ≥ 0 → over 2^18; B ≥ 10k → ≥ 2^24.7.
- **D** (offloaded Dory): interface fixed in §5; if the Dory proof is hidden add 235–256k booleans to D; if public, its 29–32 KB travel with the proof (the size floor lane J is measuring).
- Per-table exact counts (§2, est. 4–8k) come out of W2's snapshot test; every other algebra number is a closed formula in (L, K).
- Lookup-free R1CS has no cheaper booleanity than 1 row/bit; the 254-bit decompositions are therefore a hard floor of any byte-oriented transcript (1.17–1.22 × 2^18 by themselves).
