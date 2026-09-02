---
created: 2026-09-02
updated: 2026-09-02
tags: [jolt, wrapper, discovery, lane-e, prior-art]
---
# Lane E — `alberto/prover-stack/02…08` as prior art for the Spartan+HyperKZG wrapper

Read-only audit of the June-2026 stack by Markos Georghiades / Andrew Tretyakov on remote `alberto`
(= `origin/prover-stack/*`, identical tips). Worktree: a16z/jolt @ origin/main 756bddce3.
Citation format: `NN:path:line` = file at branch `alberto/prover-stack/NN-*`. "est." = estimate; everything else is read from source.
Lane A (`blindfold-and-r1cs-tooling.md`) already covers the file inventory and BlindFold/HyperKZG restore; this note goes one level down.

## Verdict (7 lines)

- The stack is a **verifier-side scaffold**: 08 verifies Spartan+HyperKZG over an *arbitrary* `ConstraintMatrices<Fr>` (real math, 22 tests, test-gated prover), but the only relation ever built is a 1-round toy; no Jolt-verifier relation, no witness generator, no production prover.
- **Dory assist (06/07) is not a byte reducer.** It re-exposes the *entire* Dory proof as `Fq` public claims and adds its own (larger) proof; it exists to cut verifier *cycles* (recursion paper), and its Dory-reduce transitions are coefficient-linear placeholders that do not encode the GT/G1/G2 group law — fixtures are all-zero synthetic witnesses.
- The pairing is never in-circuit: Miller-loop output is a public GT value, final exponentiation + 4-pair equality + 4 GT exponentiations stay native.
- 03's gadgets are sound but heavy: `FqVar::mul` = 2 716 constraints, `add` = 1 352, Grumpkin complete add ≈ 45, 254-bit scalar mul ≈ 12–24k → any Fq/Grumpkin work inside an Fr R1CS is out of budget without lookups. Native-Fr pieces (variable-challenge sumcheck = deg+2 constraints/round, Poseidon = 264/permutation) are cheap and reusable.
- The stack tip **does not build**: 07/08 are not workspace members (08:Cargo.toml:26-40), `jolt_verifier::PcsProofAssist` is never declared (07:crates/jolt-verifier/src/lib.rs:1-22), Andrew's "adapt to stack" fixes (2026-06-17) stop at 06; base e0d5d7eb2 is 143 commits behind main with a jolt-field trait reshuffle in between.
- Two design defects to fix before reuse: public inputs are pinned as **matrix constants** (per-instance key) and the verifier **absorbs every matrix entry** per proof (no VK digest).
- Reuse: port 08 verifier + `wrapper_spartan_hyperkzg` facts + 03 native gadgets (≈1.5–2.5 days est.); skip Dory assist, Hyrax, HyperKZG-zk unless the program decides on assist/ZK.

## 0. Stack map

| Layer | Tip (author, date) | Adds | Builds at tip? |
|---|---|---|---|
| 02 claims formulas | dc31a0bc8 Markos 06-09 + ce66ad5ed/c16f22953/62aab0606 Andrew 06-17 | field-inline + spartan/bytecode formulas in jolt-claims, `specs/field-inline-protocol.md` (1270), `specs/selected-verifier-integration.md` (820); +5 664/−94 vs base | adapted (CI edits) |
| 03 r1cs extensions | df912b579 Markos + cbe3ca691/774da69fd Andrew 06-17 | `jolt-r1cs/{nonnative,scalar}.rs`, `jolt-sumcheck/r1cs.rs` +490, `jolt-transcript/r1cs/*`, `jolt-poly/r1cs.rs`, `jolt-openings/r1cs.rs`, `jolt-crypto/{r1cs.rs 1888, ec/grumpkin 347}`, `jolt-field/arkworks/bn254_fq.rs`; +11 831/−986 | adapted |
| 04 hyrax | ab30576ca Markos 06-09 | `crates/jolt-hyrax` (+1 722) | in workspace |
| 05 hyperkzg-zk | bea32e1c5 Markos + fd2a7a399 Andrew 06-17 | ZK mode + SRS files in `crates/jolt-hyperkzg`, `specs/hyperkzg-zk.md` (1428); +4 889/−572 | adapted |
| 06 pcs-assist formulas | 8af0b23f2 Andrew 06-17 (squash) | `jolt-claims/protocols/dory_assist/*` (≈10.3k lines), `jolt-verifier/src/pcs_assist.rs` (100) + `stages/stage8/final_openings.rs` (595) — **both orphan files, never `mod`-declared** (07:crates/jolt-verifier/src/lib.rs:1-22, 07:…/stages/stage8/mod.rs:1-7), jolt-dory artifact API (+1 498), `specs/dory-assist-protocol.md` (2870); +15 640/−127 | formulas yes; orphans not compiled |
| 07 dory-assist verifier | bc2d34525 Markos 06-09 | `crates/jolt-dory-assist-verifier` (src ≈7.6k, tests ≈4.9k, 111 `#[test]`) | **no** — not a member; imports undeclared `jolt_verifier::PcsProofAssist` (07:…/src/verifier.rs:11) |
| 08 wrapper verifier | 439746503 Markos 06-09 | `crates/jolt-wrapper-verifier` (src ≈1.7k, tests ≈4.4k, 22 `#[test]`), `jolt-claims/protocols/wrapper_spartan_hyperkzg/*` (494), `specs/wrapper-protocol.md` (1657), `recursion_references.md` (907) | **no** — not a member |

Spec-only branches `alberto/stack/11…15` (2026-05-25, bot commits) are older drafts; the code branches carry the newer specs (dory-assist +1 392/−86, wrapper +685/−189 vs the spec branches). Commit messages are one-liners; intent lives in the specs.

## 1. Dory assist — the protocol as written (06 + 07)

### 1.1 What the Dory verifier does (the work being offloaded)
Per reduce round (state `C,D1,D2 ∈ GT`, `E1 ∈ G1`, `E2 ∈ G2`, `s1,s2 ∈ Fr`; challenges α,β; setup χ_i, δ_i) — 06:specs/dory-assist-protocol.md:1142-1175 (additive notation, i.e. GT "+" is Fq12 multiplication, "β·D2" is a GT exponentiation):
```
C'  = C + χ_i + β·D2 + β⁻¹·D1 + α·C₊ + α⁻¹·C₋            (4 GT exps + 5 GT muls)
D1' = α·D1_L + D1_R + αβ·δ1L_i + β·δ1R_i                   (3 GT exps)
D2' = α⁻¹·D2_L + D2_R + α⁻¹β⁻¹·δ2L_i + β⁻¹·δ2R_i           (3 GT exps)
E1' = E1 + β·E1β + α·E1₊ + α⁻¹·E1₋      E2' = E2 + β⁻¹·E2β + α·E2₊ + α⁻¹·E2₋
s1' = s1·fold1   s2' = s2·fold2
```
Final check (07:crates/jolt-dory-assist-verifier/src/native_final.rs:73-111):
```
rhs = C + ht^(s1·s2) + χ_0 + D2^d + D1^(d⁻¹) + D2_init^(d²)                    (4 GT exps)
e(E1 + d·g1₀, E2_fin + d⁻¹·g2₀) · e(h1, −γ·(E2 + d⁻¹s1·g2₀)) · e(−γ⁻¹·(E1 + d·s2·g1₀), h2) · e(d²·E1_init, g2₀) == rhs
```
The assist proves everything up to the Miller-loop output of that 4-pair product; the verifier does the final exponentiation and the `== rhs` comparison natively.

### 1.2 What the prover sends (`DoryAssistProof`, 07:…/src/proof.rs:18-26)
- `dimensions: DoryAssistDimensions` — GT/G1/G2/Miller/DoryReduce/Wiring/Packing log-sizes.
- `stages.stage1: Vec<Stage1RelationProof { id, sumcheck: {rounds, degree, domain}, sumcheck_proof: CompressedSumcheckProof<Fq> }>` — 24 relations, +2 (`DoryReduceStateChain`, `DoryReduceBoundary`) when reduce_rounds > 1 (07:…/stages/stage1/inputs.rs:94-138). Relations: GtExponentiation(+DigitSelector, BasePower, DigitBitness, Shift, Boundary), GtMultiplication, G1/G2 ScalarMultiplication(+Shift, Boundary), G1/G2 Addition, MillerLoop{LineStep, LineEvaluation, PairProduct, Accumulator, Boundary}, DoryReduce{Gt,G1,G2}Transition, DoryReduceScalarFold.
- `stages.stage2: Vec<DoryAssistCopyConstraint>` — the copy-edge stencil; the verifier recomputes the canonical stencil and rejects anything else (07:…/stages/stage2/verify.rs:36-45), so this is redundant data.
- `stages.stage3: { packed_eval: Fq, reduced_openings: Vec<OpeningId> }`.
- `claims`: every Stage-1 opening claim (hundreds of `Fq`) + public claims, incl. `input: DoryAssistInputPublicClaims { checked_input_digest, verifier_setup_digest, verifier_setup_artifacts, dory_proof_artifacts, jolt_commitments, jolt_evaluation_claims, dory_reduce_initial_e2, transcript_scalars }` (07:…/src/proof.rs:169-204) — i.e. **the whole Dory proof, setup and commitment re-encoded as Fq coefficients** (GT → 16 slots, G1 → 3, G2 → 5: 07:…/src/verifier.rs:522-602).
- `opening_proof: HyraxOpeningProof<Fq> { combined_row: Vec<Fq> (2^col_vars), combined_row_opening_scalar }`, `dense_commitment: HyraxCommitment<GrumpkinPoint> { rows: Vec<GrumpkinPoint> (2^row_vars) }`.
- `public_outputs.pre_final_exponentiation: Bn254Fq12` (384 B).

### 1.3 What the verifier checks, and in which field (07:…/src/verifier.rs:134-161, 743-873)
1. Shape: point length = Dory proof point length, canonical reduce shape, setup supports the round count, clear ⇔ transparent artifacts / zk ⇔ ZK artifacts (204-263); **only the canonical GT/G1/G2/Miller/wiring dimensions are accepted** (292-337).
2. Replays the Dory verifier's Fr transcript scalars (α, β, inverses, fold factors, γ, d, d⁻¹, d², σ_c) from a *clone* of the Jolt transcript via `pcs_proof.verifier_transcript_scalars` (06 jolt-dory API) and checks inverse relations (451-475).
3. Absorbs setup, full Dory proof, commitment, point (and clear eval) into the **Fr** Jolt transcript; fork-squeezes digests; expands artifacts to Fq vectors; compares with `claims.stage1.public.input` (358-424, 668-685). Field crossing = `inject_fr_to_fq` = canonical LE bytes of the Fr value reduced mod q (632-636); all assist challenges are `squeeze_fq(transcript)` of Fr squeezes.
4. Stage 1 (07:…/stages/stage1/verify.rs:24-107): per relation — absorb id/rounds/degree, sample relation challenges, evaluate the input `Expr` from claims, verify the `CompressedSumcheckProof<Fq>` on the Boolean hypercube, compare the final value with the output `Expr`, absorb opening claims. All arithmetic is **Fq** (Grumpkin scalar field). Verifier cost = O(#relations × rounds × degree) Fq ops.
5. Stage 2: copy equalities `source_value == target_value` between claims + "public folds" (MLE of public vectors at the reduce sumcheck point) (07:…/stages/stage2/verify.rs:54-89).
6. Stage 3: reduced-opening order = Stage-1 order; `packed_eval = Σ_i eq(packed_point, i)·claim_i` (07:…/stages/stage3/verify.rs:138-161); Hyrax opening verified **natively** (`DoryAssistHyrax::verify_opening_proof`, 84-90) — an MSM of 2^row_vars Grumpkin points + a Pedersen check of 2^col_vars generators.
7. Native outputs (787-873): `MillerLoopOutputGt` claims == coefficients of `public_outputs.pre_final_exponentiation`; `final_exponentiation(public_output) == rhs`, where `rhs` and the pairing inputs are rebuilt from the `NativeFinalCheckInput` public claims (final reducer state) — 4 GT exponentiations, GT muls, ≈6 G1/G2 scalar muls, one Fq12 final exponentiation. The 4-pair `multi_miller_loop` is **not** executed by the verifier (only by fixtures via `pre_final_exponentiation()`, native_final.rs:24-28).

**Pairing check status:** deferred as a public GT output + native final exp/equality. Not in any circuit. The spec says a self-contained wrapper must add an R1CS "final-check hook" (06:specs/dory-assist-protocol.md:257-267, 689-703; 08:specs/wrapper-protocol.md:1029-1056) — never implemented.

### 1.4 Bytes: before vs after
- Dory proof (unchanged, still transmitted whole — `input.pcs_proof` is the full `DoryProof`, spec 06:specs/dory-assist-protocol.md:1775-1779): per reduce round 6 GT + 3 G1 + 3 G2 (07:…/src/verifier.rs:560-574) = 6·384 + 3·32 + 3·64 = 2 592 B; VMV C, D2 (GT) + E1 (G1) = 800 B; final E1 + E2 = 96 B; ZK adds e2, y_com, scalar-product (4 GT + G1 + G2 + 3 Fr). For σ reduce rounds: ≈ 2 592·σ + 0.9 KB (σ = 13 for a 2^26-coefficient matrix → ≈34.6 KB; lane B has measured numbers).
- Assist proof (added on top; est. for the canonical dims): Stage-1 sumchecks 24–26 × (8–13 rounds × 2–5 coeffs) ≈ 20–30 KB of Fq; claims ≈ 400–600 Fq ≈ 13–19 KB; `dory_proof_artifacts` re-encoding 120 Fq/round (3.84 KB/round); Hyrax `2·2^(n/2)` elements (n = dense-witness vars; 2^20 → 64 KB). Net: **strictly more bytes than plain Dory**. The design goal is verifier cycles (recursion paper: 171–198M RV64 cycles vs 1.4–1.9B; 08:recursion_references.md:185-210), not proof size.

### 1.5 Soundness as written, and gaps
- Statement (06:specs/dory-assist-protocol.md:180-267): local correctness of every op family, directed copy edges (no permutation argument, 584-596), public-input consistency via copy edges from public claims, prefix packing to one dense Hyrax opening, native final check.
- **DoryReduce transitions are coefficient-wise linear**: `NextC(c) = CurrentC(c) + SetupChi(c) + β·CurrentD2(c) + β⁻¹·CurrentD1(c) + α·C₊(c) + α⁻¹·C₋(c)` per Fq12 coefficient `c` (06:crates/jolt-claims/src/protocols/dory_assist/formulas/dory_reduce.rs:322-396, batched by γ-powers in `transition_relation` 1711-1746); G1/G2 likewise on affine coordinates (398+; test 2264-2300). For GT (multiplicative) and for EC points this is **not the group law**. The spec knows ("group operations that the protocol decomposes through the component semantics, not raw coefficient linear equalities", 06:specs/…:1200-1204) but no copy edge links `DoryReduce*` to `GtExponentiation`/`G1ScalarMultiplication` rows (06:…/formulas/composition.rs:283-407 — `DoryReduce` appears only in `native_vars_for_relation`, 564-569). As implemented, the reduce chain is a placeholder.
- Fixtures are synthetic: Stage-1 proofs are all-zero `canonical_for_dimensions` (07:…/stages/stage1/inputs.rs:20-27, 105-111), transition claims cleared (07:tests/support/mod.rs:1327-1330), a real but tiny Dory open (`num_vars` 2 or 4, 07:tests/support/mod.rs:1201-1259). The 111 tests pin shape/binding/tamper behaviour of the verifier, not correctness of any prover. Spec: "Prover/runtime tracing still has to replace the synthetic verifier fixtures" (06:specs/dory-assist-protocol.md:≈1288).
- Canonical dims are toy-sized and hard-enforced: GT 2^7 exp steps × 2^2 exp instances × 2^3 mul instances, G1/G2 2^8 × 4 × 8, Miller 2^7 line events × 4 pairs × 2^8 ops, wiring 2^6 edges, 1 reduce round (07:…/src/proof.rs:1403-1425). A real Dory verify needs ≥10 GT exps/round × σ≈13 rounds; the verifier rejects other dims (07:…/src/verifier.rs:292-337). `MAX_DORY_ASSIST_OPENING_POINT_LEN = 64` (verifier.rs:24).
- No prover, no witness generator, no `jolt-trace` instrumentation (spec 06:specs/dory-assist-protocol.md:120-127), no wrapper R1CS hooks (steps 8–9, 06:…:2836-2870 left open). `jolt-verifier` never dispatches to `PcsProofAssist` (grep at 07: zero uses outside the orphan file).
- Cost if it were lowered into an Fr R1CS (the stack's stated plan): Fq sumcheck rounds at ≈4k constraints per degree (§3), Hyrax row-combination = 2^row_vars variable-base Grumpkin scalar muls ≈ 24k each → est. ≈40M constraints for a 2^20 dense witness. Not viable without lookup-based non-native arithmetic.

## 2. Wrapper verifier (08)

### 2.1 Exact statement
- Key/config (08:crates/jolt-wrapper-verifier/src/config.rs:13-24): `WrapperVerifierConfig<P> { transcript_label = b"JoltWrapper", key: WrapperVerifierKey { relation: ConstraintMatrices<Fr>, relation_statement: { dimensions: {variables, constraints, public_inputs} }, hyperkzg: HyperKZGVerifierSetup<P> } }`. Per-proof inputs: `WrapperVerifierInputs { public_inputs: &[Fr] }` (08:…/src/verifier.rs:35-38).
- Statement proven: ∃ W (one committed multilinear over the padded variable count) with `(A·Z)∘(B·Z) = C·Z`, `Z = W` (**no public-input column, not relaxed**), for the matrices in the key. Public inputs enter only via (a) transcript absorption and (b) constants baked into the matrices by `alloc_public_scalar`: `assert_equal(lc, LinearCombination::constant(value))` (08:…/src/r1cs_builder.rs:72-80). Consequence: **the key changes with every instance** — a real wrapper needs `Z = (1, x, W)` and a public-input term in the inner check (BlindFold's `public_column_contributions` pattern, lane A §2).
- Relation digest: `r1cs_relation::verify` absorbs protocol id, raw/padded dims, public-input layout, then **every nonzero entry of A, B, C** (column index + coefficient) and then the public inputs (08:…/src/stages/r1cs_relation/verify.rs:44-46, 162-229). O(nnz) hashing per verification; must become a precomputed VK digest.
- Transcript: generic `T: Transcript<Challenge = Fr>`; tests use `Blake2bTranscript<Fr>` as the *wrapper* transcript and `PoseidonR1csTranscript` only *inside* the mini circuit (08:…/tests/wrapper_protocol_e2e.rs:83-84). Labels: 08:crates/jolt-claims/src/protocols/wrapper_spartan_hyperkzg/protocol.rs:7-27 (`wrapper-spartan-hyperkzg-v1`, `spartan_tau`, `spartan_outer_sumcheck`, `spartan_inner_batching`, `spartan_inner_sumcheck`, `wrapper_witness_commitment`).
- Spartan (08:…/src/stages/spartan/verify.rs:33-108, 188-258): witness commitment absorbed before τ (verifier.rs:66); `τ = challenge_vector(m)`, m = log2(next_pow2(constraints)); outer sumcheck degree 3, claimed sum 0, check `final == eq(τ,rx)·(a·b − c)` with prover-supplied `a,b,c`; absorb a,b,c, sample α_a,α_b,α_c; inner sumcheck degree 2 over n = log2(next_pow2(vars)) with claim α·(a,b,c); final check `inner.value == (α_a·Ã + α_b·B̃ + α_c·C̃)(rx,ry) · Z(ry)` where `Ã,B̃,C̃` come from `ConstraintMatrices::evaluate_matrix_mles(rx, ry)` (08:crates/jolt-r1cs/src/constraint.rs:229-276) — verifier-side O(nnz) sparse evaluation that materializes both eq tables (2^m + 2^n field elements). **No SPARK, no matrix commitments, no preprocessing.**
- HyperKZG (08:…/src/stages/hyperkzg/verify.rs:23-51): `bind_opening_inputs(ry, Z(ry))` then `HyperKZGScheme::verify(commitment, ry, Z(ry), proof, setup)`.
- ZK variant (`feature = "zk"`): `WrapperZkProof { spartan: SpartanZkProof { outer, inner: CommittedSumcheckProof<VC::Output> }, hyperkzg (ZK payload), blindfold: BlindFoldProof }`; Spartan checks only committed consistency + output-row shapes; HyperKZG `verify_zk` returns a hiding `Z(ry)` commitment; a 2-stage BlindFold statement enforces `0 → eq(τ,rx)(A·B−C)` and `α·(A,B,C) → M(rx,ry)·Z(ry)` and binds `Z(ry)` to the hiding commitment (08:…/src/stages/zk/blindfold.rs:54-156, verifier.rs:89-158). Requires `BlindFoldProtocolBuilder` (exists on main) and a VC whose basis matches HyperKZG's hiding commitment (spec 08:specs/wrapper-protocol.md:1229-1247).

### 2.2 Proof object and size
`WrapperProof<P> { relation: R1csRelationStatement{3×u64}, spartan: SpartanProof { outer_sumcheck: CompressedSumcheckProof (m × 3 Fr), outer_evaluation_claims {a,b,c}, inner_sumcheck (n × 2 Fr), witness_opening_claim }, hyperkzg: HyperKzgProof { witness_commitment: G1, witness_opening_proof: HyperKZGProof { com: n−1 G1, w: 3 G1, payload Clear { v: 3 × n Fr } } } }` (08:…/src/proof.rs:22-26, 52-57, 75-78, 198-211).
Size = `(3m + 5n + 4)·32 B (Fr) + (n + 3)·32 B (G1 compressed) + 24 B`:
| m (log constraints), n (log vars) | Fr | G1 | bytes |
|---|---|---|---|
| 20, 22 | 174 | 25 | 6 392 |
| 24, 24 | 196 | 27 | 7 160 |
| 28, 28 | 228 | 31 | 8 312 |
Single-digit KB up to 2^28. ZK adds committed rounds (one VC commitment per round instead of 3/2 Fr), a BlindFold proof (≈20 KB at Jolt shape per lane A; far smaller for this 2-stage statement, est. 2–4 KB) and the ZK HyperKZG payload (`y: 3×n G1`, `y_out`).

### 2.3 Tests (22 `#[test]`)
- `tests/verifier_stages.rs` (11): dimension/public-count/unpaddable rejections, round-count and degree-bound rejections, outer-claim mismatch, "claims satisfying only the outer relation", τ-before-outer ordering, dummy HyperKZG rejected after Spartan accepts — against zero matrices (08:…/tests/verifier_stages.rs:31-260, 353-390).
- `tests/r1cs_protocols.rs` (6): raw-R1CS satisfiability of three composed gadget cases (Poseidon public arithmetic, Fq sumcheck + same-point opening reduction, Grumpkin/Pedersen Hyrax) with a tamper manifest; oracle is `verify_r1cs_witness`, not the wrapper proof (181-321).
- `tests/wrapper_protocol_e2e.rs` (5): `build_mini_protocol` (1180-1212) = 2 public scalars + 1 private scalar + 1 byte absorbed by `PoseidonR1csTranscript`, one challenge, one 1-round degree-2 native sumcheck with the challenge as a variable (1214-1246), one Grumpkin Hyrax opening over injected `FqVar`s (1264-1290), one public output equation. Test-gated prover `prove_wrapper` (1352-1416) → `prove_spartan` (2883-2960): dense `row_value_polys` (Az, Bz, Cz), sum-of-products sumcheck over materialized `eq_τ`, `combined_matrix_column_poly`, then `HyperKZGScheme::open`. Honest proofs cached on disk (`JOLT_WRAPPER_REGENERATE_FIXTURES`); tampering of publics, matrix coefficients, dims, commitment, every round poly, claims, folds, `v` entries; ZK variants incl. VC-basis mismatch and a 32-sample independence check.
- Nothing asserts constraint counts; the mini relation is a few thousand constraints (est.).

### 2.4 Missing for a real Jolt-verifier relation
1. The relation itself (stages 1–8 lowered with in-circuit Fiat–Shamir) — absent on every branch; `WrapperR1csStage` hook trait from the spec (08:specs/wrapper-protocol.md:478-500) was never written.
2. Witness generator (instrumented verifier replay) and production prover (`jolt-wrapper-prover`, explicitly deferred: spec 08:…:10-32, 299-321); the test prover is O(2^m + 2^n) dense and single-threaded.
3. Public-input column + VK digest (§2.1); transcript with a native Poseidon twin (§3.3).
4. Build: not a workspace member; depends on 03 APIs absent from main (`AssignedScalar`, `evaluate_matrix_mles`, `R1csJoltByteTranscript`), and on `jolt_claims::public` (renamed `derived` on main: origin/main:crates/jolt-claims/src/claims.rs:162).

## 3. R1CS extensions (03)

### 3.1 `FqVar` (03:crates/jolt-r1cs/src/nonnative.rs)
- Representation: 4 × 64-bit little-endian canonical integer limbs of BN254 `Fq` as `AssignedScalar<Fr>` (13-17, 40-45); `CARRY_BITS = 68`; not Montgomery.
- Range checks = **full bit decomposition**: `assert_unsigned_bits` allocates one boolean per bit (`b·(b−1)=0`) plus one recomposition equality (198-215); canonical bound via limb-wise borrow subtraction against `q−1` (`assert_limbs_less_or_equal`, 378-402: per limb 1 boolean + 64-bit difference + 1 equality).
- API: `constant / alloc / from_checked_limbs` (51-71), `inject_fr` (79-89: decompose the Fr value into limbs canonical mod r, re-use as Fq — valid because r < q), `inject_fr_challenge` (97-102), `bits_le` (114-119), `assert_equal`, `add`, `sub`, `mul`, `inverse`, `select` (121-183). Carries via `assert_terms_normalize_to` (339-368: per limb one 68-bit carry + one equality).
- Constraint counts (counted from the code paths above; each `assert_u64` = 65, `alloc`/`from_checked_limbs` = 4·65 + 4·67 + 1 = 529):
| op | constraints | breakdown |
|---|---|---|
| `alloc` / `from_checked_limbs` / `inject_fr` | 529 / 529 / 530 | 4 u64 limbs + canonical bound (+1 compose) |
| `add`, `sub` | 1 352 | out 529 + quotient bit 1 + 4 normalized u64 260 + 2 × normalize⟨4⟩ 281 |
| `mul` | **2 716** | out 529 + quotient 529 + 8 normalized u64 520 + 16 limb products + 2 × normalize⟨8⟩ 561 |
| `inverse` | 2 716 | alloc + mul relation against constant 1 |
| `select` | 538 | 1 boolean + 4 mul + 4 eq + out alloc |
| `bits_le` | 260 | 4 × 65 (re-decomposes) |
Lane A's "≈2k per mul" is low; it is 2.7k constraints and ≈2.3k fresh variables per multiplication. No lookups, no lazy reduction, no Barrett/CRT — a Miller loop (~12k Fq muls) alone would be ≈33M constraints (est.).

### 3.2 Scalar gadget + variable-challenge sumcheck
- `ScalarGadget` trait (03:crates/jolt-r1cs/src/scalar.rs:12-33): `constant, alloc, assert_equal, add, sub, mul, scale_by_constant, select`; impl for `AssignedScalar<F>` (66-125: add/sub/scale are free LC algebra, mul = 1 constraint, select = 2) and for `FqVar` (127-177). Helpers `scalar_affine_combination`, `scalar_dot_product` (35-64). `AssignedScalar { value, lc }` lives in 03:crates/jolt-r1cs/src/builder.rs:42-46 (absent on main).
- Sumcheck (03:crates/jolt-sumcheck/src/r1cs.rs): `SumcheckR1csRound::challenge() -> LinearCombination<F>` (9-22) — a **variable** challenge (the `VerifiedCommittedRound` impl keeps BlindFold's constant path). `append_round_constraints` (384-396) emits `Σ domain_weight_j·c_j == claim_in` (1 linear constraint) and Horner `((c_d·r + c_{d−1})·r + …) == claim_out` via `polynomial_eval_at_challenge` (464-483): `deg` multiplications + 1 equality → **deg + 2 constraints per round** natively (deg 3 → 5; deg 27 uni-skip round → 29). Gadget path `append_sumcheck_r1cs_gadget_constraints[_for_domain]` (216-254) + `append_gadget_round_constraints` (398-429) is generic over `ScalarGadget`; over `FqVar` a degree-d round costs d·(2 716 + 1 352) + 8 ≈ 4.1k·d. Domains via `SumcheckDomain::round_sum_coefficients` (Boolean hypercube and centered-integer both supported).
- Claim lowering over gadgets: `ScalarClaimSourceTable`, `lower_claim_expr_gadget`, `assert_claim_expr_gadget_eq` (03:crates/jolt-r1cs/src/lowering.rs:97, 405, 439) mirror the native `ClaimSourceTable` path for `jolt_claims::Expr`.

### 3.3 `PoseidonR1csTranscript` (03:crates/jolt-transcript/src/r1cs/poseidon.rs)
- Poseidon over BN254 Fr, `light_poseidon::parameters::bn254_x5`, `POSEIDON_INPUTS = 3`, width 4 (7-12); state is a **single Fr**; every absorb and every challenge is one permutation of `(0 | state, round_tag, payload)` (44-83). S-box x⁵ = 3 multiplies (348-350); MDS/ARK are linear. Per permutation: 3·(4·R_f + R_p) constraints = 264 with R_f = 8, R_p = 56 (light-poseidon t = 4; est. — parameters loaded at runtime, 385-392).
- Absorption is **Fr-native**: `absorb_scalar` = 1 permutation per element (73-83); labels are ≤32-byte LE-packed scalars (158-164); `label_with_len` packs 24 B label + 8 B BE length (174-183); bytes are packed 32 per scalar (radix 256, mod r) and take one permutation per chunk (106-155). Challenges are full Fr elements (57-69) — no 128-bit truncation.
- It mirrors **jolt-core's legacy `transcript-poseidon`** (`hash(state, n_rounds, payload)`, doc 23-27), not the modular `PoseidonTranscript = SpongeTranscript<PoseidonSponge>` (03:crates/jolt-transcript/src/lib.rs:67-69). Its only native twin is the test-local `NativeTranscript` (poseidon.rs:405-…, test 489-538). A wrapped Jolt proof needs a production native transcript with exactly this absorption schedule (and G1/GT absorbed as field elements, not bytes) — ≈300–500 lines, est.
- Trait surface (03:crates/jolt-transcript/src/r1cs/mod.rs:18-129): `R1csTranscript { new, challenge_scalar }`, `R1csAlgebraicTranscript { absorb_scalar, absorb_u64, absorb_label, absorb_label_with_len }`, `R1csJoltTranscript { append_label/u64/scalar/scalars }`, `R1csByteTranscript { absorb_bytes, absorb_constant_bytes }`. Feature `poseidon-r1cs` (Cargo.toml).

### 3.4 Other gadgets
- `jolt-poly::r1cs` (03:crates/jolt-poly/src/r1cs.rs:21-29, 31-120): `PolynomialScalarGadget` (constant/add/sub/mul) with `eq_eval` (3 muls per coordinate), `eq_evals` (full 2^k table), `scaled_eq_evals`, `inner_product`, `multilinear_eval`. No Lagrange/uni-skip/`LT` gadgets; the branch's `lagrange.rs`/`split_eq.rs` changes are native prover code.
- `jolt-openings::r1cs` (03:crates/jolt-openings/src/r1cs.rs:19-110): same-point opening-claim RLC (`reduce_same_point_opening_claims`, `assert_same_opening_point`, `reduce_opening_claim_scalars`) over any `ScalarGadget`; commitments are opaque.
- `jolt-crypto::r1cs` (03:crates/jolt-crypto/src/r1cs.rs): `GrumpkinPointVar` / `GrumpkinPointWithIdentityVar` (132-305; on-curve `y² = x³ − 17`, explicit identity flag), traits `GroupElementVar / NonExceptionalAddGroupVar / CompleteAddGroupVar / DoubleGroupVar / (Variable|Fixed)BaseScalarMulGroupVar / VectorCommitmentR1cs` (25-130), complete addition with zero-tests and gated equations (547-631, ≈45 constraints est.), fixed-/variable-base scalar mul over `FqVar::bits_le` (633-697: 254 × (select + complete add [+ complete-add double]) ≈ 12.5k / 24k est.), fixed-base MSM, Pedersen opening (720-757), `VectorCommitmentR1cs for Pedersen<GrumpkinPoint>` (395-428). Grumpkin native type: 03:crates/jolt-crypto/src/ec/grumpkin/mod.rs (347 lines, `ark-grumpkin` is already a workspace dependency on main).
- `jolt-hyrax::r1cs::verify_opening<VC>` (04:crates/jolt-hyrax/src/r1cs.rs:29): eq weights → VC linear combination → VC opening → inner product == claimed eval.
- Boolean / byte helpers only as private fns (`assert_boolean`, `assert_unsigned_bits`) and `WrapperR1csBuilder::alloc_witness_byte` (08:…/src/r1cs_builder.rs:82-97: 8 booleans + 1 recomposition).

## 4. Hyrax (04) and HyperKZG-zk (05)

**Hyrax (04, `crates/jolt-hyrax`, ~1.9k lines incl. tests).** Transparent, *zero-blinding* Hyrax over any `VectorCommitment` (`lib.rs:1-5`): a multilinear with `k` variables is split into 2^{k_row} rows of 2^{k_col} entries, each row Pedersen-committed; opening = the eq-weighted row combination sent in the clear plus its inner product with the column eq vector (`scheme.rs:26-82`). Proof size is 2^{k_col} scalars + 2^{k_row} commitments — square-root, i.e. the opposite of what the wrapper wants; its role in this stack is (a) the Dory-assist dense witness over Grumpkin (`jolt-dory-assist-verifier/src/proof.rs:18-26`) and (b) the wrapper's *test* circuit (`r1cs.rs:29`, used at 08:…/tests/wrapper_protocol_e2e.rs:1264-1290). **08 depends on 04 only as a dev-dependency** (08:crates/jolt-wrapper-verifier/Cargo.toml dev-deps `jolt-hyrax = { features = ["r1cs"] }`); the production verifier path never references it. Portable in isolation (2–4 h est.); skip unless Dory assist or the Grumpkin test circuit is wanted.

**HyperKZG-zk (05, `crates/jolt-hyperkzg`).** Adds `impl ZkOpeningScheme` with `HidingCommitment = G1`, `Blind = Fr`, `open_zk`/`open_zk_poly` (05:crates/jolt-hyperkzg/src/scheme.rs:812-867); the proof payload becomes an enum `HyperKZGProofPayload::{Clear{v}, Zk{y: G1 per fold, y_out, …}}` (types.rs), SRS gains a hiding generator (`HyperKZGSrsKind`) plus versioned file I/O (`hyperkzg_{k}.srs` / `hyperkzg_zk_{k}.srs`, `read_srs_file`, `read_srs_from_dir`, `write_*`, setup.rs:24-45, 175-305) and a test-only `setup_from_secret` (328-389). Main deleted `crates/jolt-hyperkzg` in #1795 and this worktree restored it from `d80d201d6^` (commit 992ad9d23), so the restored crate is the **pre-05, non-ZK** version: main's `ZkOpeningScheme` is `open_zk<P: MultilinearPoly>` generic (origin/main:crates/jolt-openings/src/schemes.rs) vs 05's non-generic signature → the 05 impl needs a signature port (1–2 days est. incl. tests) but only if the wrapper must be ZK. The SRS file I/O is worth cherry-picking regardless (2–3 h est.).

## 5. Rebase distance and integration cost

All seven branches share merge-base `e0d5d7eb2` (2026-06-17), **143 commits behind `origin/main` 756bddce3**; `origin/prover-stack/*` tips are identical to the local `alberto/*` tips. Upstream churn since the base on the files each layer touches (from `git diff --stat e0d5d7eb2..origin/main -- <crate>`):

| crate | upstream commits / files / +/− | 03-vs-main on shared files | portability |
|---|---|---|---|
| jolt-r1cs | 7 / 9 / +495 −436 | lowering.rs 18/461, constraint.rs 21/144, builder.rs 11/38 | **hours** — `AssignedScalar`, `ScalarGadget`, `nonnative.rs`, `scalar.rs` are additive; `evaluate_matrix_mles` must be re-added to `ConstraintMatrices` (main only has it in key.rs) |
| jolt-sumcheck | 12 / +2152 −1291 | r1cs.rs 26/498 | **hours** — variable-challenge `challenge() -> LinearCombination` is a trait-signature change; main's `VerifiedCommittedRound` impl must be adapted |
| jolt-transcript | 6 / +152 −63 | lib.rs 20/6 | **trivial** (new `r1cs/` module, additive) |
| jolt-poly | 12 / +1679 −284 | r1cs.rs new | **trivial** |
| jolt-openings | 14 / +1883 −1272 | schemes.rs 651/70 | **hours** — `r1cs.rs` additive, but `ZkOpeningScheme` drifted |
| jolt-crypto | 6 / +368 −180 | commitment.rs 70/35 | **hours–1 day** — grumpkin module + r1cs.rs additive; `FixedByteSize/FromPrimitiveInt/Invertible` traits absent on main (methods live in `crates/jolt-field/src/algebra.rs`) |
| jolt-field | 15 / 97 / +14780 −5210 | trait reshuffle | **hours** — `Fq` already exported (origin/main:crates/jolt-field/src/lib.rs:116); `from_le_bytes_mod_order` missing |
| jolt-claims | 22 / +14448 −10071 | `public`→`derived` rename | wrapper protocol module **trivial**; dory_assist claim module **days** (spec-level rework) |
| jolt-verifier | 31 / +22458 −21002 | verifier.rs 1049/474 | `pcs_assist.rs` trivial (orphan, 26 lines); anything else **rewrite** |
| jolt-dory | 10 / +1687 −230 | streaming.rs 758/67 | artifact accessors (+1.5k lines) **0.5–1 day**, only for assist |
| jolt-blindfold | 7 / +2443 −358 | verify.rs 511/431 | wrapper's 2-stage statement uses `BlindFoldProtocolBuilder` (present) — **hours** |
| jolt-hyperkzg | deleted upstream (#1795); restored here pre-05 | — | 05 ZK **1–2 days**; SRS I/O 2–3 h |
| jolt-hyrax (04) | not on main | — | **2–4 h**, skip unless needed |
| jolt-dory-assist-verifier (07) | not on main, not a workspace member | — | **rewrite/skip** (§1.5) |
| jolt-wrapper-verifier (08) | not on main, not a workspace member | — | **port + fix, 0.5–1 day** (§2.4 items 3–4) |

API drift risks (verified by grep on origin/main): absent — `AssignedScalar`, `GrumpkinPoint`/`ec::grumpkin`, `PcsProofAssist`, `ConstraintMatrices::evaluate_matrix_mles`, `FqVar::inject_fr`, `jolt_claims::public`, `from_le_bytes_mod_order`; present and reusable — `Fq`, `CanonicalBytes`, `ZkOpeningScheme` (generic), `BlindFoldProtocolBuilder`, `SumcheckR1csLayout`, `CommittedSumcheckProof`, `verify_committed_consistency`, `VerifiedCommittedRound`, `VectorCommitment`, `SumcheckClaim`, `Transcript::challenge_vector`, `LabelWithCount`/`U64Word`, `ark-grumpkin` dep, `light-poseidon` (legacy prover + fuzz only). Nothing in 03–08 is referenced by CI (`.github/workflows/rust.yml` mentions neither crate).

## 6. Intent reconstruction

- Architecture (08:specs/extended-jolt-field-inline-wrapper.md:21-57): three axes on one stack — (i) *field inline*: `jolt-claims` facts → `jolt-verifier` selected config → component R1CS crates (`jolt-r1cs` gadgets, `jolt-sumcheck::r1cs`, `jolt-transcript::r1cs`, …), each verifier component gets an in-circuit twin next to its native code; (ii) *Dory assist*: keep Dory as the Jolt PCS and move its verifier's GT/G1/G2/Miller work into a Grumpkin-native auxiliary PIOP (BN254 Fq = Grumpkin scalar field, spec 06:specs/dory-assist-protocol.md:42-64), leaving only Fr checks + final exponentiation to the wrapper/native verifier; (iii) *wrapper*: "R1CS encoding of the configured verifier computation, proved by ZK Spartan + HyperKZG first", Groth16 later (08:specs/wrapper-protocol.md:10-31).
- Dory decision: **Dory was never meant to be verified inside the wrapper R1CS** (out of scope, 06:…:66-88) — the assist exists because in-circuit GT arithmetic in Fr is prohibitive (§3.1: 2.7k constraints per Fq mul, 12 Fq per GT mul ⇒ ~150k+ per GT product). Cost of that choice: the assist proof is *larger* than the Dory proof it explains (§1.4) and the pairing final exponentiation stays native (06:…:257-267, 700) — so the design targets *verifier cycles*, not bytes. It is a recursion-oriented stack (recursion_references.md:185-210: quang/recursion-temp proved the Miller loop, final exp external), reused here without re-deriving the goal.
- Why it stalled: (a) Markos' whole stack was committed in one restack batch 2026-06-09 15:04–15:06 (the code branches carry identical author timestamps), then only 02/03/05/06 received Andrew's "adapt to stack" commits on 2026-06-17; 07/08 were never adapted, are not workspace members, and the tip does not build (orphan `pcs_assist.rs`, unresolved imports). (b) The modular prover (`jolt-prover`, 09/10*) landed upstream on a different path, making the 143-commit rebase of `jolt-verifier`/`jolt-claims` (+22k/−21k and +14k/−10k) the dominant cost. (c) The Dory-assist semantics are placeholders — coefficient-linear reduce transitions (06:crates/jolt-claims/src/protocols/dory_assist/dory_reduce.rs:322-396, 1711-1746), synthetic verifier fixtures (07:…/tests/support/mod.rs:1201-1259, 1327-1330), implementation steps 8–9 open (06:…:2797-2870) — so 07 could not be exercised against a real Dory proof. (d) Spec branches `stack/11–15` (2026-05-25) predate the code branches' specs; the design conversation moved into the code branches and stopped there.
- Last commit dates: 02–08 code 2026-06-09 (Markos, Codex-assisted per spec headers), adapt commits 2026-06-17 (Andrew), no activity since; `origin/prover-stack/*` mirrors pushed the same day.

## Reuse plan (hours are estimates)

| artifact | verdict | est. | note |
|---|---|---|---|
| `jolt-claims::protocols::wrapper_spartan_hyperkzg` | port as-is | 1 h | labels/degrees; rename `public`→`derived` |
| `jolt-wrapper-verifier` (non-zk src) | port + fix | 0.5–1 day | add public-input column `Z=(1,x,W)` + inner-check public term; replace full-matrix absorption with a VK digest; keep stage split |
| `jolt-wrapper-verifier` zk path + BlindFold 2-stage statement | port later | 0.5 day | only if ZK is required; needs VC/HyperKZG hiding-basis agreement |
| `jolt-r1cs` 03: `AssignedScalar`, `ScalarGadget`, `scalar.rs` | port as-is | 2–3 h | additive |
| `jolt-r1cs` 03: `nonnative.rs` (`FqVar`) | port + fix | 3–4 h | jolt-field trait renames; **do not** rely on it for anything pairing-sized (§3.1) |
| `jolt-r1cs` 03: lowering gadget path, `evaluate_matrix_mles` | port + fix | 2 h | re-add MLE eval to `ConstraintMatrices` (or keep in key.rs and call that) |
| `jolt-sumcheck::r1cs` variable challenge | port + fix | 2–3 h | trait signature change touches BlindFold impl |
| `jolt-transcript::r1cs` (Poseidon) | port as-is **+ write native twin** | 1–2 h + 0.5–1 day | or re-target the gadget to `SpongeTranscript<PoseidonSponge>` so the native side already exists |
| `jolt-poly::r1cs`, `jolt-openings::r1cs` | port as-is | 0.5 h each | tiny |
| `jolt-crypto::r1cs` + grumpkin | skip unless Grumpkin needed | 0.5–1 day | only for Dory assist / Hyrax test circuit |
| `jolt-hyrax` (04) | skip | 2–4 h if needed | wrong size profile for the wrapper |
| `jolt-hyperkzg` 05 ZK | defer | 1–2 days | cherry-pick SRS file I/O now (2–3 h) |
| `jolt-dory` 06 artifact accessors | skip unless assist | 0.5–1 day | |
| `jolt-claims::dory_assist`, `jolt-dory-assist-verifier` (07) | **skip / rewrite** | days–weeks | placeholder transitions, synthetic fixtures, grows bytes |
| `jolt-verifier::pcs_assist` | skip | — | 26-line orphan trait, pointless without assist |
| specs 06/08 | reuse as reading | — | wrapper-protocol.md §Statement/§R1CS Ownership/§ZK Composition are the useful parts |

Net: ≈2–3 engineer-days to have a building, main-rebased `jolt-wrapper-verifier` + gadget layer (est.); the Jolt-verifier relation, witness generator and production prover remain unwritten on every branch and dominate the real budget.

## Open questions for the architecture decision

1. **Dory in the wrapped proof.** Assist does not shrink bytes and leaves final-exp native; in-circuit Dory over `FqVar` is ~10⁷–10⁸ constraints (est.). Options: (a) verify Dory natively outside the wrapper and only wrap the Fr-side stages (proof = wrapper + Dory proof ≈ 6 KB + Dory's tens of KB — misses single-digit KB); (b) prove Jolt with HyperKZG/Zeromorph directly for the wrapped profile (kills Dory's streaming-commit advantage; pairing verifier only); (c) fold the Dory verifier into a Grumpkin-cycle recursion (quang's path) and accept two proofs; (d) Groth16 the wrapper (~200 B) — the Dory pairing checks then need a BN254-cycle anyway. Decision rule: single-digit KB with Dory retained ⇒ only (c)/(d) reach it.
2. **Public-input encoding.** Constant-pinned publics (current) vs `Z=(1,x,W)` column with the public term in the inner check; and how the VK (matrices) is digested — commitment to A/B/C (SPARK) vs trusted VK hash.
3. **Transcript.** Which Poseidon (legacy jolt-core `(state, n, payload)` vs modular `SpongeTranscript<PoseidonSponge>`), full-Fr vs 128-bit challenges, and whether G1/GT elements are absorbed as field limbs or bytes — the Jolt prover's transcript must change to match, i.e. a new `JoltProtocolConfig` transcript variant.
4. **ZK.** If the wrapped proof must be ZK, BlindFold over the wrapper's own Spartan needs a VC whose basis equals HyperKZG's hiding commitment (08:specs/wrapper-protocol.md:1229-1247) — decide before committing to the HyperKZG SRS layout.
5. **Non-native cost strategy.** Bit-decomposition `FqVar` (2.7k/mul) vs lookup-assisted range checks (Jolt-style Shout) vs pushing the verifier into a Jolt guest (RISC-V recursion). The bespoke R1CS route is only viable if all Fq work leaves the circuit (question 1).
6. **Uniform-R1CS matrix evaluation in-circuit.** Jolt's `UniformSpartanKey` evaluation at `(rx, ry)` is ~50–120 constraints per recursion_references.md:486-537 (est.) — cheap, but the wrapper's own Spartan inner check needs `evaluate_matrix_mles` over a non-uniform ~10⁶–10⁷-nnz relation natively (O(nnz) verifier) unless SPARK is added: is O(nnz) verifier time acceptable for the wrapper?
