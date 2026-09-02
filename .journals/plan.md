# Plan v2 — Spartan+HyperKZG wrapper for curve Jolt (card 135) — OFFLOADING campaign

Inputs: .journals/discovery/*.md (Phase 0), user decision 2026-09-02 16:15 (binding):
- Inner PCS stays Dory. Dory must be verified INSIDE the wrapper via OFFLOADING (Fq/Fq12/G1/G2/GT work replaced by Fr-native checks the R1CS verifies cheaply; non-native work moved out as advice/deferred checks enforced via Fr arithmetic; residual = a handful of pairing-product checks over Fr-linear combinations, deferred/public).
- Transcript MUST be Blake2b exactly as production Jolt (LegacyBlake2bTranscript, chained Blake2b-256 digest, 32-B BE Fr, 16-B challenges) — byte-identical native vs in-circuit; Blake2b compression in R1CS.
- Target: wrapper prover < 1 s on the mini (measured budget m,n ≤ 2^18 = 262,144; 2^19 = 1.25 s) AND single-digit-KB-class proof.
- Surface ONLY real blockers with numbers.

## Wave 1 — measurement gates (exact numbers before any build)
G1 Blake2b-256 compression in R1CS: exact constraint + witness count (gadget built on jolt_r1cs::R1csBuilder, tested vs `blake2` crate). Pre-estimate 45–50k constraints/compression (Blake2s R1CS ≈21k; Blake2b = 12 rounds × 8 G × (4 adds→66-bit decompositions + 4 XORs×64 bits)).
G2 Exact hidden-transcript compression count for LegacyBlake2bTranscript at L=18/20: per append_scalar/append_scalars/challenge (bytes hashed per call → compressions), over the hidden data only (round polys 939 Fr @L=20, claims 259, challenges 371). Pre-estimate ≥1,000–1,600 compressions → 50–80M constraints (≈200–300× the 2^18 budget). If confirmed → BLOCKER #1 (options: algebraic hash [rejected by user], send round polys [no size win], relax <1 s to minutes).
G3 Dory offloading study: explicit verifier algebra from the actual dory-pcs code (GT multi-exp with challenge-monomial scalars; final pairing check; subgroup checks); classify linear (batchable via RLC, offloadable) vs nonlinear (GT exp/mul, pairings, subgroup checks — each Fq mul ≥1k constraints non-native); minimal set of group elements ANY verifier must hold; candidate designs with constraints in-circuit / deferred native cost / proof bytes: (i) scalars in-circuit + GT multi-exp native (elements sent), (ii) advice+RLC for GT ops, (iii) Fq-native sub-proof (Grumpkin) verified in Fr, (iv) Dory layout (σ) tradeoffs, (v) alternative opening argument over the same Dory commitments. Pre-assessment: nonlinear Fq12 arithmetic cannot be collapsed by RLC; ≥6σ+2 GT proof elements must be held by whoever does the multi-exp → with native Dory the floor is 41+6σ+2 GT (5.2 + 8.7 KB compressed @2^18) → single-digit KB requires hiding GT elements behind an Fq-native proof (≥0.7M Fr constraints to verify, ≥5 s prover). If confirmed → BLOCKER #2.
G4 (running) compressed GT codec bytes (lane J) — the size floor for any design that sends GT elements.
G5 (running, re-steered) relation plan for everything else (sumcheck chain, stage algebra, public IO, Spartan fork, witness gen) — valid under any transcript/Dory outcome.
Oracle (GPT-5.6 pro) second opinion on G3 with the dory-pcs source attached.

## Wave 2+ (only if Wave 1 gates pass or the user relaxes a constraint)
Component ladder with budgets: Blake2b gadget (per-compression cost × exact count) → transcript replay circuit → sumcheck-chain relation (≈25–30k) → offloaded Dory verifier circuit (from G3 design) → Spartan+HyperKZG prover/verifier (public-input column, HyperKZG binding; fork of jolt-blindfold outer/inner) → wrap()/verify_wrapped() pipeline → e2e (fibonacci @2^18/2^20, sha2-chain; tamper tests) with bytes + wrapper wall-clock as first-class numbers → draft PR. Fallback per component recorded at each gate.

## Done so far
- HyperKZG restored (992ad9d23) + 5–9× faster (abab852a5, cfa02939f, 6634e39f2): 2^18 commit 137 + open 336 ms; Spartan-side 2^18 ≈156 ms → 590 ms total; 2^19 → 1,250 ms.
- Proof-size baseline 82,191 B @2^18 → 95,187 B @2^24 (commitments 15.9 KB, sumchecks 28–34 KB, claims 8.5 KB, Dory 29–37 KB).
- Poseidon transcript lane stopped (user: Blake only). Fr-only+native-Dory design (23–27 KB) rejected by user.
