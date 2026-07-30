# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-07-03

### Security

- **Fixed a critical soundness bug in ZK mode: the evaluation opening point was
  not bound by the verifier.** A ZK evaluation proof created for a point `P`
  verified at *any* point `P'`, letting a malicious prover open a committed
  polynomial to a forged evaluation at an attacker-chosen point. Transparent
  mode was unaffected.

  Root cause: the 0.3.0 ZK final check was a collapsed 1-pairing scalar-product
  argument over the *pre-fold* statement `(C, D₁, D₂)`. The opening point
  reaches the verifier only through the folded scalars `s1_acc`/`s2_acc`, and
  that check never read them — it dropped the point-binding terms (transparent
  Pairs 2 & 3 and the `s1_acc·s2_acc·HT` term) with no zero-knowledge
  replacement.

  Fix: ZK mode now follows the original Dory construction (eprint 2020/1274,
  §4.1 Fold-Scalars then §3.1 Scalar-Product). The prover applies the
  Fold-Scalars reduction to its witness (`v₁ ← v₁ + γ·s₁·H₁`,
  `v₂ ← v₂ + γ⁻¹·s₂·H₂`, `r_C ← r_C + γ·r_E2 + γ⁻¹·r_E1`) *before* running the
  scalar-product Σ-argument, so the argument now opens the point-dependent
  *folded* statement `(C′, D₁′, D₂′)`. The verifier reconstructs `(C′, D₁′, D₂′)`
  from its own point-derived `s1_acc`/`s2_acc` and the E-accumulators, batched
  into the final multi-pairing check. This pins the point coefficient to the
  committed witness while the folded coordinates `v₁[0]`/`v₂[0]` stay hidden.

### Changed

- **Breaking (proof format & API):** `DoryProof::final_message` is now
  `Option<ScalarProductMessage>`. Transparent proofs carry `Some(..)` (the
  revealed folded witness); ZK proofs carry `None` (revealing it would break
  hiding — the scalar-product Σ-proof replaces it). ZK proofs from 0.3.0 no
  longer verify and must be regenerated.
- ZK final verification is now a single 4-pairing multi-pairing, 4 ML + 1 FE
  (was a collapsed 1-pairing final check plus a standalone 2-pairing / 2-FE Σ₂
  check): the Σ₂/VMV constraint is batched into the final check at the `d²`
  slot, mirroring Pair 4 of the transparent check. The scalar-product Σ-proof
  responses `(E₁, E₂, r₁, r₂, r₃)` and the Σ₂ responses `(z₁, z₂)` are now
  absorbed into the Fiat-Shamir transcript before the batching challenge `d`
  is drawn, matching the interactive protocol; for the Σ₂ responses this is
  what makes the batching sound (unbound, they could cancel a wrong-point
  residual chosen after `d`).
- The fresh final-message blinds `r_final1`/`r_final2` (sampled in
  `compute_final_message`) are removed; they were an unconstrained degree of
  freedom that any point-agnostic fix would have left exploitable.
- **Breaking (API):** `DoryVerifierState::verify_final` now takes a single
  `FinalCheck` enum (`Transparent(&ScalarProductMessage)` or, with the `zk`
  feature, `Zk { scalar_product, sigma_c, sigma2, sigma2_c }`) instead of two
  independently optional parameters, making the mode invariant structural.
- **Breaking (behavior):** proof shapes are validated up front by the new
  `DoryProof::mode`: a proof must be fully transparent (clear final message,
  no ZK fields) or fully ZK (every ZK field present, no clear final message).
  A transparent proof carrying stray ZK fields previously verified with those
  fields silently ignored (proof-byte malleability); it is now rejected.

### Fixed

- The verifier no longer panics on a crafted proof with `nu > sigma`
  (pre-existing denial-of-service, not introduced by this fix): `nu`/`sigma`
  are untrusted proof fields and `nu` is used to slice buffers of length
  `sigma`. The layout constraint is now validated in
  `verify_evaluation_proof`, mirroring the prover.

### Added

- `DoryProverState::apply_fold_scalars` implementing the §4.1 Fold-Scalars
  witness update (used by both transparent and ZK modes).
- `absorb_scalar_product_proof` / `absorb_sigma2_proof`: verifier-side
  transcript mirrors of the Σ-proof generators, so the absorb sequences are
  maintained adjacent to their prover counterparts (and reusable by external
  protocol drivers).
- `zk_wrong_point` example and `test_zk_wrong_point_rejected` regression test:
  a ZK proof for one point must not verify at another.
- `test_zk_crafted_blind_shift_wrong_point_rejected`: an adversarial wrong-point
  proof that shifts the Σ-proof blinds by the exact public point deltas (the
  attack a naive "prove the residual is in the blinding span" fix would accept)
  is rejected.
- `test_soundness_missing_final_message` /
  `test_zk_soundness_unexpected_final_message`: the clear final message must be
  present iff the proof is transparent.
- `test_zk_crafted_sigma2_response_shift_wrong_point_rejected`: a wrong-point
  proof that shifts the Σ₂ responses by the public deltas that would cancel
  the batched final check (were the responses not transcript-bound before `d`)
  is rejected.
- `DoryProof::mode` / `ProofMode`: classifies a proof as transparent or ZK,
  validating the full field shape in one place and handing out references to
  exactly the fields that shape guarantees.
- Shape and per-component regression tests: stray ZK fields on a transparent
  proof are rejected (`test_soundness_transparent_proof_with_stray_*`);
  `test_soundness_nu_exceeds_sigma_rejected` pins the nu > sigma rejection;
  and every prover-controlled component of a ZK proof is individually
  tamper-tested — all VMV, reduce-round, Σ₁, Σ₂, and scalar-product proof
  fields, the blinded `e2`/`y_com`, the `nu`/`sigma` dimensions, and the
  round-message counts.

### Removed

- `verify_sigma2_proof`: the Σ₂ check is now performed inside
  `DoryVerifierState::verify_final` as part of the batched final multi-pairing.

## [0.3.0] - 2026-02-27

### Added

- **Zero-knowledge mode** (`zk` feature): optional hiding proofs where both commitment and proof are blinded
  - Single GT-level commitment blind (`r_d1 * HT`)
  - VMV messages (C, D2, E1, E2, y_com) blinded with OS randomness
  - Reduce-and-fold messages blinded with OS per-round randomness
  - Final message (E1, E2) blinded to hide folded witness vectors
  - Sigma1 proof: proves E2 and y_com commit to the same evaluation
  - Sigma2 proof: proves consistency of E1 and D2 blinds
  - Scalar product proof: proves (C, D1, D2) are consistent with blinded v1, v2
  - 1 ML + 1 FE verification in ZK mode (vs 4 ML + 1 FE in transparent mode)
- New `zk_e2e` example demonstrating the full ZK workflow
- New `zk_statistical` example with chi-squared uniformity and witness-independence tests (1000 trials)
- ZK test suite: end-to-end proofs, tampering resistance, sigma proof verification, soundness

### Changed

- `Polynomial::commit()` return type changed from `(GT, Vec<G1>, Option<Vec<F>>)` to `(GT, Vec<G1>, F)` — the third element is now a single GT-level blind scalar (zero in Transparent mode)
- `prove()` and `create_evaluation_proof()` now take a `commit_blind: F` parameter
- `DoryProverState::set_initial_blinds()` now takes `r_d1` as its first parameter

## [0.2.0] - 2026-01-29

### Changed
- Optimized verifier pairing check from 5 individual pairings to a size-3 multi-pairing
- Eliminated the separate VMV pairing check by batching it with the final verification
- Cache now intelligently resizes itself when setup size changes

### Fixed

- Append E final terms to the transcript in the reduce-and-fold protocol
- Use d² instead of d for batching the VMV term
- Removed `dirs` crate dependency; cache directory is now determined at runtime

### Added

- New `homomorphic_mixed_sizes` example demonstrating combination of polynomials with different matrix dimensions

## [0.1.0] - 2025-11-15

### Added

- Initial release
- Arkworks BN254 backend
- Prepared point caching for ~20-30% pairing speedup
- Support for square and non-square matrix layouts (nu ≤ sigma)
- Homomorphic commitment properties
- Comprehensive test suite including soundness tests

[0.4.0]: https://github.com/a16z/dory/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/a16z/dory/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/a16z/dory/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/a16z/dory/releases/tag/v0.1.0
