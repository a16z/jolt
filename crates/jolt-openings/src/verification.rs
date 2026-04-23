//! Batched opening verification: PCS-owned single-shot batch proving / verification.
//!
//! Each PCS implements [`OpeningVerification`] with a scheme-specific
//! [`OpeningVerification::BatchProof`] type:
//!
//! - Mock / HyperKZG / Dory: `BatchProof = Vec<Self::Proof>` — one proof
//!   per RLC-grouped (point, polys) bundle. They delegate to
//!   [`homomorphic_prove_batch`] / [`homomorphic_verify_batch_with_backend`].
//! - Hachi (future): `BatchProof = Self::BatchedProof` — a single fused
//!   proof object covering every claim in the batch. They wire their
//!   own scheme-specific helper.
//!
//! This trait subsumes the previous `OpeningReduction` trait. There is no
//! intermediate "reduce, then verify per group" step exposed in the API:
//! the entire batch is opened/verified in one call so PCSes that batch
//! monolithically (Hachi) fit naturally alongside PCSes that batch by
//! reduce-then-open (homomorphic schemes).

use std::fmt::Debug;

use jolt_crypto::HomomorphicCommitment;
use jolt_field::Field;
use jolt_transcript::{AppendToTranscript, LabelWithCount, Transcript};
use serde::{de::DeserializeOwned, Serialize};

use crate::backend::{CommitmentBackend, CommitmentOrigin};
use crate::claims::{OpeningClaim, ProverClaim};
use crate::error::OpeningsError;
use crate::schemes::{AdditivelyHomomorphic, CommitmentScheme};

/// PCS-owned batched opening prove + verify.
///
/// Implementations OWN the entire batching contract: there is no
/// "reduce step" exposed to the orchestrator. The orchestrator just
/// accumulates per-poly claims (and per-poly hints on the prover) and
/// hands the whole bag to [`prove_batch`] / [`verify_batch_with_backend`]
/// once.
///
/// **Prover/verifier transcript invariant.** [`prove_batch`] and
/// [`verify_batch_with_backend`] MUST drive the Fiat-Shamir transcript
/// through byte-identical sequences for any matching pair of input
/// claims. Verifier-side parity tests (one per PCS, e.g.
/// `crates/jolt-verifier-backend/tests/verification_parity.rs`) cross-
/// check this against `Native` runs.
///
/// **Joint-eval surface.** [`prove_batch`] returns a `Vec<Self::Field>`
/// of "binding evals" alongside the [`Self::BatchProof`]. Today the
/// orchestrator routes `binding_evals[0]` into `Op::BindOpeningInputs`
/// (the post-opening Dory transcript bind). For homomorphic schemes
/// this is the per-group reduced eval (length = number of distinct
/// opening points = 1 in current jolt-zkvm). For non-homomorphic
/// schemes (Hachi), implementations may return whatever scalars they
/// wish to expose for downstream binding.
///
/// [`prove_batch`]: OpeningVerification::prove_batch
/// [`verify_batch_with_backend`]: OpeningVerification::verify_batch_with_backend
pub trait OpeningVerification: CommitmentScheme {
    /// Single proof object covering the entire batch.
    ///
    /// - Mock / HyperKZG / Dory: `Vec<Self::Proof>` — one per RLC group.
    /// - Hachi (future): `Self::BatchedProof` — single fused proof.
    type BatchProof: Clone + Debug + Send + Sync + Serialize + DeserializeOwned;

    /// Prove an entire batch of opening claims in one shot.
    ///
    /// The orchestrator passes per-poly claims (with materialised
    /// polynomials inside [`ProverClaim`]) and a parallel `hints`
    /// vector (`hints[i]` is the commit-time hint for `claims[i]`).
    ///
    /// Returns `(batch_proof, binding_evals)`; see the trait docs for
    /// the meaning of `binding_evals`.
    fn prove_batch<T: Transcript<Challenge = Self::Field>>(
        claims: Vec<ProverClaim<Self::Field>>,
        hints: Vec<Self::OpeningHint>,
        setup: &Self::ProverSetup,
        transcript: &mut T,
    ) -> (Self::BatchProof, Vec<Self::Field>);

    /// Verify an entire batch of opening claims against a single
    /// [`Self::BatchProof`].
    ///
    /// Drives the Fiat-Shamir transcript through the same byte sequence
    /// as [`prove_batch`] for any matching input. The backend-side
    /// transcript can be a `Tracing` recorder — the homomorphic helper
    /// re-wraps any combined commitments through
    /// [`CommitmentBackend::wrap_commitment`] so they thread into the
    /// AST as named nodes.
    fn verify_batch_with_backend<B>(
        backend: &mut B,
        vk: &Self::VerifierSetup,
        claims: Vec<OpeningClaim<B, Self>>,
        batch_proof: &Self::BatchProof,
        transcript: &mut B::Transcript,
    ) -> Result<(), OpeningsError>
    where
        B: CommitmentBackend<Self, F = Self::Field>,
        Self::Output: AppendToTranscript;
}

/// RLC-based batched proving for [`AdditivelyHomomorphic`] schemes.
///
/// Drop-in body for `OpeningVerification::prove_batch` for Mock /
/// HyperKZG / Dory. Mirrors [`homomorphic_verify_batch_with_backend`]
/// on the verifier side: groups by point, draws ρ per group, combines
/// polynomials and hints, opens once per group.
///
/// Returns `(per_group_proofs, per_group_evals)`. The per-group evals
/// are the RLC-combined evaluations (one per distinct opening point),
/// suitable for routing into a downstream `bind_opening_inputs` call
/// via the trait's `binding_evals` return slot.
#[allow(clippy::type_complexity)]
#[tracing::instrument(skip_all, name = "homomorphic_prove_batch")]
pub fn homomorphic_prove_batch<PCS, T>(
    claims: Vec<ProverClaim<PCS::Field>>,
    hints: Vec<PCS::OpeningHint>,
    setup: &PCS::ProverSetup,
    transcript: &mut T,
) -> (Vec<PCS::Proof>, Vec<PCS::Field>)
where
    PCS: AdditivelyHomomorphic,
    PCS::Output: HomomorphicCommitment<PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    if claims.is_empty() {
        debug_assert!(
            hints.is_empty(),
            "homomorphic_prove_batch: hints must be empty when claims are empty"
        );
        return (Vec::new(), Vec::new());
    }
    assert_eq!(
        claims.len(),
        hints.len(),
        "homomorphic_prove_batch: hints and claims must be parallel"
    );

    transcript.append(&LabelWithCount(b"rlc_claims", claims.len() as u64));
    for claim in &claims {
        claim.eval.append_to_transcript(transcript);
    }

    let groups = group_prover_claims_with_hints::<PCS>(claims, hints);
    let mut proofs = Vec::with_capacity(groups.len());
    let mut binding_evals = Vec::with_capacity(groups.len());

    for (point, group_claims, group_hints) in groups {
        let rho: PCS::Field = transcript.challenge();

        let eval_slices: Vec<&[PCS::Field]> = group_claims
            .iter()
            .map(|c| c.polynomial.evaluations())
            .collect();
        let evals: Vec<PCS::Field> = group_claims.iter().map(|c| c.eval).collect();

        let combined_evals = rlc_combine(&eval_slices, rho);
        let combined_eval = rlc_combine_scalars(&evals, rho);

        let powers = rho_powers(rho, group_hints.len());
        let combined_hint = PCS::combine_hints(group_hints, &powers);

        let combined_poly: PCS::Polynomial = combined_evals.into();
        let proof = PCS::open(
            &combined_poly,
            &point,
            combined_eval,
            setup,
            Some(combined_hint),
            transcript,
        );

        proofs.push(proof);
        binding_evals.push(combined_eval);
    }

    (proofs, binding_evals)
}

/// RLC-based batched verification for [`AdditivelyHomomorphic`] schemes.
///
/// Drop-in body for `OpeningVerification::verify_batch_with_backend` for
/// Mock / HyperKZG / Dory. Drives the same transcript bytes as
/// [`homomorphic_prove_batch`]: header → per-claim eval absorb →
/// per-group ρ squeeze → per-group `verify_opening` (which itself drives
/// the proof-bytes part of the transcript).
///
/// Combined commitments are produced via `PCS::combine` on values
/// unwrapped out of the backend, then re-wrapped via
/// [`CommitmentBackend::wrap_commitment`] so the resulting handle
/// references the corresponding AST node when the backend is `Tracing`.
/// Combined evals are computed via backend-side Horner evaluation so the
/// scalar threads through the AST instead of arriving as a fresh wrap.
#[allow(clippy::type_complexity)]
#[tracing::instrument(skip_all, name = "homomorphic_verify_batch_with_backend")]
pub fn homomorphic_verify_batch_with_backend<PCS, B>(
    backend: &mut B,
    vk: &PCS::VerifierSetup,
    claims: Vec<OpeningClaim<B, PCS>>,
    batch_proof: &[PCS::Proof],
    transcript: &mut B::Transcript,
) -> Result<(), OpeningsError>
where
    PCS: AdditivelyHomomorphic + OpeningVerification,
    PCS::Output: HomomorphicCommitment<PCS::Field> + AppendToTranscript,
    B: CommitmentBackend<PCS, F = PCS::Field>,
{
    if claims.is_empty() {
        if batch_proof.is_empty() {
            return Ok(());
        }
        return Err(OpeningsError::VerificationFailed);
    }

    transcript.append(&LabelWithCount(b"rlc_claims", claims.len() as u64));
    for claim in &claims {
        let eval_value = backend.unwrap(&claim.eval).expect(
            "homomorphic_verify_batch_with_backend: \
             backend must expose concrete eval values for transcript binding",
        );
        eval_value.append_to_transcript(transcript);
    }

    let groups = group_backend_claims_by_point::<PCS, B>(backend, claims);

    if groups.len() != batch_proof.len() {
        return Err(OpeningsError::VerificationFailed);
    }

    for ((point_scalars, group_claims), proof) in groups.into_iter().zip(batch_proof.iter()) {
        let (rho, rho_scalar) = backend.squeeze(transcript, "rlc_rho");

        let commitment_outputs: Vec<PCS::Output> = group_claims
            .iter()
            .map(|c| backend.unwrap_commitment(&c.commitment))
            .collect();
        let powers = rho_powers(rho, commitment_outputs.len());
        let combined_commitment = PCS::combine(&commitment_outputs, &powers);
        let combined_handle =
            backend.wrap_commitment(combined_commitment, CommitmentOrigin::Proof, "rlc_combined");

        let combined_eval = horner_combine_eval_scalars::<PCS, B>(backend, &group_claims, &rho_scalar);

        backend.verify_opening(
            vk,
            &combined_handle,
            &point_scalars,
            &combined_eval,
            proof,
            transcript,
        )?;
    }

    Ok(())
}

/// Backend-side Horner evaluation: returns
/// `evals[0] + ρ · evals[1] + ρ² · evals[2] + ...`, threaded through
/// `backend.add` / `backend.mul` so the result references the same AST
/// nodes the original eval scalars do.
fn horner_combine_eval_scalars<PCS, B>(
    backend: &mut B,
    group_claims: &[OpeningClaim<B, PCS>],
    rho_scalar: &B::Scalar,
) -> B::Scalar
where
    PCS: AdditivelyHomomorphic,
    PCS::Output: HomomorphicCommitment<PCS::Field> + AppendToTranscript,
    B: CommitmentBackend<PCS, F = PCS::Field>,
{
    let mut iter = group_claims.iter().rev();
    let last = iter
        .next()
        .expect("homomorphic_verify_batch_with_backend: empty group");
    let mut acc = last.eval.clone();
    for claim in iter {
        let scaled = backend.mul(&acc, rho_scalar);
        acc = backend.add(&scaled, &claim.eval);
    }
    acc
}

#[allow(clippy::type_complexity)]
fn group_backend_claims_by_point<PCS, B>(
    backend: &B,
    claims: Vec<OpeningClaim<B, PCS>>,
) -> Vec<(Vec<B::Scalar>, Vec<OpeningClaim<B, PCS>>)>
where
    PCS: CommitmentScheme,
    PCS::Output: AppendToTranscript,
    B: CommitmentBackend<PCS, F = PCS::Field>,
{
    // Group by the underlying field-valued point. Two claims belong to
    // the same group iff their points unwrap to the same `Vec<F>` —
    // matches the prover-side `Vec<F> == Vec<F>` comparison while still
    // letting each group keep its original backend-scalar handles for
    // downstream ops (so the AST keeps referencing the original nodes
    // instead of fresh wraps).
    let mut groups: Vec<(Vec<PCS::Field>, Vec<B::Scalar>, Vec<OpeningClaim<B, PCS>>)> = Vec::new();
    for claim in claims {
        let point_values: Vec<PCS::Field> = claim
            .point
            .iter()
            .map(|s| {
                backend.unwrap(s).expect(
                    "homomorphic_verify_batch_with_backend: backend must expose concrete \
                     point values for grouping",
                )
            })
            .collect();
        if let Some((_, _, group)) = groups.iter_mut().find(|(p, _, _)| *p == point_values) {
            group.push(claim);
        } else {
            let canonical_point_scalars = claim.point.clone();
            groups.push((point_values, canonical_point_scalars, vec![claim]));
        }
    }
    groups
        .into_iter()
        .map(|(_, point_scalars, group)| (point_scalars, group))
        .collect()
}

/// result[i] = p_1[i] + ρ · p_2[i] + ρ² · p_3[i] + ... (Horner evaluation).
#[tracing::instrument(skip_all, name = "rlc_combine")]
pub fn rlc_combine<F: Field>(polynomials: &[&[F]], rho: F) -> Vec<F> {
    assert!(!polynomials.is_empty(), "need at least one polynomial");
    let len = polynomials[0].len();

    let mut result = polynomials.last().unwrap().to_vec();
    for p in polynomials.iter().rev().skip(1) {
        assert_eq!(p.len(), len);
        for (r, &val) in result.iter_mut().zip(p.iter()) {
            *r = *r * rho + val;
        }
    }
    result
}

/// v_1 + ρ · v_2 + ρ² · v_3 + ... (Horner evaluation).
pub fn rlc_combine_scalars<F: Field>(evals: &[F], rho: F) -> F {
    assert!(!evals.is_empty(), "need at least one evaluation");
    let mut result = F::zero();
    for &v in evals.iter().rev() {
        result = result * rho + v;
    }
    result
}

fn rho_powers<F: Field>(rho: F, n: usize) -> Vec<F> {
    std::iter::successors(Some(F::from_u64(1)), |prev| Some(*prev * rho))
        .take(n)
        .collect()
}

type ProverPointGroup<F, H> = Vec<(Vec<F>, Vec<ProverClaim<F>>, Vec<H>)>;

fn group_prover_claims_with_hints<PCS>(
    claims: Vec<ProverClaim<PCS::Field>>,
    hints: Vec<PCS::OpeningHint>,
) -> ProverPointGroup<PCS::Field, PCS::OpeningHint>
where
    PCS: CommitmentScheme,
{
    let mut groups: ProverPointGroup<PCS::Field, PCS::OpeningHint> = Vec::new();
    for (claim, hint) in claims.into_iter().zip(hints.into_iter()) {
        if let Some((_, group, hint_group)) =
            groups.iter_mut().find(|(point, _, _)| *point == claim.point)
        {
            group.push(claim);
            hint_group.push(hint);
        } else {
            let point = claim.point.clone();
            groups.push((point, vec![claim], vec![hint]));
        }
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Fr;
    use jolt_poly::Polynomial;

    #[test]
    fn rlc_combine_single_polynomial_is_identity() {
        let evals: Vec<Fr> = (0..4).map(|i| Fr::from_u64(i + 1)).collect();
        let rho = Fr::from_u64(7);
        let result = rlc_combine(&[&evals], rho);
        assert_eq!(result, evals);
    }

    #[test]
    fn rlc_combine_two_polynomials() {
        let p1: Vec<Fr> = (1..=4).map(Fr::from_u64).collect();
        let p2: Vec<Fr> = (5..=8).map(Fr::from_u64).collect();
        let rho = Fr::from_u64(3);
        let result = rlc_combine(&[&p1, &p2], rho);
        for i in 0..4 {
            assert_eq!(result[i], p1[i] + rho * p2[i], "mismatch at index {i}");
        }
    }

    #[test]
    fn rlc_combine_three_polynomials_horner() {
        let p1 = [Fr::from_u64(1)];
        let p2 = [Fr::from_u64(2)];
        let p3 = [Fr::from_u64(3)];
        let rho = Fr::from_u64(5);
        let result = rlc_combine(&[&p1[..], &p2[..], &p3[..]], rho);
        assert_eq!(result[0], Fr::from_u64(86));
    }

    #[test]
    fn rlc_combine_scalars_matches_manual() {
        let evals: Vec<Fr> = vec![Fr::from_u64(10), Fr::from_u64(20), Fr::from_u64(30)];
        let rho = Fr::from_u64(2);
        assert_eq!(rlc_combine_scalars(&evals, rho), Fr::from_u64(170));
    }

    #[test]
    fn rlc_combine_scalars_single() {
        assert_eq!(
            rlc_combine_scalars(&[Fr::from_u64(42)], Fr::from_u64(999)),
            Fr::from_u64(42),
        );
    }

    #[test]
    fn rlc_combine_scalars_consistent_with_rlc_combine() {
        use rand_chacha::rand_core::SeedableRng;
        use rand_chacha::ChaCha20Rng;

        let mut rng = ChaCha20Rng::seed_from_u64(555);
        let num_vars = 3;
        let rho = Fr::from_u64(7);

        let p1 = Polynomial::<Fr>::random(num_vars, &mut rng);
        let p2 = Polynomial::<Fr>::random(num_vars, &mut rng);
        let p3 = Polynomial::<Fr>::random(num_vars, &mut rng);

        let point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();

        let eval1 = p1.evaluate(&point);
        let eval2 = p2.evaluate(&point);
        let eval3 = p3.evaluate(&point);

        let combined = rlc_combine(&[p1.evaluations(), p2.evaluations(), p3.evaluations()], rho);
        let combined_poly = Polynomial::new(combined);
        let result_via_poly = combined_poly.evaluate(&point);
        let result_via_scalars = rlc_combine_scalars(&[eval1, eval2, eval3], rho);

        assert_eq!(result_via_poly, result_via_scalars);
    }
}
