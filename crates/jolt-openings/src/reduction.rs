//! Opening claim reduction via random linear combination (RLC).

use jolt_field::Field;
use jolt_transcript::{AppendToTranscript, LabelWithCount, Transcript};

use crate::backend::{CommitmentBackend, CommitmentOrigin};
use crate::claims::{ProverClaim, VerifierClaim};
use crate::error::OpeningsError;
use crate::schemes::{AdditivelyHomomorphic, CommitmentScheme};
use jolt_crypto::HomomorphicCommitment;

/// One backend-side opening claim: `(commitment_handle, point, eval)`.
///
/// Mirrors [`VerifierClaim`] but with the commitment, point, and
/// evaluation lifted into a [`CommitmentBackend`]'s associated handles.
/// Used as both input and output of
/// [`OpeningReduction::reduce_verifier_with_backend`].
pub type BackendVerifierClaim<B, PCS> = (
    <B as CommitmentBackend<PCS>>::Commitment,
    Vec<<B as crate::backend::FieldBackend>::Scalar>,
    <B as crate::backend::FieldBackend>::Scalar,
);

/// Reduces many opening claims into fewer. Each PCS provides its own
/// implementation, since the natural batching strategy is scheme-specific
/// (RLC for homomorphic schemes, FRI/DEEP-ALI batching for hash-based
/// schemes, etc.).
///
/// Homomorphic schemes (Dory, HyperKZG, Mock) can delegate their
/// [`OpeningReduction`] impls to [`homomorphic_reduce_prover`] /
/// [`homomorphic_reduce_verifier`] / [`homomorphic_reduce_verifier_with_backend`].
#[allow(clippy::type_complexity)]
pub trait OpeningReduction: CommitmentScheme {
    fn reduce_prover<T: Transcript<Challenge = Self::Field>>(
        claims: Vec<ProverClaim<Self::Field>>,
        transcript: &mut T,
    ) -> Vec<ProverClaim<Self::Field>>;

    fn reduce_verifier<T: Transcript<Challenge = Self::Field>>(
        claims: Vec<VerifierClaim<Self::Field, Self::Output>>,
        transcript: &mut T,
    ) -> Result<Vec<VerifierClaim<Self::Field, Self::Output>>, OpeningsError>;

    /// Backend-aware mirror of [`Self::reduce_verifier`].
    ///
    /// Called from `verify_with_backend` (top-level Jolt verifier) so the
    /// reduction is recorded into whatever backend the verifier is running
    /// against (Native: direct combine; Tracing: emits the corresponding
    /// AST nodes via `wrap_commitment` / `absorb_commitment`).
    ///
    /// The output Fiat-Shamir transcript bytes (and squeezed challenges)
    /// MUST be byte-identical to [`Self::reduce_verifier`] on `Native` for
    /// any input — this is what makes the
    /// `modular_self_verify_via_tracing_backend` round-trip work.
    fn reduce_verifier_with_backend<B>(
        backend: &mut B,
        claims: Vec<BackendVerifierClaim<B, Self>>,
        transcript: &mut B::Transcript,
    ) -> Result<Vec<BackendVerifierClaim<B, Self>>, OpeningsError>
    where
        B: CommitmentBackend<Self, F = Self::Field>,
        Self::Output: AppendToTranscript;
}

/// RLC-based prover-side reduction for [`AdditivelyHomomorphic`] schemes.
///
/// Groups claims by point, draws ρ per group, combines: p = Σ ρ^i · p_i.
/// Each homomorphic scheme should delegate its
/// [`OpeningReduction::reduce_prover`] impl to this helper.
#[tracing::instrument(skip_all, name = "homomorphic_reduce_prover")]
pub fn homomorphic_reduce_prover<PCS, T>(
    claims: Vec<ProverClaim<PCS::Field>>,
    transcript: &mut T,
) -> Vec<ProverClaim<PCS::Field>>
where
    PCS: AdditivelyHomomorphic,
    PCS::Output: HomomorphicCommitment<PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    if claims.is_empty() {
        return Vec::new();
    }

    transcript.append(&LabelWithCount(b"rlc_claims", claims.len() as u64));
    for claim in &claims {
        claim.eval.append_to_transcript(transcript);
    }

    let groups = group_prover_claims_by_point(claims);
    let mut reduced = Vec::with_capacity(groups.len());

    for (point, group_claims) in groups {
        let rho: PCS::Field = transcript.challenge();

        let eval_slices: Vec<&[PCS::Field]> = group_claims
            .iter()
            .map(|c| c.polynomial.evaluations())
            .collect();
        let evals: Vec<PCS::Field> = group_claims.iter().map(|c| c.eval).collect();

        let combined_evals = rlc_combine(&eval_slices, rho);
        let combined_eval = rlc_combine_scalars(&evals, rho);

        reduced.push(ProverClaim {
            polynomial: combined_evals.into(),
            point,
            eval: combined_eval,
        });
    }

    reduced
}

/// RLC-based verifier-side reduction for [`AdditivelyHomomorphic`] schemes.
///
/// Groups claims by point, draws ρ per group, combines commitments via
/// `PCS::combine`. Each homomorphic scheme should delegate its
/// [`OpeningReduction::reduce_verifier`] impl to this helper.
#[allow(clippy::type_complexity)]
#[tracing::instrument(skip_all, name = "homomorphic_reduce_verifier")]
pub fn homomorphic_reduce_verifier<PCS, T>(
    claims: Vec<VerifierClaim<PCS::Field, PCS::Output>>,
    transcript: &mut T,
) -> Result<Vec<VerifierClaim<PCS::Field, PCS::Output>>, OpeningsError>
where
    PCS: AdditivelyHomomorphic,
    PCS::Output: HomomorphicCommitment<PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    if claims.is_empty() {
        return Ok(Vec::new());
    }

    transcript.append(&LabelWithCount(b"rlc_claims", claims.len() as u64));
    for claim in &claims {
        claim.eval.append_to_transcript(transcript);
    }

    let groups = group_verifier_claims_by_point(claims);
    let mut reduced = Vec::with_capacity(groups.len());

    for (point, group_claims) in groups {
        let rho: PCS::Field = transcript.challenge();

        let commitments: Vec<PCS::Output> =
            group_claims.iter().map(|c| c.commitment.clone()).collect();
        let evals: Vec<PCS::Field> = group_claims.iter().map(|c| c.eval).collect();

        let powers = rho_powers(rho, commitments.len());
        let combined_commitment = PCS::combine(&commitments, &powers);
        let combined_eval = rlc_combine_scalars(&evals, rho);

        reduced.push(VerifierClaim {
            commitment: combined_commitment,
            point,
            eval: combined_eval,
        });
    }

    Ok(reduced)
}

/// Backend-aware mirror of [`homomorphic_reduce_verifier`].
///
/// Each homomorphic scheme delegates its
/// [`OpeningReduction::reduce_verifier_with_backend`] impl here. The
/// backend's transcript is driven through the same byte sequence as the
/// native helper (`LabelWithCount` header + per-claim eval absorbs +
/// per-group ρ challenge squeeze), so squeezed challenges remain
/// bit-identical to the native path. Commitments are unwrapped via
/// [`CommitmentBackend::unwrap_commitment`], combined via `PCS::combine`,
/// and re-wrapped with [`CommitmentBackend::wrap_commitment`]. Combined
/// evaluations are computed as a backend-side Horner evaluation so the
/// resulting scalar threads through the AST instead of arriving as a
/// fresh wrap.
#[allow(clippy::type_complexity)]
#[tracing::instrument(skip_all, name = "homomorphic_reduce_verifier_with_backend")]
pub fn homomorphic_reduce_verifier_with_backend<PCS, B>(
    backend: &mut B,
    claims: Vec<BackendVerifierClaim<B, PCS>>,
    transcript: &mut B::Transcript,
) -> Result<Vec<BackendVerifierClaim<B, PCS>>, OpeningsError>
where
    PCS: AdditivelyHomomorphic,
    PCS::Output: HomomorphicCommitment<PCS::Field> + AppendToTranscript,
    B: CommitmentBackend<PCS, F = PCS::Field>,
{
    if claims.is_empty() {
        return Ok(Vec::new());
    }

    transcript.append(&LabelWithCount(b"rlc_claims", claims.len() as u64));
    for (_, _, eval_scalar) in &claims {
        let eval_value = backend.unwrap(eval_scalar).expect(
            "homomorphic_reduce_verifier_with_backend: \
             backend must expose concrete eval values for transcript binding",
        );
        eval_value.append_to_transcript(transcript);
    }

    let groups = group_backend_claims_by_point::<PCS, B>(backend, claims);
    let mut reduced = Vec::with_capacity(groups.len());

    for (point_scalars, group_claims) in groups {
        let (rho, rho_scalar) = backend.squeeze(transcript, "rlc_rho");

        let commitment_outputs: Vec<PCS::Output> = group_claims
            .iter()
            .map(|(c, _, _)| backend.unwrap_commitment(c))
            .collect();
        let powers = rho_powers(rho, commitment_outputs.len());
        let combined_commitment = PCS::combine(&commitment_outputs, &powers);
        let combined_handle =
            backend.wrap_commitment(combined_commitment, CommitmentOrigin::Proof, "rlc_combined");

        let combined_eval = horner_combine_scalars(backend, &group_claims, &rho_scalar);

        reduced.push((combined_handle, point_scalars, combined_eval));
    }

    Ok(reduced)
}

/// Backend-side Horner evaluation: returns
/// `evals[0] + ρ · evals[1] + ρ² · evals[2] + ...`, threaded through
/// `backend.add` / `backend.mul` so the result references the same AST
/// nodes the original eval scalars do.
fn horner_combine_scalars<PCS, B>(
    backend: &mut B,
    group_claims: &[BackendVerifierClaim<B, PCS>],
    rho_scalar: &B::Scalar,
) -> B::Scalar
where
    PCS: AdditivelyHomomorphic,
    PCS::Output: HomomorphicCommitment<PCS::Field> + AppendToTranscript,
    B: CommitmentBackend<PCS, F = PCS::Field>,
{
    let mut iter = group_claims.iter().rev();
    let (_, _, last_eval) = iter
        .next()
        .expect("homomorphic_reduce_verifier_with_backend: empty group");
    let mut acc = last_eval.clone();
    for (_, _, eval) in iter {
        let scaled = backend.mul(&acc, rho_scalar);
        acc = backend.add(&scaled, eval);
    }
    acc
}

#[allow(clippy::type_complexity)]
fn group_backend_claims_by_point<PCS, B>(
    backend: &B,
    claims: Vec<BackendVerifierClaim<B, PCS>>,
) -> Vec<(Vec<B::Scalar>, Vec<BackendVerifierClaim<B, PCS>>)>
where
    PCS: CommitmentScheme,
    PCS::Output: AppendToTranscript,
    B: CommitmentBackend<PCS, F = PCS::Field>,
{
    // Group by the underlying field-valued point. Two claims belong to the
    // same group iff their points unwrap to the same `Vec<F>` — this matches
    // the native helper's `Vec<F> == Vec<F>` comparison while still letting
    // each group keep its original backend-scalar handles for downstream
    // ops.
    let mut groups: Vec<(Vec<PCS::Field>, Vec<B::Scalar>, Vec<BackendVerifierClaim<B, PCS>>)> =
        Vec::new();
    for (commitment, point_scalars, eval) in claims {
        let point_values: Vec<PCS::Field> = point_scalars
            .iter()
            .map(|s| {
                backend.unwrap(s).expect(
                    "homomorphic_reduce_verifier_with_backend: backend must expose concrete \
                     point values for grouping",
                )
            })
            .collect();
        if let Some((_, _, group)) = groups.iter_mut().find(|(p, _, _)| *p == point_values) {
            group.push((commitment, point_scalars, eval));
        } else {
            let canonical_point_scalars = point_scalars.clone();
            groups.push((
                point_values,
                canonical_point_scalars,
                vec![(commitment, point_scalars, eval)],
            ));
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

type PointGroup<F> = Vec<(Vec<F>, Vec<ProverClaim<F>>)>;
type VerifierPointGroup<F, C> = Vec<(Vec<F>, Vec<VerifierClaim<F, C>>)>;

fn group_prover_claims_by_point<F: Field>(claims: Vec<ProverClaim<F>>) -> PointGroup<F> {
    let mut groups: PointGroup<F> = Vec::new();
    for claim in claims {
        if let Some((_, group)) = groups.iter_mut().find(|(point, _)| *point == claim.point) {
            group.push(claim);
        } else {
            let point = claim.point.clone();
            groups.push((point, vec![claim]));
        }
    }
    groups
}

fn group_verifier_claims_by_point<F: Field, C>(
    claims: Vec<VerifierClaim<F, C>>,
) -> VerifierPointGroup<F, C> {
    let mut groups: VerifierPointGroup<F, C> = Vec::new();
    for claim in claims {
        if let Some((_, group)) = groups.iter_mut().find(|(point, _)| *point == claim.point) {
            group.push(claim);
        } else {
            let point = claim.point.clone();
            groups.push((point, vec![claim]));
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

    #[test]
    fn group_prover_claims_same_point() {
        let point = vec![Fr::from_u64(1), Fr::from_u64(2)];
        let claims = vec![
            ProverClaim {
                polynomial: vec![Fr::from_u64(10)].into(),
                point: point.clone(),
                eval: Fr::from_u64(10),
            },
            ProverClaim {
                polynomial: vec![Fr::from_u64(20)].into(),
                point: point.clone(),
                eval: Fr::from_u64(20),
            },
        ];
        let groups = group_prover_claims_by_point(claims);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].1.len(), 2);
    }

    #[test]
    fn group_prover_claims_different_points() {
        let claims = vec![
            ProverClaim {
                polynomial: vec![Fr::from_u64(10)].into(),
                point: vec![Fr::from_u64(1)],
                eval: Fr::from_u64(10),
            },
            ProverClaim {
                polynomial: vec![Fr::from_u64(20)].into(),
                point: vec![Fr::from_u64(2)],
                eval: Fr::from_u64(20),
            },
        ];
        let groups = group_prover_claims_by_point(claims);
        assert_eq!(groups.len(), 2);
    }
}
