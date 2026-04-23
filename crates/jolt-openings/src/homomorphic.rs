//! Homomorphic batched-opening helpers.
//!
//! These free functions implement the standard "group-by-point + RLC"
//! batched opening for any [`AdditivelyHomomorphic`] PCS. They drive the
//! Fiat-Shamir transcript through the same byte sequence on prover and
//! verifier sides and are the recommended bodies for `prove_batch` /
//! `verify_batch` on schemes that fall into this family. Hachi-style
//! fused-batch schemes can implement those methods directly without
//! touching this module.

use jolt_crypto::HomomorphicCommitment;
use jolt_field::Field;
use jolt_transcript::{AppendToTranscript, LabelWithCount, Transcript};

use crate::claims::{OpeningClaim, ProverClaim};
use crate::error::OpeningsError;
use crate::schemes::{
    AdditivelyHomomorphic, AdditivelyHomomorphicVerifier, CommitmentSchemeVerifier,
};

/// Prover-side fused batched opening for additively-homomorphic schemes.
///
/// Groups claims by point, draws one challenge per group, RLC-combines
/// polynomials/evals/hints, then opens each combined claim via
/// [`AdditivelyHomomorphic::open`]. Returns the per-group proofs (the
/// scheme's `BatchProof = Vec<Self::Proof>`) and the per-group joint
/// evaluations (consumed downstream by `Op::BindOpeningInputs`).
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
        return (Vec::new(), Vec::new());
    }
    assert_eq!(
        claims.len(),
        hints.len(),
        "claims and hints must have the same length"
    );

    transcript.append(&LabelWithCount(b"rlc_claims", claims.len() as u64));
    for c in &claims {
        c.eval.append_to_transcript(transcript);
    }

    let groups = group_prover_claims_by_point(claims, hints);
    let mut proofs = Vec::with_capacity(groups.len());
    let mut joint_evals = Vec::with_capacity(groups.len());

    for (point, group_claims, group_hints) in groups {
        let rho: PCS::Field = transcript.challenge();

        let eval_slices: Vec<&[PCS::Field]> = group_claims
            .iter()
            .map(|c| c.polynomial.evaluations())
            .collect();
        let scalar_evals: Vec<PCS::Field> = group_claims.iter().map(|c| c.eval).collect();

        let combined_evals = rlc_combine(&eval_slices, rho);
        let combined_eval = rlc_combine_scalars(&scalar_evals, rho);
        let combined_poly: PCS::Polynomial = combined_evals.into();

        let powers = rho_powers(rho, group_hints.len());
        let combined_hint = PCS::combine_hints(group_hints, &powers);

        let proof = PCS::open(
            &combined_poly,
            &point,
            combined_eval,
            setup,
            Some(combined_hint),
            transcript,
        );
        proofs.push(proof);
        joint_evals.push(combined_eval);
    }

    (proofs, joint_evals)
}

/// Verifier-side fused batched verification for additively-homomorphic schemes.
///
/// Mirrors [`homomorphic_prove_batch`] step-for-step on the verifier side:
/// groups claims by point, draws the same per-group challenges, builds the
/// RLC-combined commitment via [`AdditivelyHomomorphicVerifier::combine`],
/// and verifies each group's proof via
/// [`AdditivelyHomomorphicVerifier::verify`]. Errors propagate as
/// [`OpeningsError::VerificationFailed`].
#[tracing::instrument(skip_all, name = "homomorphic_verify_batch")]
pub fn homomorphic_verify_batch<PCS, T>(
    claims: Vec<OpeningClaim<PCS::Field, PCS>>,
    proofs: &[PCS::Proof],
    setup: &PCS::VerifierSetup,
    transcript: &mut T,
) -> Result<(), OpeningsError>
where
    PCS: AdditivelyHomomorphicVerifier,
    PCS::Output: HomomorphicCommitment<PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    if claims.is_empty() {
        if proofs.is_empty() {
            return Ok(());
        }
        return Err(OpeningsError::VerificationFailed);
    }

    transcript.append(&LabelWithCount(b"rlc_claims", claims.len() as u64));
    for c in &claims {
        c.eval.append_to_transcript(transcript);
    }

    let groups = group_verifier_claims_by_point::<PCS>(claims);
    if groups.len() != proofs.len() {
        return Err(OpeningsError::VerificationFailed);
    }

    for ((point, group_claims), proof) in groups.into_iter().zip(proofs.iter()) {
        let rho: PCS::Field = transcript.challenge();

        let commitments: Vec<PCS::Output> =
            group_claims.iter().map(|c| c.commitment.clone()).collect();
        let evals: Vec<PCS::Field> = group_claims.iter().map(|c| c.eval).collect();

        let powers = rho_powers(rho, commitments.len());
        let combined_commitment = PCS::combine(&commitments, &powers);
        let combined_eval = rlc_combine_scalars(&evals, rho);

        PCS::verify(
            &combined_commitment,
            &point,
            combined_eval,
            proof,
            setup,
            transcript,
        )?;
    }

    Ok(())
}

/// `result[i] = p_1[i] + ρ · p_2[i] + ρ² · p_3[i] + ...` (Horner).
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

/// `v_1 + ρ · v_2 + ρ² · v_3 + ...` (Horner).
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

type ProverGroup<F, H> = Vec<(Vec<F>, Vec<ProverClaim<F>>, Vec<H>)>;
type VerifierGroup<PCS> = Vec<(
    Vec<<PCS as CommitmentSchemeVerifier>::Field>,
    Vec<OpeningClaim<<PCS as CommitmentSchemeVerifier>::Field, PCS>>,
)>;

fn group_prover_claims_by_point<F: Field, H>(
    claims: Vec<ProverClaim<F>>,
    hints: Vec<H>,
) -> ProverGroup<F, H> {
    let mut groups: ProverGroup<F, H> = Vec::new();
    for (claim, hint) in claims.into_iter().zip(hints) {
        if let Some((_, group, group_hints)) = groups
            .iter_mut()
            .find(|(point, _, _)| *point == claim.point)
        {
            group.push(claim);
            group_hints.push(hint);
        } else {
            let point = claim.point.clone();
            groups.push((point, vec![claim], vec![hint]));
        }
    }
    groups
}

fn group_verifier_claims_by_point<PCS: CommitmentSchemeVerifier>(
    claims: Vec<OpeningClaim<PCS::Field, PCS>>,
) -> VerifierGroup<PCS> {
    let mut groups: VerifierGroup<PCS> = Vec::new();
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
        let hints: Vec<()> = vec![(), ()];
        let groups = group_prover_claims_by_point(claims, hints);
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
        let hints: Vec<()> = vec![(), ()];
        let groups = group_prover_claims_by_point(claims, hints);
        assert_eq!(groups.len(), 2);
    }
}
