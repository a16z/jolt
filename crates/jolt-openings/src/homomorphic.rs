//! Homomorphic batched-opening helper via random linear combination.

use std::iter::successors;

use jolt_crypto::HomomorphicCommitment;
use jolt_field::Field;
use jolt_transcript::{AppendToTranscript, LabelWithCount, Transcript};

use crate::claims::{OpeningClaim, ProverClaim};
use crate::error::OpeningsError;
use crate::schemes::{
    AdditivelyHomomorphic, AdditivelyHomomorphicVerifier, CommitmentScheme,
    CommitmentSchemeVerifier,
};
use crate::sources::{materialize_source_evaluations, CommitmentSource};

/// Groups prover claims by point, RLC-combines each group, and opens one proof
/// per group.
#[tracing::instrument(skip_all, name = "homomorphic_prove_batch")]
pub fn homomorphic_prove_batch<PCS, T, S>(
    claims: Vec<ProverClaim<PCS::Field, S>>,
    hints: Vec<PCS::OpeningHint>,
    setup: &PCS::ProverSetup,
    transcript: &mut T,
) -> Vec<PCS::Proof>
where
    PCS: AdditivelyHomomorphic,
    PCS::Output: HomomorphicCommitment<PCS::Field>,
    S: CommitmentSource<PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    assert_eq!(
        claims.len(),
        hints.len(),
        "one opening hint is required for each prover claim",
    );
    if claims.is_empty() {
        return Vec::new();
    }

    if claims.len() == 1 {
        let mut claims = claims.into_iter();
        let mut hints = hints.into_iter();
        let Some(claim) = claims.next() else {
            unreachable!("single claim exists after len check");
        };
        let Some(hint) = hints.next() else {
            unreachable!("single hint exists after len check");
        };
        return vec![PCS::open(
            &claim.polynomial,
            &claim.point,
            claim.eval,
            setup,
            Some(hint),
            transcript,
        )];
    }

    bind_batch_claims::<PCS::Field, _, _>(&claims, transcript);

    let groups = group_prover_claims_by_point::<PCS, S>(claims.into_iter().zip(hints).collect());
    let mut proofs = Vec::with_capacity(groups.len());
    for (point, group_claims) in groups {
        let rho: PCS::Field = transcript.challenge();
        let powers = rho_powers(rho, group_claims.len());

        let mut eval_slices = Vec::with_capacity(group_claims.len());
        let mut evals = Vec::with_capacity(group_claims.len());
        let mut hints = Vec::with_capacity(group_claims.len());
        for (claim, hint) in &group_claims {
            eval_slices.push(source_evaluations(&claim.polynomial));
            evals.push(claim.eval);
            hints.push(hint.clone());
        }
        let eval_slices: Vec<&[PCS::Field]> = eval_slices.iter().map(Vec::as_slice).collect();

        let combined_evals = rlc_combine(&eval_slices, rho);
        let combined_eval = rlc_combine_scalars(&evals, rho);
        let combined_hint = PCS::combine_hints(hints, &powers);
        proofs.push(PCS::open(
            &combined_evals,
            &point,
            combined_eval,
            setup,
            Some(combined_hint),
            transcript,
        ));
    }
    proofs
}

/// Groups verifier claims by point, RLC-combines each group, and verifies one
/// proof per group.
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

    if claims.len() == 1 {
        let [proof] = proofs else {
            return Err(OpeningsError::VerificationFailed);
        };
        let mut claims = claims.into_iter();
        let Some(claim) = claims.next() else {
            unreachable!("single claim exists after len check");
        };
        return PCS::verify(
            &claim.commitment,
            &claim.point,
            claim.eval,
            proof,
            setup,
            transcript,
        );
    }

    bind_batch_claims::<PCS::Field, _, _>(&claims, transcript);

    let groups = group_opening_claims_by_point::<PCS>(claims);
    if groups.len() != proofs.len() {
        return Err(OpeningsError::VerificationFailed);
    }

    for ((point, group_claims), proof) in groups.into_iter().zip(proofs.iter()) {
        let rho: PCS::Field = transcript.challenge();
        let powers = rho_powers(rho, group_claims.len());

        let commitments: Vec<PCS::Output> = group_claims
            .iter()
            .map(|claim| claim.commitment.clone())
            .collect();
        let evals: Vec<PCS::Field> = group_claims.iter().map(|claim| claim.eval).collect();

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

/// result[i] = p_1[i] + ρ · p_2[i] + ρ² · p_3[i] + ... .
#[expect(
    clippy::expect_used,
    reason = "empty polynomials is an API contract violation"
)]
#[tracing::instrument(skip_all, name = "rlc_combine")]
pub fn rlc_combine<F: Field>(polynomials: &[&[F]], rho: F) -> Vec<F> {
    let (last, rest) = polynomials
        .split_last()
        .expect("rlc_combine requires at least one polynomial");
    let len = last.len();

    let mut result = last.to_vec();
    for p in rest.iter().rev() {
        assert_eq!(p.len(), len);
        for (r, &val) in result.iter_mut().zip(p.iter()) {
            *r = *r * rho + val;
        }
    }
    result
}

/// v_1 + ρ · v_2 + ρ² · v_3 + ... .
pub fn rlc_combine_scalars<F: Field>(evals: &[F], rho: F) -> F {
    assert!(!evals.is_empty(), "need at least one evaluation");
    let mut result = F::zero();
    for &v in evals.iter().rev() {
        result = result * rho + v;
    }
    result
}

fn bind_batch_claims<F, C, T>(claims: &[C], transcript: &mut T)
where
    F: Field,
    C: ClaimEval<F>,
    T: Transcript<Challenge = F>,
{
    transcript.append(&LabelWithCount(b"rlc_claims", claims.len() as u64));
    for claim in claims {
        claim.eval().append_to_transcript(transcript);
    }
}

trait ClaimEval<F: Field> {
    fn eval(&self) -> F;
}

impl<F, P> ClaimEval<F> for ProverClaim<F, P>
where
    F: Field,
{
    fn eval(&self) -> F {
        self.eval
    }
}

impl<F, PCS> ClaimEval<F> for OpeningClaim<F, PCS>
where
    F: Field,
    PCS: CommitmentSchemeVerifier<Field = F>,
{
    fn eval(&self) -> F {
        self.eval
    }
}

fn rho_powers<F: Field>(rho: F, n: usize) -> Vec<F> {
    successors(Some(F::from_u64(1)), |prev| Some(*prev * rho))
        .take(n)
        .collect()
}

fn source_evaluations<F, S>(source: &S) -> Vec<F>
where
    F: Field,
    S: CommitmentSource<F> + ?Sized,
{
    materialize_source_evaluations(source)
}

type ProverPointGroup<F, PCS, S> = Vec<(Vec<F>, Vec<ProverClaimWithHint<F, PCS, S>>)>;

type ProverClaimWithHint<F, PCS, S> = (ProverClaim<F, S>, <PCS as CommitmentScheme>::OpeningHint);

fn group_prover_claims_by_point<PCS, S>(
    claims: Vec<ProverClaimWithHint<PCS::Field, PCS, S>>,
) -> ProverPointGroup<PCS::Field, PCS, S>
where
    PCS: CommitmentScheme,
    S: CommitmentSource<PCS::Field>,
{
    let mut groups: ProverPointGroup<PCS::Field, PCS, S> = Vec::new();
    for (claim, hint) in claims {
        if let Some((_, group)) = groups.iter_mut().find(|(point, _)| *point == claim.point) {
            group.push((claim, hint));
        } else {
            let point = claim.point.clone();
            groups.push((point, vec![(claim, hint)]));
        }
    }
    groups
}

type OpeningPointGroup<F, PCS> = Vec<(Vec<F>, Vec<OpeningClaim<F, PCS>>)>;

fn group_opening_claims_by_point<PCS>(
    claims: Vec<OpeningClaim<PCS::Field, PCS>>,
) -> OpeningPointGroup<PCS::Field, PCS>
where
    PCS: CommitmentSchemeVerifier,
{
    let mut groups: OpeningPointGroup<PCS::Field, PCS> = Vec::new();
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
    use jolt_field::{Fr, FromPrimitiveInt, RandomSampling};
    use jolt_poly::Polynomial;
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha20Rng;

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
    fn rlc_combine_scalars_consistent_with_rlc_combine() {
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
