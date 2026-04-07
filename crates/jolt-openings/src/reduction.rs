//! Opening claim reduction via random linear combination (RLC).

use jolt_field::Field;
use jolt_transcript::{AppendToTranscript, LabelWithCount, Transcript};

use crate::claims::{ProverClaim, VerifierClaim};
use crate::error::OpeningsError;
use crate::schemes::{AdditivelyHomomorphic, CommitmentScheme};
use jolt_crypto::HomomorphicCommitment;

/// Reduces many opening claims into fewer. Blanket-implemented for
/// [`AdditivelyHomomorphic`] PCS; non-homomorphic schemes must provide their own impl.
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
}

/// Groups claims by point, draws ρ per group, combines: p = Σ ρ^i · p_i.
#[allow(clippy::type_complexity)]
impl<PCS: AdditivelyHomomorphic> OpeningReduction for PCS
where
    PCS::Output: HomomorphicCommitment<PCS::Field>,
{
    #[tracing::instrument(skip_all, name = "OpeningReduction::reduce_prover")]
    fn reduce_prover<T: Transcript<Challenge = PCS::Field>>(
        claims: Vec<ProverClaim<PCS::Field>>,
        transcript: &mut T,
    ) -> Vec<ProverClaim<PCS::Field>> {
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

    #[tracing::instrument(skip_all, name = "OpeningReduction::reduce_verifier")]
    fn reduce_verifier<T: Transcript<Challenge = PCS::Field>>(
        claims: Vec<VerifierClaim<PCS::Field, PCS::Output>>,
        transcript: &mut T,
    ) -> Result<Vec<VerifierClaim<PCS::Field, PCS::Output>>, OpeningsError> {
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
