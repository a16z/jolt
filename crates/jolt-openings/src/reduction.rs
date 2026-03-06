//! Opening claim reduction: trait and RLC-based implementation.
//!
//! An [`OpeningReduction`] transforms a batch of opening claims (many) into
//! fewer claims suitable for direct PCS opening. The output type equals the
//! input type, so reductions compose.
//!
//! [`RlcReduction`] is the standard implementation for additively homomorphic
//! schemes: it groups claims by evaluation point and combines each group via
//! random linear combination (RLC).
//!
//! # RLC utilities
//!
//! [`rlc_combine`] and [`rlc_combine_scalars`] are standalone utility functions
//! for computing random linear combinations of polynomial evaluation tables
//! and scalar evaluations respectively.

use jolt_field::Field;
use jolt_transcript::Transcript;
use serde::{de::DeserializeOwned, Serialize};

use crate::claims::{ProverClaim, VerifierClaim};
use crate::error::OpeningsError;
use crate::traits::{AdditivelyHomomorphic, CommitmentScheme};

#[allow(clippy::type_complexity)]
/// Transforms a batch of opening claims into fewer claims.
///
/// Implementations define the batching strategy. The output type equals the
/// input type, so reductions can be chained:
///
/// ```text
/// many claims → Reduction₁ → fewer claims → Reduction₂ → even fewer claims → PCS::open
/// ```
///
/// The associated [`ReductionProof`](Self::ReductionProof) captures any proof
/// artifact produced by the reduction itself. For deterministic reductions
/// like RLC this is `()`.
///
/// The `challenge_fn` parameter converts a transcript challenge into a field
/// element, following the same pattern as sumcheck. This keeps the trait
/// generic over the transcript's challenge type.
pub trait OpeningReduction<PCS: CommitmentScheme> {
    /// Proof artifact produced by the reduction (if any).
    type ReductionProof: Clone + Send + Sync + Serialize + DeserializeOwned;

    /// Reduces prover-side claims.
    ///
    /// Returns the reduced claims and any proof artifact.
    ///
    /// **Precondition:** the transcript must already contain all claim data
    /// (commitments, points, evaluations) that needs to be bound by
    /// Fiat-Shamir. The reduction draws challenges from the transcript
    /// without absorbing claims itself.
    fn reduce_prover<T: Transcript>(
        claims: Vec<ProverClaim<PCS::Field>>,
        transcript: &mut T,
        challenge_fn: impl Fn(T::Challenge) -> PCS::Field,
    ) -> (Vec<ProverClaim<PCS::Field>>, Self::ReductionProof);

    /// Reduces verifier-side claims.
    ///
    /// Returns the reduced claims or an error if the reduction detects
    /// inconsistency.
    ///
    /// **Precondition:** same transcript state requirement as
    /// [`reduce_prover`](Self::reduce_prover).
    fn reduce_verifier<T: Transcript>(
        claims: Vec<VerifierClaim<PCS::Field, PCS::Output>>,
        proof: &Self::ReductionProof,
        transcript: &mut T,
        challenge_fn: impl Fn(T::Challenge) -> PCS::Field,
    ) -> Result<Vec<VerifierClaim<PCS::Field, PCS::Output>>, OpeningsError>;
}

/// Random linear combination (RLC) reduction for additively homomorphic PCS.
///
/// Groups claims by evaluation point and combines each group:
///
/// $$p_{\text{batch}}(x) = \sum_{i=0}^{k-1} \rho^i \cdot p_i(x), \quad
///   C_{\text{batch}} = \sum_{i=0}^{k-1} \rho^i \cdot C_i, \quad
///   v_{\text{batch}} = \sum_{i=0}^{k-1} \rho^i \cdot v_i$$
///
/// where $\rho$ is a Fiat-Shamir challenge drawn independently per group.
/// Using separate challenges (rather than a single global $\rho$) allows
/// groups to be reduced in parallel. Produces one reduced claim per
/// distinct evaluation point.
///
/// The reduction is deterministic given the transcript state, so
/// [`ReductionProof`](OpeningReduction::ReductionProof) is `()`.
pub struct RlcReduction;

impl<PCS: AdditivelyHomomorphic> OpeningReduction<PCS> for RlcReduction {
    type ReductionProof = ();

    fn reduce_prover<T: Transcript>(
        claims: Vec<ProverClaim<PCS::Field>>,
        transcript: &mut T,
        challenge_fn: impl Fn(T::Challenge) -> PCS::Field,
    ) -> (Vec<ProverClaim<PCS::Field>>, ()) {
        if claims.is_empty() {
            return (Vec::new(), ());
        }

        let groups = group_prover_claims_by_point(claims);
        let mut reduced = Vec::with_capacity(groups.len());

        for (point, group_claims) in groups {
            debug_assert!(
                group_claims.iter().all(|c| c.evaluations.len().is_power_of_two()
                    && c.evaluations.len() == 1 << c.point.len()),
                "evaluation table size must equal 2^num_vars"
            );

            let rho = challenge_fn(transcript.challenge());

            let eval_slices: Vec<&[PCS::Field]> = group_claims
                .iter()
                .map(|c| c.evaluations.as_slice())
                .collect();
            let evals: Vec<PCS::Field> = group_claims.iter().map(|c| c.eval).collect();

            let combined_evals = rlc_combine(&eval_slices, rho);
            let combined_eval = rlc_combine_scalars(&evals, rho);

            reduced.push(ProverClaim {
                evaluations: combined_evals,
                point,
                eval: combined_eval,
            });
        }

        (reduced, ())
    }

    fn reduce_verifier<T: Transcript>(
        claims: Vec<VerifierClaim<PCS::Field, PCS::Output>>,
        _proof: &(),
        transcript: &mut T,
        challenge_fn: impl Fn(T::Challenge) -> PCS::Field,
    ) -> Result<Vec<VerifierClaim<PCS::Field, PCS::Output>>, OpeningsError> {
        if claims.is_empty() {
            return Ok(Vec::new());
        }

        let groups = group_verifier_claims_by_point(claims);
        let mut reduced = Vec::with_capacity(groups.len());

        for (point, group_claims) in groups {
            let rho = challenge_fn(transcript.challenge());

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

/// Computes the RLC of polynomial evaluation tables.
///
/// Given evaluation tables $p_1, \ldots, p_k$ (each of length $2^n$) and a
/// Fiat-Shamir challenge $\rho$, returns the evaluation table of:
///
/// $$p_{\text{combined}}(x) = p_1(x) + \rho \cdot p_2(x) + \rho^2 \cdot p_3(x) + \cdots + \rho^{k-1} \cdot p_k(x)$$
///
/// # Panics
///
/// Panics if `polynomials` is empty or if the evaluation tables have different lengths.
pub fn rlc_combine<F: Field>(polynomials: &[&[F]], rho: F) -> Vec<F> {
    assert!(!polynomials.is_empty(), "need at least one polynomial");
    let len = polynomials[0].len();
    for (i, p) in polynomials.iter().enumerate().skip(1) {
        assert_eq!(
            p.len(),
            len,
            "polynomial {i} has length {} but expected {len}",
            p.len()
        );
    }

    // Horner's method: iterate from the last polynomial backwards.
    // result = p_k, then result = result * rho + p_{k-1}, ...
    let mut result = polynomials.last().unwrap().to_vec();

    for p in polynomials.iter().rev().skip(1) {
        for (r, &val) in result.iter_mut().zip(p.iter()) {
            *r = *r * rho + val;
        }
    }

    result
}

/// Computes the RLC of scalar evaluations.
///
/// Given claimed evaluations $v_1, \ldots, v_k$ and challenge $\rho$, returns:
///
/// $$v_{\text{combined}} = \sum_{i=0}^{k-1} \rho^i \cdot v_i$$
///
/// Uses Horner's method for $O(k)$ multiplications.
///
/// # Panics
///
/// Panics if `evals` is empty.
pub fn rlc_combine_scalars<F: Field>(evals: &[F], rho: F) -> F {
    assert!(!evals.is_empty(), "need at least one evaluation");

    let mut result = F::zero();
    for &v in evals.iter().rev() {
        result = result * rho + v;
    }
    result
}

/// Computes $[1, \rho, \rho^2, \ldots, \rho^{n-1}]$.
///
/// Used by the verifier path to call `PCS::combine` with explicit scalars.
/// The prover path computes the same linear combination implicitly via
/// [`rlc_combine`]'s Horner evaluation.
fn rho_powers<F: Field>(rho: F, n: usize) -> Vec<F> {
    std::iter::successors(Some(F::from_u64(1)), |prev| Some(*prev * rho))
        .take(n)
        .collect()
}

type PointGroup<F> = Vec<(Vec<F>, Vec<ProverClaim<F>>)>;
type VerifierPointGroup<F, C> = Vec<(Vec<F>, Vec<VerifierClaim<F, C>>)>;

/// Groups prover claims by evaluation point using exact equality.
///
/// Returns `(point, claims)` pairs preserving insertion order.
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

/// Groups verifier claims by evaluation point using exact equality.
///
/// Returns `(point, claims)` pairs preserving insertion order.
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
    use jolt_field::Field;
    use jolt_field::Fr;

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
            let expected = p1[i] + rho * p2[i];
            assert_eq!(result[i], expected, "mismatch at index {i}");
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
        let result = rlc_combine_scalars(&evals, rho);
        assert_eq!(result, Fr::from_u64(170));
    }

    #[test]
    fn rlc_combine_scalars_single() {
        let result = rlc_combine_scalars(&[Fr::from_u64(42)], Fr::from_u64(999));
        assert_eq!(result, Fr::from_u64(42));
    }

    #[test]
    fn rlc_combine_with_zero_rho() {
        let p1: Vec<Fr> = vec![Fr::from_u64(5), Fr::from_u64(10)];
        let p2: Vec<Fr> = vec![Fr::from_u64(99), Fr::from_u64(99)];
        let rho = Fr::from_u64(0);

        let result = rlc_combine(&[&p1, &p2], rho);
        assert_eq!(result, p1);
    }

    #[test]
    fn rlc_combine_rho_one_equal_weight() {
        let p1: Vec<Fr> = vec![Fr::from_u64(1), Fr::from_u64(2)];
        let p2: Vec<Fr> = vec![Fr::from_u64(3), Fr::from_u64(4)];
        let p3: Vec<Fr> = vec![Fr::from_u64(5), Fr::from_u64(6)];
        let rho = Fr::from_u64(1);

        let result = rlc_combine(&[&p1, &p2, &p3], rho);
        assert_eq!(result[0], Fr::from_u64(9));
        assert_eq!(result[1], Fr::from_u64(12));
    }

    #[test]
    fn rlc_combine_scalars_consistent_with_rlc_combine() {
        use jolt_poly::Polynomial;
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

        let evals1 = p1.evaluations();
        let evals2 = p2.evaluations();
        let evals3 = p3.evaluations();
        let combined = rlc_combine(&[evals1, evals2, evals3], rho);
        let combined_poly = Polynomial::new(combined);
        let result_via_poly = combined_poly.evaluate(&point);

        let result_via_scalars = rlc_combine_scalars(&[eval1, eval2, eval3], rho);

        assert_eq!(
            result_via_poly, result_via_scalars,
            "rlc_combine then evaluate must equal evaluate then rlc_combine_scalars"
        );
    }

    #[test]
    fn group_prover_claims_same_point() {
        let point = vec![Fr::from_u64(1), Fr::from_u64(2)];
        let claims = vec![
            ProverClaim {
                evaluations: vec![Fr::from_u64(10)],
                point: point.clone(),
                eval: Fr::from_u64(10),
            },
            ProverClaim {
                evaluations: vec![Fr::from_u64(20)],
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
                evaluations: vec![Fr::from_u64(10)],
                point: vec![Fr::from_u64(1)],
                eval: Fr::from_u64(10),
            },
            ProverClaim {
                evaluations: vec![Fr::from_u64(20)],
                point: vec![Fr::from_u64(2)],
                eval: Fr::from_u64(20),
            },
        ];

        let groups = group_prover_claims_by_point(claims);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].1.len(), 1);
        assert_eq!(groups[1].1.len(), 1);
    }
}
