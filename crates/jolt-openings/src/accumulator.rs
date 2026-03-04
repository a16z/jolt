//! Opening proof accumulators for batched commitment scheme operations.
//!
//! During a multi-round interactive proof, the prover and verifier accumulate
//! opening claims $(C_i, r_i, v_i)$ — a commitment, evaluation point, and
//! claimed value. At the end of the protocol, all claims are reduced via
//! random linear combination (RLC) and proved/verified in a single batch.

use std::any::Any;
use std::collections::HashMap;

use jolt_field::Field;
use jolt_poly::{DensePolynomial, MultilinearPolynomial};
use jolt_transcript::Transcript;

use crate::error::OpeningsError;
use crate::reduction::{rlc_combine, rlc_combine_scalars};
use crate::traits::HomomorphicCommitmentScheme;

/// A single prover-side opening claim: polynomial evaluations, point, and claimed value.
struct ProverClaim<F: Field> {
    evaluations: Vec<F>,
    point: Vec<F>,
    eval: F,
}

/// Prover-side accumulator that collects opening claims during proof generation.
///
/// Claims are accumulated with [`accumulate`](Self::accumulate), then reduced
/// and proved in batch via [`reduce_and_prove`](Self::reduce_and_prove).
/// Claims sharing the same evaluation point are automatically grouped and
/// combined via RLC, minimizing the number of opening proofs.
pub struct ProverOpeningAccumulator<F: Field> {
    claims: Vec<ProverClaim<F>>,
}

impl<F: Field> Default for ProverOpeningAccumulator<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Field> ProverOpeningAccumulator<F> {
    pub fn new() -> Self {
        Self { claims: Vec::new() }
    }

    /// Accumulates an opening claim for a polynomial at the given point.
    ///
    /// The polynomial's evaluations are cloned into the accumulator so the
    /// caller can continue to use or mutate the polynomial.
    pub fn accumulate(&mut self, poly: &dyn MultilinearPolynomial<F>, point: Vec<F>, eval: F) {
        self.claims.push(ProverClaim {
            evaluations: poly.evaluations().into_owned(),
            point,
            eval,
        });
    }

    pub fn len(&self) -> usize {
        self.claims.len()
    }

    pub fn is_empty(&self) -> bool {
        self.claims.is_empty()
    }

    /// Reduces all accumulated claims via RLC and produces batched opening proofs.
    ///
    /// Claims are grouped by evaluation point. Within each group, polynomials
    /// are combined as:
    ///
    /// $$p_{\text{batch}} = \sum_{i=0}^{k-1} \rho^i \cdot p_i$$
    ///
    /// where $\rho$ is a Fiat-Shamir challenge drawn from the transcript.
    /// One batched proof is produced per distinct evaluation point.
    ///
    /// # Panics
    ///
    /// Panics if the accumulator is empty.
    pub fn reduce_and_prove<PCS, T>(
        self,
        setup: &PCS::ProverSetup,
        transcript: &mut T,
    ) -> Vec<PCS::BatchedProof>
    where
        PCS: HomomorphicCommitmentScheme<Field = F>,
        T: Transcript<Challenge = u128>,
    {
        assert!(!self.claims.is_empty(), "no claims to prove");

        let groups = group_claims_by_point(self.claims);

        let mut proofs = Vec::with_capacity(groups.len());
        for (point, group_claims) in groups {
            let rho = F::from_u128(transcript.challenge());

            let eval_slices: Vec<&[F]> = group_claims
                .iter()
                .map(|c| c.evaluations.as_slice())
                .collect();
            let evals: Vec<F> = group_claims.iter().map(|c| c.eval).collect();

            let combined_evals = rlc_combine(&eval_slices, rho);
            let combined_eval = rlc_combine_scalars(&evals, rho);
            let combined_poly = DensePolynomial::new(combined_evals);

            let proof = PCS::batch_prove(
                &[&combined_poly as &dyn MultilinearPolynomial<F>],
                &[point],
                &[combined_eval],
                setup,
                transcript,
            );
            proofs.push(proof);
        }
        proofs
    }
}

/// A single verifier-side opening claim: type-erased commitment, point, and claimed value.
struct VerifierClaim<F: Field> {
    commitment: Box<dyn Any + Send + Sync>,
    point: Vec<F>,
    eval: F,
}

/// Verifier-side accumulator that collects opening claims during proof verification.
///
/// Mirrors [`ProverOpeningAccumulator`] on the verifier side. Commitments are
/// stored as type-erased `Box<dyn Any>` and downcast during
/// [`reduce_and_verify`](Self::reduce_and_verify).
pub struct VerifierOpeningAccumulator<F: Field> {
    claims: Vec<VerifierClaim<F>>,
}

impl<F: Field> Default for VerifierOpeningAccumulator<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Field> VerifierOpeningAccumulator<F> {
    pub fn new() -> Self {
        Self { claims: Vec::new() }
    }

    /// Accumulates a verifier-side opening claim.
    ///
    /// The commitment type `C` is erased — it will be downcast to
    /// `PCS::Commitment` during [`reduce_and_verify`](Self::reduce_and_verify).
    pub fn accumulate<C: Clone + Send + Sync + 'static>(
        &mut self,
        commitment: C,
        point: Vec<F>,
        eval: F,
    ) {
        self.claims.push(VerifierClaim {
            commitment: Box::new(commitment),
            point,
            eval,
        });
    }

    pub fn len(&self) -> usize {
        self.claims.len()
    }

    pub fn is_empty(&self) -> bool {
        self.claims.is_empty()
    }

    /// Reduces all accumulated claims and verifies batched proofs.
    ///
    /// Claims are grouped by evaluation point, commitments are combined via
    /// RLC using the same Fiat-Shamir challenge as the prover, and each group
    /// is verified against the corresponding batched proof.
    ///
    /// # Errors
    ///
    /// Returns [`OpeningsError::VerificationFailed`] if any batched proof is invalid.
    ///
    /// # Panics
    ///
    /// Panics if a stored commitment cannot be downcast to `PCS::Commitment`,
    /// or if the accumulator is empty.
    pub fn reduce_and_verify<PCS, T>(
        self,
        proofs: &[PCS::BatchedProof],
        setup: &PCS::VerifierSetup,
        transcript: &mut T,
    ) -> Result<(), OpeningsError>
    where
        PCS: HomomorphicCommitmentScheme<Field = F>,
        T: Transcript<Challenge = u128>,
    {
        assert!(!self.claims.is_empty(), "no claims to verify");

        let groups = group_verifier_claims_by_point(self.claims);

        assert_eq!(
            groups.len(),
            proofs.len(),
            "number of proofs ({}) does not match number of point groups ({})",
            proofs.len(),
            groups.len()
        );

        for ((point, group_claims), proof) in groups.into_iter().zip(proofs.iter()) {
            let rho = F::from_u128(transcript.challenge());

            let commitments: Vec<PCS::Commitment> = group_claims
                .iter()
                .map(|c| {
                    c.commitment
                        .downcast_ref::<PCS::Commitment>()
                        .expect("commitment type mismatch during downcast")
                        .clone()
                })
                .collect();

            let evals: Vec<F> = group_claims.iter().map(|c| c.eval).collect();

            let powers = rho_powers::<F>(rho, commitments.len());
            let combined_commitment = PCS::combine_commitments(&commitments, &powers);
            let combined_eval = rlc_combine_scalars(&evals, rho);

            PCS::batch_verify(
                &[combined_commitment],
                &[point],
                &[combined_eval],
                proof,
                setup,
                transcript,
            )?;
        }
        Ok(())
    }
}

/// Computes $[1, \rho, \rho^2, \ldots, \rho^{n-1}]$.
fn rho_powers<F: Field>(rho: F, n: usize) -> Vec<F> {
    let mut powers = Vec::with_capacity(n);
    let mut current = F::from_u64(1);
    for _ in 0..n {
        powers.push(current);
        current *= rho;
    }
    powers
}

/// Groups prover claims by their evaluation point, preserving insertion order per group.
///
/// Uses byte-serialized points as hash keys to avoid requiring `Hash` on `F`.
fn group_claims_by_point<F: Field>(
    claims: Vec<ProverClaim<F>>,
) -> Vec<(Vec<F>, Vec<ProverClaim<F>>)> {
    let mut order: Vec<Vec<F>> = Vec::new();
    let mut map: HashMap<u64, Vec<ProverClaim<F>>> = HashMap::new();

    for claim in claims {
        let key = point_key(&claim.point);
        if !map.contains_key(&key) {
            order.push(claim.point.clone());
        }
        map.entry(key).or_default().push(claim);
    }

    order
        .into_iter()
        .map(|point| {
            let key = point_key(&point);
            let claims = map.remove(&key).unwrap();
            (point, claims)
        })
        .collect()
}

/// Groups verifier claims by their evaluation point, preserving insertion order per group.
fn group_verifier_claims_by_point<F: Field>(
    claims: Vec<VerifierClaim<F>>,
) -> Vec<(Vec<F>, Vec<VerifierClaim<F>>)> {
    let mut order: Vec<Vec<F>> = Vec::new();
    let mut map: HashMap<u64, Vec<VerifierClaim<F>>> = HashMap::new();

    for claim in claims {
        let key = point_key(&claim.point);
        if !map.contains_key(&key) {
            order.push(claim.point.clone());
        }
        map.entry(key).or_default().push(claim);
    }

    order
        .into_iter()
        .map(|point| {
            let key = point_key(&point);
            let claims = map.remove(&key).unwrap();
            (point, claims)
        })
        .collect()
}

/// Hashes a point for use as a hash map key.
///
/// Since `F: Hash`, we use a deterministic hasher to produce a stable key.
fn point_key<F: Field>(point: &[F]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    point.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::MockCommitmentScheme;
    use jolt_field::Field;
    use jolt_field::Fr;
    use jolt_transcript::Blake2bTranscript;
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    type MockPCS = MockCommitmentScheme<Fr>;

    #[test]
    fn prover_accumulate_and_reduce_single_claim() {
        let mut rng = ChaCha20Rng::seed_from_u64(10);
        let poly = DensePolynomial::<Fr>::random(3, &mut rng);
        let point: Vec<Fr> = (0..3).map(|_| Fr::random(&mut rng)).collect();
        let eval = poly.evaluate(&point);

        let mut acc = ProverOpeningAccumulator::new();
        acc.accumulate(&poly, point, eval);
        assert_eq!(acc.len(), 1);

        let mut transcript = Blake2bTranscript::new(b"test-acc");
        let proofs = acc.reduce_and_prove::<MockPCS, _>(&(), &mut transcript);
        assert_eq!(proofs.len(), 1);
    }

    #[test]
    fn prover_groups_claims_at_same_point() {
        let mut rng = ChaCha20Rng::seed_from_u64(20);
        let poly1 = DensePolynomial::<Fr>::random(3, &mut rng);
        let poly2 = DensePolynomial::<Fr>::random(3, &mut rng);
        let point: Vec<Fr> = (0..3).map(|_| Fr::random(&mut rng)).collect();
        let eval1 = poly1.evaluate(&point);
        let eval2 = poly2.evaluate(&point);

        let mut acc = ProverOpeningAccumulator::new();
        acc.accumulate(&poly1, point.clone(), eval1);
        acc.accumulate(&poly2, point, eval2);
        assert_eq!(acc.len(), 2);

        let mut transcript = Blake2bTranscript::new(b"test-acc");
        let proofs = acc.reduce_and_prove::<MockPCS, _>(&(), &mut transcript);
        // Both claims share the same point, so one batched proof
        assert_eq!(proofs.len(), 1);
    }

    #[test]
    fn prover_separate_proofs_for_different_points() {
        let mut rng = ChaCha20Rng::seed_from_u64(30);
        let poly1 = DensePolynomial::<Fr>::random(3, &mut rng);
        let poly2 = DensePolynomial::<Fr>::random(3, &mut rng);
        let point1: Vec<Fr> = (0..3).map(|_| Fr::random(&mut rng)).collect();
        let point2: Vec<Fr> = (0..3).map(|_| Fr::random(&mut rng)).collect();
        let eval1 = poly1.evaluate(&point1);
        let eval2 = poly2.evaluate(&point2);

        let mut acc = ProverOpeningAccumulator::new();
        acc.accumulate(&poly1, point1, eval1);
        acc.accumulate(&poly2, point2, eval2);

        let mut transcript = Blake2bTranscript::new(b"test-acc");
        let proofs = acc.reduce_and_prove::<MockPCS, _>(&(), &mut transcript);
        // Different points, so two batched proofs
        assert_eq!(proofs.len(), 2);
    }

    #[test]
    fn verifier_accumulate_len_and_empty() {
        let mut acc = VerifierOpeningAccumulator::<Fr>::new();
        assert!(acc.is_empty());
        assert_eq!(acc.len(), 0);

        let commitment = MockCommitment {
            fingerprint: 42,
            len: 8,
        };
        acc.accumulate(commitment, vec![Fr::from_u64(1)], Fr::from_u64(2));
        assert!(!acc.is_empty());
        assert_eq!(acc.len(), 1);
    }

    use crate::mock::MockCommitment;

    /// Helper: builds a prover and verifier accumulator with matching claims,
    /// runs reduce_and_prove / reduce_and_verify, and returns the verification result.
    fn prove_and_verify(
        prover_polys: &[(DensePolynomial<Fr>, Vec<Fr>)],
        verifier_evals: Option<&[Fr]>,
    ) -> Result<(), OpeningsError> {
        let mut prover_acc = ProverOpeningAccumulator::new();
        let mut verifier_acc = VerifierOpeningAccumulator::new();

        for (i, (poly, point)) in prover_polys.iter().enumerate() {
            let eval = poly.evaluate(point);
            prover_acc.accumulate(poly, point.clone(), eval);

            let commitment = <MockPCS as CommitmentScheme>::commit(poly, &());
            let v_eval = verifier_evals.map_or(eval, |overrides| overrides[i]);
            verifier_acc.accumulate(commitment, point.clone(), v_eval);
        }

        let mut transcript_p = Blake2bTranscript::new(b"e2e-test");
        let proofs = prover_acc.reduce_and_prove::<MockPCS, _>(&(), &mut transcript_p);

        let mut transcript_v = Blake2bTranscript::new(b"e2e-test");
        verifier_acc.reduce_and_verify::<MockPCS, _>(&proofs, &(), &mut transcript_v)
    }

    #[test]
    fn e2e_single_claim_roundtrip() {
        let mut rng = ChaCha20Rng::seed_from_u64(100);
        let poly = DensePolynomial::<Fr>::random(4, &mut rng);
        let point: Vec<Fr> = (0..4).map(|_| Fr::random(&mut rng)).collect();

        prove_and_verify(&[(poly, point)], None).expect("single claim e2e should verify");
    }

    #[test]
    fn e2e_multiple_claims_shared_and_distinct_points() {
        let mut rng = ChaCha20Rng::seed_from_u64(200);
        let num_vars = 3;

        let poly_a = DensePolynomial::<Fr>::random(num_vars, &mut rng);
        let poly_b = DensePolynomial::<Fr>::random(num_vars, &mut rng);
        let poly_c = DensePolynomial::<Fr>::random(num_vars, &mut rng);
        let poly_d = DensePolynomial::<Fr>::random(num_vars, &mut rng);

        let point1: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let point2: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();

        // poly_a and poly_b share point1; poly_c and poly_d share point2
        let claims = vec![
            (poly_a, point1.clone()),
            (poly_b, point1),
            (poly_c, point2.clone()),
            (poly_d, point2),
        ];

        prove_and_verify(&claims, None).expect("multi-claim e2e should verify");
    }

    #[test]
    fn e2e_rejects_tampered_evaluation() {
        let mut rng = ChaCha20Rng::seed_from_u64(300);
        let num_vars = 3;

        let poly_a = DensePolynomial::<Fr>::random(num_vars, &mut rng);
        let poly_b = DensePolynomial::<Fr>::random(num_vars, &mut rng);
        let poly_c = DensePolynomial::<Fr>::random(num_vars, &mut rng);

        let point1: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let point2: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();

        let eval_a = poly_a.evaluate(&point1);
        let eval_b = poly_b.evaluate(&point1);
        let eval_c = poly_c.evaluate(&point2);

        // Tamper with the second evaluation
        let tampered_evals = [eval_a, eval_b + Fr::from_u64(1), eval_c];

        let claims = vec![(poly_a, point1.clone()), (poly_b, point1), (poly_c, point2)];

        let result = prove_and_verify(&claims, Some(&tampered_evals));
        assert!(
            result.is_err(),
            "tampered evaluation should cause rejection"
        );
    }

    use crate::traits::CommitmentScheme;
}
