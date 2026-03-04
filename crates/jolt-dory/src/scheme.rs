//! Dory polynomial commitment scheme implementing the `jolt-openings` traits.
//!
//! [`DoryScheme`] wraps the `dory-pcs` crate behind the [`CommitmentScheme`] and
//! [`HomomorphicCommitmentScheme`] trait interfaces, using instance-local
//! [`DoryParams`] instead of global state.

use ark_bn254::Fr;
use dory::backends::arkworks::{ArkworksProverSetup, G1Routines, G2Routines};
use dory::mode::Transparent;
use dory::primitives::arithmetic::{
    DoryRoutines, Field as DoryField, Group as DoryGroup, PairingCurve,
};
use dory::primitives::poly::{MultilinearLagrange, Polynomial as DoryPolynomial};
use jolt_openings::{CommitmentScheme, HomomorphicCommitmentScheme, OpeningsError};
use jolt_poly::MultilinearPolynomial;
use jolt_transcript::Transcript;

use crate::params::DoryParams;
use crate::transcript::JoltToDoryTranscript;
use crate::types::{
    ark_to_jolt_fr, jolt_fr_to_ark, DoryBatchedProof, DoryCommitment, DoryProof, DoryProverSetup,
    DoryVerifierSetup,
};

type InnerFr = dory::backends::arkworks::ArkFr;
type InnerGT = dory::backends::arkworks::ArkGT;
type InnerBN254 = dory::backends::arkworks::BN254;

/// Dory polynomial commitment scheme with instance-local parameters.
///
/// Wraps the `dory-pcs` arkworks backend behind the `jolt-openings` trait
/// hierarchy. Each `DoryScheme` instance carries its own [`DoryParams`],
/// eliminating the need for global mutable state.
///
/// # Usage
///
/// ```ignore
/// use jolt_dory::{DoryScheme, DoryParams};
/// use jolt_openings::CommitmentScheme;
///
/// let params = DoryParams::from_dimensions(4, 4);
/// let scheme = DoryScheme::new(params);
/// let prover_setup = DoryScheme::setup_prover(4);
/// ```
#[derive(Clone)]
pub struct DoryScheme {
    params: DoryParams,
}

impl DoryScheme {
    /// Creates a new Dory scheme instance with the given parameters.
    pub fn new(params: DoryParams) -> Self {
        Self { params }
    }

    /// Returns a reference to the scheme's parameters.
    pub fn params(&self) -> &DoryParams {
        &self.params
    }
}

impl CommitmentScheme for DoryScheme {
    type Field = Fr;
    type Commitment = DoryCommitment;
    type Proof = DoryProof;
    type ProverSetup = DoryProverSetup;
    type VerifierSetup = DoryVerifierSetup;

    fn protocol_name() -> &'static str {
        "Dory"
    }

    /// Generates prover SRS (G1/G2 generator vectors) from a deterministic
    /// SHA3-seeded RNG. `max_num_vars` is the maximum polynomial size
    /// ($\log_2$ of evaluations) that can be committed.
    fn setup_prover(max_num_vars: usize) -> Self::ProverSetup {
        DoryProverSetup(ArkworksProverSetup::new_from_urs(max_num_vars))
    }

    /// Derives the verifier SRS (a subset of the prover SRS).
    fn setup_verifier(max_num_vars: usize) -> Self::VerifierSetup {
        let prover_setup = Self::setup_prover(max_num_vars);
        DoryVerifierSetup(prover_setup.0.to_verifier_setup())
    }

    /// Commits to a multilinear polynomial via two-tier Dory:
    /// tier-1 computes per-row MSMs against G1 generators, then tier-2
    /// pairs those row commitments with G2 generators into a single GT element.
    fn commit(
        poly: &impl MultilinearPolynomial<Fr>,
        setup: &Self::ProverSetup,
    ) -> Self::Commitment {
        let num_vars = poly.num_vars();
        let sigma = num_vars.div_ceil(2);

        let evals = poly.evaluations();
        let row_commitments = commit_rows_msm(&evals, sigma, &setup.0);

        let g2_bases = &setup.0.g2_vec[..row_commitments.len()];
        let tier_2 = <InnerBN254 as PairingCurve>::multi_pair_g2_setup(&row_commitments, g2_bases);

        DoryCommitment(tier_2)
    }

    /// Generates an opening proof for `poly` at `point`.
    ///
    /// The evaluation point is reversed before passing to `dory-pcs` because
    /// Jolt uses MSB-first variable ordering while `dory-pcs` uses LSB-first.
    fn prove(
        poly: &impl MultilinearPolynomial<Fr>,
        point: &[Fr],
        _eval: Fr,
        setup: &Self::ProverSetup,
        transcript: &mut impl Transcript,
    ) -> Self::Proof {
        let wrapped = DoryPolyAdapter::from_poly(poly);
        let num_vars = poly.num_vars();
        let sigma = num_vars.div_ceil(2);
        let nu = num_vars - sigma;

        let row_commitments = commit_rows_msm(&wrapped.evals, sigma, &setup.0);

        // dory-pcs expects the point in reversed order relative to Jolt convention
        let ark_point: Vec<InnerFr> = point.iter().rev().map(jolt_fr_to_ark).collect();

        let mut dory_transcript = JoltToDoryTranscript::new(transcript);

        let (proof, _blind) =
            dory::prove::<InnerFr, InnerBN254, G1Routines, G2Routines, _, _, Transparent>(
                &wrapped,
                &ark_point,
                row_commitments,
                <InnerFr as DoryField>::zero(),
                nu,
                sigma,
                &setup.0,
                &mut dory_transcript,
            )
            .expect("Dory proof generation should not fail");

        DoryProof(proof)
    }

    /// Verifies an opening proof against a commitment, point, and claimed evaluation.
    fn verify(
        commitment: &Self::Commitment,
        point: &[Fr],
        eval: Fr,
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript,
    ) -> Result<(), OpeningsError> {
        let ark_point: Vec<InnerFr> = point.iter().rev().map(jolt_fr_to_ark).collect();
        let ark_eval = jolt_fr_to_ark(&eval);

        let mut dory_transcript = JoltToDoryTranscript::new(transcript);

        dory::verify::<InnerFr, InnerBN254, G1Routines, G2Routines, _>(
            commitment.0,
            ark_eval,
            &ark_point,
            &proof.0,
            setup.0.clone().into_inner(),
            &mut dory_transcript,
        )
        .map_err(|_| OpeningsError::VerificationFailed)
    }
}

impl HomomorphicCommitmentScheme for DoryScheme {
    type BatchedProof = DoryBatchedProof;

    /// Computes $\sum_i s_i \cdot C_i$ in GT, used for RLC batching.
    fn combine_commitments(
        commitments: &[Self::Commitment],
        scalars: &[Self::Field],
    ) -> Self::Commitment {
        assert_eq!(
            commitments.len(),
            scalars.len(),
            "commitments and scalars must have the same length"
        );

        let combined = commitments
            .iter()
            .zip(scalars.iter())
            .map(|(c, s)| {
                let ark_s = jolt_fr_to_ark(s);
                ark_s * c.0
            })
            .fold(InnerGT::identity(), |acc, x| acc + x);

        DoryCommitment(combined)
    }

    /// Produces one independent opening proof per polynomial.
    fn batch_prove(
        polynomials: &[&dyn MultilinearPolynomial<Fr>],
        points: &[Vec<Fr>],
        evals: &[Fr],
        setup: &Self::ProverSetup,
        transcript: &mut impl Transcript,
    ) -> Self::BatchedProof {
        assert_eq!(polynomials.len(), points.len());
        assert_eq!(polynomials.len(), evals.len());

        let proofs: Vec<_> = polynomials
            .iter()
            .zip(points.iter())
            .zip(evals.iter())
            .map(|((poly, point), eval)| prove_inner(*poly, point, *eval, setup, transcript))
            .collect();

        DoryBatchedProof(proofs)
    }

    /// Verifies each proof in the batch independently.
    fn batch_verify(
        commitments: &[Self::Commitment],
        points: &[Vec<Fr>],
        evals: &[Fr],
        proof: &Self::BatchedProof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript,
    ) -> Result<(), OpeningsError> {
        assert_eq!(commitments.len(), points.len());
        assert_eq!(commitments.len(), evals.len());
        assert_eq!(commitments.len(), proof.0.len());

        for (((c, p), e), pf) in commitments
            .iter()
            .zip(points.iter())
            .zip(evals.iter())
            .zip(proof.0.iter())
        {
            DoryScheme::verify(c, p, *e, pf, setup, transcript)?;
        }

        Ok(())
    }
}

/// Internal prove implementation that works with `&dyn MultilinearPolynomial<Fr>`,
/// needed by `batch_prove` which receives trait objects.
fn prove_inner(
    poly: &dyn MultilinearPolynomial<Fr>,
    point: &[Fr],
    eval: Fr,
    setup: &DoryProverSetup,
    transcript: &mut impl Transcript,
) -> DoryProof {
    let wrapped = DoryPolyAdapter::from_dyn(poly);
    let num_vars = poly.num_vars();
    let sigma = num_vars.div_ceil(2);
    let nu = num_vars - sigma;

    let row_commitments = commit_rows_msm(&wrapped.evals, sigma, &setup.0);

    let ark_point: Vec<InnerFr> = point.iter().rev().map(jolt_fr_to_ark).collect();
    let _ = eval;

    let mut dory_transcript = JoltToDoryTranscript::new(transcript);

    let (proof, _blind) =
        dory::prove::<InnerFr, InnerBN254, G1Routines, G2Routines, _, _, Transparent>(
            &wrapped,
            &ark_point,
            row_commitments,
            <InnerFr as DoryField>::zero(),
            nu,
            sigma,
            &setup.0,
            &mut dory_transcript,
        )
        .expect("Dory proof generation should not fail");

    DoryProof(proof)
}

/// Computes row-level Pedersen commitments (tier-1) for a flattened evaluation vector.
///
/// Chunks the evaluations into rows of `2^sigma` elements each, then computes
/// an MSM against the G1 generators for each row.
fn commit_rows_msm(
    evals: &[Fr],
    sigma: usize,
    setup: &ArkworksProverSetup,
) -> Vec<dory::backends::arkworks::ArkG1> {
    let num_cols = 1usize << sigma;
    let g1_bases = &setup.g1_vec[..num_cols];

    evals
        .chunks(num_cols)
        .map(|row| {
            let scalars: Vec<InnerFr> = row.iter().map(jolt_fr_to_ark).collect();
            G1Routines::msm(&g1_bases[..scalars.len()], &scalars)
        })
        .collect()
}

/// Owned adapter providing dory-pcs trait implementations for polynomial
/// evaluations without violating the orphan rule.
///
/// Materializes the evaluations as `Vec<Fr>` so we have owned data that
/// dory-pcs can operate on.
struct DoryPolyAdapter {
    evals: Vec<Fr>,
    num_vars: usize,
}

impl DoryPolyAdapter {
    fn from_poly(poly: &impl MultilinearPolynomial<Fr>) -> Self {
        Self {
            evals: poly.evaluations().to_vec(),
            num_vars: poly.num_vars(),
        }
    }

    fn from_dyn(poly: &dyn MultilinearPolynomial<Fr>) -> Self {
        Self {
            evals: poly.evaluations().to_vec(),
            num_vars: poly.num_vars(),
        }
    }
}

impl DoryPolynomial<InnerFr> for DoryPolyAdapter {
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn evaluate(&self, point: &[InnerFr]) -> InnerFr {
        let native_point: Vec<Fr> = point.iter().map(ark_to_jolt_fr).collect();
        let dense = jolt_poly::DensePolynomial::new(self.evals.clone());
        let result = MultilinearPolynomial::evaluate(&dense, &native_point);
        jolt_fr_to_ark(&result)
    }

    fn commit<E, Mo, M1>(
        &self,
        _nu: usize,
        sigma: usize,
        setup: &dory::setup::ProverSetup<E>,
    ) -> Result<(E::GT, Vec<E::G1>, InnerFr), dory::error::DoryError>
    where
        E: PairingCurve,
        Mo: dory::mode::Mode,
        M1: DoryRoutines<E::G1>,
        E::G1: DoryGroup<Scalar = InnerFr>,
    {
        let num_cols = 1usize << sigma;

        let row_commitments: Vec<E::G1> = self
            .evals
            .chunks(num_cols)
            .map(|row| {
                let scalars: Vec<InnerFr> = row.iter().map(jolt_fr_to_ark).collect();
                M1::msm(&setup.g1_vec[..scalars.len()], &scalars)
            })
            .collect();

        let g2_bases = &setup.g2_vec[..row_commitments.len()];
        let tier_2 = E::multi_pair_g2_setup(&row_commitments, g2_bases);

        Ok((tier_2, row_commitments, <InnerFr as DoryField>::zero()))
    }
}

impl MultilinearLagrange<InnerFr> for DoryPolyAdapter {
    fn vector_matrix_product(&self, left_vec: &[InnerFr], nu: usize, sigma: usize) -> Vec<InnerFr> {
        use ark_ff::Zero as _;

        let num_cols = 1usize << sigma;
        let num_rows = 1usize << nu;
        let left_native: Vec<Fr> = left_vec.iter().map(ark_to_jolt_fr).collect();

        let mut result = vec![Fr::zero(); num_cols];
        for (col_idx, dest) in result.iter_mut().enumerate() {
            let mut sum = Fr::zero();
            for (left_val, row_idx) in left_native.iter().zip(0..num_rows) {
                let coeff_idx = row_idx * num_cols + col_idx;
                if coeff_idx < self.evals.len() {
                    sum += self.evals[coeff_idx] * *left_val;
                }
            }
            *dest = sum;
        }

        result.iter().map(jolt_fr_to_ark).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::One;
    use jolt_field::Field;
    use jolt_poly::DensePolynomial;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    #[test]
    fn scheme_construction() {
        let params = DoryParams::from_dimensions(4, 4);
        let scheme = DoryScheme::new(params.clone());
        assert_eq!(scheme.params(), &params);
    }

    #[test]
    fn protocol_name() {
        assert_eq!(DoryScheme::protocol_name(), "Dory");
    }

    #[test]
    fn commit_prove_verify_round_trip() {
        let num_vars = 4;
        let mut rng = ChaCha20Rng::seed_from_u64(42);

        let prover_setup = DoryScheme::setup_prover(num_vars);
        let verifier_setup = DoryVerifierSetup(prover_setup.0.to_verifier_setup());

        let poly = DensePolynomial::<Fr>::random(num_vars, &mut rng);
        let point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let eval = MultilinearPolynomial::evaluate(&poly, &point);

        let commitment = DoryScheme::commit(&poly, &prover_setup);

        let mut prove_transcript = jolt_transcript::Blake2bTranscript::new(b"test");
        let proof = DoryScheme::prove(&poly, &point, eval, &prover_setup, &mut prove_transcript);

        let mut verify_transcript = jolt_transcript::Blake2bTranscript::new(b"test");
        let result = DoryScheme::verify(
            &commitment,
            &point,
            eval,
            &proof,
            &verifier_setup,
            &mut verify_transcript,
        );
        assert!(result.is_ok(), "Verification failed: {result:?}");
    }

    #[test]
    fn batch_prove_verify_round_trip() {
        let num_vars = 3;
        let num_polys = 3;
        let mut rng = ChaCha20Rng::seed_from_u64(100);

        let prover_setup = DoryScheme::setup_prover(num_vars);
        let verifier_setup = DoryVerifierSetup(prover_setup.0.to_verifier_setup());

        let polys: Vec<DensePolynomial<Fr>> = (0..num_polys)
            .map(|_| DensePolynomial::random(num_vars, &mut rng))
            .collect();

        let points: Vec<Vec<Fr>> = (0..num_polys)
            .map(|_| (0..num_vars).map(|_| Fr::random(&mut rng)).collect())
            .collect();

        let evals: Vec<Fr> = polys
            .iter()
            .zip(points.iter())
            .map(|(p, pt)| MultilinearPolynomial::evaluate(p, pt))
            .collect();

        let commitments: Vec<DoryCommitment> = polys
            .iter()
            .map(|p| DoryScheme::commit(p, &prover_setup))
            .collect();

        let poly_refs: Vec<&dyn MultilinearPolynomial<Fr>> = polys
            .iter()
            .map(|p| p as &dyn MultilinearPolynomial<Fr>)
            .collect();

        let mut prove_transcript = jolt_transcript::Blake2bTranscript::new(b"batch-test");
        let batched_proof = DoryScheme::batch_prove(
            &poly_refs,
            &points,
            &evals,
            &prover_setup,
            &mut prove_transcript,
        );

        assert_eq!(batched_proof.0.len(), num_polys);

        let mut verify_transcript = jolt_transcript::Blake2bTranscript::new(b"batch-test");
        let result = DoryScheme::batch_verify(
            &commitments,
            &points,
            &evals,
            &batched_proof,
            &verifier_setup,
            &mut verify_transcript,
        );
        assert!(result.is_ok(), "Batch verification failed: {result:?}");
    }

    #[test]
    fn batch_verify_rejects_wrong_eval() {
        let num_vars = 2;
        let num_polys = 2;
        let mut rng = ChaCha20Rng::seed_from_u64(200);

        let prover_setup = DoryScheme::setup_prover(num_vars);
        let verifier_setup = DoryVerifierSetup(prover_setup.0.to_verifier_setup());

        let polys: Vec<DensePolynomial<Fr>> = (0..num_polys)
            .map(|_| DensePolynomial::random(num_vars, &mut rng))
            .collect();

        let points: Vec<Vec<Fr>> = (0..num_polys)
            .map(|_| (0..num_vars).map(|_| Fr::random(&mut rng)).collect())
            .collect();

        let evals: Vec<Fr> = polys
            .iter()
            .zip(points.iter())
            .map(|(p, pt)| MultilinearPolynomial::evaluate(p, pt))
            .collect();

        let commitments: Vec<DoryCommitment> = polys
            .iter()
            .map(|p| DoryScheme::commit(p, &prover_setup))
            .collect();

        let poly_refs: Vec<&dyn MultilinearPolynomial<Fr>> = polys
            .iter()
            .map(|p| p as &dyn MultilinearPolynomial<Fr>)
            .collect();

        let mut prove_transcript = jolt_transcript::Blake2bTranscript::new(b"batch-bad");
        let batched_proof = DoryScheme::batch_prove(
            &poly_refs,
            &points,
            &evals,
            &prover_setup,
            &mut prove_transcript,
        );

        // Tamper with the second evaluation
        let mut bad_evals = evals;
        bad_evals[1] += Fr::one();

        let mut verify_transcript = jolt_transcript::Blake2bTranscript::new(b"batch-bad");
        let result = DoryScheme::batch_verify(
            &commitments,
            &points,
            &bad_evals,
            &batched_proof,
            &verifier_setup,
            &mut verify_transcript,
        );
        assert!(
            result.is_err(),
            "Batch verification should reject tampered eval"
        );
    }

    #[test]
    fn combine_commitments_homomorphic() {
        let num_vars = 2;
        let mut rng = ChaCha20Rng::seed_from_u64(300);

        let prover_setup = DoryScheme::setup_prover(num_vars);

        let poly_a = DensePolynomial::<Fr>::random(num_vars, &mut rng);
        let poly_b = DensePolynomial::<Fr>::random(num_vars, &mut rng);

        let commit_a = DoryScheme::commit(&poly_a, &prover_setup);
        let commit_b = DoryScheme::commit(&poly_b, &prover_setup);

        // Commit to sum polynomial directly
        let sum_evals: Vec<Fr> = poly_a
            .evaluations()
            .iter()
            .zip(poly_b.evaluations().iter())
            .map(|(a, b)| *a + *b)
            .collect();
        let poly_sum = DensePolynomial::new(sum_evals);
        let commit_sum_direct = DoryScheme::commit(&poly_sum, &prover_setup);

        // Combine commitments with scalars [1, 1]
        let combined =
            DoryScheme::combine_commitments(&[commit_a, commit_b], &[Fr::one(), Fr::one()]);

        assert_eq!(
            commit_sum_direct, combined,
            "combine_commitments([1,1]) must match commitment to sum"
        );
    }
}
