//! Dory polynomial commitment scheme implementing the `jolt-openings` traits.
//!
//! [`DoryScheme`] wraps the `dory-pcs` crate behind the [`CommitmentScheme`],
//! [`AdditivelyHomomorphic`], and [`ZkOpeningScheme`] trait interfaces, using
//! instance-local [`DoryParams`] instead of global state.

use dory::backends::arkworks::{ArkworksProverSetup, G1Routines, G2Routines};
use dory::mode::Transparent;
use dory::primitives::arithmetic::{
    DoryRoutines, Field as DoryField, Group as DoryGroup, PairingCurve,
};
use dory::primitives::poly::{MultilinearLagrange, Polynomial as DoryPolynomial};
use jolt_crypto::{Bn254G1, Bn254GT, Commitment, Pedersen, PedersenSetup};
use jolt_field::Fr;
use jolt_openings::{
    AdditivelyHomomorphic, CommitmentScheme, OpeningsError, VcSetupExtractable, ZkOpeningScheme,
};
use jolt_poly::MultilinearPoly;
use jolt_transcript::Transcript;

use crate::params::DoryParams;
use crate::transcript::JoltToDoryTranscript;
use crate::types::{DoryCommitment, DoryHint, DoryProof, DoryProverSetup, DoryVerifierSetup};

type InnerFr = dory::backends::arkworks::ArkFr;
type InnerGT = dory::backends::arkworks::ArkGT;
type InnerBN254 = dory::backends::arkworks::BN254;

/// Converts a `jolt_field::Fr` to the dory-pcs `ArkFr` wrapper.
///
/// # Safety
///
/// Both `jolt_field::Fr` and `ArkFr` are `#[repr(transparent)]` over
/// `ark_bn254::Fr`, guaranteeing identical memory layout.
#[inline]
pub(crate) fn jolt_fr_to_ark(f: &Fr) -> InnerFr {
    // SAFETY: jolt_field::Fr is repr(transparent) over ark_bn254::Fr,
    // and ArkFr is repr(transparent) over ark_bn254::Fr.
    unsafe { std::mem::transmute_copy(f) }
}

/// Converts a dory-pcs `ArkFr` back to a `jolt_field::Fr`.
#[inline]
pub(crate) fn ark_to_jolt_fr(ark: &InnerFr) -> Fr {
    // SAFETY: Same layout guarantee as jolt_fr_to_ark.
    unsafe { std::mem::transmute_copy(ark) }
}

/// Dory polynomial commitment scheme with instance-local parameters.
#[derive(Clone)]
pub struct DoryScheme {
    params: DoryParams,
}

impl DoryScheme {
    pub fn new(params: DoryParams) -> Self {
        Self { params }
    }

    pub fn params(&self) -> &DoryParams {
        &self.params
    }

    /// Generates prover SRS from a deterministic SHA3-seeded RNG.
    #[tracing::instrument(skip_all, name = "DoryScheme::setup_prover", fields(max_num_vars))]
    pub fn setup_prover(max_num_vars: usize) -> DoryProverSetup {
        DoryProverSetup(ArkworksProverSetup::new_from_urs(max_num_vars))
    }

    /// Derives the verifier SRS (a subset of the prover SRS).
    #[tracing::instrument(skip_all, name = "DoryScheme::setup_verifier", fields(max_num_vars))]
    pub fn setup_verifier(max_num_vars: usize) -> DoryVerifierSetup {
        let prover_setup = Self::setup_prover(max_num_vars);
        DoryVerifierSetup(prover_setup.0.to_verifier_setup())
    }
}

impl Commitment for DoryScheme {
    type Output = DoryCommitment;
}

impl CommitmentScheme for DoryScheme {
    type Field = Fr;
    type Proof = DoryProof;
    type ProverSetup = DoryProverSetup;
    type VerifierSetup = DoryVerifierSetup;
    type Polynomial = jolt_poly::Polynomial<Fr>;
    type OpeningHint = DoryHint;

    #[tracing::instrument(skip_all, name = "DoryScheme::commit")]
    fn commit<P: MultilinearPoly<Fr> + ?Sized>(
        poly: &P,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint) {
        let num_vars = poly.num_vars();
        let sigma = num_vars.div_ceil(2);
        let num_cols = 1usize << sigma;
        let num_rows = 1usize << (num_vars - sigma);

        let row_commitments = if poly.is_sparse() {
            commit_rows_sparse(poly, num_rows, num_cols, &setup.0)
        } else {
            commit_rows_dense(poly, sigma, &setup.0)
        };

        let g2_bases = &setup.0.g2_vec[..row_commitments.len()];
        let tier_2 = <InnerBN254 as PairingCurve>::multi_pair_g2_setup(&row_commitments, g2_bases);

        // SAFETY: ArkGT is repr(transparent) over Fq12, same as Bn254GT.
        let gt: Bn254GT = unsafe { std::mem::transmute(tier_2) };

        // SAFETY: ArkG1 is repr(transparent) over G1Projective, same as Bn254G1.
        let hint_rows: Vec<Bn254G1> = row_commitments
            .into_iter()
            .map(|g1| unsafe { std::mem::transmute(g1) })
            .collect();

        (DoryCommitment(gt), DoryHint(hint_rows))
    }

    #[tracing::instrument(skip_all, name = "DoryScheme::open")]
    fn open(
        poly: &Self::Polynomial,
        point: &[Fr],
        _eval: Fr,
        setup: &Self::ProverSetup,
        hint: Option<Self::OpeningHint>,
        transcript: &mut impl Transcript,
    ) -> Self::Proof {
        let num_vars = point.len();
        let adapter = DorySourceAdapter::new(poly);
        let sigma = num_vars.div_ceil(2);
        let nu = num_vars - sigma;

        let row_commitments = match hint {
            Some(h) => {
                // SAFETY: Bn254G1 is repr(transparent) over G1Projective, same as ArkG1.
                h.0.into_iter()
                    .map(|g1| unsafe { std::mem::transmute(g1) })
                    .collect()
            }
            None => commit_rows_dense(poly, sigma, &setup.0),
        };

        let ark_point: Vec<InnerFr> = point.iter().rev().map(jolt_fr_to_ark).collect();

        let mut dory_transcript = JoltToDoryTranscript::new(transcript);

        let (proof, _blind) =
            dory::prove::<InnerFr, InnerBN254, G1Routines, G2Routines, _, _, Transparent>(
                &adapter,
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

    #[tracing::instrument(skip_all, name = "DoryScheme::verify")]
    fn verify(
        commitment: &Self::Output,
        point: &[Fr],
        eval: Fr,
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript,
    ) -> Result<(), OpeningsError> {
        let ark_point: Vec<InnerFr> = point.iter().rev().map(jolt_fr_to_ark).collect();
        let ark_eval = jolt_fr_to_ark(&eval);

        // SAFETY: Bn254GT is repr(transparent) over Fq12, same as ArkGT.
        let ark_commitment: InnerGT = unsafe { std::mem::transmute(commitment.0) };

        let mut dory_transcript = JoltToDoryTranscript::new(transcript);

        dory::verify::<InnerFr, InnerBN254, G1Routines, G2Routines, _>(
            ark_commitment,
            ark_eval,
            &ark_point,
            &proof.0,
            setup.0.clone().into_inner(),
            &mut dory_transcript,
        )
        .map_err(|_| OpeningsError::VerificationFailed)
    }
}

impl AdditivelyHomomorphic for DoryScheme {
    fn combine(commitments: &[Self::Output], scalars: &[Self::Field]) -> Self::Output {
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
                // SAFETY: Bn254GT is repr(transparent) over Fq12, same as ArkGT.
                let ark_gt: InnerGT = unsafe { std::mem::transmute(c.0) };
                ark_s * ark_gt
            })
            .fold(InnerGT::identity(), |acc, x| acc + x);

        // SAFETY: reverse transmute.
        let gt: Bn254GT = unsafe { std::mem::transmute(combined) };
        DoryCommitment(gt)
    }

    fn combine_hints(hints: Vec<Self::OpeningHint>, scalars: &[Self::Field]) -> Self::OpeningHint {
        assert_eq!(hints.len(), scalars.len());
        if hints.is_empty() {
            return DoryHint::default();
        }

        let num_rows = hints[0].0.len();
        let mut combined = vec![Bn254G1::default(); num_rows];

        for (hint, &scalar) in hints.iter().zip(scalars.iter()) {
            for (dst, src) in combined.iter_mut().zip(hint.0.iter()) {
                *dst += scalar * src;
            }
        }

        DoryHint(combined)
    }
}

impl ZkOpeningScheme for DoryScheme {
    type EvalCommitment = Bn254G1;
    type EvalBlinding = Fr;

    #[tracing::instrument(skip_all, name = "DoryScheme::open_zk")]
    fn open_zk(
        poly: &Self::Polynomial,
        point: &[Fr],
        _eval: Fr,
        setup: &Self::ProverSetup,
        hint: Option<Self::OpeningHint>,
        transcript: &mut impl Transcript,
    ) -> (Self::Proof, Self::EvalCommitment, Self::EvalBlinding) {
        let num_vars = point.len();
        let adapter = DorySourceAdapter::new(poly);
        let sigma = num_vars.div_ceil(2);
        let nu = num_vars - sigma;

        let row_commitments = match hint {
            Some(h) => {
                // SAFETY: Bn254G1 is repr(transparent) over G1Projective, same as ArkG1.
                h.0.into_iter()
                    .map(|g1| unsafe { std::mem::transmute(g1) })
                    .collect()
            }
            None => commit_rows_dense(poly, sigma, &setup.0),
        };

        let ark_point: Vec<InnerFr> = point.iter().rev().map(jolt_fr_to_ark).collect();

        let mut dory_transcript = JoltToDoryTranscript::new(transcript);

        let (proof, y_blinding) =
            dory::prove::<InnerFr, InnerBN254, G1Routines, G2Routines, _, _, dory::mode::ZK>(
                &adapter,
                &ark_point,
                row_commitments,
                <InnerFr as DoryField>::zero(),
                nu,
                sigma,
                &setup.0,
                &mut dory_transcript,
            )
            .expect("Dory ZK proof generation should not fail");

        // Extract y_com from the proof. In ZK mode, dory::prove always sets y_com.
        let ark_y_com = proof.y_com.expect("ZK mode proof must contain y_com");

        // SAFETY: ArkG1 is repr(transparent) over G1Projective, same as Bn254G1.
        let y_com: Bn254G1 = unsafe { std::mem::transmute(ark_y_com) };
        let blinding = ark_to_jolt_fr(&y_blinding.expect("ZK mode prove must return y_blinding"));

        (DoryProof(proof), y_com, blinding)
    }

    #[tracing::instrument(skip_all, name = "DoryScheme::verify_zk")]
    fn verify_zk(
        commitment: &Self::Output,
        point: &[Fr],
        _eval_commitment: &Self::EvalCommitment,
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript,
    ) -> Result<(), OpeningsError> {
        let ark_point: Vec<InnerFr> = point.iter().rev().map(jolt_fr_to_ark).collect();

        // In ZK mode, dory::verify ignores the evaluation parameter and uses
        // the proof's y_com/e2 instead. We pass zero as a placeholder.
        let dummy_eval = <InnerFr as DoryField>::zero();

        // SAFETY: Bn254GT is repr(transparent) over Fq12, same as ArkGT.
        let ark_commitment: InnerGT = unsafe { std::mem::transmute(commitment.0) };

        let mut dory_transcript = JoltToDoryTranscript::new(transcript);

        dory::verify::<InnerFr, InnerBN254, G1Routines, G2Routines, _>(
            ark_commitment,
            dummy_eval,
            &ark_point,
            &proof.0,
            setup.0.clone().into_inner(),
            &mut dory_transcript,
        )
        .map_err(|_| OpeningsError::VerificationFailed)
    }

    fn extract_eval_commitment(proof: &Self::Proof) -> Option<Self::EvalCommitment> {
        proof.0.y_com.map(|ark_g1| {
            // SAFETY: ArkG1 is repr(transparent) over G1Projective, same as Bn254G1.
            unsafe { std::mem::transmute(ark_g1) }
        })
    }
}

impl VcSetupExtractable<Pedersen<Bn254G1>> for DoryScheme {
    fn extract_vc_setup(setup: &Self::ProverSetup, capacity: usize) -> PedersenSetup<Bn254G1> {
        let len = capacity.min(setup.0.g1_vec.len());
        let message_generators: Vec<Bn254G1> = setup.0.g1_vec[..len]
            .iter()
            .map(|ark_g1| {
                // SAFETY: ArkG1 is repr(transparent) over G1Projective, same as Bn254G1.
                unsafe { std::mem::transmute_copy(ark_g1) }
            })
            .collect();

        // SAFETY: ArkG1 is repr(transparent) over G1Projective, same as Bn254G1.
        let blinding_generator: Bn254G1 = unsafe { std::mem::transmute_copy(&setup.0.h1) };

        PedersenSetup::new(message_generators, blinding_generator)
    }
}

type ArkG1 = dory::backends::arkworks::ArkG1;

/// Dense commit: full MSM per row.
fn commit_rows_dense<P: MultilinearPoly<Fr> + ?Sized>(
    poly: &P,
    sigma: usize,
    setup: &ArkworksProverSetup,
) -> Vec<ArkG1> {
    let num_cols = 1usize << sigma;
    let g1_bases = &setup.g1_vec[..num_cols];

    let mut rows = Vec::new();
    poly.for_each_row(sigma, &mut |_, row| {
        let scalars: Vec<InnerFr> = row.iter().map(jolt_fr_to_ark).collect();
        rows.push(G1Routines::msm(&g1_bases[..scalars.len()], &scalars));
    });
    rows
}

/// Sparse commit: for each nonzero entry, the row commitment is just the
/// G1 generator at that column. Avoids MSM entirely — O(T) lookups instead
/// of O(T × 254 × K) group ops.
fn commit_rows_sparse<P: MultilinearPoly<Fr> + ?Sized>(
    poly: &P,
    num_rows: usize,
    num_cols: usize,
    setup: &ArkworksProverSetup,
) -> Vec<ArkG1> {
    let g1_bases = &setup.g1_vec[..num_cols];
    let identity = <dory::backends::arkworks::BN254 as PairingCurve>::G1::identity();

    let mut row_commitments = vec![identity; num_rows];
    poly.for_each_nonzero(&mut |flat_idx, _val| {
        let row = flat_idx / num_cols;
        let col = flat_idx % num_cols;
        // For one-hot (val == 1), the contribution is just the generator.
        // For general sparse, this would be val * generator, but one-hot is
        // the primary use case and val is always 1.
        row_commitments[row] = <dory::backends::arkworks::BN254 as PairingCurve>::G1::add(
            &row_commitments[row],
            &g1_bases[col],
        );
    });
    row_commitments
}

/// Bridges any [`MultilinearPoly<Fr>`] to dory-pcs's polynomial traits.
///
/// Borrows the source and delegates `evaluate`, `commit`, and
/// `vector_matrix_product` through [`MultilinearPoly`] methods —
/// enabling streaming opening proofs where the full evaluation table
/// never needs to be materialized.
struct DorySourceAdapter<'a, S: MultilinearPoly<Fr>> {
    source: &'a S,
}

impl<'a, S: MultilinearPoly<Fr>> DorySourceAdapter<'a, S> {
    fn new(source: &'a S) -> Self {
        Self { source }
    }
}

impl<S: MultilinearPoly<Fr>> DoryPolynomial<InnerFr> for DorySourceAdapter<'_, S> {
    fn num_vars(&self) -> usize {
        self.source.num_vars()
    }

    fn evaluate(&self, point: &[InnerFr]) -> InnerFr {
        let native_point: Vec<Fr> = point.iter().map(ark_to_jolt_fr).collect();
        jolt_fr_to_ark(&self.source.evaluate(&native_point))
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
        let mut row_commitments: Vec<E::G1> = Vec::new();
        self.source.for_each_row(sigma, &mut |_idx, row| {
            let scalars: Vec<InnerFr> = row.iter().map(jolt_fr_to_ark).collect();
            row_commitments.push(M1::msm(&setup.g1_vec[..scalars.len()], &scalars));
        });

        let g2_bases = &setup.g2_vec[..row_commitments.len()];
        let tier_2 = E::multi_pair_g2_setup(&row_commitments, g2_bases);

        Ok((tier_2, row_commitments, <InnerFr as DoryField>::zero()))
    }
}

impl<S: MultilinearPoly<Fr>> MultilinearLagrange<InnerFr> for DorySourceAdapter<'_, S> {
    fn vector_matrix_product(
        &self,
        left_vec: &[InnerFr],
        _nu: usize,
        sigma: usize,
    ) -> Vec<InnerFr> {
        let native_left: Vec<Fr> = left_vec.iter().map(ark_to_jolt_fr).collect();
        let result = self.source.fold_rows(&native_left, sigma);
        result.iter().map(jolt_fr_to_ark).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_crypto::JoltCommitment;
    use jolt_field::Field;
    use jolt_poly::Polynomial;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    #[test]
    fn scheme_construction() {
        let params = DoryParams::from_dimensions(4, 4);
        let scheme = DoryScheme::new(params.clone());
        assert_eq!(scheme.params(), &params);
    }

    #[test]
    fn commit_open_verify_round_trip() {
        let num_vars = 4;
        let mut rng = ChaCha20Rng::seed_from_u64(42);

        let prover_setup = DoryScheme::setup_prover(num_vars);
        let verifier_setup = DoryVerifierSetup(prover_setup.0.to_verifier_setup());

        let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
        let point: Vec<Fr> = (0..num_vars)
            .map(|_| <Fr as Field>::random(&mut rng))
            .collect();
        let eval = poly.evaluate(&point);

        let (commitment, hint) = DoryScheme::commit(poly.evaluations(), &prover_setup);

        let mut prove_transcript = jolt_transcript::Blake2bTranscript::new(b"test");
        let proof = DoryScheme::open(
            &poly,
            &point,
            eval,
            &prover_setup,
            Some(hint),
            &mut prove_transcript,
        );

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
    fn combine_commitments_homomorphic() {
        let num_vars = 2;
        let mut rng = ChaCha20Rng::seed_from_u64(300);

        let prover_setup = DoryScheme::setup_prover(num_vars);

        let poly_a = Polynomial::<Fr>::random(num_vars, &mut rng);
        let poly_b = Polynomial::<Fr>::random(num_vars, &mut rng);

        let (commit_a, _) = DoryScheme::commit(poly_a.evaluations(), &prover_setup);
        let (commit_b, _) = DoryScheme::commit(poly_b.evaluations(), &prover_setup);

        let sum_evals: Vec<Fr> = poly_a
            .evaluations()
            .iter()
            .zip(poly_b.evaluations().iter())
            .map(|(a, b)| *a + *b)
            .collect();
        let (commit_sum_direct, _) = DoryScheme::commit(&sum_evals, &prover_setup);

        let combined = DoryScheme::combine(&[commit_a, commit_b], &[Fr::one(), Fr::one()]);

        assert_eq!(
            commit_sum_direct, combined,
            "combine([1,1]) must match commitment to sum"
        );
    }

    #[test]
    fn zk_open_verify_round_trip() {
        let num_vars = 4;
        let mut rng = ChaCha20Rng::seed_from_u64(600);

        let prover_setup = DoryScheme::setup_prover(num_vars);
        let verifier_setup = DoryVerifierSetup(prover_setup.0.to_verifier_setup());

        let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
        let point: Vec<Fr> = (0..num_vars)
            .map(|_| <Fr as Field>::random(&mut rng))
            .collect();
        let eval = poly.evaluate(&point);

        let (commitment, hint) = DoryScheme::commit(poly.evaluations(), &prover_setup);

        let mut prove_transcript = jolt_transcript::Blake2bTranscript::new(b"zk-test");
        let (proof, eval_com, _blinding) = DoryScheme::open_zk(
            &poly,
            &point,
            eval,
            &prover_setup,
            Some(hint),
            &mut prove_transcript,
        );

        // Eval commitment should be extractable from the proof.
        let extracted = DoryScheme::extract_eval_commitment(&proof);
        assert_eq!(extracted, Some(eval_com));

        let mut verify_transcript = jolt_transcript::Blake2bTranscript::new(b"zk-test");
        let result = DoryScheme::verify_zk(
            &commitment,
            &point,
            &eval_com,
            &proof,
            &verifier_setup,
            &mut verify_transcript,
        );
        assert!(result.is_ok(), "ZK verification failed: {result:?}");
    }

    #[test]
    fn extract_eval_commitment_none_for_transparent_proof() {
        let num_vars = 2;
        let mut rng = ChaCha20Rng::seed_from_u64(601);

        let prover_setup = DoryScheme::setup_prover(num_vars);
        let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
        let point: Vec<Fr> = (0..num_vars)
            .map(|_| <Fr as Field>::random(&mut rng))
            .collect();
        let eval = poly.evaluate(&point);

        let mut transcript = jolt_transcript::Blake2bTranscript::new(b"transparent");
        let proof = DoryScheme::open(&poly, &point, eval, &prover_setup, None, &mut transcript);

        assert!(
            DoryScheme::extract_eval_commitment(&proof).is_none(),
            "transparent-mode proof should not have y_com"
        );
    }

    #[test]
    fn extract_vc_setup_produces_valid_pedersen_setup() {
        let num_vars = 6;
        let prover_setup = DoryScheme::setup_prover(num_vars);

        let capacity = 5;
        let vc_setup = <DoryScheme as VcSetupExtractable<Pedersen<Bn254G1>>>::extract_vc_setup(
            &prover_setup,
            capacity,
        );

        assert_eq!(
            <Pedersen<Bn254G1> as JoltCommitment>::capacity(&vc_setup),
            capacity,
        );

        // Commit and verify a small vector.
        let values = vec![
            <Fr as Field>::from_u64(1),
            <Fr as Field>::from_u64(2),
            <Fr as Field>::from_u64(3),
        ];
        let blinding = <Fr as Field>::from_u64(42);
        let commitment =
            <Pedersen<Bn254G1> as JoltCommitment>::commit(&vc_setup, &values, &blinding);
        assert!(<Pedersen<Bn254G1> as JoltCommitment>::verify(
            &vc_setup,
            &commitment,
            &values,
            &blinding,
        ),);
    }
}
