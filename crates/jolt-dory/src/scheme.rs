//! Dory polynomial commitment scheme implementing the `jolt-openings` traits.
//!
//! [`DoryScheme`] wraps the `dory-pcs` crate behind the [`CommitmentScheme`] and
//! [`AdditivelyHomomorphic`] trait interfaces, using instance-local
//! [`DoryParams`] instead of global state.

use dory::backends::arkworks::{ArkworksProverSetup, G1Routines, G2Routines};
use dory::mode::Transparent;
use dory::primitives::arithmetic::{
    DoryRoutines, Field as DoryField, Group as DoryGroup, PairingCurve,
};
use dory::primitives::poly::{MultilinearLagrange, Polynomial as DoryPolynomial};
use jolt_crypto::{Bn254G1, Bn254GT, Commitment};
use jolt_field::Fr;
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme, OpeningsError};
use jolt_poly::EvaluationSource;
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
    pub fn setup_prover(max_num_vars: usize) -> DoryProverSetup {
        DoryProverSetup(ArkworksProverSetup::new_from_urs(max_num_vars))
    }

    /// Derives the verifier SRS (a subset of the prover SRS).
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

    fn commit(evaluations: &[Fr], setup: &Self::ProverSetup) -> (Self::Output, Self::OpeningHint) {
        let num_vars = evaluations.len().trailing_zeros() as usize;
        let sigma = num_vars.div_ceil(2);

        let row_commitments = commit_rows_msm(evaluations, sigma, &setup.0);

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
            None => commit_rows_msm(poly.evaluations(), sigma, &setup.0),
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

/// Bridges any [`EvaluationSource<Fr>`] to dory-pcs's polynomial traits.
///
/// Replaces the old `DoryPolyAdapter` which cloned evaluations eagerly.
/// `DorySourceAdapter` borrows the source and delegates `evaluate`,
/// `commit`, and `vector_matrix_product` through [`EvaluationSource`]
/// methods — enabling streaming opening proofs where the full evaluation
/// table never needs to be materialized.
struct DorySourceAdapter<'a, S: EvaluationSource<Fr>> {
    source: &'a S,
}

impl<'a, S: EvaluationSource<Fr>> DorySourceAdapter<'a, S> {
    fn new(source: &'a S) -> Self {
        Self { source }
    }
}

impl<S: EvaluationSource<Fr>> DoryPolynomial<InnerFr> for DorySourceAdapter<'_, S> {
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

impl<S: EvaluationSource<Fr>> MultilinearLagrange<InnerFr> for DorySourceAdapter<'_, S> {
    fn vector_matrix_product(&self, left_vec: &[InnerFr], _nu: usize, sigma: usize) -> Vec<InnerFr> {
        let native_left: Vec<Fr> = left_vec.iter().map(ark_to_jolt_fr).collect();
        let result = self.source.fold_rows(&native_left, sigma);
        result.iter().map(jolt_fr_to_ark).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
}
