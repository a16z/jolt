//! Dory polynomial commitment scheme implementation

use super::dory_globals::DoryGlobals;
use super::jolt_dory_routines::{JoltG1Routines, JoltG2Routines};
use super::wrappers::{
    jolt_to_ark, ArkDoryProof, ArkFr, ArkG1, ArkGT, ArkworksProverSetup, ArkworksVerifierSetup,
    JoltToDoryTranscript, BN254,
};
use crate::{
    field::JoltField,
    msm::VariableBaseMSM,
    poly::commitment::commitment_scheme::{CommitmentScheme, StreamingCommitmentScheme},
    poly::multilinear_polynomial::MultilinearPolynomial,
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math, small_scalar::SmallScalar},
};
use ark_bn254::{G1Affine, G1Projective};
use ark_ec::CurveGroup;
use ark_ff::Zero;
use dory::primitives::{
    arithmetic::{Group, PairingCurve},
    poly::Polynomial,
};
use rand_core::OsRng;
use rayon::prelude::*;
use std::borrow::Borrow;
use tracing::trace_span;

/// Simplified cached data for Dory streaming commitments
pub struct DoryCachedData {
    /// Pre-computed G1 bases in affine form
    pub g1_bases: Vec<G1Affine>,
}

#[derive(Clone)]
pub struct DoryCommitmentScheme;

impl CommitmentScheme for DoryCommitmentScheme {
    type Field = ark_bn254::Fr;
    type ProverSetup = ArkworksProverSetup;
    type VerifierSetup = ArkworksVerifierSetup;
    type Commitment = ArkGT;
    type Proof = ArkDoryProof;
    type BatchedProof = Vec<ArkDoryProof>;
    type OpeningProofHint = Vec<ArkG1>;

    fn setup_prover(max_num_vars: usize) -> Self::ProverSetup {
        let _span = trace_span!("DoryCommitmentScheme::setup_prover").entered();
        let setup = ArkworksProverSetup::new(&mut OsRng, max_num_vars);

        // Initialize the prepared point cache for faster multi-pairings
        // Skips cache initialization during tests to avoid shared state issues
        #[cfg(not(test))]
        DoryGlobals::init_prepared_cache(&setup.g1_vec, &setup.g2_vec);

        setup
    }

    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup {
        let _span = trace_span!("DoryCommitmentScheme::setup_verifier").entered();
        setup.to_verifier_setup()
    }

    fn commit(
        poly: &MultilinearPolynomial<ark_bn254::Fr>,
        setup: &Self::ProverSetup,
    ) -> (Self::Commitment, Self::OpeningProofHint) {
        let _span = trace_span!("DoryCommitmentScheme::commit").entered();

        let num_cols = DoryGlobals::get_num_columns();
        let num_rows = DoryGlobals::get_max_num_rows();
        let sigma = num_cols.log_2();
        let nu = num_rows.log_2();

        let (tier_2, row_commitments) = <MultilinearPolynomial<ark_bn254::Fr> as Polynomial<
            ArkFr,
        >>::commit::<BN254, JoltG1Routines>(
            poly, nu, sigma, setup
        )
        .expect("commitment should succeed");

        (tier_2, row_commitments)
    }

    fn batch_commit<U>(
        polys: &[U],
        gens: &Self::ProverSetup,
    ) -> Vec<(Self::Commitment, Self::OpeningProofHint)>
    where
        U: std::borrow::Borrow<MultilinearPolynomial<ark_bn254::Fr>> + Sync,
    {
        let _span = trace_span!("DoryCommitmentScheme::batch_commit").entered();

        polys
            .par_iter()
            .map(|poly| Self::commit(poly.borrow(), gens))
            .collect()
    }

    fn prove<ProofTranscript: Transcript>(
        setup: &Self::ProverSetup,
        poly: &MultilinearPolynomial<ark_bn254::Fr>,
        opening_point: &[<ark_bn254::Fr as JoltField>::Challenge],
        row_commitments: Self::OpeningProofHint,
        transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        let _span = trace_span!("DoryCommitmentScheme::prove").entered();

        let num_cols = DoryGlobals::get_num_columns();
        let num_rows = DoryGlobals::get_max_num_rows();
        let sigma = num_cols.log_2();
        let nu = num_rows.log_2();

        // Dory uses the opposite endian-ness as Jolt
        let ark_point: Vec<ArkFr> = opening_point
            .iter()
            .rev()  // Reverse the order for Dory
            .map(|p| {
                let f_val: ark_bn254::Fr = (*p).into();
                jolt_to_ark(&f_val)
            })
            .collect();

        let mut dory_transcript = JoltToDoryTranscript::<ProofTranscript>::new(transcript);

        dory::prove::<ArkFr, BN254, JoltG1Routines, JoltG2Routines, _, _>(
            poly,
            &ark_point,
            row_commitments,
            nu,
            sigma,
            setup,
            &mut dory_transcript,
        )
        .expect("proof generation should succeed")
    }

    fn verify<ProofTranscript: Transcript>(
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut ProofTranscript,
        opening_point: &[<ark_bn254::Fr as JoltField>::Challenge],
        opening: &ark_bn254::Fr,
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError> {
        let _span = trace_span!("DoryCommitmentScheme::verify").entered();

        // Dory uses the opposite endian-ness as Jolt
        let ark_point: Vec<ArkFr> = opening_point
            .iter()
            .rev()  // Reverse the order for Dory
            .map(|p| {
                let f_val: ark_bn254::Fr = (*p).into();
                jolt_to_ark(&f_val)
            })
            .collect();
        let ark_eval: ArkFr = jolt_to_ark(opening);

        let mut dory_transcript = JoltToDoryTranscript::<ProofTranscript>::new(transcript);

        dory::verify::<ArkFr, BN254, JoltG1Routines, JoltG2Routines, _>(
            *commitment,
            ark_eval,
            &ark_point,
            proof,
            setup.clone().into_inner(),
            &mut dory_transcript,
        )
        .map_err(|_| ProofVerifyError::InternalError)?;

        Ok(())
    }

    fn protocol_name() -> &'static [u8] {
        b"Dory"
    }

    /// In Dory, the opening proof hint consists of the Pedersen commitments to the rows
    /// of the polynomial coefficient matrix. In the context of a batch opening proof, we
    /// can homomorphically combine the row commitments for multiple polynomials into the
    /// row commitments for the RLC of those polynomials. This is more efficient than computing
    /// the row commitments for the RLC from scratch.
    #[tracing::instrument(skip_all, name = "DoryCommitmentScheme::combine_hints")]
    fn combine_hints(
        hints: Vec<Self::OpeningProofHint>,
        coeffs: &[Self::Field],
    ) -> Self::OpeningProofHint {
        let num_rows = DoryGlobals::get_max_num_rows();

        let mut rlc_hint = vec![ArkG1(G1Projective::zero()); num_rows];
        for (coeff, mut hint) in coeffs.iter().zip(hints.into_iter()) {
            hint.resize(num_rows, ArkG1(G1Projective::zero()));

            let row_commitments: &mut [G1Projective] = unsafe {
                std::slice::from_raw_parts_mut(hint.as_mut_ptr() as *mut G1Projective, hint.len())
            };

            let rlc_row_commitments: &[G1Projective] = unsafe {
                std::slice::from_raw_parts(rlc_hint.as_ptr() as *const G1Projective, rlc_hint.len())
            };

            let _span = trace_span!("vector_scalar_mul_add_gamma_g1_online");
            let _enter = _span.enter();

            // Scales the row commitments for the current polynomial by
            // its coefficient
            jolt_optimizations::vector_scalar_mul_add_gamma_g1_online(
                row_commitments,
                *coeff,
                rlc_row_commitments,
            );

            let _ = std::mem::replace(&mut rlc_hint, hint);
        }

        rlc_hint
    }

    /// Homomorphically combines multiple commitments using a random linear combination.
    /// Computes: sum_i(coeff_i * commitment_i) for the GT elements.
    #[tracing::instrument(skip_all, name = "DoryCommitmentScheme::combine_commitments")]
    fn combine_commitments<C: Borrow<Self::Commitment>>(
        commitments: &[C],
        coeffs: &[Self::Field],
    ) -> Self::Commitment {
        let _span = trace_span!("DoryCommitmentScheme::combine_commitments").entered();

        // Combine GT elements using parallel RLC
        let commitments_vec: Vec<&ArkGT> = commitments.iter().map(|c| c.borrow()).collect();
        coeffs
            .par_iter()
            .zip(commitments_vec.par_iter())
            .map(|(coeff, commitment)| {
                let ark_coeff = jolt_to_ark(coeff);
                ark_coeff * **commitment
            })
            .reduce(|| ArkGT::identity(), |a, b| a + b)
    }
}

impl StreamingCommitmentScheme for DoryCommitmentScheme {
    type ChunkState = Vec<ArkG1>; // Row commitments for this chunk
    type CachedData = DoryCachedData;

    #[tracing::instrument(skip_all, name = "DoryCommitmentScheme::prepare_cached_data")]
    fn prepare_cached_data(setup: &Self::ProverSetup) -> Self::CachedData {
        let row_len = DoryGlobals::get_num_columns();
        let g1_slice = unsafe {
            std::slice::from_raw_parts(setup.g1_vec.as_ptr() as *const ArkG1, setup.g1_vec.len())
        };

        let g1_bases: Vec<G1Affine> = g1_slice[..row_len]
            .par_iter()
            .map(|g| g.0.into_affine())
            .collect();

        DoryCachedData { g1_bases }
    }

    #[tracing::instrument(skip_all, name = "DoryCommitmentScheme::compute_tier1_commitment")]
    fn compute_tier1_commitment<T: SmallScalar>(
        cached_data: &Self::CachedData,
        chunk: &[T],
    ) -> Self::ChunkState {
        debug_assert_eq!(chunk.len(), DoryGlobals::get_num_columns());

        let row_commitment = ArkG1(
            T::msm(&cached_data.g1_bases[..chunk.len()], chunk).expect("MSM calculation failed."),
        );
        vec![row_commitment]
    }

    #[tracing::instrument(skip_all, name = "DoryCommitmentScheme::compute_tier1_commitment_field")]
    fn compute_tier1_commitment_field(
        cached_data: &Self::CachedData,
        chunk: &[Self::Field],
    ) -> Self::ChunkState {
        debug_assert_eq!(chunk.len(), DoryGlobals::get_num_columns());

        let row_commitment = ArkG1(
            VariableBaseMSM::msm_field_elements(&cached_data.g1_bases[..chunk.len()], chunk)
                .expect("MSM calculation failed."),
        );
        vec![row_commitment]
    }

    #[tracing::instrument(skip_all, name = "DoryCommitmentScheme::compute_tier1_commitment_onehot")]
    fn compute_tier1_commitment_onehot(
        cached_data: &Self::CachedData,
        onehot_k: usize,
        chunk: &[Option<usize>],
    ) -> Self::ChunkState {
        let K = onehot_k;

        let mut indices_per_k: Vec<Vec<usize>> = vec![Vec::new(); K];
        for (col_index, k) in chunk.iter().enumerate() {
            // All the nonzero coefficients are 1, so we simply add
            // the associated base to the result.
            if let Some(k) = k {
                indices_per_k[*k].push(col_index);
            }
        }

        let results =
            jolt_optimizations::batch_g1_additions_multi(&cached_data.g1_bases, &indices_per_k);

        let mut row_commitments = vec![ArkG1(G1Projective::zero()); K];
        for (k, result) in results.into_iter().enumerate() {
            if !indices_per_k[k].is_empty() {
                row_commitments[k] = ArkG1(G1Projective::from(result));
            }
        }
        row_commitments
    }

    #[tracing::instrument(skip_all, name = "DoryCommitmentScheme::compute_tier2_commitment")]
    fn compute_tier2_commitment(
        _cached_data: &Self::CachedData,
        setup: &Self::ProverSetup,
        onehot_k: Option<usize>,
        chunks: &[Self::ChunkState],
    ) -> (Self::Commitment, Self::OpeningProofHint) {

        if let Some(K) = onehot_k {
            // Handle one-hot polynomial case with transpose logic
            let row_len = DoryGlobals::get_num_columns();
            let T = DoryGlobals::get_T();
            let rows_per_k = T / row_len;
            let num_rows = K * T / row_len;

            let mut row_commitments = vec![ArkG1(G1Projective::zero()); num_rows];
            for (chunk_index, commitments) in chunks.iter().enumerate() {
                row_commitments
                    .par_iter_mut()
                    .skip(chunk_index)
                    .step_by(rows_per_k)
                    .zip(commitments.par_iter())
                    .for_each(|(dest, src)| *dest = *src);
            }

            let g2_bases = &setup.g2_vec[..row_commitments.len()];
            let tier_2 = <BN254 as PairingCurve>::multi_pair(&row_commitments, g2_bases);

            (tier_2, row_commitments)
        } else {
            // Regular polynomial case - just flatten the chunks
            let row_commitments: Vec<ArkG1> =
                chunks.iter().flat_map(|chunk| chunk.clone()).collect();

            let g2_bases = &setup.g2_vec[..row_commitments.len()];
            let tier_2 = <BN254 as PairingCurve>::multi_pair(&row_commitments, g2_bases);

            (tier_2, row_commitments)
        }
    }
}
