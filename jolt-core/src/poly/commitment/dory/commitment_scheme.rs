//! Dory polynomial commitment scheme implementation

use super::dory_globals::{DoryGlobals, DoryLayout};
use super::jolt_dory_routines::{JoltG1Routines, JoltG2Routines};
use super::wrappers::{
    jolt_to_ark, ArkDoryProof, ArkFr, ArkG1, ArkGT, ArkworksProverSetup, ArkworksVerifierSetup,
    JoltToDoryTranscript, BN254,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use crate::{
    field::JoltField,
    poly::commitment::commitment_scheme::{CommitmentScheme, StreamingCommitmentScheme},
    poly::multilinear_polynomial::MultilinearPolynomial,
    poly::opening_proof::BatchPolynomialSource,
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
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use rayon::prelude::*;
use sha3::{Digest, Sha3_256};
use tracing::trace_span;

#[derive(Clone, Default)]
pub struct DoryCommitmentScheme {
    pub layout: DoryLayout,
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct DoryBatchedProof {
    pub proof: ArkDoryProof,
    pub layout: DoryLayout,
}

/// Split `total_vars` into balanced `(sigma, nu)` where sigma = ceil(total_vars / 2)
/// and nu = total_vars - sigma. sigma is the number of column variables,
/// nu is the number of row variables.
#[inline]
pub fn balanced_sigma_nu(total_vars: usize) -> (usize, usize) {
    let sigma = total_vars.div_ceil(2);
    let nu = total_vars - sigma;
    (sigma, nu)
}

impl CommitmentScheme for DoryCommitmentScheme {
    type Field = ark_bn254::Fr;
    type Config = DoryLayout;
    type ProverSetup = ArkworksProverSetup;
    type VerifierSetup = ArkworksVerifierSetup;
    type Commitment = ArkGT;
    type Proof = ArkDoryProof;
    type BatchedProof = DoryBatchedProof;
    type OpeningProofHint = Vec<ArkG1>;

    fn setup_prover(max_num_vars: usize) -> Self::ProverSetup {
        let _span = trace_span!("DoryCommitmentScheme::setup_prover").entered();
        let mut hasher = Sha3_256::new();
        hasher.update(b"Jolt Dory URS seed");
        let hash_result = hasher.finalize();
        let seed: [u8; 32] = hash_result.into();
        let mut rng = ChaCha20Rng::from_seed(seed);
        let setup = ArkworksProverSetup::new_from_urs(&mut rng, max_num_vars);

        // The prepared-point cache in dory-pcs is global and can only be initialized once.
        // In unit tests, multiple setups with different sizes are created, so initializing the
        // cache with a small setup can break later tests that need more generators.
        // We therefore disable cache initialization in `cfg(test)` builds.
        #[cfg(not(test))]
        DoryGlobals::init_prepared_cache(&setup.g1_vec, &setup.g2_vec);

        setup
    }

    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup {
        let _span = trace_span!("DoryCommitmentScheme::setup_verifier").entered();
        setup.to_verifier_setup()
    }

    fn from_proof(proof: &DoryBatchedProof) -> Self {
        Self {
            layout: proof.layout,
        }
    }

    fn config(&self) -> &DoryLayout {
        &self.layout
    }

    fn commit(
        &self,
        poly: &MultilinearPolynomial<ark_bn254::Fr>,
        setup: &Self::ProverSetup,
    ) -> (Self::Commitment, Self::OpeningProofHint) {
        let _span = trace_span!("DoryCommitmentScheme::commit").entered();

        let total_vars = poly.len().log_2();
        let (sigma, nu) = balanced_sigma_nu(total_vars);

        let (tier_2, row_commitments) = <MultilinearPolynomial<ark_bn254::Fr> as Polynomial<
            ArkFr,
        >>::commit::<BN254, JoltG1Routines>(
            poly, nu, sigma, setup
        )
        .expect("commitment should succeed");

        (tier_2, row_commitments)
    }

    fn batch_commit<U>(
        &self,
        polys: &[U],
        gens: &Self::ProverSetup,
    ) -> Vec<(Self::Commitment, Self::OpeningProofHint)>
    where
        U: std::borrow::Borrow<MultilinearPolynomial<ark_bn254::Fr>> + Sync,
    {
        let _span = trace_span!("DoryCommitmentScheme::batch_commit").entered();

        polys
            .par_iter()
            .map(|poly| self.commit(poly.borrow(), gens))
            .collect()
    }

    fn prove<ProofTranscript: Transcript>(
        &self,
        setup: &Self::ProverSetup,
        poly: &MultilinearPolynomial<ark_bn254::Fr>,
        opening_point: &[<ark_bn254::Fr as JoltField>::Challenge],
        hint: Option<Self::OpeningProofHint>,
        transcript: &mut ProofTranscript,
        _commitment: &Self::Commitment,
    ) -> Self::Proof {
        let _span = trace_span!("DoryCommitmentScheme::prove").entered();

        let row_commitments = hint.unwrap_or_else(|| {
            let (_commitment, row_commitments) = self.commit(poly, setup);
            row_commitments
        });

        let total_vars = poly.len().log_2();
        let (sigma, nu) = balanced_sigma_nu(total_vars);

        let reordered_point =
            reorder_opening_point_for_layout::<ark_bn254::Fr>(self.layout, opening_point);

        // Dory uses the opposite endian-ness as Jolt
        let ark_point: Vec<ArkFr> = reordered_point
            .iter()
            .rev()
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
        &self,
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut ProofTranscript,
        opening_point: &[<ark_bn254::Fr as JoltField>::Challenge],
        opening: &ark_bn254::Fr,
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError> {
        let _span = trace_span!("DoryCommitmentScheme::verify").entered();

        let reordered_point =
            reorder_opening_point_for_layout::<ark_bn254::Fr>(self.layout, opening_point);

        // Dory uses the opposite endian-ness as Jolt
        let ark_point: Vec<ArkFr> = reordered_point
            .iter()
            .rev()
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

    fn batch_prove<ProofTranscript: Transcript, S: BatchPolynomialSource<Self::Field>>(
        &self,
        setup: &Self::ProverSetup,
        poly_source: &S,
        hints: Vec<Self::OpeningProofHint>,
        commitments: &[&Self::Commitment],
        opening_point: &[<Self::Field as JoltField>::Challenge],
        _claims: &[Self::Field],
        coeffs: &[Self::Field],
        transcript: &mut ProofTranscript,
    ) -> Self::BatchedProof {
        let joint_poly = poly_source.build_joint_polynomial(coeffs);
        let total_vars = joint_poly.len().log_2();
        let (_sigma, nu) = balanced_sigma_nu(total_vars);
        let max_num_rows = 1 << nu;
        let combined_hint = Self::combine_hints_internal(hints, coeffs, max_num_rows);
        let joint_commitment = Self::combine_commitments_internal(commitments, coeffs);
        let proof = self.prove(
            setup,
            &joint_poly,
            opening_point,
            Some(combined_hint),
            transcript,
            &joint_commitment,
        );
        DoryBatchedProof {
            proof,
            layout: self.layout,
        }
    }

    fn batch_verify<ProofTranscript: Transcript>(
        &self,
        proof: &Self::BatchedProof,
        setup: &Self::VerifierSetup,
        transcript: &mut ProofTranscript,
        opening_point: &[<Self::Field as JoltField>::Challenge],
        commitments: &[&Self::Commitment],
        claims: &[Self::Field],
        coeffs: &[Self::Field],
    ) -> Result<(), ProofVerifyError> {
        let joint_commitment = Self::combine_commitments_internal(commitments, coeffs);
        let joint_claim: ark_bn254::Fr = coeffs.iter().zip(claims).map(|(c, v)| *c * *v).sum();
        self.verify(
            &proof.proof,
            setup,
            transcript,
            opening_point,
            &joint_claim,
            &joint_commitment,
        )
    }

    fn protocol_name() -> &'static [u8] {
        b"Dory"
    }
}

impl DoryCommitmentScheme {
    /// Homomorphically combines row commitment hints using a random linear combination.
    #[tracing::instrument(skip_all, name = "DoryCommitmentScheme::combine_hints_internal")]
    pub(crate) fn combine_hints_internal(
        hints: Vec<Vec<ArkG1>>,
        coeffs: &[ark_bn254::Fr],
        max_num_rows: usize,
    ) -> Vec<ArkG1> {
        let mut rlc_hint = vec![ArkG1(G1Projective::zero()); max_num_rows];
        for (coeff, mut hint) in coeffs.iter().zip(hints.into_iter()) {
            hint.resize(max_num_rows, ArkG1(G1Projective::zero()));

            // SAFETY: ArkG1 is repr(transparent) over G1Projective
            let row_commitments: &mut [G1Projective] = unsafe {
                std::slice::from_raw_parts_mut(hint.as_mut_ptr() as *mut G1Projective, hint.len())
            };

            let rlc_row_commitments: &[G1Projective] = unsafe {
                std::slice::from_raw_parts(rlc_hint.as_ptr() as *const G1Projective, rlc_hint.len())
            };

            let _span = trace_span!("vector_scalar_mul_add_gamma_g1_online");
            let _enter = _span.enter();

            jolt_optimizations::vector_scalar_mul_add_gamma_g1_online(
                row_commitments,
                *coeff,
                rlc_row_commitments,
            );

            let _ = std::mem::replace(&mut rlc_hint, hint);
        }

        rlc_hint
    }

    /// Homomorphically combines multiple GT commitments using a random linear combination.
    #[tracing::instrument(skip_all, name = "DoryCommitmentScheme::combine_commitments_internal")]
    pub(crate) fn combine_commitments_internal(
        commitments: &[&ArkGT],
        coeffs: &[ark_bn254::Fr],
    ) -> ArkGT {
        coeffs
            .par_iter()
            .zip(commitments.par_iter())
            .map(|(coeff, commitment)| {
                let ark_coeff = jolt_to_ark(coeff);
                ark_coeff * **commitment
            })
            .reduce(ArkGT::identity, |a, b| a + b)
    }
}

impl StreamingCommitmentScheme for DoryCommitmentScheme {
    type ChunkState = Vec<ArkG1>;

    #[tracing::instrument(skip_all, name = "DoryCommitmentScheme::compute_tier1_commitment")]
    fn process_chunk<T: SmallScalar>(&self, setup: &Self::ProverSetup, chunk: &[T]) -> Self::ChunkState {
        let row_len = chunk.len();
        let g1_slice =
            unsafe { std::slice::from_raw_parts(setup.g1_vec.as_ptr(), setup.g1_vec.len()) };

        let g1_bases: Vec<G1Affine> = g1_slice[..row_len]
            .iter()
            .map(|g| g.0.into_affine())
            .collect();

        let row_commitment =
            ArkG1(T::msm(&g1_bases[..chunk.len()], chunk).expect("MSM calculation failed."));
        vec![row_commitment]
    }

    #[tracing::instrument(
        skip_all,
        name = "DoryCommitmentScheme::compute_tier1_commitment_onehot"
    )]
    fn process_chunk_onehot(
        &self,
        setup: &Self::ProverSetup,
        onehot_k: usize,
        chunk: &[Option<usize>],
    ) -> Self::ChunkState {
        let K = onehot_k;
        let row_len = chunk.len();

        let g1_slice =
            unsafe { std::slice::from_raw_parts(setup.g1_vec.as_ptr(), setup.g1_vec.len()) };

        let g1_bases: Vec<G1Affine> = g1_slice[..row_len]
            .iter()
            .map(|g| g.0.into_affine())
            .collect();

        let mut indices_per_k: Vec<Vec<usize>> = vec![Vec::new(); K];
        for (col_index, k) in chunk.iter().enumerate() {
            if let Some(k) = k {
                indices_per_k[*k].push(col_index);
            }
        }

        let results = jolt_optimizations::batch_g1_additions_multi(&g1_bases, &indices_per_k);

        let mut row_commitments = vec![ArkG1(G1Projective::zero()); K];
        for (k, result) in results.into_iter().enumerate() {
            if !indices_per_k[k].is_empty() {
                row_commitments[k] = ArkG1(G1Projective::from(result));
            }
        }
        row_commitments
    }

    #[tracing::instrument(skip_all, name = "DoryCommitmentScheme::compute_tier2_commitment")]
    fn aggregate_chunks(
        &self,
        setup: &Self::ProverSetup,
        onehot_k: Option<usize>,
        chunks: &[Self::ChunkState],
    ) -> (Self::Commitment, Self::OpeningProofHint) {
        if let Some(K) = onehot_k {
            let rows_per_k = chunks.len();
            let num_rows = K * rows_per_k;

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
            let tier_2 = <BN254 as PairingCurve>::multi_pair_g2_setup(&row_commitments, g2_bases);

            (tier_2, row_commitments)
        } else {
            let row_commitments: Vec<ArkG1> =
                chunks.iter().flat_map(|chunk| chunk.clone()).collect();

            let g2_bases = &setup.g2_vec[..row_commitments.len()];
            let tier_2 = <BN254 as PairingCurve>::multi_pair_g2_setup(&row_commitments, g2_bases);

            (tier_2, row_commitments)
        }
    }
}

/// Reorders opening_point for AddressMajor layout.
///
/// For AddressMajor layout, reorders opening_point from [r_address, r_cycle] to [r_cycle, r_address].
/// This ensures that after Dory's reversal and splitting:
/// - Column (right) vector gets address variables (matching AddressMajor column indexing)
/// - Row (left) vector gets cycle variables (matching AddressMajor row indexing)
///
/// For CycleMajor layout, returns the point unchanged.
fn reorder_opening_point_for_layout<F: JoltField>(
    layout: DoryLayout,
    opening_point: &[F::Challenge],
) -> Vec<F::Challenge> {
    if layout == DoryLayout::AddressMajor {
        // For AddressMajor, T is needed to split the point.
        // Fall back to DoryGlobals for now; will be eliminated in Phase 2d.
        let log_T = DoryGlobals::get_T().log_2();
        let log_K = opening_point.len().saturating_sub(log_T);
        let (r_address, r_cycle) = opening_point.split_at(log_K);
        [r_cycle, r_address].concat()
    } else {
        opening_point.to_vec()
    }
}
