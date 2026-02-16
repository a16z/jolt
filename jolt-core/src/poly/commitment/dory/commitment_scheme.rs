//! Dory polynomial commitment scheme implementation

use super::dory_globals::{DoryGlobals, DoryLayout};
use super::jolt_dory_routines::{JoltG1Routines, JoltG2Routines};
use super::wrappers::{
    ark_to_jolt, jolt_to_ark, ArkDoryProof, ArkFr, ArkG1, ArkGT, ArkworksProverSetup,
    ArkworksVerifierSetup, JoltToDoryTranscript, BN254,
};
use crate::poly::commitment::dory::DoryContext;
use crate::{
    curve::JoltCurve,
    field::JoltField,
    poly::commitment::commitment_scheme::{
        CommitmentScheme, StreamingCommitmentScheme, ZkEvalCommitment,
    },
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
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use rayon::prelude::*;
use sha3::{Digest, Sha3_256};
use std::borrow::Borrow;
use tracing::trace_span;

#[derive(Clone)]
pub struct DoryCommitmentScheme;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DoryHintPad {
    Replicate,
    ZeroPad,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DoryOpeningProofHint {
    row_commitments: Vec<ArkG1>,
    pad: DoryHintPad,
}

impl DoryOpeningProofHint {
    fn new(row_commitments: Vec<ArkG1>) -> Self {
        let pad = match DoryGlobals::current_context() {
            DoryContext::Main => DoryHintPad::Replicate,
            DoryContext::TrustedAdvice | DoryContext::UntrustedAdvice => DoryHintPad::ZeroPad,
        };
        Self {
            row_commitments,
            pad,
        }
    }

    fn into_rows(self) -> Vec<ArkG1> {
        self.row_commitments
    }
}

pub fn bind_opening_inputs<F: JoltField, ProofTranscript: Transcript>(
    transcript: &mut ProofTranscript,
    opening_point: &[F::Challenge],
    opening: &F,
) {
    let mut point_scalars = Vec::with_capacity(opening_point.len());
    for point in opening_point {
        let scalar: F = (*point).into();
        point_scalars.push(scalar);
    }
    transcript.append_scalars(b"dory_opening_point", &point_scalars);

    transcript.append_scalar(b"dory_opening_eval", opening);
}

#[cfg(feature = "zk")]
pub fn bind_opening_inputs_zk<F: JoltField, C: JoltCurve, ProofTranscript: Transcript>(
    transcript: &mut ProofTranscript,
    opening_point: &[F::Challenge],
    y_com: &C::G1,
) {
    let mut point_scalars = Vec::with_capacity(opening_point.len());
    for point in opening_point {
        let scalar: F = (*point).into();
        point_scalars.push(scalar);
    }
    transcript.append_scalars(b"dory_opening_point", &point_scalars);

    transcript.append_point(b"dory_eval_commitment", y_com);
}

impl CommitmentScheme for DoryCommitmentScheme {
    type Field = ark_bn254::Fr;
    type ProverSetup = ArkworksProverSetup;
    type VerifierSetup = ArkworksVerifierSetup;
    type Commitment = ArkGT;
    type Proof = ArkDoryProof;
    type BatchedProof = Vec<ArkDoryProof>;
    type OpeningProofHint = DoryOpeningProofHint;

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

        (tier_2, DoryOpeningProofHint::new(row_commitments))
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
        hint: Option<Self::OpeningProofHint>,
        transcript: &mut ProofTranscript,
    ) -> (Self::Proof, Option<Self::Field>) {
        let _span = trace_span!("DoryCommitmentScheme::prove").entered();
        let mut rng = rand::thread_rng();

        let row_commitments = hint
            .map(DoryOpeningProofHint::into_rows)
            .unwrap_or_else(|| {
                let (_commitment, row_commitments) = Self::commit(poly, setup);
                row_commitments.into_rows()
            });

        let num_cols = DoryGlobals::get_num_columns();
        let num_rows = DoryGlobals::get_max_num_rows();
        let sigma = num_cols.log_2();
        let nu = num_rows.log_2();

        let reordered_point = reorder_opening_point_for_layout::<ark_bn254::Fr>(opening_point);
        let ark_point: Vec<ArkFr> = reordered_point
            .iter()
            .rev()
            .map(|p| {
                let f_val: ark_bn254::Fr = (*p).into();
                jolt_to_ark(&f_val)
            })
            .collect();

        let mut dory_transcript = JoltToDoryTranscript::<ProofTranscript>::new(transcript);

        #[cfg(feature = "zk")]
        type DoryMode = dory::ZK;
        #[cfg(not(feature = "zk"))]
        type DoryMode = dory::Transparent;

        let (proof, y_blinding) =
            dory::prove::<ArkFr, BN254, JoltG1Routines, JoltG2Routines, _, _, DoryMode, _>(
                poly,
                &ark_point,
                row_commitments,
                nu,
                sigma,
                setup,
                &mut dory_transcript,
                &mut rng,
            )
            .expect("proof generation should succeed");

        (proof, y_blinding.map(|b| ark_to_jolt(&b)))
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

        let reordered_point = reorder_opening_point_for_layout::<ark_bn254::Fr>(opening_point);

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

    fn protocol_name() -> &'static [u8] {
        b"Dory"
    }

    /// In Dory, the opening proof hint consists of the Pedersen commitments to the rows
    /// of the polynomial coefficient matrix. In the context of a batch opening proof, we
    /// can homomorphically combine the row commitments for multiple polynomials into the
    /// row commitments for the RLC of those polynomials. This is more efficient than computing
    /// the row commitments for the RLC from scratch.
    ///
    /// For shorter polynomials (e.g., T-length dense polys in a K*T RLC), we replicate their
    /// row commitments K times to match the K*T matrix layout. This is because a dense poly
    /// p(cycle) embedded in K*T is constant in the address dimension, so its row commitments
    /// repeat with period T/num_cols.
    #[tracing::instrument(skip_all, name = "DoryCommitmentScheme::combine_hints")]
    fn combine_hints(
        hints: Vec<Self::OpeningProofHint>,
        coeffs: &[Self::Field],
    ) -> Self::OpeningProofHint {
        let num_rows = DoryGlobals::get_max_num_rows();

        let mut rlc_hint = vec![ArkG1(G1Projective::zero()); num_rows];
        for (coeff, hint) in coeffs.iter().zip(hints.into_iter()) {
            let DoryOpeningProofHint {
                row_commitments: hint_rows,
                pad,
            } = hint;
            // Replicate shorter hints to match num_rows for dense poly embedding.
            // For advice polynomials, we zero-pad instead (they occupy the top-left block only).
            let mut expanded_hint = if hint_rows.len() < num_rows && !hint_rows.is_empty() {
                match pad {
                    DoryHintPad::ZeroPad => {
                        let mut h = hint_rows;
                        h.resize(num_rows, ArkG1(G1Projective::zero()));
                        h
                    }
                    DoryHintPad::Replicate => {
                        let replication_factor = num_rows / hint_rows.len();
                        let mut expanded = Vec::with_capacity(num_rows);
                        for _ in 0..replication_factor {
                            expanded.extend(hint_rows.iter().cloned());
                        }
                        expanded
                    }
                }
            } else {
                let mut h = hint_rows;
                h.resize(num_rows, ArkG1(G1Projective::zero()));
                h
            };

            let row_commitments: &mut [G1Projective] = unsafe {
                std::slice::from_raw_parts_mut(
                    expanded_hint.as_mut_ptr() as *mut G1Projective,
                    expanded_hint.len(),
                )
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

            let _ = std::mem::replace(&mut rlc_hint, expanded_hint);
        }

        DoryOpeningProofHint::new(rlc_hint)
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
            .reduce(ArkGT::identity, |a, b| a + b)
    }
}

impl StreamingCommitmentScheme for DoryCommitmentScheme {
    type ChunkState = Vec<ArkG1>; // Tier 1 commitment chunks

    #[tracing::instrument(skip_all, name = "DoryCommitmentScheme::compute_tier1_commitment")]
    fn process_chunk<T: SmallScalar>(setup: &Self::ProverSetup, chunk: &[T]) -> Self::ChunkState {
        debug_assert_eq!(chunk.len(), DoryGlobals::get_num_columns());

        let row_len = DoryGlobals::get_num_columns();
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
        setup: &Self::ProverSetup,
        onehot_k: usize,
        chunk: &[Option<usize>],
    ) -> Self::ChunkState {
        let K = onehot_k;

        let row_len = DoryGlobals::get_num_columns();
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
        setup: &Self::ProverSetup,
        onehot_k: Option<usize>,
        chunks: &[Self::ChunkState],
    ) -> (Self::Commitment, Self::OpeningProofHint) {
        let num_rows = DoryGlobals::get_max_num_rows();

        if let Some(_K) = onehot_k {
            let row_len = DoryGlobals::get_num_columns();
            let T = DoryGlobals::get_T();
            let rows_per_k = T / row_len;

            let mut row_commitments = vec![ArkG1(G1Projective::zero()); num_rows];
            for (chunk_index, commitments) in chunks.iter().enumerate() {
                row_commitments
                    .par_iter_mut()
                    .skip(chunk_index)
                    .step_by(rows_per_k)
                    .zip(commitments.par_iter())
                    .for_each(|(dest, src)| *dest = *src);
            }

            let g2_bases = &setup.g2_vec[..num_rows];
            let tier_2 = <BN254 as PairingCurve>::multi_pair_g2_setup(&row_commitments, g2_bases);

            (tier_2, DoryOpeningProofHint::new(row_commitments))
        } else {
            // Dense polynomial: replicate row commitments to match K*T matrix layout
            // This ensures homomorphic combination works correctly with one-hot polynomials
            let dense_rows: Vec<ArkG1> = chunks.iter().flat_map(|chunk| chunk.clone()).collect();

            let dense_row_count = dense_rows.len();
            if dense_row_count > 0 && dense_row_count < num_rows {
                // Replicate dense rows to fill the full matrix
                let replication_factor = num_rows / dense_row_count;
                let mut row_commitments = Vec::with_capacity(num_rows);
                for _ in 0..replication_factor {
                    row_commitments.extend(dense_rows.iter().cloned());
                }

                let g2_bases = &setup.g2_vec[..num_rows];
                let tier_2 =
                    <BN254 as PairingCurve>::multi_pair_g2_setup(&row_commitments, g2_bases);

                (tier_2, DoryOpeningProofHint::new(row_commitments))
            } else {
                // No replication needed (dense_row_count == num_rows or edge case)
                let g2_bases = &setup.g2_vec[..dense_rows.len()];
                let tier_2 = <BN254 as PairingCurve>::multi_pair_g2_setup(&dense_rows, g2_bases);

                (tier_2, DoryOpeningProofHint::new(dense_rows))
            }
        }
    }
}

impl<C: JoltCurve> ZkEvalCommitment<C> for DoryCommitmentScheme
where
    C::G1: From<ArkG1>,
{
    fn eval_commitment(proof: &Self::Proof) -> Option<C::G1> {
        #[cfg(feature = "zk")]
        {
            proof.y_com.as_ref().copied().map(C::G1::from)
        }
        #[cfg(not(feature = "zk"))]
        {
            let _ = proof;
            None
        }
    }

    fn eval_commitment_gens(setup: &Self::ProverSetup) -> Option<(C::G1, C::G1)> {
        let g1_0 = setup.0.g1_vec.first().copied().map(C::G1::from)?;
        let h1 = C::G1::from(setup.0.h1);
        Some((g1_0, h1))
    }

    fn eval_commitment_gens_verifier(setup: &Self::VerifierSetup) -> Option<(C::G1, C::G1)> {
        let g1_0 = C::G1::from(setup.0.g1_0);
        let h1 = C::G1::from(setup.0.h1);
        Some((g1_0, h1))
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
    opening_point: &[F::Challenge],
) -> Vec<F::Challenge> {
    if DoryGlobals::get_layout() == DoryLayout::AddressMajor {
        let log_T = DoryGlobals::get_T().log_2();
        let log_K = opening_point.len().saturating_sub(log_T);
        let (r_address, r_cycle) = opening_point.split_at(log_K);
        [r_cycle, r_address].concat()
    } else {
        opening_point.to_vec()
    }
}
