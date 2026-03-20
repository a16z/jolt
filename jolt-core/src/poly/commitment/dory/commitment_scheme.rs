//! Dory polynomial commitment scheme implementation

use super::dory_globals::DoryGlobals;
use super::jolt_dory_routines::{JoltG1Routines, JoltG2Routines};
use super::wrappers::{
    ark_to_jolt, jolt_to_ark, ArkDoryProof, ArkFr, ArkG1, ArkGT, ArkworksProverSetup,
    ArkworksVerifierSetup, JoltToDoryTranscript, BN254,
};
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
    arithmetic::{Field as DoryField, Group, PairingCurve},
    poly::Polynomial,
};
use rayon::prelude::*;
use std::borrow::Borrow;
use tracing::trace_span;

fn debug_disable_dory_setup_cache() -> bool {
    std::env::var("JOLT_DEBUG_DISABLE_DORY_SETUP_CACHE")
        .map(|v| {
            let value = v.trim().to_ascii_lowercase();
            !matches!(value.as_str(), "" | "0" | "false" | "off")
        })
        .unwrap_or(false)
}

#[derive(Clone)]
pub struct DoryCommitmentScheme;

#[derive(Clone, Debug, PartialEq)]
pub struct DoryOpeningProofHint(Vec<ArkG1>);

impl DoryOpeningProofHint {
    fn new(row_commitments: Vec<ArkG1>) -> Self {
        Self(row_commitments)
    }

    fn into_rows(self) -> Vec<ArkG1> {
        self.0
    }
}

#[inline]
fn canonical_setup_log_n(max_num_vars: usize) -> usize {
    // Dory's generator count depends on ceil(max_log_n / 2), so odd/even pairs like
    // 23 and 24 share the same generator bucket. Canonicalizing to the even bucket
    // representative keeps those runs on a single URS file.
    if max_num_vars.is_multiple_of(2) {
        max_num_vars
    } else {
        max_num_vars + 1
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
pub fn bind_opening_inputs_zk<F: JoltField, C: JoltCurve<F = F>, ProofTranscript: Transcript>(
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

    transcript.append_commitment(b"dory_eval_commitment", y_com);
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
        let canonical_max_num_vars = canonical_setup_log_n(max_num_vars);
        #[cfg(not(target_arch = "wasm32"))]
        let setup = ArkworksProverSetup::new_from_urs(canonical_max_num_vars);
        #[cfg(target_arch = "wasm32")]
        let setup = ArkworksProverSetup::new(canonical_max_num_vars);

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

        let (tier_2, row_commitments, _commit_blind) =
            <MultilinearPolynomial<ark_bn254::Fr> as Polynomial<ArkFr>>::commit::<
                BN254,
                dory::Transparent,
                JoltG1Routines,
            >(poly, nu, sigma, setup)
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

        let (row_commitments, commit_blind) = hint
            .map(|h| (h.into_rows(), DoryField::zero()))
            .unwrap_or_else(|| {
                let (_commitment, row_commitments) = Self::commit(poly, setup);
                (row_commitments.into_rows(), DoryField::zero())
            });

        let num_cols = DoryGlobals::get_num_columns();
        let num_rows = DoryGlobals::get_max_num_rows();
        let sigma = num_cols.log_2();
        let nu = num_rows.log_2();

        let ark_point: Vec<ArkFr> = opening_point
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
            dory::prove::<ArkFr, BN254, JoltG1Routines, JoltG2Routines, _, _, DoryMode>(
                poly,
                &ark_point,
                row_commitments,
                commit_blind,
                nu,
                sigma,
                setup,
                &mut dory_transcript,
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

        // Dory uses the opposite endian-ness as Jolt
        let ark_point: Vec<ArkFr> = opening_point
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
    #[tracing::instrument(skip_all, name = "DoryCommitmentScheme::combine_hints")]
    fn combine_hints(
        hints: Vec<Self::OpeningProofHint>,
        coeffs: &[Self::Field],
    ) -> Self::OpeningProofHint {
        let num_rows = DoryGlobals::get_max_num_rows();

        let mut rlc_hint = vec![ArkG1(G1Projective::zero()); num_rows];
        for (coeff, mut hint) in coeffs.iter().zip(hints.into_iter()) {
            hint.0.resize(num_rows, ArkG1(G1Projective::zero()));

            let row_commitments: &mut [G1Projective] = unsafe {
                std::slice::from_raw_parts_mut(
                    hint.0.as_mut_ptr() as *mut G1Projective,
                    hint.0.len(),
                )
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

            let _ = std::mem::replace(&mut rlc_hint, hint.0);
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
            let tier_2 = if debug_disable_dory_setup_cache() {
                <BN254 as PairingCurve>::multi_pair(&row_commitments, g2_bases)
            } else {
                <BN254 as PairingCurve>::multi_pair_g2_setup(&row_commitments, g2_bases)
            };

            (tier_2, DoryOpeningProofHint::new(row_commitments))
        } else {
            let row_commitments: Vec<ArkG1> =
                chunks.iter().flat_map(|chunk| chunk.clone()).collect();

            let g2_bases = &setup.g2_vec[..row_commitments.len()];
            let tier_2 = if debug_disable_dory_setup_cache() {
                <BN254 as PairingCurve>::multi_pair(&row_commitments, g2_bases)
            } else {
                <BN254 as PairingCurve>::multi_pair_g2_setup(&row_commitments, g2_bases)
            };

            (tier_2, DoryOpeningProofHint::new(row_commitments))
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

    #[cfg(feature = "zk")]
    fn zk_generators(setup: &Self::ProverSetup, count: usize) -> Option<(Vec<C::G1>, C::G1)> {
        let count = std::cmp::min(count, setup.0.g1_vec.len());
        let g1s = setup.0.g1_vec[..count]
            .iter()
            .map(|g| C::G1::from(*g))
            .collect();
        let h1 = C::G1::from(setup.0.h1);
        Some((g1s, h1))
    }
}
