//! Dory PCS implementing the `jolt-openings` trait hierarchy.

#![expect(
    clippy::expect_used,
    clippy::panic,
    clippy::unimplemented,
    reason = "ZK proof y_com/y_blinding are Dory-mode invariants; dory::prove/verify errors are caller-precondition violations surfaced via panic; the dory adapter's commit is unreachable because DoryScheme pre-computes row commitments"
)]

use std::collections::BTreeMap;
use std::iter::successors;
use std::mem::{transmute, transmute_copy};

#[cfg(not(test))]
use dory::backends::arkworks::{init_cache, is_cached};
use dory::backends::arkworks::{
    ArkFr as DoryArkFr, ArkG1 as ArkG1Struct, ArkGT as DoryArkGT, ArkworksProverSetup,
    BN254 as DoryBN254,
};
use dory::mode::Transparent;
use dory::primitives::arithmetic::{
    DoryRoutines, Field as DoryField, Group as DoryGroup, PairingCurve,
};
use dory::primitives::poly::{MultilinearLagrange, Polynomial as DoryPolynomial};
use dory::primitives::transcript::Transcript as DoryTranscript;
use dory::setup::ProverSetup as DoryNativeProverSetup;
use dory::{error::DoryError, mode::Mode as DoryMode};
use dory::{prove, verify, Mode, ZK};
use jolt_crypto::{Bn254G1, Bn254GT, Commitment, DeriveSetup, JoltGroup, PedersenSetup};
use jolt_field::{Fr, FromPrimitiveInt};
use jolt_openings::{
    homomorphic_prove_batch, homomorphic_verify_batch, AdditivelyHomomorphic,
    AdditivelyHomomorphicVerifier, BatchCommitmentSource, BatchOpeningProverResult,
    BatchOpeningPublic, BatchOpeningSource, BatchOutputExpression, BatchOutputRelation,
    BatchOutputValue, CommitmentScheme, CommitmentSchemeVerifier, CommitmentSource,
    EvaluationCommitmentProver, EvaluationCommitmentScheme, LinearCombinationOpeningSource,
    LinearOpeningScheme, LinearOpeningSchemeVerifier, LinearSourceTerm, OneHotEntries, OneHotRow,
    OpenedBatchOutput, OpeningClaim, OpeningsError, ProverBatchOpeningTerm, ProverClaim,
    PublicVerifierSetup, SourceId, SourceRow, VerifierBatchOpeningTerm, ZkBatchOpeningProverResult,
    ZkBatchOpeningWitness, ZkLinearOpeningScheme, ZkLinearOpeningSchemeVerifier, ZkOpeningScheme,
    ZkOpeningSchemeVerifier,
};
use jolt_optimizations::batch_g1_additions_multi;
use jolt_transcript::{AppendToTranscript, Label, LabelWithCount, Transcript};
use rayon::prelude::*;

use ark_bn254::{G1Affine, G1Projective};
use ark_ec::scalar_mul::variable_base::{msm_i128, msm_u64};
use ark_ec::CurveGroup;

use crate::routines::{JoltG1Routines, JoltG2Routines};
use crate::transcript::JoltToDoryTranscript;
use crate::types::{DoryCommitment, DoryHint, DoryProof, DoryProverSetup, DoryVerifierSetup};

// All jolt types below are #[repr(transparent)] over the same arkworks
// inner type as their dory-pcs counterpart, guaranteeing identical layout.

/// Dory-pcs's arkworks scalar wrapper.
///
/// Most callers should use the backend-neutral `jolt_field::Fr` APIs on
/// `DoryScheme`. This type is exposed for Dory-native integration points that
/// already implement dory-pcs polynomial traits and need to avoid converting
/// large opening-time vectors through the generic source abstraction.
pub type ArkFr = DoryArkFr;
pub(crate) type ArkG1 = ArkG1Struct;
pub(crate) type ArkGT = DoryArkGT;
type InnerBN254 = DoryBN254;

// The dory-pcs prepared-point cache is global and tied to one URS. Jolt
// initializes one large Dory setup during preprocessing, so caching that setup
// preserves the old in-core performance path. Small standalone callers can
// create several independent toy URS sizes in one process, so avoid seeding the
// global cache from those setups.
#[cfg(not(test))]
const PREPARED_CACHE_MIN_NUM_VARS: usize = 16;

// All conversion functions below rely on repr(transparent) layout identity
// between jolt and dory-pcs wrappers over the same arkworks inner type.

#[inline]
pub(crate) fn jolt_fr_to_ark(f: &Fr) -> ArkFr {
    // SAFETY: Fr and ArkFr are both repr(transparent) over ark_bn254::Fr.
    unsafe { transmute_copy(f) }
}

#[inline]
pub(crate) fn ark_to_jolt_fr(ark: &ArkFr) -> Fr {
    // SAFETY: same layout as jolt_fr_to_ark.
    unsafe { transmute_copy(ark) }
}

#[inline]
pub(crate) fn jolt_gt_to_ark(gt: &Bn254GT) -> ArkGT {
    // SAFETY: Bn254GT and ArkGT are both repr(transparent) over Fq12.
    unsafe { transmute_copy(gt) }
}

#[inline]
pub(crate) fn ark_to_jolt_gt(ark: &ArkGT) -> Bn254GT {
    // SAFETY: same layout as jolt_gt_to_ark.
    unsafe { transmute_copy(ark) }
}

#[inline]
pub(crate) fn jolt_g1_vec_to_ark(v: Vec<Bn254G1>) -> Vec<ArkG1> {
    // SAFETY: Bn254G1 and ArkG1 have identical size/align (repr(transparent)
    // over G1Projective), so Vec layout is identical.
    unsafe { transmute(v) }
}

#[inline]
pub(crate) fn ark_to_jolt_g1_vec(v: Vec<ArkG1>) -> Vec<Bn254G1> {
    // SAFETY: same layout as jolt_g1_vec_to_ark.
    unsafe { transmute(v) }
}

#[inline]
pub(crate) fn ark_to_jolt_g1(ark: ArkG1) -> Bn254G1 {
    // SAFETY: Bn254G1 and ArkG1 are both repr(transparent) over G1Projective.
    unsafe { transmute(ark) }
}

#[derive(Clone)]
pub struct DoryScheme;

impl DoryScheme {
    #[tracing::instrument(skip_all, name = "DoryScheme::setup_prover", fields(max_num_vars))]
    pub fn setup_prover(max_num_vars: usize) -> DoryProverSetup {
        #[cfg(not(target_arch = "wasm32"))]
        let setup = ArkworksProverSetup::new_from_urs(max_num_vars);
        #[cfg(target_arch = "wasm32")]
        let setup = ArkworksProverSetup::new(max_num_vars);
        #[cfg(not(test))]
        if max_num_vars >= PREPARED_CACHE_MIN_NUM_VARS && !is_cached() {
            init_cache(&setup.g1_vec, &setup.g2_vec);
        }
        DoryProverSetup(setup)
    }

    /// Derives the verifier SRS (a subset of the prover SRS).
    #[tracing::instrument(skip_all, name = "DoryScheme::setup_verifier", fields(max_num_vars))]
    pub fn setup_verifier(max_num_vars: usize) -> DoryVerifierSetup {
        let prover_setup = Self::setup_prover(max_num_vars);
        DoryVerifierSetup(prover_setup.0.to_verifier_setup())
    }

    fn commit_with_mode<S, M>(source: &S, setup: &ArkworksProverSetup) -> (DoryCommitment, DoryHint)
    where
        S: CommitmentSource<Fr> + ?Sized,
        M: Mode,
    {
        let chunk_len = natural_or_balanced_chunk_len(source);
        let row_commitments = compute_row_commitments(source, chunk_len, setup);
        finish_row_commitments::<M>(row_commitments, chunk_len, setup)
    }

    fn commit_batch_with_mode<B, M>(
        batch: &B,
        ids: &[B::Id],
        setup: &ArkworksProverSetup,
    ) -> Vec<(DoryCommitment, DoryHint)>
    where
        B: BatchCommitmentSource<Fr>,
        M: Mode,
    {
        if ids.is_empty() {
            return Vec::new();
        }

        let max_num_vars = ids
            .iter()
            .map(|&id| batch.num_vars(id))
            .max()
            .expect("ids is non-empty");
        let chunk_len = batch
            .natural_chunk_len(ids)
            .unwrap_or_else(|| balanced_chunk_len(max_num_vars));
        validate_chunk_len(chunk_len);
        let ctx = CommitRowContext::new(setup, chunk_len);
        let row_major = batch.map_rows(chunk_len, ids, |_, row| commit_source_row(row, &ctx));

        let mut chunks_by_source: Vec<Vec<DoryChunkCommitment>> = (0..ids.len())
            .map(|_| Vec::with_capacity(row_major.len()))
            .collect();
        for row in row_major {
            assert_eq!(
                row.len(),
                ids.len(),
                "batch source returned a ragged row of committed sources",
            );
            for (source_chunks, chunk) in chunks_by_source.iter_mut().zip(row) {
                source_chunks.push(chunk);
            }
        }

        chunks_by_source
            .into_par_iter()
            .map(|chunks| aggregate_batch_chunks::<M>(chunks, chunk_len, setup))
            .collect()
    }

    fn open_source_with_mode<S, T, M>(
        source: &S,
        point: &[Fr],
        setup: &ArkworksProverSetup,
        hint: DoryHint,
        transcript: &mut T,
    ) -> (DoryProof, Option<Fr>)
    where
        S: CommitmentSource<Fr> + ?Sized,
        T: DoryTranscript<Curve = InnerBN254>,
        M: Mode,
    {
        let adapter = DorySourceAdapter::new(source);
        let ark_point: Vec<ArkFr> = point.iter().rev().map(jolt_fr_to_ark).collect();
        let (nu, sigma) = hint_shape(&hint);

        Self::open_dory_source_with_mode::<_, _, M>(
            &adapter, &ark_point, nu, sigma, setup, hint, transcript,
        )
    }

    fn open_dory_source_with_mode<S, T, M>(
        source: &S,
        ark_point: &[ArkFr],
        nu: usize,
        sigma: usize,
        setup: &ArkworksProverSetup,
        hint: DoryHint,
        transcript: &mut T,
    ) -> (DoryProof, Option<Fr>)
    where
        S: DoryPolynomial<ArkFr> + MultilinearLagrange<ArkFr>,
        T: DoryTranscript<Curve = InnerBN254>,
        M: Mode,
    {
        let (row_commitments, commit_blind) = hint.into_ark_parts();
        let (proof, y_blinding) =
            prove::<ArkFr, InnerBN254, JoltG1Routines, JoltG2Routines, _, _, M>(
                source,
                ark_point,
                row_commitments,
                commit_blind,
                nu,
                sigma,
                setup,
                transcript,
            )
            .unwrap_or_else(|e| panic!("dory::prove failed: {e:?}"));

        (
            DoryProof(proof),
            y_blinding.map(|blind| ark_to_jolt_fr(&blind)),
        )
    }

    /// Opens a transparent Dory commitment using the traversal recorded in the hint.
    #[tracing::instrument(skip_all, name = "DoryScheme::open_source_with_hint")]
    fn open_source_with_hint<S, T>(
        source: &S,
        point: &[Fr],
        setup: &DoryProverSetup,
        hint: DoryHint,
        transcript: &mut T,
    ) -> DoryProof
    where
        S: CommitmentSource<Fr> + ?Sized,
        T: DoryTranscript<Curve = InnerBN254>,
    {
        let (proof, _blind) = Self::open_source_with_mode::<S, T, Transparent>(
            source, point, &setup.0, hint, transcript,
        );
        proof
    }

    /// Opens a ZK/hiding Dory commitment using the traversal recorded in the hint.
    #[tracing::instrument(skip_all, name = "DoryScheme::open_zk_source_with_hint")]
    fn open_zk_source_with_hint<S, T>(
        source: &S,
        point: &[Fr],
        setup: &DoryProverSetup,
        hint: DoryHint,
        transcript: &mut T,
    ) -> (DoryProof, Bn254G1, Fr)
    where
        S: CommitmentSource<Fr> + ?Sized,
        T: DoryTranscript<Curve = InnerBN254>,
    {
        let (proof, y_blinding) =
            Self::open_source_with_mode::<S, T, ZK>(source, point, &setup.0, hint, transcript);
        let y_com = ark_to_jolt_g1(proof.0.y_com.expect("ZK proof must contain y_com"));
        let blinding = y_blinding.expect("ZK proof must return y_blinding");
        (proof, y_com, blinding)
    }

    fn prove_source_backed_batch_with_mode<B, ClaimId, T, M>(
        terms: Vec<ProverBatchOpeningTerm<Fr, ClaimId, B::Id>>,
        source_batch: &mut B,
        setup: &DoryProverSetup,
        transcript: &mut T,
    ) -> (
        DoryProof,
        BatchOpeningPublic<Fr, M::OutputValue, ClaimId>,
        Fr,
        Option<Fr>,
    )
    where
        B: LinearCombinationOpeningSource<Fr, DoryHint>,
        T: Transcript<Challenge = Fr>,
        M: DoryBatchOpeningMode,
    {
        assert!(
            !terms.is_empty(),
            "Dory source-backed batch opening requires at least one term",
        );
        let (public_point, proof_point) = prover_batch_points(&terms);

        if M::ABSORB_PUBLIC_EVALS {
            bind_source_backed_evals(&terms, transcript);
        }
        let gamma_powers = challenge_powers(transcript, terms.len());
        let joint_claim = joint_claim(&terms, &gamma_powers);
        let source_terms = aggregate_source_terms(&terms, &gamma_powers);
        let combined_hint = combine_batch_opening_hints(source_batch, &source_terms);
        let combined_source = source_batch.linear_combination(&source_terms);

        let adapter = DoryBatchOpeningAdapter {
            source: &combined_source,
            output_eval: joint_claim,
            num_vars: proof_point.len(),
        };

        let ark_point: Vec<ArkFr> = proof_point.iter().rev().map(jolt_fr_to_ark).collect();
        let (nu, sigma) = hint_shape(&combined_hint);
        let mut dory_transcript = JoltToDoryTranscript::new(transcript);
        let (proof, y_blinding) = Self::open_dory_source_with_mode::<_, _, M::DoryMode>(
            &adapter,
            &ark_point,
            nu,
            sigma,
            &setup.0,
            combined_hint,
            &mut dory_transcript,
        );

        let output_value = M::output_value(&proof, joint_claim);
        M::bind_output(transcript, &public_point, &output_value);
        let public = batch_opening_public(terms, gamma_powers, public_point, output_value);
        (proof, public, joint_claim, y_blinding)
    }

    fn verify_source_backed_batch_with_mode<ClaimId, SourceIdT, T, M>(
        terms: Vec<VerifierBatchOpeningTerm<Fr, Self, ClaimId, SourceIdT>>,
        proof: &Vec<DoryProof>,
        setup: &DoryVerifierSetup,
        transcript: &mut T,
    ) -> Result<BatchOpeningPublic<Fr, M::OutputValue, ClaimId>, OpeningsError>
    where
        SourceIdT: SourceId,
        T: Transcript<Challenge = Fr>,
        M: DoryBatchOpeningMode,
    {
        if terms.is_empty() {
            return Err(OpeningsError::VerificationFailed);
        }
        let [proof] = proof.as_slice() else {
            return Err(OpeningsError::VerificationFailed);
        };
        let (public_point, proof_point) = verifier_batch_points(&terms)?;

        if M::ABSORB_PUBLIC_EVALS {
            bind_source_backed_evals(&terms, transcript);
        }
        let gamma_powers = challenge_powers(transcript, terms.len());
        let joint_claim = joint_claim(&terms, &gamma_powers);
        let joint_commitment = combine_batch_opening_commitments(&terms, &gamma_powers);

        let output_value = M::output_value(proof, joint_claim);
        match &output_value {
            BatchOutputValue::Public(eval) => {
                Self::verify(
                    &joint_commitment,
                    &proof_point,
                    *eval,
                    proof,
                    setup,
                    transcript,
                )?;
            }
            BatchOutputValue::Hidden(_) => {
                Self::verify_zk(&joint_commitment, &proof_point, proof, setup, transcript)?;
            }
        }
        M::bind_output(transcript, &public_point, &output_value);
        Ok(batch_opening_public(
            terms,
            gamma_powers,
            public_point,
            output_value,
        ))
    }

    /// Verifies a transparent Dory opening using an already Dory-compatible
    /// transcript adapter.
    ///
    /// This is the verifier-side counterpart to opening with a hint for
    /// protocol layers that still own their transcript type but delegate Dory
    /// verification to this crate.
    #[tracing::instrument(skip_all, name = "DoryScheme::verify_with_transcript")]
    pub fn verify_with_transcript<T>(
        commitment: &DoryCommitment,
        point: &[Fr],
        eval: Fr,
        proof: &DoryProof,
        setup: &DoryVerifierSetup,
        transcript: &mut T,
    ) -> Result<(), OpeningsError>
    where
        T: DoryTranscript<Curve = InnerBN254>,
    {
        let ark_point: Vec<ArkFr> = point.iter().rev().map(jolt_fr_to_ark).collect();
        let ark_eval = jolt_fr_to_ark(&eval);
        let ark_commitment = jolt_gt_to_ark(&commitment.0);

        verify::<ArkFr, InnerBN254, JoltG1Routines, JoltG2Routines, _>(
            ark_commitment,
            ark_eval,
            &ark_point,
            &proof.0,
            setup.0.clone().into_inner(),
            transcript,
        )
        .map_err(|_| OpeningsError::VerificationFailed)
    }

    /// Verifies a ZK/hiding Dory opening using an already Dory-compatible
    /// transcript adapter.
    ///
    /// In ZK mode the evaluation is hidden and Dory verifies against the
    /// evaluation commitment embedded in the proof.
    #[tracing::instrument(skip_all, name = "DoryScheme::verify_zk_with_transcript")]
    pub fn verify_zk_with_transcript<T>(
        commitment: &DoryCommitment,
        point: &[Fr],
        proof: &DoryProof,
        setup: &DoryVerifierSetup,
        transcript: &mut T,
    ) -> Result<(), OpeningsError>
    where
        T: DoryTranscript<Curve = InnerBN254>,
    {
        let ark_point: Vec<ArkFr> = point.iter().rev().map(jolt_fr_to_ark).collect();
        let dummy_eval = <ArkFr as DoryField>::zero();
        let ark_commitment = jolt_gt_to_ark(&commitment.0);

        verify::<ArkFr, InnerBN254, JoltG1Routines, JoltG2Routines, _>(
            ark_commitment,
            dummy_eval,
            &ark_point,
            &proof.0,
            setup.0.clone().into_inner(),
            transcript,
        )
        .map_err(|_| OpeningsError::VerificationFailed)
    }
}

trait DoryBatchOpeningMode {
    type DoryMode: Mode;
    type OutputValue: Clone + std::fmt::Debug + Eq + Send + Sync + 'static;

    const ABSORB_PUBLIC_EVALS: bool;

    fn output_value(proof: &DoryProof, joint_claim: Fr) -> BatchOutputValue<Fr, Self::OutputValue>;

    fn bind_output(
        transcript: &mut impl Transcript<Challenge = Fr>,
        public_point: &[Fr],
        value: &BatchOutputValue<Fr, Self::OutputValue>,
    );
}

struct TransparentBatchOpening;

impl DoryBatchOpeningMode for TransparentBatchOpening {
    type DoryMode = Transparent;
    type OutputValue = ();

    const ABSORB_PUBLIC_EVALS: bool = true;

    fn output_value(
        _proof: &DoryProof,
        joint_claim: Fr,
    ) -> BatchOutputValue<Fr, Self::OutputValue> {
        BatchOutputValue::Public(joint_claim)
    }

    fn bind_output(
        transcript: &mut impl Transcript<Challenge = Fr>,
        public_point: &[Fr],
        value: &BatchOutputValue<Fr, Self::OutputValue>,
    ) {
        let BatchOutputValue::Public(eval) = value else {
            unreachable!("transparent Dory batch opening returns a public scalar");
        };
        DoryScheme::bind_opening_inputs(transcript, public_point, eval);
    }
}

struct ZkBatchOpening;

impl DoryBatchOpeningMode for ZkBatchOpening {
    type DoryMode = ZK;
    type OutputValue = Bn254G1;

    const ABSORB_PUBLIC_EVALS: bool = false;

    fn output_value(
        proof: &DoryProof,
        _joint_claim: Fr,
    ) -> BatchOutputValue<Fr, Self::OutputValue> {
        let y_com = proof
            .0
            .y_com
            .expect("ZK source-backed Dory proof must contain y_com");
        BatchOutputValue::Hidden(ark_to_jolt_g1(y_com))
    }

    fn bind_output(
        transcript: &mut impl Transcript<Challenge = Fr>,
        public_point: &[Fr],
        value: &BatchOutputValue<Fr, Self::OutputValue>,
    ) {
        let BatchOutputValue::Hidden(y_com) = value else {
            unreachable!("ZK Dory batch opening returns a hidden output");
        };
        DoryScheme::bind_zk_opening_inputs(transcript, public_point, y_com);
    }
}

fn challenge_powers<T>(transcript: &mut T, len: usize) -> Vec<Fr>
where
    T: Transcript<Challenge = Fr>,
{
    let gamma = transcript.challenge();
    successors(Some(Fr::from_u64(1)), |prev| Some(*prev * gamma))
        .take(len)
        .collect()
}

fn bind_source_backed_evals<Term, T>(terms: &[Term], transcript: &mut T)
where
    Term: DoryBatchTerm,
    T: Transcript<Challenge = Fr>,
{
    transcript.append(&LabelWithCount(b"rlc_claims", terms.len() as u64));
    for term in terms {
        let scaled_eval = term.eval() * term.eval_scale();
        scaled_eval.append_to_transcript(transcript);
    }
}

fn joint_claim<Term>(terms: &[Term], gamma_powers: &[Fr]) -> Fr
where
    Term: DoryBatchTerm,
{
    terms
        .iter()
        .zip(gamma_powers.iter())
        .map(|(term, gamma)| *gamma * term.eval_scale() * term.eval())
        .sum()
}

fn batch_opening_public<ClaimId, HidingCommitment, Term>(
    terms: Vec<Term>,
    gamma_powers: Vec<Fr>,
    point: Vec<Fr>,
    value: BatchOutputValue<Fr, HidingCommitment>,
) -> BatchOpeningPublic<Fr, HidingCommitment, ClaimId>
where
    Term: IntoDoryBatchTerm<ClaimId>,
{
    let relation_terms = terms
        .into_iter()
        .zip(gamma_powers)
        .map(|(term, gamma)| {
            let (claim_id, eval_scale) = term.into_claim_id_and_scale();
            (claim_id, gamma * eval_scale)
        })
        .collect();

    BatchOpeningPublic {
        outputs: vec![OpenedBatchOutput { point, value }],
        relations: vec![BatchOutputRelation {
            output_index: 0,
            expression: BatchOutputExpression::Linear(relation_terms),
        }],
    }
}

fn aggregate_source_terms<Term, SourceIdT>(
    terms: &[Term],
    gamma_powers: &[Fr],
) -> Vec<LinearSourceTerm<Fr, SourceIdT>>
where
    Term: DoryBatchTerm<SourceId = SourceIdT>,
    SourceIdT: SourceId,
{
    let mut terms_by_source = BTreeMap::new();
    for (term, gamma) in terms.iter().zip(gamma_powers.iter()) {
        *terms_by_source
            .entry(term.source_id())
            .or_insert_with(|| Fr::from_u64(0)) += *gamma;
    }

    terms_by_source
        .into_iter()
        .map(|(source_id, coefficient)| LinearSourceTerm {
            source_id,
            coefficient,
        })
        .collect()
}

fn combine_batch_opening_hints<B>(
    source_batch: &B,
    source_terms: &[LinearSourceTerm<Fr, B::Id>],
) -> DoryHint
where
    B: BatchOpeningSource<Fr, DoryHint>,
{
    let hints: Vec<&DoryHint> = source_terms
        .iter()
        .map(|term| source_batch.opening_hint(term.source_id))
        .collect();
    let scalars: Vec<Fr> = source_terms.iter().map(|term| term.coefficient).collect();
    combine_hint_refs(&hints, &scalars)
}

fn combine_batch_opening_commitments<ClaimId, SourceIdT>(
    terms: &[VerifierBatchOpeningTerm<Fr, DoryScheme, ClaimId, SourceIdT>],
    gamma_powers: &[Fr],
) -> DoryCommitment
where
    SourceIdT: SourceId,
{
    let mut terms_by_source = BTreeMap::<SourceIdT, (DoryCommitment, Fr)>::new();
    for (term, gamma) in terms.iter().zip(gamma_powers.iter()) {
        let _ = terms_by_source
            .entry(term.source_id)
            .and_modify(|(_, coefficient)| *coefficient += *gamma)
            .or_insert_with(|| (term.commitment.clone(), *gamma));
    }

    let (commitments, scalars): (Vec<_>, Vec<_>) = terms_by_source.into_values().unzip();
    DoryScheme::combine(&commitments, &scalars)
}

fn combine_hint_refs(hints: &[&DoryHint], scalars: &[Fr]) -> DoryHint {
    assert_eq!(hints.len(), scalars.len());
    assert!(!hints.is_empty(), "combine_hints: empty hint set");

    let num_rows = hints
        .iter()
        .map(|hint| hint.row_commitments.len())
        .max()
        .unwrap_or(0);

    let combined_blind = hints
        .iter()
        .zip(scalars.iter())
        .map(|(hint, &scalar)| scalar * hint.commit_blind)
        .sum();

    let combined: Vec<Bn254G1> = (0..num_rows)
        .into_par_iter()
        .map(|row| {
            let mut acc = Bn254G1::default();
            for (hint, &scalar) in hints.iter().zip(scalars.iter()) {
                if let Some(row_commitment) = hint.row_commitments.get(row) {
                    acc += row_commitment.scalar_mul(&scalar);
                }
            }
            acc
        })
        .collect();

    let chunk_len = hints
        .iter()
        .map(|hint| hint.chunk_len)
        .max()
        .unwrap_or_default();
    DoryHint::new(combined, combined_blind, chunk_len)
}

fn prover_batch_points<ClaimId, SourceIdT>(
    terms: &[ProverBatchOpeningTerm<Fr, ClaimId, SourceIdT>],
) -> (Vec<Fr>, Vec<Fr>) {
    let first = &terms[0].point;
    for term in terms.iter().skip(1) {
        assert_eq!(
            term.point.public, first.public,
            "Dory source-backed batch opening expects one public point",
        );
        assert_eq!(
            term.point.proof, first.proof,
            "Dory source-backed batch opening expects one proof point",
        );
    }
    (first.public.clone(), first.proof.clone())
}

fn verifier_batch_points<ClaimId, SourceIdT>(
    terms: &[VerifierBatchOpeningTerm<Fr, DoryScheme, ClaimId, SourceIdT>],
) -> Result<(Vec<Fr>, Vec<Fr>), OpeningsError> {
    let first = &terms[0].point;
    if terms
        .iter()
        .skip(1)
        .any(|term| term.point.public != first.public || term.point.proof != first.proof)
    {
        return Err(OpeningsError::VerificationFailed);
    }
    Ok((first.public.clone(), first.proof.clone()))
}

trait DoryBatchTerm {
    type SourceId: SourceId;

    fn source_id(&self) -> Self::SourceId;
    fn eval(&self) -> Fr;
    fn eval_scale(&self) -> Fr;
}

impl<ClaimId, SourceIdT> DoryBatchTerm for ProverBatchOpeningTerm<Fr, ClaimId, SourceIdT>
where
    SourceIdT: SourceId,
{
    type SourceId = SourceIdT;

    fn source_id(&self) -> Self::SourceId {
        self.source_id
    }

    fn eval(&self) -> Fr {
        self.eval
    }

    fn eval_scale(&self) -> Fr {
        self.eval_scale
    }
}

impl<ClaimId, SourceIdT> DoryBatchTerm
    for VerifierBatchOpeningTerm<Fr, DoryScheme, ClaimId, SourceIdT>
where
    SourceIdT: SourceId,
{
    type SourceId = SourceIdT;

    fn source_id(&self) -> Self::SourceId {
        self.source_id
    }

    fn eval(&self) -> Fr {
        self.eval
    }

    fn eval_scale(&self) -> Fr {
        self.eval_scale
    }
}

trait IntoDoryBatchTerm<ClaimId> {
    fn into_claim_id_and_scale(self) -> (ClaimId, Fr);
}

impl<ClaimId, SourceIdT> IntoDoryBatchTerm<ClaimId>
    for ProverBatchOpeningTerm<Fr, ClaimId, SourceIdT>
{
    fn into_claim_id_and_scale(self) -> (ClaimId, Fr) {
        (self.claim_id, self.eval_scale)
    }
}

impl<ClaimId, SourceIdT> IntoDoryBatchTerm<ClaimId>
    for VerifierBatchOpeningTerm<Fr, DoryScheme, ClaimId, SourceIdT>
{
    fn into_claim_id_and_scale(self) -> (ClaimId, Fr) {
        (self.claim_id, self.eval_scale)
    }
}

struct DoryBatchOpeningAdapter<'a, S>
where
    S: CommitmentSource<Fr> + ?Sized,
{
    source: &'a S,
    output_eval: Fr,
    num_vars: usize,
}

impl<S> DoryPolynomial<ArkFr> for DoryBatchOpeningAdapter<'_, S>
where
    S: CommitmentSource<Fr> + ?Sized,
{
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn evaluate(&self, _point: &[ArkFr]) -> ArkFr {
        jolt_fr_to_ark(&self.output_eval)
    }

    fn commit<E, Mo, M1>(
        &self,
        _nu: usize,
        _sigma: usize,
        _setup: &DoryNativeProverSetup<E>,
    ) -> Result<(E::GT, Vec<E::G1>, ArkFr), DoryError>
    where
        E: PairingCurve,
        Mo: DoryMode,
        M1: DoryRoutines<E::G1>,
        E::G1: DoryGroup<Scalar = ArkFr>,
    {
        unimplemented!(
            "DoryScheme pre-computes source-backed row commitments before invoking dory::prove; \
             dory::Polynomial::commit on this adapter is not exercised"
        )
    }
}

impl<S> MultilinearLagrange<ArkFr> for DoryBatchOpeningAdapter<'_, S>
where
    S: CommitmentSource<Fr> + ?Sized,
{
    fn vector_matrix_product(&self, left_vec: &[ArkFr], _nu: usize, sigma: usize) -> Vec<ArkFr> {
        let native_left: Vec<Fr> = left_vec.iter().map(ark_to_jolt_fr).collect();
        let result = self.source.fold_rows(&native_left, 1usize << sigma);
        result.iter().map(jolt_fr_to_ark).collect()
    }
}

impl DeriveSetup<DoryProverSetup> for PedersenSetup<Bn254G1> {
    fn derive(source: &DoryProverSetup, capacity: usize) -> Self {
        assert!(
            capacity <= source.0.g1_vec.len(),
            "Pedersen capacity ({}) exceeds Dory SRS size ({})",
            capacity,
            source.0.g1_vec.len(),
        );
        let generators = ark_to_jolt_g1_vec(source.0.g1_vec[..capacity].to_vec());
        let blinding = ark_to_jolt_g1(source.0.h1);
        PedersenSetup::new(generators, blinding)
    }
}

impl Commitment for DoryScheme {
    type Output = DoryCommitment;
}

impl CommitmentSchemeVerifier for DoryScheme {
    type Field = Fr;
    type Proof = DoryProof;
    type BatchProof = Vec<DoryProof>;
    type VerifierSetup = DoryVerifierSetup;

    #[tracing::instrument(skip_all, name = "DoryScheme::verify")]
    fn verify(
        commitment: &Self::Output,
        point: &[Fr],
        eval: Fr,
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError> {
        let mut dory_transcript = JoltToDoryTranscript::new(transcript);
        Self::verify_with_transcript(commitment, point, eval, proof, setup, &mut dory_transcript)
    }

    fn verify_batch(
        claims: Vec<OpeningClaim<Self::Field, Self>>,
        proof: &Self::BatchProof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError> {
        homomorphic_verify_batch::<Self, _>(claims, proof, setup, transcript)
    }

    fn bind_opening_inputs(
        transcript: &mut impl Transcript<Challenge = Self::Field>,
        point: &[Self::Field],
        eval: &Self::Field,
    ) {
        transcript.append(&LabelWithCount(b"dory_opening_point", point.len() as u64));
        for p in point {
            p.append_to_transcript(transcript);
        }
        transcript.append(&Label(b"dory_opening_eval"));
        eval.append_to_transcript(transcript);
    }
}

impl PublicVerifierSetup for DoryScheme {
    type PublicParams = usize;

    fn verifier_setup(max_num_vars: Self::PublicParams) -> DoryVerifierSetup {
        Self::setup_verifier(max_num_vars)
    }
}

impl CommitmentScheme for DoryScheme {
    type ProverSetup = DoryProverSetup;
    type OpeningHint = DoryHint;
    type SetupParams = usize;

    fn setup(max_num_vars: Self::SetupParams) -> (DoryProverSetup, DoryVerifierSetup) {
        let prover = Self::setup_prover(max_num_vars);
        let verifier = Self::project_verifier_setup(&prover);
        (prover, verifier)
    }

    fn project_verifier_setup(prover_setup: &DoryProverSetup) -> DoryVerifierSetup {
        DoryVerifierSetup(prover_setup.0.to_verifier_setup())
    }

    #[tracing::instrument(skip_all, name = "DoryScheme::commit")]
    fn commit<S: CommitmentSource<Fr> + ?Sized>(
        source: &S,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint) {
        Self::commit_with_mode::<S, Transparent>(source, &setup.0)
    }

    #[tracing::instrument(skip_all, name = "DoryScheme::commit_batch")]
    fn commit_batch<B: BatchCommitmentSource<Self::Field>>(
        batch: &B,
        ids: &[B::Id],
        setup: &Self::ProverSetup,
    ) -> Vec<(Self::Output, Self::OpeningHint)> {
        Self::commit_batch_with_mode::<B, Transparent>(batch, ids, &setup.0)
    }

    #[tracing::instrument(skip_all, name = "DoryScheme::open")]
    fn open<S>(
        poly: &S,
        point: &[Fr],
        _eval: Fr,
        setup: &Self::ProverSetup,
        hint: Option<Self::OpeningHint>,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Self::Proof
    where
        S: CommitmentSource<Self::Field> + ?Sized,
    {
        let hint = if let Some(hint) = hint {
            hint
        } else {
            let chunk_len = natural_or_balanced_chunk_len(poly);
            DoryHint::new(
                ark_to_jolt_g1_vec(compute_row_commitments(poly, chunk_len, &setup.0)),
                Fr::from_u64(0),
                chunk_len,
            )
        };
        debug_assert!(
            hint.commit_blind == Fr::from_u64(0),
            "commit_blind should be 0 for transparent mode"
        );
        let mut dory_transcript = JoltToDoryTranscript::new(transcript);
        Self::open_source_with_hint(poly, point, setup, hint, &mut dory_transcript)
    }

    fn prove_batch<S>(
        claims: Vec<ProverClaim<Self::Field, S>>,
        hints: Vec<Self::OpeningHint>,
        setup: &Self::ProverSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Self::BatchProof
    where
        S: CommitmentSource<Self::Field>,
    {
        homomorphic_prove_batch::<Self, _, _>(claims, hints, setup, transcript)
    }
}

impl LinearOpeningSchemeVerifier for DoryScheme {
    fn verify_batch_opening<ClaimId, SourceIdT>(
        terms: Vec<VerifierBatchOpeningTerm<Self::Field, Self, ClaimId, SourceIdT>>,
        proof: &Self::BatchProof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<BatchOpeningPublic<Self::Field, (), ClaimId>, OpeningsError>
    where
        SourceIdT: SourceId,
    {
        Self::verify_source_backed_batch_with_mode::<_, _, _, TransparentBatchOpening>(
            terms, proof, setup, transcript,
        )
    }
}

impl LinearOpeningScheme for DoryScheme {
    fn prove_batch_opening<B, ClaimId>(
        terms: Vec<ProverBatchOpeningTerm<Self::Field, ClaimId, B::Id>>,
        source_batch: &mut B,
        setup: &Self::ProverSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> BatchOpeningProverResult<Self, ClaimId>
    where
        B: LinearCombinationOpeningSource<Self::Field, Self::OpeningHint>,
    {
        let (proof, public, _joint_claim, _blind) =
            Self::prove_source_backed_batch_with_mode::<_, _, _, TransparentBatchOpening>(
                terms,
                source_batch,
                setup,
                transcript,
            );
        BatchOpeningProverResult {
            proof: vec![proof],
            public,
        }
    }
}

impl AdditivelyHomomorphicVerifier for DoryScheme {
    #[tracing::instrument(skip_all, name = "DoryScheme::combine")]
    fn combine(commitments: &[Self::Output], scalars: &[Self::Field]) -> Self::Output {
        assert_eq!(commitments.len(), scalars.len());

        let combined = commitments
            .par_iter()
            .zip(scalars.par_iter())
            .map(|(c, s)| jolt_fr_to_ark(s) * jolt_gt_to_ark(&c.0))
            .reduce(ArkGT::identity, |acc, x| acc + x);

        DoryCommitment(ark_to_jolt_gt(&combined))
    }
}

impl AdditivelyHomomorphic for DoryScheme {
    #[tracing::instrument(skip_all, name = "DoryScheme::combine_hints")]
    fn combine_hints(hints: Vec<Self::OpeningHint>, scalars: &[Self::Field]) -> Self::OpeningHint {
        let hint_refs: Vec<&DoryHint> = hints.iter().collect();
        combine_hint_refs(&hint_refs, scalars)
    }
}

impl ZkOpeningSchemeVerifier for DoryScheme {
    type HidingCommitment = Bn254G1;

    #[tracing::instrument(skip_all, name = "DoryScheme::verify_zk")]
    fn verify_zk(
        commitment: &Self::Output,
        point: &[Fr],
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError> {
        let mut dory_transcript = JoltToDoryTranscript::new(transcript);
        Self::verify_zk_with_transcript(commitment, point, proof, setup, &mut dory_transcript)
    }

    fn verify_batch_zk(
        claims: Vec<OpeningClaim<Self::Field, Self>>,
        proof: &Self::BatchProof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError> {
        let [claim] = claims.as_slice() else {
            return Err(OpeningsError::VerificationFailed);
        };
        let [proof] = proof.as_slice() else {
            return Err(OpeningsError::VerificationFailed);
        };
        Self::verify_zk(&claim.commitment, &claim.point, proof, setup, transcript)
    }

    fn bind_zk_opening_inputs(
        transcript: &mut impl Transcript<Challenge = Self::Field>,
        point: &[Self::Field],
        hiding_commitment: &Self::HidingCommitment,
    ) {
        transcript.append(&LabelWithCount(b"dory_opening_point", point.len() as u64));
        for p in point {
            p.append_to_transcript(transcript);
        }
        transcript.append(&Label(b"dory_eval_commitment"));
        hiding_commitment.append_to_transcript(transcript);
    }
}

impl ZkOpeningScheme for DoryScheme {
    type Blind = Fr;

    fn commit_zk<S: CommitmentSource<Fr> + ?Sized>(
        source: &S,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint) {
        Self::commit_with_mode::<S, ZK>(source, &setup.0)
    }

    #[tracing::instrument(skip_all, name = "DoryScheme::commit_batch_zk")]
    fn commit_batch_zk<B: BatchCommitmentSource<Self::Field>>(
        batch: &B,
        ids: &[B::Id],
        setup: &Self::ProverSetup,
    ) -> Vec<(Self::Output, Self::OpeningHint)> {
        Self::commit_batch_with_mode::<B, ZK>(batch, ids, &setup.0)
    }

    #[tracing::instrument(skip_all, name = "DoryScheme::open_zk")]
    fn open_zk<S>(
        poly: &S,
        point: &[Fr],
        _eval: Fr,
        setup: &Self::ProverSetup,
        hint: Self::OpeningHint,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> (Self::Proof, Self::HidingCommitment, Self::Blind)
    where
        S: CommitmentSource<Self::Field> + ?Sized,
    {
        let mut dory_transcript = JoltToDoryTranscript::new(transcript);
        Self::open_zk_source_with_hint(poly, point, setup, hint, &mut dory_transcript)
    }

    fn prove_batch_zk<S>(
        claims: Vec<ProverClaim<Self::Field, S>>,
        hints: Vec<Self::OpeningHint>,
        setup: &Self::ProverSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> (Self::BatchProof, Self::HidingCommitment, Self::Blind)
    where
        S: CommitmentSource<Self::Field>,
    {
        let [claim] = claims.as_slice() else {
            panic!("Dory ZK batch opening expects one already-combined claim");
        };
        let [hint] = hints.as_slice() else {
            panic!("Dory ZK batch opening expects one already-combined hint");
        };
        let (proof, y_com, y_blinding) = Self::open_zk(
            &claim.polynomial,
            &claim.point,
            claim.eval,
            setup,
            hint.clone(),
            transcript,
        );
        (vec![proof], y_com, y_blinding)
    }
}

impl ZkLinearOpeningSchemeVerifier for DoryScheme {
    fn verify_batch_opening_zk<ClaimId, SourceIdT>(
        terms: Vec<VerifierBatchOpeningTerm<Self::Field, Self, ClaimId, SourceIdT>>,
        proof: &Self::BatchProof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<BatchOpeningPublic<Self::Field, Self::HidingCommitment, ClaimId>, OpeningsError>
    where
        SourceIdT: SourceId,
    {
        Self::verify_source_backed_batch_with_mode::<_, _, _, ZkBatchOpening>(
            terms, proof, setup, transcript,
        )
    }
}

impl ZkLinearOpeningScheme for DoryScheme {
    fn prove_batch_opening_zk<B, ClaimId>(
        terms: Vec<ProverBatchOpeningTerm<Self::Field, ClaimId, B::Id>>,
        source_batch: &mut B,
        setup: &Self::ProverSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> ZkBatchOpeningProverResult<Self, ClaimId>
    where
        B: LinearCombinationOpeningSource<Self::Field, Self::OpeningHint>,
    {
        let (proof, public, joint_claim, y_blinding) =
            Self::prove_source_backed_batch_with_mode::<_, _, _, ZkBatchOpening>(
                terms,
                source_batch,
                setup,
                transcript,
            );
        let y_blinding = y_blinding.expect("ZK source-backed Dory proof must return y_blinding");
        ZkBatchOpeningProverResult {
            proof: vec![proof],
            public,
            witness: ZkBatchOpeningWitness {
                output_values: vec![joint_claim],
                output_blinds: vec![y_blinding],
            },
        }
    }
}

impl EvaluationCommitmentScheme<Bn254G1> for DoryScheme {
    fn batch_eval_commitment(proof: &Self::BatchProof) -> Option<Bn254G1> {
        let [proof] = proof.as_slice() else {
            return None;
        };
        proof.0.y_com.as_ref().copied().map(ark_to_jolt_g1)
    }

    fn eval_commitment_gens_verifier(setup: &Self::VerifierSetup) -> Option<(Bn254G1, Bn254G1)> {
        Some((ark_to_jolt_g1(setup.0.g1_0), ark_to_jolt_g1(setup.0.h1)))
    }
}

impl EvaluationCommitmentProver<Bn254G1> for DoryScheme {
    fn eval_commitment_gens(setup: &Self::ProverSetup) -> Option<(Bn254G1, Bn254G1)> {
        let g1_0 = setup.0.g1_vec.first().copied().map(ark_to_jolt_g1)?;
        Some((g1_0, ark_to_jolt_g1(setup.0.h1)))
    }

    fn zk_generators(setup: &Self::ProverSetup, count: usize) -> Option<(Vec<Bn254G1>, Bn254G1)> {
        let count = count.min(setup.0.g1_vec.len());
        let g1s = ark_to_jolt_g1_vec(setup.0.g1_vec[..count].to_vec());
        Some((g1s, ark_to_jolt_g1(setup.0.h1)))
    }
}

enum DoryChunkCommitment {
    Dense(ArkG1),
    OneHot(Vec<ArkG1>),
}

struct CommitRowContext<'a> {
    setup: &'a ArkworksProverSetup,
    g1_bases_affine: Vec<G1Affine>,
}

impl<'a> CommitRowContext<'a> {
    fn new(setup: &'a ArkworksProverSetup, row_len: usize) -> Self {
        Self {
            setup,
            g1_bases_affine: g1_bases_affine(setup, row_len),
        }
    }
}

/// Dense commit: full MSM per row, parallel over rows.
fn commit_rows_dense<S: CommitmentSource<Fr> + ?Sized>(
    source: &S,
    chunk_len: usize,
    setup: &ArkworksProverSetup,
) -> Vec<ArkG1> {
    let ctx = CommitRowContext::new(setup, chunk_len);

    let chunks = source.map_rows(chunk_len, |_, row| commit_source_row(row, &ctx));
    flatten_chunks(chunks)
}

fn commit_field_row(values: &[Fr], setup: &ArkworksProverSetup) -> ArkG1 {
    assert!(
        values.len() <= setup.g1_vec.len(),
        "Dory row length ({}) exceeds G1 SRS size ({})",
        values.len(),
        setup.g1_vec.len(),
    );
    let scalars: Vec<ArkFr> = values.iter().map(jolt_fr_to_ark).collect();
    JoltG1Routines::msm(&setup.g1_vec[..scalars.len()], &scalars)
}

fn strided_affine_bases(
    values_len: usize,
    column_stride: usize,
    ctx: &CommitRowContext<'_>,
) -> Vec<G1Affine> {
    assert!(
        column_stride > 0,
        "Dory strided row column stride must be nonzero"
    );
    assert!(
        values_len == 0 || (values_len - 1) * column_stride < ctx.g1_bases_affine.len(),
        "Dory strided row length ({values_len}) with stride ({column_stride}) exceeds G1 SRS row size ({})",
        ctx.g1_bases_affine.len(),
    );
    ctx.g1_bases_affine
        .iter()
        .step_by(column_stride)
        .take(values_len)
        .copied()
        .collect()
}

fn commit_strided_field_row(
    values: &[Fr],
    column_stride: usize,
    ctx: &CommitRowContext<'_>,
) -> ArkG1 {
    assert!(
        column_stride > 0,
        "Dory strided row column stride must be nonzero"
    );
    assert!(
        values.is_empty() || (values.len() - 1) * column_stride < ctx.setup.g1_vec.len(),
        "Dory strided row length ({}) with stride ({column_stride}) exceeds G1 SRS size ({})",
        values.len(),
        ctx.setup.g1_vec.len(),
    );
    let bases: Vec<ArkG1Struct> = ctx
        .setup
        .g1_vec
        .iter()
        .step_by(column_stride)
        .take(values.len())
        .copied()
        .collect();
    let scalars: Vec<ArkFr> = values.iter().map(jolt_fr_to_ark).collect();
    JoltG1Routines::msm(&bases, &scalars)
}

fn commit_i128_row(values: &[i128], ctx: &CommitRowContext<'_>) -> ArkG1 {
    assert!(
        values.len() <= ctx.g1_bases_affine.len(),
        "Dory row length ({}) exceeds G1 SRS size ({})",
        values.len(),
        ctx.g1_bases_affine.len(),
    );
    ArkG1Struct(msm_i128::<G1Projective>(
        &ctx.g1_bases_affine[..values.len()],
        values,
        true,
    ))
}

fn commit_strided_i128_row(
    values: &[i128],
    column_stride: usize,
    ctx: &CommitRowContext<'_>,
) -> ArkG1 {
    let bases = strided_affine_bases(values.len(), column_stride, ctx);
    ArkG1Struct(msm_i128::<G1Projective>(&bases, values, true))
}

fn commit_u64_row(values: &[u64], ctx: &CommitRowContext<'_>) -> ArkG1 {
    assert!(
        values.len() <= ctx.g1_bases_affine.len(),
        "Dory row length ({}) exceeds G1 SRS size ({})",
        values.len(),
        ctx.g1_bases_affine.len(),
    );
    ArkG1Struct(msm_u64::<G1Projective>(
        &ctx.g1_bases_affine[..values.len()],
        values,
        true,
    ))
}

fn commit_strided_u64_row(
    values: &[u64],
    column_stride: usize,
    ctx: &CommitRowContext<'_>,
) -> ArkG1 {
    let bases = strided_affine_bases(values.len(), column_stride, ctx);
    ArkG1Struct(msm_u64::<G1Projective>(&bases, values, true))
}

/// One-hot commit: O(T) group additions for unit-valued one-hot polynomials.
fn commit_rows_one_hot<S: CommitmentSource<Fr> + ?Sized>(
    source: &S,
    num_rows: usize,
    num_cols: usize,
    setup: &ArkworksProverSetup,
) -> Vec<ArkG1> {
    let g1_bases = &setup.g1_vec[..num_cols];

    let mut cols_per_row: Vec<Vec<usize>> = vec![Vec::new(); num_rows];
    source.for_each_one(|flat_idx: usize| {
        let row = flat_idx / num_cols;
        let col = flat_idx % num_cols;
        debug_assert!(
            row < num_rows && col < num_cols,
            "for_each_one out-of-bounds flat_idx: row={row} num_rows={num_rows} col={col} num_cols={num_cols}",
        );
        cols_per_row[row].push(col);
    });

    cols_per_row
        .par_iter()
        .map(|cols| {
            cols.iter()
                .fold(<InnerBN254 as PairingCurve>::G1::identity(), |acc, &col| {
                    <InnerBN254 as PairingCurve>::G1::add(&acc, &g1_bases[col])
                })
        })
        .collect()
}

fn commit_one_hot_row(row: OneHotRow<'_>, ctx: &CommitRowContext<'_>) -> Vec<ArkG1> {
    let k = 1usize << row.log_domain_size;
    let num_columns = match row.entries {
        OneHotEntries::OnePerColumn(indices) => indices.len(),
        OneHotEntries::MaybeZero(indices) => indices.len(),
    };
    assert!(
        num_columns <= ctx.g1_bases_affine.len(),
        "Dory one-hot row length ({}) exceeds G1 SRS size ({})",
        num_columns,
        ctx.g1_bases_affine.len(),
    );

    let mut columns_by_hot_index: Vec<Vec<usize>> = vec![Vec::new(); k];
    match row.entries {
        OneHotEntries::OnePerColumn(indices) => {
            for (column, hot_index) in indices.iter().enumerate() {
                columns_by_hot_index[hot_index.get()].push(column);
            }
        }
        OneHotEntries::MaybeZero(indices) => {
            for (column, hot_index) in indices.iter().enumerate() {
                if let Some(hot_index) = hot_index {
                    columns_by_hot_index[hot_index.get()].push(column);
                }
            }
        }
    }

    batch_g1_additions_multi(&ctx.g1_bases_affine[..num_columns], &columns_by_hot_index)
        .into_iter()
        .map(|affine| ArkG1Struct(affine.into()))
        .collect()
}

fn g1_bases_affine(setup: &ArkworksProverSetup, len: usize) -> Vec<G1Affine> {
    setup.g1_vec[..len]
        .par_iter()
        .map(|base| base.0.into_affine())
        .collect()
}

fn commit_source_row(row: SourceRow<'_, Fr>, ctx: &CommitRowContext<'_>) -> DoryChunkCommitment {
    match row {
        SourceRow::FieldElements(values) => {
            DoryChunkCommitment::Dense(commit_field_row(values, ctx.setup))
        }
        SourceRow::StridedFieldElements {
            values,
            column_stride,
        } => DoryChunkCommitment::Dense(commit_strided_field_row(values, column_stride, ctx)),
        SourceRow::I128(values) => DoryChunkCommitment::Dense(commit_i128_row(values, ctx)),
        SourceRow::StridedI128 {
            values,
            column_stride,
        } => DoryChunkCommitment::Dense(commit_strided_i128_row(values, column_stride, ctx)),
        SourceRow::U64(values) => DoryChunkCommitment::Dense(commit_u64_row(values, ctx)),
        SourceRow::StridedU64 {
            values,
            column_stride,
        } => DoryChunkCommitment::Dense(commit_strided_u64_row(values, column_stride, ctx)),
        SourceRow::OneHot(row) => DoryChunkCommitment::OneHot(commit_one_hot_row(row, ctx)),
    }
}

fn flatten_chunks(chunks: Vec<DoryChunkCommitment>) -> Vec<ArkG1> {
    let Some(first) = chunks.first() else {
        return Vec::new();
    };

    match first {
        DoryChunkCommitment::Dense(_) => chunks
            .into_iter()
            .map(|chunk| match chunk {
                DoryChunkCommitment::Dense(row_commitment) => row_commitment,
                DoryChunkCommitment::OneHot(_) => {
                    panic!("source mixed dense and one-hot rows during commitment");
                }
            })
            .collect(),
        DoryChunkCommitment::OneHot(first) => {
            let rows_per_hot_index = chunks.len();
            let k = first.len();
            let mut row_commitments =
                vec![<InnerBN254 as PairingCurve>::G1::identity(); rows_per_hot_index * k];
            for (chunk_index, chunk) in chunks.into_iter().enumerate() {
                match chunk {
                    DoryChunkCommitment::OneHot(commitments) => {
                        assert_eq!(
                            commitments.len(),
                            k,
                            "source changed one-hot domain size during commitment",
                        );
                        for (hot_index, row_commitment) in commitments.into_iter().enumerate() {
                            row_commitments[chunk_index + hot_index * rows_per_hot_index] =
                                row_commitment;
                        }
                    }
                    DoryChunkCommitment::Dense(_) => {
                        panic!("source mixed dense and one-hot rows during commitment");
                    }
                }
            }
            row_commitments
        }
    }
}

fn aggregate_batch_chunks<M: Mode>(
    chunks: Vec<DoryChunkCommitment>,
    chunk_len: usize,
    setup: &ArkworksProverSetup,
) -> (DoryCommitment, DoryHint) {
    assert!(!chunks.is_empty(), "cannot aggregate an empty source");

    match &chunks[0] {
        DoryChunkCommitment::Dense(_) => {
            let mut row_commitments = Vec::with_capacity(chunks.len());
            for chunk in chunks {
                match chunk {
                    DoryChunkCommitment::Dense(row_commitment) => {
                        row_commitments.push(row_commitment);
                    }
                    DoryChunkCommitment::OneHot(_) => {
                        panic!("batch source mixed dense and one-hot rows for one source");
                    }
                }
            }
            finish_row_commitments::<M>(row_commitments, chunk_len, setup)
        }
        DoryChunkCommitment::OneHot(first) => {
            let rows_per_hot_index = chunks.len();
            let k = first.len();
            let mut row_commitments =
                vec![<InnerBN254 as PairingCurve>::G1::identity(); rows_per_hot_index * k];

            let chunks: Vec<Vec<ArkG1>> = chunks
                .into_iter()
                .map(|chunk| match chunk {
                    DoryChunkCommitment::OneHot(commitments) => {
                        assert_eq!(
                            commitments.len(),
                            k,
                            "batch source changed one-hot domain size within one source",
                        );
                        commitments
                    }
                    DoryChunkCommitment::Dense(_) => {
                        panic!("batch source mixed dense and one-hot rows for one source");
                    }
                })
                .collect();

            for (chunk_index, commitments) in chunks.iter().enumerate() {
                row_commitments
                    .par_iter_mut()
                    .skip(chunk_index)
                    .step_by(rows_per_hot_index)
                    .zip(commitments.par_iter())
                    .for_each(|(dest, src)| *dest = *src);
            }
            finish_row_commitments::<M>(row_commitments, chunk_len, setup)
        }
    }
}

fn finish_row_commitments<M: Mode>(
    row_commitments: Vec<ArkG1>,
    chunk_len: usize,
    setup: &ArkworksProverSetup,
) -> (DoryCommitment, DoryHint) {
    let (tier_2, commit_blind) = commit_rows_tier_2::<M>(&row_commitments, setup);
    (
        DoryCommitment(ark_to_jolt_gt(&tier_2)),
        DoryHint::new(
            ark_to_jolt_g1_vec(row_commitments),
            ark_to_jolt_fr(&commit_blind),
            chunk_len,
        ),
    )
}

fn balanced_chunk_len(num_vars: usize) -> usize {
    1usize << num_vars.div_ceil(2)
}

fn validate_chunk_len(chunk_len: usize) {
    assert!(
        chunk_len.is_power_of_two(),
        "Dory commitment chunk length ({chunk_len}) must be a power of two",
    );
}

fn natural_or_balanced_chunk_len<S: CommitmentSource<Fr> + ?Sized>(source: &S) -> usize {
    let chunk_len = source
        .natural_chunk_len()
        .unwrap_or_else(|| balanced_chunk_len(source.num_vars()));
    validate_chunk_len(chunk_len);
    chunk_len
}

fn hint_shape(hint: &DoryHint) -> (usize, usize) {
    validate_chunk_len(hint.chunk_len);
    assert!(
        hint.row_commitments.len().is_power_of_two(),
        "Dory hint row count ({}) must be a power of two",
        hint.row_commitments.len(),
    );
    let sigma = hint.chunk_len.trailing_zeros() as usize;
    let nu = hint.row_commitments.len().trailing_zeros() as usize;
    (nu, sigma)
}

fn compute_row_commitments<S: CommitmentSource<Fr> + ?Sized>(
    source: &S,
    chunk_len: usize,
    setup: &ArkworksProverSetup,
) -> Vec<ArkG1> {
    let num_vars = source.num_vars();
    let sigma = chunk_len.trailing_zeros() as usize;
    let num_rows = 1usize << num_vars.saturating_sub(sigma);
    if source.is_one_hot() {
        commit_rows_one_hot(source, num_rows, chunk_len, setup)
    } else {
        commit_rows_dense(source, chunk_len, setup)
    }
}

pub(crate) fn commit_rows_tier_2<M: Mode>(
    row_commitments: &[ArkG1],
    setup: &ArkworksProverSetup,
) -> (ArkGT, ArkFr) {
    let g2_bases = &setup.g2_vec[..row_commitments.len()];
    let tier_2 = <InnerBN254 as PairingCurve>::multi_pair_g2_setup(row_commitments, g2_bases);
    let commit_blind = M::sample::<ArkFr>();
    let tier_2 = M::mask(tier_2, &setup.ht, &commit_blind);
    (tier_2, commit_blind)
}

impl DoryHint {
    fn into_ark_parts(self) -> (Vec<ArkG1>, ArkFr) {
        (
            jolt_g1_vec_to_ark(self.row_commitments),
            jolt_fr_to_ark(&self.commit_blind),
        )
    }
}

/// Adapts [`CommitmentSource<Fr>`] to dory-pcs's polynomial traits
/// without materializing the full evaluation table.
struct DorySourceAdapter<'a, S: CommitmentSource<Fr> + ?Sized> {
    source: &'a S,
}

impl<'a, S: CommitmentSource<Fr> + ?Sized> DorySourceAdapter<'a, S> {
    fn new(source: &'a S) -> Self {
        Self { source }
    }
}

impl<S: CommitmentSource<Fr> + ?Sized> DoryPolynomial<ArkFr> for DorySourceAdapter<'_, S> {
    fn num_vars(&self) -> usize {
        self.source.num_vars()
    }

    fn evaluate(&self, point: &[ArkFr]) -> ArkFr {
        let native_point: Vec<Fr> = point.iter().rev().map(ark_to_jolt_fr).collect();
        jolt_fr_to_ark(&self.source.evaluate(&native_point))
    }

    fn commit<E, Mo, M1>(
        &self,
        _nu: usize,
        _sigma: usize,
        _setup: &DoryNativeProverSetup<E>,
    ) -> Result<(E::GT, Vec<E::G1>, ArkFr), DoryError>
    where
        E: PairingCurve,
        Mo: DoryMode,
        M1: DoryRoutines<E::G1>,
        E::G1: DoryGroup<Scalar = ArkFr>,
    {
        unimplemented!(
            "DoryScheme pre-computes row commitments before invoking dory::prove; \
             dory::Polynomial::commit on this adapter is not exercised"
        )
    }
}

impl<S: CommitmentSource<Fr> + ?Sized> MultilinearLagrange<ArkFr> for DorySourceAdapter<'_, S> {
    fn vector_matrix_product(&self, left_vec: &[ArkFr], _nu: usize, sigma: usize) -> Vec<ArkFr> {
        let native_left: Vec<Fr> = left_vec.iter().map(ark_to_jolt_fr).collect();
        let result = self.source.fold_rows(&native_left, 1usize << sigma);
        result.iter().map(jolt_fr_to_ark).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_crypto::{Bn254, JoltGroup, Pedersen, VectorCommitment};
    use jolt_field::{FromPrimitiveInt, RandomSampling};
    use jolt_openings::{
        BatchOpeningPoint, BatchOpeningSource, LinearCombinationOpeningSource,
        MaterializedLinearCombination, ProverBatchOpeningTerm, SourceRow, VerifierBatchOpeningTerm,
    };
    use jolt_poly::{MultilinearPoly, Polynomial};
    use jolt_transcript::Blake2bTranscript;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    struct FoldOnlySource {
        poly: Polynomial<Fr>,
    }

    struct U64Source {
        evaluations: Vec<u64>,
    }

    struct StridedU64Source {
        rows: Vec<Vec<u64>>,
        dense: Polynomial<Fr>,
        column_stride: usize,
    }

    struct TestOpeningBatch {
        polynomials: Vec<Polynomial<Fr>>,
        hints: Vec<DoryHint>,
    }

    impl BatchOpeningSource<Fr, DoryHint> for TestOpeningBatch {
        type Id = usize;
        type Source<'a>
            = &'a Polynomial<Fr>
        where
            Self: 'a;

        fn source(&self, id: Self::Id) -> Self::Source<'_> {
            &self.polynomials[id]
        }

        fn opening_hint(&self, id: Self::Id) -> &DoryHint {
            &self.hints[id]
        }
    }

    impl LinearCombinationOpeningSource<Fr, DoryHint> for TestOpeningBatch {
        type LinearCombination<'a>
            = MaterializedLinearCombination<Fr>
        where
            Self: 'a;

        fn linear_combination<'a>(
            &'a mut self,
            terms: &[LinearSourceTerm<Fr, Self::Id>],
        ) -> Self::LinearCombination<'a> {
            MaterializedLinearCombination::new(self, terms)
        }
    }

    impl U64Source {
        fn field_evaluation(&self, point: &[Fr]) -> Fr {
            let dense: Vec<Fr> = self
                .evaluations
                .iter()
                .map(|&value| Fr::from_u64(value))
                .collect();
            MultilinearPoly::evaluate(&dense, point)
        }
    }

    impl CommitmentSource<Fr> for U64Source {
        fn num_vars(&self) -> usize {
            self.evaluations.len().ilog2() as usize
        }

        fn evaluate(&self, point: &[Fr]) -> Fr {
            self.field_evaluation(point)
        }

        fn for_each_row<V>(&self, chunk_len: usize, mut visit: V)
        where
            V: for<'row> FnMut(usize, SourceRow<'row, Fr>),
        {
            for (row_index, row) in self.evaluations.chunks(chunk_len).enumerate() {
                visit(row_index, SourceRow::U64(row));
            }
        }

        fn fold_rows(&self, left: &[Fr], chunk_len: usize) -> Vec<Fr> {
            let mut result = vec![Fr::from_u64(0); chunk_len];
            for (row_index, row) in self.evaluations.chunks(chunk_len).enumerate() {
                let weight = left[row_index];
                for (dest, &value) in result.iter_mut().zip(row) {
                    *dest += Fr::from_u64(value) * weight;
                }
            }
            result
        }
    }

    impl CommitmentSource<Fr> for StridedU64Source {
        fn num_vars(&self) -> usize {
            self.dense.num_vars()
        }

        fn evaluate(&self, point: &[Fr]) -> Fr {
            self.dense.evaluate(point)
        }

        fn natural_chunk_len(&self) -> Option<usize> {
            Some(self.rows[0].len() * self.column_stride)
        }

        fn for_each_row<V>(&self, _chunk_len: usize, mut visit: V)
        where
            V: for<'row> FnMut(usize, SourceRow<'row, Fr>),
        {
            for (row_index, row) in self.rows.iter().enumerate() {
                visit(
                    row_index,
                    SourceRow::StridedU64 {
                        values: row,
                        column_stride: self.column_stride,
                    },
                );
            }
        }

        fn fold_rows(&self, left: &[Fr], chunk_len: usize) -> Vec<Fr> {
            let sigma = chunk_len.trailing_zeros() as usize;
            MultilinearPoly::fold_rows(&self.dense, left, sigma)
        }
    }

    impl CommitmentSource<Fr> for FoldOnlySource {
        fn num_vars(&self) -> usize {
            self.poly.num_vars()
        }

        fn evaluate(&self, point: &[Fr]) -> Fr {
            self.poly.evaluate(point)
        }

        fn for_each_row<V>(&self, _chunk_len: usize, _visit: V)
        where
            V: for<'row> FnMut(usize, SourceRow<'row, Fr>),
        {
            panic!("single-claim prove_batch must not materialize source rows")
        }

        fn fold_rows(&self, left: &[Fr], chunk_len: usize) -> Vec<Fr> {
            let sigma = chunk_len.trailing_zeros() as usize;
            MultilinearPoly::fold_rows(&self.poly, left, sigma)
        }
    }

    #[test]
    fn commit_open_verify_round_trip() {
        let num_vars = 4;
        let mut rng = ChaCha20Rng::seed_from_u64(42);

        let prover_setup = DoryScheme::setup_prover(num_vars);
        let verifier_setup = DoryVerifierSetup(prover_setup.0.to_verifier_setup());

        let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
        let point: Vec<Fr> = (0..num_vars)
            .map(|_| <Fr as RandomSampling>::random(&mut rng))
            .collect();
        let eval = poly.evaluate(&point);

        let (commitment, hint) = DoryScheme::commit(poly.evaluations(), &prover_setup);

        let mut prove_transcript = Blake2bTranscript::new(b"test");
        let proof = DoryScheme::open(
            &poly,
            &point,
            eval,
            &prover_setup,
            Some(hint),
            &mut prove_transcript,
        );

        let mut verify_transcript = Blake2bTranscript::new(b"test");
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
    fn u64_source_commit_open_verify_round_trip() {
        let num_vars = 4;
        let prover_setup = DoryScheme::setup_prover(num_vars);
        let verifier_setup = DoryVerifierSetup(prover_setup.0.to_verifier_setup());

        let source = U64Source {
            evaluations: (0..(1 << num_vars)).map(|value| value as u64).collect(),
        };
        let point: Vec<Fr> = (0..num_vars)
            .map(|idx| Fr::from_u64((idx + 2) as u64))
            .collect();
        let eval = source.evaluate(&point);

        let (commitment, hint) = DoryScheme::commit(&source, &prover_setup);
        let mut prove_transcript = Blake2bTranscript::new(b"test-u64");
        let proof = DoryScheme::open(
            &source,
            &point,
            eval,
            &prover_setup,
            Some(hint),
            &mut prove_transcript,
        );

        let mut verify_transcript = Blake2bTranscript::new(b"test-u64");
        let result = DoryScheme::verify(
            &commitment,
            &point,
            eval,
            &proof,
            &verifier_setup,
            &mut verify_transcript,
        );

        assert!(result.is_ok());
    }

    #[test]
    fn strided_u64_commit_matches_dense_zero_padded_rows() {
        let prover_setup = DoryScheme::setup_prover(6);
        let mut dense = vec![Fr::from_u64(0); 16];
        dense[0] = Fr::from_u64(3);
        dense[4] = Fr::from_u64(5);
        dense[8] = Fr::from_u64(7);
        dense[12] = Fr::from_u64(11);

        let source = StridedU64Source {
            rows: vec![vec![3, 5], vec![7, 11]],
            dense: Polynomial::new(dense.clone()),
            column_stride: 4,
        };
        let dense_source = Polynomial::new(dense);

        let (strided_commitment, strided_hint) = DoryScheme::commit(&source, &prover_setup);
        let dense_wrapped = StridedU64Source {
            rows: vec![vec![3, 0, 0, 0, 5, 0, 0, 0], vec![7, 0, 0, 0, 11, 0, 0, 0]],
            dense: dense_source,
            column_stride: 1,
        };
        let (dense_commitment, dense_hint) = DoryScheme::commit(&dense_wrapped, &prover_setup);

        assert_eq!(strided_commitment, dense_commitment);
        assert_eq!(strided_hint.row_commitments, dense_hint.row_commitments);
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

        let combined = DoryScheme::combine(
            &[commit_a, commit_b],
            &[
                <Fr as FromPrimitiveInt>::from_u64(1),
                <Fr as FromPrimitiveInt>::from_u64(1),
            ],
        );

        assert_eq!(
            commit_sum_direct, combined,
            "combine([1,1]) must match commitment to sum"
        );
    }

    #[test]
    fn combine_hints_zero_pads_ragged_rows() {
        let g = Bn254::g1_generator();
        let h = g.scalar_mul(&Fr::from_u64(11));
        let k = g.scalar_mul(&Fr::from_u64(13));
        let a = Fr::from_u64(2);
        let b = Fr::from_u64(7);

        let hint_a = DoryHint::new(vec![g], Fr::from_u64(3), 4);
        let hint_b = DoryHint::new(vec![h, k], Fr::from_u64(5), 8);

        let combined = DoryScheme::combine_hints(vec![hint_a, hint_b], &[a, b]);

        assert_eq!(combined.row_commitments.len(), 2);
        assert_eq!(
            combined.row_commitments[0],
            g.scalar_mul(&a) + h.scalar_mul(&b)
        );
        assert_eq!(combined.row_commitments[1], k.scalar_mul(&b));
        assert_eq!(
            combined.commit_blind,
            a * Fr::from_u64(3) + b * Fr::from_u64(5),
            "combined hint blind must match the same linear combination"
        );
        assert_eq!(combined.chunk_len, 8);
    }

    #[test]
    fn generic_open_preserves_jolt_point_order() {
        let num_vars = 3;
        let prover_setup = DoryScheme::setup_prover(num_vars);
        let verifier_setup = DoryVerifierSetup(prover_setup.0.to_verifier_setup());

        // f(x0, x1, x2) = x0. Reversing the opening point evaluates x2
        // instead, so this catches accidental Dory/Jolt point-order swaps.
        let evals = vec![
            Fr::from_u64(0),
            Fr::from_u64(0),
            Fr::from_u64(0),
            Fr::from_u64(0),
            Fr::from_u64(1),
            Fr::from_u64(1),
            Fr::from_u64(1),
            Fr::from_u64(1),
        ];
        let poly = Polynomial::new(evals);
        let point = vec![Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(5)];
        let eval = poly.evaluate(&point);

        let (commitment, hint) = DoryScheme::commit(poly.evaluations(), &prover_setup);

        let mut prove_transcript = Blake2bTranscript::new(b"point-order");
        let proof = DoryScheme::open(
            &poly,
            &point,
            eval,
            &prover_setup,
            Some(hint),
            &mut prove_transcript,
        );

        let mut verify_transcript = Blake2bTranscript::new(b"point-order");
        let result = DoryScheme::verify(
            &commitment,
            &point,
            eval,
            &proof,
            &verifier_setup,
            &mut verify_transcript,
        );
        assert!(
            result.is_ok(),
            "generic Dory opening must evaluate at the caller's Jolt-order point"
        );
    }

    #[test]
    fn single_claim_prove_batch_opens_source_without_materializing_rows() {
        let num_vars = 3;
        let mut rng = ChaCha20Rng::seed_from_u64(414);
        let prover_setup = DoryScheme::setup_prover(num_vars);
        let verifier_setup = DoryVerifierSetup(prover_setup.0.to_verifier_setup());

        let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
        let source = FoldOnlySource { poly: poly.clone() };
        let point: Vec<Fr> = (0..num_vars)
            .map(|_| <Fr as RandomSampling>::random(&mut rng))
            .collect();
        let eval = poly.evaluate(&point);
        let (commitment, hint) = DoryScheme::commit(&poly, &prover_setup);

        let mut prove_transcript = Blake2bTranscript::new(b"single-batch");
        let proof = DoryScheme::prove_batch(
            vec![ProverClaim {
                polynomial: source,
                point: point.clone(),
                eval,
            }],
            vec![hint],
            &prover_setup,
            &mut prove_transcript,
        );

        let mut verify_transcript = Blake2bTranscript::new(b"single-batch");
        DoryScheme::verify_batch(
            vec![OpeningClaim {
                commitment: commitment.clone(),
                point: point.clone(),
                eval,
            }],
            &proof,
            &verifier_setup,
            &mut verify_transcript,
        )
        .expect("single-claim batch proof should verify");

        let [single_proof] = proof.as_slice() else {
            panic!("single-claim batch should contain one proof");
        };
        let mut direct_verify_transcript = Blake2bTranscript::new(b"single-batch");
        DoryScheme::verify(
            &commitment,
            &point,
            eval,
            single_proof,
            &verifier_setup,
            &mut direct_verify_transcript,
        )
        .expect("single-claim batch should use the raw single-opening transcript");
    }

    #[test]
    fn zk_open_verify_round_trip() {
        let num_vars = 4;
        let mut rng = ChaCha20Rng::seed_from_u64(600);

        let prover_setup = DoryScheme::setup_prover(num_vars);
        let verifier_setup = DoryVerifierSetup(prover_setup.0.to_verifier_setup());

        let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
        let point: Vec<Fr> = (0..num_vars)
            .map(|_| <Fr as RandomSampling>::random(&mut rng))
            .collect();
        let eval = poly.evaluate(&point);

        let (commitment, hint) =
            <DoryScheme as ZkOpeningScheme>::commit_zk(poly.evaluations(), &prover_setup);

        let mut prove_transcript = Blake2bTranscript::new(b"zk-test");
        let (proof, _eval_com, _blinding) = DoryScheme::open_zk(
            &poly,
            &point,
            eval,
            &prover_setup,
            hint,
            &mut prove_transcript,
        );

        let mut verify_transcript = Blake2bTranscript::new(b"zk-test");
        let result = DoryScheme::verify_zk(
            &commitment,
            &point,
            &proof,
            &verifier_setup,
            &mut verify_transcript,
        );
        assert!(result.is_ok(), "ZK verification failed: {result:?}");
    }

    #[test]
    fn zk_single_claim_batch_round_trip() {
        let num_vars = 3;
        let mut rng = ChaCha20Rng::seed_from_u64(601);

        let prover_setup = DoryScheme::setup_prover(num_vars);
        let verifier_setup = DoryScheme::project_verifier_setup(&prover_setup);

        let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
        let point: Vec<Fr> = (0..num_vars)
            .map(|_| <Fr as RandomSampling>::random(&mut rng))
            .collect();
        let eval = poly.evaluate(&point);
        let (commitment, hint) = DoryScheme::commit_zk(&poly, &prover_setup);

        let mut prove_transcript = Blake2bTranscript::new(b"zk-batch");
        let (proof, y_com, _blind) = DoryScheme::prove_batch_zk(
            vec![ProverClaim {
                polynomial: poly,
                point: point.clone(),
                eval,
            }],
            vec![hint],
            &prover_setup,
            &mut prove_transcript,
        );
        assert_eq!(
            DoryScheme::batch_eval_commitment(&proof),
            Some(y_com),
            "batch proof should expose the hidden evaluation commitment"
        );

        let mut verify_transcript = Blake2bTranscript::new(b"zk-batch");
        DoryScheme::verify_batch_zk(
            vec![OpeningClaim {
                commitment,
                point,
                eval,
            }],
            &proof,
            &verifier_setup,
            &mut verify_transcript,
        )
        .expect("ZK batch proof should verify");
    }

    #[test]
    fn source_backed_batch_round_trip() {
        let num_vars = 3;
        let mut rng = ChaCha20Rng::seed_from_u64(603);

        let prover_setup = DoryScheme::setup_prover(num_vars);
        let verifier_setup = DoryScheme::project_verifier_setup(&prover_setup);

        let p1 = Polynomial::<Fr>::random(num_vars, &mut rng);
        let p2 = Polynomial::<Fr>::random(num_vars, &mut rng);
        let point: Vec<Fr> = (0..num_vars)
            .map(|_| <Fr as RandomSampling>::random(&mut rng))
            .collect();
        let eval1 = p1.evaluate(&point);
        let eval2 = p2.evaluate(&point);

        let (c1, h1) = DoryScheme::commit(&p1, &prover_setup);
        let (c2, h2) = DoryScheme::commit(&p2, &prover_setup);
        let mut batch = TestOpeningBatch {
            polynomials: vec![p1, p2],
            hints: vec![h1, h2],
        };

        let prover_terms = vec![
            ProverBatchOpeningTerm {
                claim_id: 0u8,
                source_id: 0usize,
                point: BatchOpeningPoint::same(point.clone()),
                eval: eval1,
                eval_scale: Fr::from_u64(1),
            },
            ProverBatchOpeningTerm {
                claim_id: 1u8,
                source_id: 1usize,
                point: BatchOpeningPoint::same(point.clone()),
                eval: eval2,
                eval_scale: Fr::from_u64(1),
            },
        ];
        let verifier_terms = vec![
            VerifierBatchOpeningTerm::<Fr, DoryScheme, _, _> {
                claim_id: 0u8,
                source_id: 0usize,
                commitment: c1,
                point: BatchOpeningPoint::same(point.clone()),
                eval: eval1,
                eval_scale: Fr::from_u64(1),
            },
            VerifierBatchOpeningTerm::<Fr, DoryScheme, _, _> {
                claim_id: 1u8,
                source_id: 1usize,
                commitment: c2,
                point: BatchOpeningPoint::same(point),
                eval: eval2,
                eval_scale: Fr::from_u64(1),
            },
        ];

        let mut prove_transcript = Blake2bTranscript::new(b"source-backed-batch");
        let prover_result = DoryScheme::prove_batch_opening(
            prover_terms,
            &mut batch,
            &prover_setup,
            &mut prove_transcript,
        );

        let mut verify_transcript = Blake2bTranscript::new(b"source-backed-batch");
        let verifier_public = DoryScheme::verify_batch_opening(
            verifier_terms,
            &prover_result.proof,
            &verifier_setup,
            &mut verify_transcript,
        )
        .expect("source-backed Dory batch proof should verify");

        assert_eq!(prover_result.public, verifier_public);
        assert_eq!(verifier_public.outputs.len(), 1);
        assert_eq!(verifier_public.relations.len(), 1);
    }

    #[test]
    fn source_backed_zk_batch_round_trip() {
        let num_vars = 3;
        let mut rng = ChaCha20Rng::seed_from_u64(604);

        let prover_setup = DoryScheme::setup_prover(num_vars);
        let verifier_setup = DoryScheme::project_verifier_setup(&prover_setup);

        let p1 = Polynomial::<Fr>::random(num_vars, &mut rng);
        let p2 = Polynomial::<Fr>::random(num_vars, &mut rng);
        let point: Vec<Fr> = (0..num_vars)
            .map(|_| <Fr as RandomSampling>::random(&mut rng))
            .collect();
        let eval1 = p1.evaluate(&point);
        let eval2 = p2.evaluate(&point);

        let (c1, h1) = DoryScheme::commit_zk(&p1, &prover_setup);
        let (c2, h2) = DoryScheme::commit_zk(&p2, &prover_setup);
        let mut batch = TestOpeningBatch {
            polynomials: vec![p1, p2],
            hints: vec![h1, h2],
        };

        let prover_terms = vec![
            ProverBatchOpeningTerm {
                claim_id: 0u8,
                source_id: 0usize,
                point: BatchOpeningPoint::same(point.clone()),
                eval: eval1,
                eval_scale: Fr::from_u64(1),
            },
            ProverBatchOpeningTerm {
                claim_id: 1u8,
                source_id: 1usize,
                point: BatchOpeningPoint::same(point.clone()),
                eval: eval2,
                eval_scale: Fr::from_u64(1),
            },
        ];
        let verifier_terms = vec![
            VerifierBatchOpeningTerm::<Fr, DoryScheme, _, _> {
                claim_id: 0u8,
                source_id: 0usize,
                commitment: c1,
                point: BatchOpeningPoint::same(point.clone()),
                eval: eval1,
                eval_scale: Fr::from_u64(1),
            },
            VerifierBatchOpeningTerm::<Fr, DoryScheme, _, _> {
                claim_id: 1u8,
                source_id: 1usize,
                commitment: c2,
                point: BatchOpeningPoint::same(point),
                eval: eval2,
                eval_scale: Fr::from_u64(1),
            },
        ];

        let mut prove_transcript = Blake2bTranscript::new(b"source-backed-zk-batch");
        let prover_result = DoryScheme::prove_batch_opening_zk(
            prover_terms,
            &mut batch,
            &prover_setup,
            &mut prove_transcript,
        );

        let mut verify_transcript = Blake2bTranscript::new(b"source-backed-zk-batch");
        let verifier_public = DoryScheme::verify_batch_opening_zk(
            verifier_terms,
            &prover_result.proof,
            &verifier_setup,
            &mut verify_transcript,
        )
        .expect("source-backed ZK Dory batch proof should verify");

        assert_eq!(prover_result.public, verifier_public);
        assert_eq!(verifier_public.outputs.len(), 1);
        assert_eq!(verifier_public.relations.len(), 1);
        assert_eq!(prover_result.witness.output_values.len(), 1);
        assert_eq!(prover_result.witness.output_blinds.len(), 1);
    }

    #[test]
    fn extract_vc_setup_produces_valid_pedersen_setup() {
        let num_vars = 6;
        let prover_setup = DoryScheme::setup_prover(num_vars);

        let capacity = 5;
        let vc_setup = PedersenSetup::<Bn254G1>::derive(&prover_setup, capacity);

        assert_eq!(
            <Pedersen<Bn254G1> as VectorCommitment>::capacity(&vc_setup),
            capacity,
        );

        let values = vec![
            <Fr as FromPrimitiveInt>::from_u64(1),
            <Fr as FromPrimitiveInt>::from_u64(2),
            <Fr as FromPrimitiveInt>::from_u64(3),
        ];
        let blinding = <Fr as FromPrimitiveInt>::from_u64(42);
        let commitment =
            <Pedersen<Bn254G1> as VectorCommitment>::commit(&vc_setup, &values, &blinding);
        assert!(<Pedersen<Bn254G1> as VectorCommitment>::verify(
            &vc_setup,
            &commitment,
            &values,
            &blinding,
        ),);
    }
}
