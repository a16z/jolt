//! Dory PCS implementing the `jolt-openings` trait hierarchy.

#![expect(
    clippy::expect_used,
    clippy::panic,
    clippy::unimplemented,
    reason = "ZK proof y_com/y_blinding are Dory-mode invariants; dory::prove/verify errors are caller-precondition violations surfaced via panic; the dory adapter's commit is unreachable because DoryScheme pre-computes row commitments"
)]

use ark_ec::CurveGroup;
use dory::backends::arkworks::ArkworksProverSetup;
use dory::mode::Transparent;
use dory::primitives::arithmetic::{
    DoryRoutines, Field as DoryField, Group as DoryGroup, PairingCurve,
};
use dory::primitives::poly::{MultilinearLagrange, Polynomial as DoryPolynomial};
use dory::Mode;
use jolt_crypto::ec::bn254::batch_addition::batch_g1_additions_multi_affine;
use jolt_crypto::ec::bn254::glv;
use jolt_crypto::{Bn254G1, Bn254G2, Bn254GT, Commitment, DeriveSetup, JoltGroup, PedersenSetup};
use jolt_field::Fr;
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme, OpeningsError, ZkOpeningScheme};
use jolt_poly::{MultilinearPoly, OneHotIndexOrder};
use jolt_transcript::{AppendToTranscript, Label, LabelWithCount, Transcript};
use rayon::prelude::*;

use crate::transcript::JoltToDoryTranscript;
use crate::types::{DoryCommitment, DoryHint, DoryProof, DoryProverSetup, DoryVerifierSetup};

// All jolt types below are #[repr(transparent)] over the same arkworks
// inner type as their dory-pcs counterpart, guaranteeing identical layout.

pub(crate) type ArkFr = dory::backends::arkworks::ArkFr;
pub(crate) type ArkG1 = dory::backends::arkworks::ArkG1;
pub(crate) type ArkG2 = dory::backends::arkworks::ArkG2;
pub(crate) type ArkGT = dory::backends::arkworks::ArkGT;
type InnerBN254 = dory::backends::arkworks::BN254;

// All conversion functions below rely on repr(transparent) layout identity
// between jolt and dory-pcs wrappers over the same arkworks inner type.

#[inline]
pub(crate) fn jolt_fr_to_ark(f: &Fr) -> ArkFr {
    // SAFETY: Fr and ArkFr are both repr(transparent) over ark_bn254::Fr.
    unsafe { std::mem::transmute_copy(f) }
}

#[inline]
pub(crate) fn ark_to_jolt_fr(ark: &ArkFr) -> Fr {
    // SAFETY: same layout as jolt_fr_to_ark.
    unsafe { std::mem::transmute_copy(ark) }
}

#[inline]
pub(crate) fn jolt_gt_to_ark(gt: &Bn254GT) -> ArkGT {
    // SAFETY: Bn254GT and ArkGT are both repr(transparent) over Fq12.
    unsafe { std::mem::transmute_copy(gt) }
}

#[inline]
pub(crate) fn ark_to_jolt_gt(ark: &ArkGT) -> Bn254GT {
    // SAFETY: same layout as jolt_gt_to_ark.
    unsafe { std::mem::transmute_copy(ark) }
}

#[inline]
pub(crate) fn jolt_g1_vec_to_ark(v: Vec<Bn254G1>) -> Vec<ArkG1> {
    // SAFETY: Bn254G1 and ArkG1 have identical size/align (repr(transparent)
    // over G1Projective), so Vec layout is identical.
    unsafe { std::mem::transmute(v) }
}

#[inline]
pub(crate) fn ark_to_jolt_g1_vec(v: Vec<ArkG1>) -> Vec<Bn254G1> {
    // SAFETY: same layout as jolt_g1_vec_to_ark.
    unsafe { std::mem::transmute(v) }
}

#[inline]
pub(crate) fn ark_to_jolt_g1(ark: ArkG1) -> Bn254G1 {
    // SAFETY: Bn254G1 and ArkG1 are both repr(transparent) over G1Projective.
    unsafe { std::mem::transmute(ark) }
}

#[inline]
pub(crate) fn ark_to_jolt_g2(ark: ArkG2) -> Bn254G2 {
    // SAFETY: Bn254G2 and ArkG2 are both repr(transparent) over G2Projective.
    unsafe { std::mem::transmute(ark) }
}

#[inline]
fn jolt_g1_to_ark(value: Bn254G1) -> ArkG1 {
    // SAFETY: Bn254G1 and ArkG1 are both repr(transparent) over G1Projective.
    unsafe { std::mem::transmute(value) }
}

#[inline]
fn jolt_g2_to_ark(value: Bn254G2) -> ArkG2 {
    // SAFETY: Bn254G2 and ArkG2 are both repr(transparent) over G2Projective.
    unsafe { std::mem::transmute(value) }
}

#[inline]
fn ark_fr_slice_to_jolt(values: &[ArkFr]) -> &[Fr] {
    // SAFETY: Fr and ArkFr are both repr(transparent) over ark_bn254::Fr.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<Fr>(), values.len()) }
}

#[inline]
fn ark_g1_slice_to_jolt(values: &[ArkG1]) -> &[Bn254G1] {
    // SAFETY: Bn254G1 and ArkG1 are both repr(transparent) over G1Projective.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<Bn254G1>(), values.len()) }
}

#[inline]
fn ark_g1_slice_to_jolt_mut(values: &mut [ArkG1]) -> &mut [Bn254G1] {
    // SAFETY: Bn254G1 and ArkG1 are both repr(transparent) over G1Projective.
    unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<Bn254G1>(), values.len()) }
}

#[inline]
fn ark_g2_slice_to_jolt(values: &[ArkG2]) -> &[Bn254G2] {
    // SAFETY: Bn254G2 and ArkG2 are both repr(transparent) over G2Projective.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<Bn254G2>(), values.len()) }
}

#[inline]
fn ark_g2_slice_to_jolt_mut(values: &mut [ArkG2]) -> &mut [Bn254G2] {
    // SAFETY: Bn254G2 and ArkG2 are both repr(transparent) over G2Projective.
    unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<Bn254G2>(), values.len()) }
}

fn fold_field_vectors(left: &mut [ArkFr], right: &[ArkFr], scalar: &ArkFr) {
    assert_eq!(left.len(), right.len(), "field vector lengths must match");
    left.par_iter_mut()
        .zip(right.par_iter())
        .for_each(|(left, right)| {
            *left = *left * *scalar + *right;
        });
}

pub struct JoltG1Routines;

impl DoryRoutines<ArkG1> for JoltG1Routines {
    fn msm(bases: &[ArkG1], scalars: &[ArkFr]) -> ArkG1 {
        jolt_g1_to_ark(Bn254G1::msm(
            ark_g1_slice_to_jolt(bases),
            ark_fr_slice_to_jolt(scalars),
        ))
    }

    fn fixed_base_vector_scalar_mul(base: &ArkG1, scalars: &[ArkFr]) -> Vec<ArkG1> {
        if scalars.is_empty() {
            return Vec::new();
        }
        jolt_g1_vec_to_ark(glv::fixed_base_vector_msm_g1(
            &ark_to_jolt_g1(*base),
            ark_fr_slice_to_jolt(scalars),
        ))
    }

    fn fixed_scalar_mul_bases_then_add(bases: &[ArkG1], vs: &mut [ArkG1], scalar: &ArkFr) {
        assert_eq!(bases.len(), vs.len(), "bases and vs must match");
        glv::vector_add_scalar_mul_g1(
            ark_g1_slice_to_jolt_mut(vs),
            ark_g1_slice_to_jolt(bases),
            ark_to_jolt_fr(scalar),
        );
    }

    fn fixed_scalar_mul_vs_then_add(vs: &mut [ArkG1], addends: &[ArkG1], scalar: &ArkFr) {
        assert_eq!(vs.len(), addends.len(), "vs and addends must match");
        glv::vector_scalar_mul_add_gamma_g1(
            ark_g1_slice_to_jolt_mut(vs),
            ark_to_jolt_fr(scalar),
            ark_g1_slice_to_jolt(addends),
        );
    }

    fn fold_field_vectors(left: &mut [ArkFr], right: &[ArkFr], scalar: &ArkFr) {
        fold_field_vectors(left, right, scalar);
    }
}

pub struct JoltG2Routines;

impl DoryRoutines<ArkG2> for JoltG2Routines {
    fn msm(bases: &[ArkG2], scalars: &[ArkFr]) -> ArkG2 {
        jolt_g2_to_ark(Bn254G2::msm(
            ark_g2_slice_to_jolt(bases),
            ark_fr_slice_to_jolt(scalars),
        ))
    }

    fn fixed_base_vector_scalar_mul(base: &ArkG2, scalars: &[ArkFr]) -> Vec<ArkG2> {
        if scalars.is_empty() {
            return Vec::new();
        }
        let base = base.0;
        scalars
            .par_iter()
            .map(|scalar| {
                // SAFETY: ArkFr is repr(transparent) over ark_bn254::Fr.
                let raw_scalar: ark_bn254::Fr = unsafe { std::mem::transmute_copy(scalar) };
                let mut result = glv::glv_four::glv_four_scalar_mul_online(raw_scalar, &[base]);
                dory::backends::arkworks::ArkG2(
                    result
                        .pop()
                        .expect("single-base G2 GLV multiplication returns one result"),
                )
            })
            .collect()
    }

    fn fixed_scalar_mul_bases_then_add(bases: &[ArkG2], vs: &mut [ArkG2], scalar: &ArkFr) {
        assert_eq!(bases.len(), vs.len(), "bases and vs must match");
        glv::vector_add_scalar_mul_g2(
            ark_g2_slice_to_jolt_mut(vs),
            ark_g2_slice_to_jolt(bases),
            ark_to_jolt_fr(scalar),
        );
    }

    fn fixed_scalar_mul_vs_then_add(vs: &mut [ArkG2], addends: &[ArkG2], scalar: &ArkFr) {
        assert_eq!(vs.len(), addends.len(), "vs and addends must match");
        glv::vector_scalar_mul_add_gamma_g2(
            ark_g2_slice_to_jolt_mut(vs),
            ark_to_jolt_fr(scalar),
            ark_g2_slice_to_jolt(addends),
        );
    }

    fn fold_field_vectors(left: &mut [ArkFr], right: &[ArkFr], scalar: &ArkFr) {
        fold_field_vectors(left, right, scalar);
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DoryScheme;

impl DoryScheme {
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

    fn commit_with_mode<P, M>(poly: &P, setup: &DoryProverSetup) -> (DoryCommitment, DoryHint)
    where
        P: MultilinearPoly<Fr> + ?Sized,
        M: Mode,
    {
        let row_commitments = compute_row_commitments(poly, setup);
        let (tier_2, commit_blind) = commit_rows_tier_2::<M>(&row_commitments, setup);

        (
            DoryCommitment(ark_to_jolt_gt(&tier_2)),
            DoryHint::new(
                ark_to_jolt_g1_vec(row_commitments),
                ark_to_jolt_fr(&commit_blind),
            ),
        )
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

impl CommitmentScheme for DoryScheme {
    type Field = Fr;
    type Proof = DoryProof;
    type ProverSetup = DoryProverSetup;
    type VerifierSetup = DoryVerifierSetup;
    type Polynomial = jolt_poly::Polynomial<Fr>;
    type OpeningHint = DoryHint;
    type SetupParams = usize;

    fn setup(max_num_vars: Self::SetupParams) -> (DoryProverSetup, DoryVerifierSetup) {
        let prover = Self::setup_prover(max_num_vars);
        let verifier = Self::verifier_setup(&prover);
        (prover, verifier)
    }

    fn verifier_setup(prover_setup: &DoryProverSetup) -> DoryVerifierSetup {
        DoryVerifierSetup(prover_setup.0.to_verifier_setup())
    }

    #[tracing::instrument(skip_all, name = "DoryScheme::commit")]
    fn commit<P: MultilinearPoly<Fr> + ?Sized>(
        poly: &P,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint) {
        Self::commit_with_mode::<P, Transparent>(poly, setup)
    }

    #[tracing::instrument(skip_all, name = "DoryScheme::open")]
    fn open(
        poly: &Self::Polynomial,
        point: &[Fr],
        _eval: Fr,
        setup: &Self::ProverSetup,
        hint: Option<Self::OpeningHint>,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Self::Proof {
        Self::open_poly(poly, point, _eval, setup, hint, transcript)
    }

    #[tracing::instrument(skip_all, name = "DoryScheme::open_poly")]
    fn open_poly<P: MultilinearPoly<Self::Field>>(
        poly: &P,
        point: &[Fr],
        _eval: Fr,
        setup: &Self::ProverSetup,
        hint: Option<Self::OpeningHint>,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Self::Proof {
        let num_vars = point.len();
        let adapter = DorySourceAdapter::with_claimed_eval(poly, point, _eval);
        let sigma = num_vars.div_ceil(2);
        let nu = num_vars - sigma;

        let (row_commitments, commit_blind) = match hint {
            Some(h) => h.into_ark_parts(),
            None => (
                compute_row_commitments(poly, setup),
                <ArkFr as DoryField>::zero(),
            ),
        };
        debug_assert!(
            commit_blind.is_zero(),
            "commit_blind should be 0 for transparent mode"
        );

        let ark_point: Vec<ArkFr> = point.iter().rev().map(jolt_fr_to_ark).collect();
        let mut dory_transcript = JoltToDoryTranscript::new(transcript);

        let (proof, _blind) =
            dory::prove::<ArkFr, InnerBN254, JoltG1Routines, JoltG2Routines, _, _, Transparent>(
                &adapter,
                &ark_point,
                row_commitments,
                commit_blind,
                nu,
                sigma,
                &setup.0,
                &mut dory_transcript,
            )
            .unwrap_or_else(|e| panic!("dory::prove failed: {e:?}"));

        DoryProof(proof)
    }

    #[tracing::instrument(skip_all, name = "DoryScheme::verify")]
    fn verify(
        commitment: &Self::Output,
        point: &[Fr],
        eval: Fr,
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError> {
        let ark_point: Vec<ArkFr> = point.iter().rev().map(jolt_fr_to_ark).collect();
        let ark_eval = jolt_fr_to_ark(&eval);
        let ark_commitment = jolt_gt_to_ark(&commitment.0);
        let mut dory_transcript = JoltToDoryTranscript::new(transcript);

        dory::verify::<ArkFr, InnerBN254, JoltG1Routines, JoltG2Routines, _>(
            ark_commitment,
            ark_eval,
            &ark_point,
            &proof.0,
            setup.0.clone().into_inner(),
            &mut dory_transcript,
        )
        .map_err(|_| OpeningsError::VerificationFailed)
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

impl AdditivelyHomomorphic for DoryScheme {
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

    #[tracing::instrument(skip_all, name = "DoryScheme::combine_hints")]
    fn combine_hints(hints: Vec<Self::OpeningHint>, scalars: &[Self::Field]) -> Self::OpeningHint {
        assert_eq!(hints.len(), scalars.len());
        assert!(!hints.is_empty(), "combine_hints: empty hint set");

        // Hints may be ragged: dense (e.g. increment) commitments use a smaller
        // matrix than one-hot commitments, so their row-commitment vectors are
        // shorter. Resize to the maximum row count (matching core, which resizes
        // to the global max), treating absent rows as the identity commitment.
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

        let mut combined = vec![Bn254G1::default(); num_rows];
        for (mut row_commitments, &scalar) in hints
            .into_iter()
            .map(|hint| hint.row_commitments)
            .zip(scalars.iter())
        {
            row_commitments.resize(num_rows, Bn254G1::default());
            glv::vector_scalar_mul_add_gamma_g1(&mut row_commitments, scalar, &combined);
            combined = row_commitments;
        }

        DoryHint::new(combined, combined_blind)
    }
}

impl ZkOpeningScheme for DoryScheme {
    type HidingCommitment = Bn254G1;
    type Blind = Fr;

    fn commit_zk<P: MultilinearPoly<Fr> + ?Sized>(
        poly: &P,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint) {
        Self::commit_with_mode::<P, dory::ZK>(poly, setup)
    }

    #[tracing::instrument(skip_all, name = "DoryScheme::open_zk")]
    fn open_zk(
        poly: &Self::Polynomial,
        point: &[Fr],
        eval: Fr,
        setup: &Self::ProverSetup,
        hint: Self::OpeningHint,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> (Self::Proof, Self::HidingCommitment, Self::Blind) {
        Self::open_zk_poly(poly, point, eval, setup, hint, transcript)
    }

    #[tracing::instrument(skip_all, name = "DoryScheme::open_zk_poly")]
    fn open_zk_poly<P: MultilinearPoly<Self::Field>>(
        poly: &P,
        point: &[Fr],
        _eval: Fr,
        setup: &Self::ProverSetup,
        hint: Self::OpeningHint,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> (Self::Proof, Self::HidingCommitment, Self::Blind) {
        let num_vars = point.len();
        let adapter = DorySourceAdapter::with_claimed_eval(poly, point, _eval);
        let sigma = num_vars.div_ceil(2);
        let nu = num_vars - sigma;
        let (row_commitments, commit_blind) = hint.into_ark_parts();

        let ark_point: Vec<ArkFr> = point.iter().rev().map(jolt_fr_to_ark).collect();
        let mut dory_transcript = JoltToDoryTranscript::new(transcript);

        let (proof, y_blinding) =
            dory::prove::<ArkFr, InnerBN254, JoltG1Routines, JoltG2Routines, _, _, dory::mode::ZK>(
                &adapter,
                &ark_point,
                row_commitments,
                commit_blind,
                nu,
                sigma,
                &setup.0,
                &mut dory_transcript,
            )
            .unwrap_or_else(|e| panic!("dory::prove (ZK) failed: {e:?}"));

        let y_com = ark_to_jolt_g1(proof.y_com.expect("ZK proof must contain y_com"));
        let blinding = ark_to_jolt_fr(&y_blinding.expect("ZK proof must return y_blinding"));

        (DoryProof(proof), y_com, blinding)
    }

    #[tracing::instrument(skip_all, name = "DoryScheme::verify_zk")]
    fn verify_zk(
        commitment: &Self::Output,
        point: &[Fr],
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<Self::HidingCommitment, OpeningsError> {
        let ark_point: Vec<ArkFr> = point.iter().rev().map(jolt_fr_to_ark).collect();
        // In ZK mode dory::verify reads the evaluation commitment from `proof.y_com`,
        // so the caller-side eval is unused here.
        let dummy_eval = <ArkFr as DoryField>::zero();
        let ark_commitment = jolt_gt_to_ark(&commitment.0);
        let mut dory_transcript = JoltToDoryTranscript::new(transcript);
        let hiding_commitment = proof
            .0
            .y_com
            .map(ark_to_jolt_g1)
            .ok_or(OpeningsError::VerificationFailed)?;

        dory::verify::<ArkFr, InnerBN254, JoltG1Routines, JoltG2Routines, _>(
            ark_commitment,
            dummy_eval,
            &ark_point,
            &proof.0,
            setup.0.clone().into_inner(),
            &mut dory_transcript,
        )
        .map_err(|_| OpeningsError::VerificationFailed)?;

        Ok(hiding_commitment)
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

/// Dense commit: full MSM per row, parallel over rows.
fn commit_rows_dense<P: MultilinearPoly<Fr> + ?Sized>(
    poly: &P,
    sigma: usize,
    setup: &ArkworksProverSetup,
) -> Vec<ArkG1> {
    let num_cols = 1usize << sigma;
    let g1_bases = &setup.g1_vec[..num_cols];

    let mut rows: Vec<Vec<Fr>> = Vec::new();
    poly.for_each_row(sigma, &mut |_, row| rows.push(row.to_vec()));

    rows.par_iter()
        .map(|row| {
            let scalars: Vec<ArkFr> = row.iter().map(jolt_fr_to_ark).collect();
            JoltG1Routines::msm(&g1_bases[..scalars.len()], &scalars)
        })
        .collect()
}

/// One-hot commit: O(T) group additions for unit-valued one-hot polynomials.
fn commit_rows_one_hot<P: MultilinearPoly<Fr> + ?Sized>(
    poly: &P,
    num_rows: usize,
    num_cols: usize,
    setup: &ArkworksProverSetup,
) -> Vec<ArkG1> {
    let g1_bases: Vec<_> = setup.g1_vec[..num_cols]
        .par_iter()
        .map(|base| base.0.into_affine())
        .collect();

    if let (Some(k), Some(indices), Some(OneHotIndexOrder::ColumnMajor)) = (
        poly.one_hot_k(),
        poly.one_hot_indices(),
        poly.one_hot_index_order(),
    ) {
        let rows_per_k = indices.len() / num_cols;
        if rows_per_k > 0
            && indices.len().is_multiple_of(num_cols)
            && k.checked_mul(rows_per_k) == Some(num_rows)
            && rows_per_k >= rayon::current_num_threads()
        {
            let chunk_commitments = indices
                .par_chunks(num_cols)
                .map(|chunk| {
                    let mut indices_per_k: Vec<Vec<usize>> = vec![Vec::new(); k];
                    for (col_index, hot_column) in chunk.iter().copied().enumerate() {
                        if let Some(hot_column) = hot_column {
                            indices_per_k[hot_column as usize].push(col_index);
                        }
                    }
                    let additions = batch_g1_additions_multi_affine(&g1_bases, &indices_per_k);
                    let mut row_commitments = vec![<InnerBN254 as PairingCurve>::G1::identity(); k];
                    for (row_commitment, (indices, addition)) in row_commitments
                        .iter_mut()
                        .zip(indices_per_k.iter().zip(additions))
                    {
                        if !indices.is_empty() {
                            *row_commitment = dory::backends::arkworks::ArkG1(addition.into());
                        }
                    }
                    row_commitments
                })
                .collect::<Vec<_>>();

            let mut result = vec![<InnerBN254 as PairingCurve>::G1::identity(); num_rows];
            for (chunk_index, commitments) in chunk_commitments.iter().enumerate() {
                result
                    .par_iter_mut()
                    .skip(chunk_index)
                    .step_by(rows_per_k)
                    .zip(commitments.par_iter())
                    .for_each(|(dest, src)| *dest = *src);
            }
            return result;
        }
    }

    let mut cols_per_row: Vec<Vec<usize>> = vec![Vec::new(); num_rows];
    poly.for_each_one(&mut |flat_idx| {
        let row = flat_idx / num_cols;
        let col = flat_idx % num_cols;
        debug_assert!(
            row < num_rows && col < num_cols,
            "for_each_one out-of-bounds flat_idx: row={row} num_rows={num_rows} col={col} num_cols={num_cols}",
        );
        cols_per_row[row].push(col);
    });

    let num_chunks = rayon::current_num_threads().next_power_of_two();
    let chunk_size = num_rows.div_ceil(num_chunks).max(1);
    let mut result = vec![<InnerBN254 as PairingCurve>::G1::identity(); num_rows];

    result
        .par_chunks_mut(chunk_size)
        .zip(cols_per_row.par_chunks(chunk_size))
        .for_each(|(result_chunk, indices_chunk)| {
            let additions = batch_g1_additions_multi_affine(&g1_bases, indices_chunk);
            for (row_result, batch_result) in result_chunk.iter_mut().zip(additions) {
                *row_result = dory::backends::arkworks::ArkG1(batch_result.into());
            }
        });

    result
}

fn compute_row_commitments<P: MultilinearPoly<Fr> + ?Sized>(
    poly: &P,
    setup: &DoryProverSetup,
) -> Vec<ArkG1> {
    let num_vars = poly.num_vars();
    let sigma = num_vars.div_ceil(2);
    let num_cols = 1usize << sigma;
    let num_rows = 1usize << (num_vars - sigma);

    if poly.is_one_hot() {
        commit_rows_one_hot(poly, num_rows, num_cols, &setup.0)
    } else {
        commit_rows_dense(poly, sigma, &setup.0)
    }
}

pub(crate) fn commit_rows_tier_2<M: Mode>(
    row_commitments: &[ArkG1],
    setup: &DoryProverSetup,
) -> (ArkGT, ArkFr) {
    let tier_2 = if row_commitments
        .par_iter()
        .all(|row| *row == <InnerBN254 as PairingCurve>::G1::identity())
    {
        ArkGT::identity()
    } else {
        let g2_bases = &setup.0.g2_vec[..row_commitments.len()];
        <InnerBN254 as PairingCurve>::multi_pair_g2_setup(row_commitments, g2_bases)
    };
    let commit_blind = M::sample::<ArkFr>();
    let tier_2 = M::mask(tier_2, &setup.0.ht, &commit_blind);
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

/// Bridges [`MultilinearPoly<Fr>`] to dory-pcs's polynomial traits
/// without materializing the full evaluation table.
struct DorySourceAdapter<'a, S: MultilinearPoly<Fr>> {
    source: &'a S,
    claimed_eval: Option<(&'a [Fr], Fr)>,
}

impl<'a, S: MultilinearPoly<Fr>> DorySourceAdapter<'a, S> {
    fn with_claimed_eval(source: &'a S, point: &'a [Fr], eval: Fr) -> Self {
        Self {
            source,
            claimed_eval: Some((point, eval)),
        }
    }
}

impl<S: MultilinearPoly<Fr>> DoryPolynomial<ArkFr> for DorySourceAdapter<'_, S> {
    fn num_vars(&self) -> usize {
        self.source.num_vars()
    }

    fn evaluate(&self, point: &[ArkFr]) -> ArkFr {
        if let Some((claimed_point, claimed_eval)) = self.claimed_eval {
            let matches_claimed_point = point.len() == claimed_point.len()
                && point
                    .iter()
                    .zip(claimed_point.iter().rev())
                    .all(|(actual, expected)| *actual == jolt_fr_to_ark(expected));
            if matches_claimed_point {
                return jolt_fr_to_ark(&claimed_eval);
            }
        }
        let native_point: Vec<Fr> = point.iter().rev().map(ark_to_jolt_fr).collect();
        jolt_fr_to_ark(&self.source.evaluate(&native_point))
    }

    fn commit<E, Mo, M1>(
        &self,
        _nu: usize,
        _sigma: usize,
        _setup: &dory::setup::ProverSetup<E>,
    ) -> Result<(E::GT, Vec<E::G1>, ArkFr), dory::error::DoryError>
    where
        E: PairingCurve,
        Mo: dory::mode::Mode,
        M1: DoryRoutines<E::G1>,
        E::G1: DoryGroup<Scalar = ArkFr>,
    {
        unimplemented!(
            "DoryScheme pre-computes row commitments before invoking dory::prove; \
             dory::Polynomial::commit on this adapter is not exercised"
        )
    }
}

impl<S: MultilinearPoly<Fr>> MultilinearLagrange<ArkFr> for DorySourceAdapter<'_, S> {
    fn vector_matrix_product(&self, left_vec: &[ArkFr], _nu: usize, sigma: usize) -> Vec<ArkFr> {
        let native_left: Vec<Fr> = left_vec.iter().map(ark_to_jolt_fr).collect();
        let result = self.source.fold_rows(&native_left, sigma);
        result.iter().map(jolt_fr_to_ark).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_crypto::{Pedersen, VectorCommitment};
    use jolt_field::{FromPrimitiveInt, RandomSampling};
    use jolt_poly::Polynomial;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

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

        let mut prove_transcript = jolt_transcript::Blake2bTranscript::new(b"zk-test");
        let (proof, _eval_com, _blinding) = DoryScheme::open_zk(
            &poly,
            &point,
            eval,
            &prover_setup,
            hint,
            &mut prove_transcript,
        );

        let mut verify_transcript = jolt_transcript::Blake2bTranscript::new(b"zk-test");
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
