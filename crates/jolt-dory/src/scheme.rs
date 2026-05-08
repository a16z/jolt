//! Dory PCS implementing the `jolt-openings` trait hierarchy.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};

use dory::backends::arkworks::{ArkworksProverSetup, G1Routines};
use dory::mode::Transparent;
use dory::primitives::arithmetic::{
    DoryRoutines, Field as DoryField, Group as DoryGroup, PairingCurve,
};
use dory::primitives::poly::{MultilinearLagrange, Polynomial as DoryPolynomial};
use jolt_crypto::{Bn254G1, Bn254GT, Commitment, DeriveSetup, JoltGroup, PedersenSetup};
use jolt_field::{Field, Fr};
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme, OpeningsError, ZkOpeningScheme};
use jolt_poly::MultilinearPoly;
use jolt_transcript::{AppendToTranscript, Label, LabelWithCount, Transcript};

use crate::routines::{JoltG1Routines, JoltG2Routines};
use crate::transcript::JoltToDoryTranscript;
use crate::types::{DoryCommitment, DoryHint, DoryProof, DoryProverSetup, DoryVerifierSetup};

// All jolt types below are #[repr(transparent)] over the same arkworks
// inner type as their dory-pcs counterpart, guaranteeing identical layout.

pub(crate) type ArkFr = dory::backends::arkworks::ArkFr;
pub(crate) type ArkG1 = dory::backends::arkworks::ArkG1;
pub(crate) type ArkG2 = dory::backends::arkworks::ArkG2;
pub(crate) type ArkGT = dory::backends::arkworks::ArkGT;
type InnerBN254 = dory::backends::arkworks::BN254;

type ArkG1Affine = ark_bn254::G1Affine;
type AffineBaseCache = Mutex<HashMap<(usize, usize), Arc<Vec<ArkG1Affine>>>>;

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
fn ark_fr_slice_to_jolt(slice: &[ArkFr]) -> &[Fr] {
    // SAFETY: Fr and ArkFr are both repr(transparent) over ark_bn254::Fr.
    unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<Fr>(), slice.len()) }
}

#[inline]
fn jolt_fr_vec_to_ark(vec: Vec<Fr>) -> Vec<ArkFr> {
    // SAFETY: Fr and ArkFr have identical size/alignment and transparent layout.
    unsafe { std::mem::transmute(vec) }
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
pub(crate) fn jolt_gt_ref_to_ark(gt: &Bn254GT) -> &ArkGT {
    // SAFETY: Bn254GT and ArkGT are both repr(transparent) over Fq12.
    unsafe { &*(std::ptr::from_ref(gt).cast::<ArkGT>()) }
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

fn cached_affine_g1_bases(g1_bases: &[ArkG1]) -> Arc<Vec<ArkG1Affine>> {
    static CACHE: OnceLock<AffineBaseCache> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (g1_bases.as_ptr() as usize, g1_bases.len());
    {
        let guard = cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(cached) = guard.get(&key).cloned() {
            return cached;
        }
    }

    let bases = g1_bases
        .iter()
        .copied()
        .map(ark_to_jolt_g1)
        .collect::<Vec<_>>();
    let affines = Arc::new(jolt_crypto::ec::bn254::batch_addition::normalize_g1_bases(
        &bases,
    ));
    let mut guard = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let _ = guard.insert(key, Arc::clone(&affines));
    affines
}

#[derive(Clone)]
pub struct DoryScheme;

impl DoryScheme {
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

impl DeriveSetup<DoryProverSetup> for PedersenSetup<Bn254G1> {
    fn derive(source: &DoryProverSetup, capacity: usize) -> Self {
        let len = capacity.min(source.0.g1_vec.len());
        let generators = ark_to_jolt_g1_vec(source.0.g1_vec[..len].to_vec());
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

        (
            DoryCommitment(ark_to_jolt_gt(&tier_2)),
            DoryHint(ark_to_jolt_g1_vec(row_commitments)),
        )
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
    fn combine(commitments: &[Self::Output], scalars: &[Self::Field]) -> Self::Output {
        assert_eq!(commitments.len(), scalars.len());

        let combined = commitments
            .iter()
            .zip(scalars.iter())
            .map(|(c, s)| jolt_fr_to_ark(s) * jolt_gt_to_ark(&c.0))
            .fold(ArkGT::identity(), |acc, x| acc + x);

        DoryCommitment(ark_to_jolt_gt(&combined))
    }

    fn combine_hints(hints: Vec<Self::OpeningHint>, scalars: &[Self::Field]) -> Self::OpeningHint {
        let hint_refs = hints.iter().collect::<Vec<_>>();
        Self::combine_hint_refs(&hint_refs, scalars)
    }
}

impl DoryScheme {
    #[tracing::instrument(skip_all, name = "DoryScheme::open_poly")]
    pub fn open_poly<P>(
        poly: &P,
        point: &[Fr],
        _eval: Fr,
        setup: &DoryProverSetup,
        hint: Option<DoryHint>,
        transcript: &mut impl Transcript<Challenge = Fr>,
    ) -> DoryProof
    where
        P: MultilinearPoly<Fr>,
    {
        let num_vars = point.len();
        let _adapter_span = tracing::info_span!("DoryScheme::open.adapter").entered();
        let adapter = DorySourceAdapter::new(poly);
        drop(_adapter_span);
        let sigma = num_vars.div_ceil(2);
        let nu = num_vars - sigma;

        let _rows_span = tracing::info_span!("DoryScheme::open.row_commitments").entered();
        let row_commitments = match hint {
            Some(h) => jolt_g1_vec_to_ark(h.0),
            None => commit_rows_dense(poly, sigma, &setup.0),
        };
        drop(_rows_span);

        let _point_span = tracing::info_span!("DoryScheme::open.point").entered();
        let ark_point: Vec<ArkFr> = point.iter().rev().map(jolt_fr_to_ark).collect();
        let mut dory_transcript = JoltToDoryTranscript::new(transcript);
        drop(_point_span);

        let _prove_span = tracing::info_span!("DoryScheme::open.prove").entered();
        let (proof, _blind) = match dory::prove::<
            ArkFr,
            InnerBN254,
            JoltG1Routines,
            JoltG2Routines,
            _,
            _,
            Transparent,
        >(
            &adapter,
            &ark_point,
            row_commitments,
            <ArkFr as DoryField>::zero(),
            nu,
            sigma,
            &setup.0,
            &mut dory_transcript,
        ) {
            Ok(proof) => proof,
            Err(_) => std::process::abort(),
        };
        drop(_prove_span);

        DoryProof(proof)
    }

    pub fn combine_hint_refs(hints: &[&DoryHint], scalars: &[Fr]) -> DoryHint {
        assert_eq!(hints.len(), scalars.len());
        if hints.is_empty() {
            return DoryHint::default();
        }

        let num_rows = hints.iter().map(|hint| hint.0.len()).max().unwrap_or(0);
        let mut combined = vec![Bn254G1::identity(); num_rows];

        for (hint, &scalar) in hints.iter().zip(scalars.iter()) {
            jolt_crypto::ec::bn254::glv::vector_add_scalar_mul_g1(
                &mut combined[..hint.0.len()],
                &hint.0,
                scalar,
            );
        }

        DoryHint(combined)
    }
}

impl ZkOpeningScheme for DoryScheme {
    type HidingCommitment = Bn254G1;
    type Blind = Fr;

    #[tracing::instrument(skip_all, name = "DoryScheme::open_zk")]
    fn open_zk(
        poly: &Self::Polynomial,
        point: &[Fr],
        _eval: Fr,
        setup: &Self::ProverSetup,
        hint: Option<Self::OpeningHint>,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> (Self::Proof, Self::HidingCommitment, Self::Blind) {
        let num_vars = point.len();
        let adapter = DorySourceAdapter::new(poly);
        let sigma = num_vars.div_ceil(2);
        let nu = num_vars - sigma;

        let row_commitments = match hint {
            Some(h) => jolt_g1_vec_to_ark(h.0),
            None => commit_rows_dense(poly, sigma, &setup.0),
        };

        let ark_point: Vec<ArkFr> = point.iter().rev().map(jolt_fr_to_ark).collect();
        let mut dory_transcript = JoltToDoryTranscript::new(transcript);

        let (proof, y_blinding) = match dory::prove::<
            ArkFr,
            InnerBN254,
            JoltG1Routines,
            JoltG2Routines,
            _,
            _,
            dory::mode::ZK,
        >(
            &adapter,
            &ark_point,
            row_commitments,
            <ArkFr as DoryField>::zero(),
            nu,
            sigma,
            &setup.0,
            &mut dory_transcript,
        ) {
            Ok(proof) => proof,
            Err(_) => std::process::abort(),
        };

        let Some(proof_y_com) = proof.y_com else {
            std::process::abort();
        };
        let Some(y_blinding) = y_blinding else {
            std::process::abort();
        };
        let y_com = ark_to_jolt_g1(proof_y_com);
        let blinding = ark_to_jolt_fr(&y_blinding);

        (DoryProof(proof), y_com, blinding)
    }

    #[tracing::instrument(skip_all, name = "DoryScheme::verify_zk")]
    fn verify_zk(
        commitment: &Self::Output,
        point: &[Fr],
        _eval_commitment: &Self::HidingCommitment,
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError> {
        let ark_point: Vec<ArkFr> = point.iter().rev().map(jolt_fr_to_ark).collect();
        // In ZK mode, dory::verify uses proof's y_com/e2 instead of evaluation.
        let dummy_eval = <ArkFr as DoryField>::zero();
        let ark_commitment = jolt_gt_to_ark(&commitment.0);
        let mut dory_transcript = JoltToDoryTranscript::new(transcript);

        dory::verify::<ArkFr, InnerBN254, JoltG1Routines, JoltG2Routines, _>(
            ark_commitment,
            dummy_eval,
            &ark_point,
            &proof.0,
            setup.0.clone().into_inner(),
            &mut dory_transcript,
        )
        .map_err(|_| OpeningsError::VerificationFailed)
    }
}

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
        let scalars: Vec<ArkFr> = row.iter().map(jolt_fr_to_ark).collect();
        rows.push(G1Routines::msm(&g1_bases[..scalars.len()], &scalars));
    });
    rows
}

/// Sparse commit: O(T) group additions for one-hot polynomials.
fn commit_rows_sparse<P: MultilinearPoly<Fr> + ?Sized>(
    poly: &P,
    num_rows: usize,
    num_cols: usize,
    setup: &ArkworksProverSetup,
) -> Vec<ArkG1> {
    let g1_bases = &setup.g1_vec[..num_cols];
    let identity = <InnerBN254 as PairingCurve>::G1::identity();

    let mut row_indices = vec![Vec::<usize>::new(); num_rows];
    let mut all_one = true;
    poly.for_each_nonzero(&mut |flat_idx, _val| {
        let row = flat_idx / num_cols;
        let col = flat_idx % num_cols;
        row_indices[row].push(col);
        all_one &= _val == Fr::from_u64(1);
    });

    if all_one {
        return jolt_g1_vec_to_ark(
            jolt_crypto::ec::bn254::batch_addition::batch_g1_additions_multi_affine(
                &cached_affine_g1_bases(g1_bases),
                &row_indices,
            )
            .into_iter()
            .map(|affine| Bn254G1::from(ark_bn254::G1Projective::from(affine)))
            .collect(),
        );
    }

    let mut row_commitments = vec![identity; num_rows];
    poly.for_each_nonzero(&mut |flat_idx, _val| {
        let row = flat_idx / num_cols;
        let col = flat_idx % num_cols;
        let scaled = jolt_fr_to_ark(&_val) * g1_bases[col];
        row_commitments[row] =
            <InnerBN254 as PairingCurve>::G1::add(&row_commitments[row], &scaled);
    });
    row_commitments
}

/// Bridges [`MultilinearPoly<Fr>`] to dory-pcs's polynomial traits
/// without materializing the full evaluation table.
struct DorySourceAdapter<'a, S: MultilinearPoly<Fr>> {
    source: &'a S,
}

impl<'a, S: MultilinearPoly<Fr>> DorySourceAdapter<'a, S> {
    fn new(source: &'a S) -> Self {
        Self { source }
    }
}

impl<S: MultilinearPoly<Fr>> DoryPolynomial<ArkFr> for DorySourceAdapter<'_, S> {
    fn num_vars(&self) -> usize {
        self.source.num_vars()
    }

    fn evaluate(&self, point: &[ArkFr]) -> ArkFr {
        // Dory calls this with its little-endian point. Jolt polynomials are
        // evaluated with the original high-to-low variable order.
        let native_point: Vec<Fr> = point.iter().rev().map(ark_to_jolt_fr).collect();
        jolt_fr_to_ark(&self.source.evaluate(&native_point))
    }

    fn commit<E, Mo, M1>(
        &self,
        _nu: usize,
        sigma: usize,
        setup: &dory::setup::ProverSetup<E>,
    ) -> Result<(E::GT, Vec<E::G1>, ArkFr), dory::error::DoryError>
    where
        E: PairingCurve,
        Mo: dory::mode::Mode,
        M1: DoryRoutines<E::G1>,
        E::G1: DoryGroup<Scalar = ArkFr>,
    {
        let mut row_commitments: Vec<E::G1> = Vec::new();
        self.source.for_each_row(sigma, &mut |_idx, row| {
            let scalars: Vec<ArkFr> = row.iter().map(jolt_fr_to_ark).collect();
            row_commitments.push(M1::msm(&setup.g1_vec[..scalars.len()], &scalars));
        });

        let g2_bases = &setup.g2_vec[..row_commitments.len()];
        let tier_2 = E::multi_pair_g2_setup(&row_commitments, g2_bases);

        Ok((tier_2, row_commitments, <ArkFr as DoryField>::zero()))
    }
}

impl<S: MultilinearPoly<Fr>> MultilinearLagrange<ArkFr> for DorySourceAdapter<'_, S> {
    fn vector_matrix_product(&self, left_vec: &[ArkFr], _nu: usize, sigma: usize) -> Vec<ArkFr> {
        let _span = tracing::info_span!(
            "DorySourceAdapter::vector_matrix_product",
            left_len = left_vec.len(),
            sigma = sigma
        )
        .entered();
        jolt_fr_vec_to_ark(self.source.fold_rows(ark_fr_slice_to_jolt(left_vec), sigma))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_crypto::{Pedersen, VectorCommitment};
    use jolt_field::Field;
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

        let combined = DoryScheme::combine(
            &[commit_a, commit_b],
            &[<Fr as Field>::from_u64(1), <Fr as Field>::from_u64(1)],
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
            <Fr as Field>::from_u64(1),
            <Fr as Field>::from_u64(2),
            <Fr as Field>::from_u64(3),
        ];
        let blinding = <Fr as Field>::from_u64(42);
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
