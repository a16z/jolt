#![expect(
    clippy::unimplemented,
    reason = "the dory adapter's commit is unreachable because DoryScheme pre-computes row commitments"
)]

use dory::backends::arkworks::ArkworksProverSetup;
use dory::mode::{Transparent, ZK};
use dory::primitives::arithmetic::{
    DoryRoutines, Field as DoryField, Group as DoryGroup, PairingCurve,
};
use dory::primitives::poly::{MultilinearLagrange, Polynomial as DoryPolynomial};
use dory::Mode;
use jolt_crypto::{Bn254G1, Bn254GT, Commitment, DeriveSetup, JoltGroup, PedersenSetup};
use jolt_field::Fr;
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme, OpeningsError, ZkOpeningScheme};
use jolt_poly::MultilinearPoly;
use jolt_transcript::Transcript;
use rayon::prelude::*;

use crate::routines::{JoltG1Routines, JoltG2Routines};
use crate::transcript::JoltToDoryTranscript;
use crate::types::{DoryCommitment, DoryHint, DoryProof, DoryProverSetup, DoryVerifierSetup};

// All jolt types below are #[repr(transparent)] over the same arkworks
// inner type as their dory-pcs counterpart, guaranteeing identical layout.

pub(crate) type ArkFr = dory::backends::arkworks::ArkFr;
pub(crate) type ArkG1 = dory::backends::arkworks::ArkG1;
pub(crate) type ArkGT = dory::backends::arkworks::ArkGT;
type InnerBN254 = dory::backends::arkworks::BN254;

// All conversion functions below rely on repr(transparent) layout identity
// between jolt and dory-pcs wrappers over the same arkworks inner type. The
// size and alignment halves of that reliance are checked at compile time; a
// wrapper gaining a field or losing repr(transparent) fails these asserts
// before any transmute can misbehave.
const _: () = {
    use std::mem::{align_of, size_of};
    assert!(size_of::<Fr>() == size_of::<ArkFr>());
    assert!(align_of::<Fr>() == align_of::<ArkFr>());
    assert!(size_of::<Bn254GT>() == size_of::<ArkGT>());
    assert!(align_of::<Bn254GT>() == align_of::<ArkGT>());
    assert!(size_of::<Bn254G1>() == size_of::<ArkG1>());
    assert!(align_of::<Bn254G1>() == align_of::<ArkG1>());
};

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
    let mut v = std::mem::ManuallyDrop::new(v);
    let (ptr, len, capacity) = (v.as_mut_ptr(), v.len(), v.capacity());
    // SAFETY: Bn254G1 and ArkG1 are both repr(transparent) over G1Projective
    // (size/align compile-checked above), so the buffer's elements can be
    // reinterpreted in place. `Vec`'s internal layout across distinct element
    // types is unspecified, so the container is rebuilt from raw parts
    // rather than transmuted whole; ManuallyDrop keeps the original from
    // freeing the buffer it hands over.
    unsafe { Vec::from_raw_parts(ptr.cast::<ArkG1>(), len, capacity) }
}

#[inline]
pub(crate) fn ark_to_jolt_g1_vec(v: Vec<ArkG1>) -> Vec<Bn254G1> {
    let mut v = std::mem::ManuallyDrop::new(v);
    let (ptr, len, capacity) = (v.as_mut_ptr(), v.len(), v.capacity());
    // SAFETY: same layout facts as jolt_g1_vec_to_ark.
    unsafe { Vec::from_raw_parts(ptr.cast::<Bn254G1>(), len, capacity) }
}

#[inline]
pub(crate) fn ark_to_jolt_g1(ark: ArkG1) -> Bn254G1 {
    // SAFETY: Bn254G1 and ArkG1 are both repr(transparent) over G1Projective.
    unsafe { std::mem::transmute(ark) }
}

/// First `n` bases of an SRS table. Callers size `n` from the polynomial
/// being committed; the assert keeps a descriptive panic if it outgrows the
/// setup instead of a bare slice-bounds failure.
#[inline]
pub(crate) fn srs_prefix<T>(bases: &[T], n: usize) -> &[T] {
    assert!(
        n <= bases.len(),
        "requested {n} SRS bases but the setup has {}",
        bases.len(),
    );
    bases.split_at(n).0
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DoryScheme;

/// Dory's generator count depends on `ceil(max_log_n / 2)`, so odd/even pairs
/// like 21 and 22 share the same generator bucket — but the URS derivation
/// seeds on the exact size, so their generator VALUES differ. Canonicalizing
/// to the even bucket representative keeps every run on a single URS (and
/// matches the legacy prover's setups byte-for-byte).
#[inline]
fn canonical_setup_log_n(max_num_vars: usize) -> usize {
    if max_num_vars.is_multiple_of(2) {
        max_num_vars
    } else {
        max_num_vars + 1
    }
}

/// The matrix layout `(nu, sigma)` of a `num_vars`-variable polynomial:
/// `2^nu` rows of `2^sigma` coefficients.
///
/// Commit, open and verify must all use it. dory reads `nu` and `sigma` from
/// the proof and only checks that they sum to the point's length, with
/// `nu <= sigma`. A commitment whose matrix is zero outside the top
/// `2^nu'` rows is also a commitment under any wider layout `(nu', sigma')`,
/// to a different polynomial. So the verifier rejects any other layout.
fn matrix_shape(num_vars: usize) -> (usize, usize) {
    let sigma = num_vars.div_ceil(2);
    (num_vars - sigma, sigma)
}

impl DoryScheme {
    #[tracing::instrument(skip_all, name = "DoryScheme::setup_prover", fields(max_num_vars))]
    pub fn setup_prover(max_num_vars: usize) -> DoryProverSetup {
        let canonical_max_num_vars = canonical_setup_log_n(max_num_vars);
        #[cfg(not(target_arch = "wasm32"))]
        let setup = {
            // dory persists a freshly OsRng-generated URS with an unlocked
            // truncating write, so concurrent processes generate different
            // random SRS and last-writer-wins. A single run calls setup more
            // than once, so it can commit under its own SRS and then load a
            // sibling process's, diverging the generators mid-proof.
            let _urs_lock = crate::urs_lock::lock_urs_cache();
            ArkworksProverSetup::new_from_urs(canonical_max_num_vars)
        };
        #[cfg(target_arch = "wasm32")]
        let setup = ArkworksProverSetup::new(canonical_max_num_vars);

        // Deliberately NOT priming dory's global prepared-point cache (the
        // legacy prover does): the cache only grows and its consumers
        // prefix-match blindly, so any process touching two setup sizes —
        // the URS seeds generators on the exact size — silently pairs
        // against the wrong generators. The saving is ~0.2 s wall of G2
        // Miller preprocessing per 2^20 proof; not worth the footgun.
        DoryProverSetup(setup)
    }

    /// Derives the verifier SRS (a subset of the prover SRS).
    #[tracing::instrument(skip_all, name = "DoryScheme::setup_verifier", fields(max_num_vars))]
    pub fn setup_verifier(max_num_vars: usize) -> DoryVerifierSetup {
        let prover_setup = Self::setup_prover(max_num_vars);
        DoryVerifierSetup(prover_setup.0.to_verifier_setup())
    }

    fn commit_with_mode<P, M>(
        poly: &P,
        setup: &DoryProverSetup,
    ) -> Result<(DoryCommitment, DoryHint), OpeningsError>
    where
        P: MultilinearPoly<Fr> + ?Sized,
        M: Mode,
    {
        let row_commitments = compute_row_commitments(poly, setup)?;
        let (tier_2, commit_blind) = commit_rows_tier_2::<M>(&row_commitments, setup);

        Ok((
            DoryCommitment(ark_to_jolt_gt(&tier_2)),
            DoryHint::new(
                ark_to_jolt_g1_vec(row_commitments),
                ark_to_jolt_fr(&commit_blind),
            ),
        ))
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
        let generators =
            ark_to_jolt_g1_vec(source.0.g1_vec.iter().take(capacity).copied().collect());
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
    type OpeningHint = DoryHint;
    type SetupParams = usize;

    fn setup(
        max_num_vars: Self::SetupParams,
    ) -> Result<(DoryProverSetup, DoryVerifierSetup), OpeningsError> {
        let prover = Self::setup_prover(max_num_vars);
        let verifier = Self::verifier_setup(&prover);
        Ok((prover, verifier))
    }

    fn verifier_setup(prover_setup: &DoryProverSetup) -> DoryVerifierSetup {
        DoryVerifierSetup(prover_setup.0.to_verifier_setup())
    }

    #[tracing::instrument(skip_all, name = "DoryScheme::commit")]
    fn commit<P: MultilinearPoly<Fr> + ?Sized>(
        poly: &P,
        setup: &Self::ProverSetup,
    ) -> Result<(Self::Output, Self::OpeningHint), OpeningsError> {
        Self::commit_with_mode::<P, Transparent>(poly, setup)
    }

    #[tracing::instrument(skip_all, name = "DoryScheme::open")]
    fn open<P: MultilinearPoly<Fr> + ?Sized>(
        poly: &P,
        point: &[Fr],
        _eval: Fr,
        setup: &Self::ProverSetup,
        hint: Option<Self::OpeningHint>,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<Self::Proof, OpeningsError> {
        let adapter = DorySourceAdapter::new(poly);
        let (nu, sigma) = matrix_shape(point.len());

        let (row_commitments, commit_blind) = match hint {
            Some(h) => h.into_ark_parts(),
            None => (
                compute_row_commitments(poly, setup)?,
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
            .map_err(|e| OpeningsError::ProveFailed(format!("dory::prove failed: {e:?}")))?;

        Ok(DoryProof(proof))
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

        if (proof.0.nu, proof.0.sigma) != matrix_shape(point.len()) {
            return Err(OpeningsError::VerificationFailed);
        }
        if proof.0.e2.is_some()
            || proof.0.y_com.is_some()
            || proof.0.sigma1_proof.is_some()
            || proof.0.sigma2_proof.is_some()
            || proof.0.scalar_product_proof.is_some()
        {
            return Err(OpeningsError::VerificationFailed);
        }

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

        // Hints may be ragged: a polynomial narrower than the shared
        // commitment grid commits fewer rows, and its zero-embedding's
        // missing rows are identity commitments — combine over the widest
        // hint's row count, treating absent rows as identity.
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

        DoryHint::new(combined, combined_blind)
    }
}

impl ZkOpeningScheme for DoryScheme {
    type HidingCommitment = Bn254G1;
    type Blind = Fr;

    fn commit_zk<P: MultilinearPoly<Fr> + ?Sized>(
        poly: &P,
        setup: &Self::ProverSetup,
    ) -> Result<(Self::Output, Self::OpeningHint), OpeningsError> {
        Self::commit_with_mode::<P, dory::ZK>(poly, setup)
    }

    #[expect(
        clippy::type_complexity,
        reason = "ZK openings return the native proof, hiding commitment, and blind"
    )]
    #[tracing::instrument(skip_all, name = "DoryScheme::open_zk")]
    fn open_zk<P: MultilinearPoly<Fr> + ?Sized>(
        poly: &P,
        point: &[Fr],
        _eval: Fr,
        setup: &Self::ProverSetup,
        hint: Self::OpeningHint,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(Self::Proof, Self::HidingCommitment, Self::Blind), OpeningsError> {
        let adapter = DorySourceAdapter::new(poly);
        let (nu, sigma) = matrix_shape(point.len());
        let (row_commitments, commit_blind) = hint.into_ark_parts();

        let ark_point: Vec<ArkFr> = point.iter().rev().map(jolt_fr_to_ark).collect();
        let mut dory_transcript = JoltToDoryTranscript::new(transcript);

        let (proof, y_blinding) =
            dory::prove::<ArkFr, InnerBN254, JoltG1Routines, JoltG2Routines, _, _, ZK>(
                &adapter,
                &ark_point,
                row_commitments,
                commit_blind,
                nu,
                sigma,
                &setup.0,
                &mut dory_transcript,
            )
            .map_err(|e| OpeningsError::ProveFailed(format!("dory::prove (ZK) failed: {e:?}")))?;

        let y_com = proof
            .y_com
            .map(ark_to_jolt_g1)
            .ok_or_else(|| OpeningsError::ProveFailed("ZK proof must contain y_com".to_owned()))?;
        let blinding = y_blinding.as_ref().map(ark_to_jolt_fr).ok_or_else(|| {
            OpeningsError::ProveFailed("ZK proof must return y_blinding".to_owned())
        })?;

        Ok((DoryProof(proof), y_com, blinding))
    }

    #[tracing::instrument(skip_all, name = "DoryScheme::verify_zk")]
    fn verify_zk(
        commitment: &Self::Output,
        point: &[Fr],
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<Self::HidingCommitment, OpeningsError> {
        if (proof.0.nu, proof.0.sigma) != matrix_shape(point.len()) {
            return Err(OpeningsError::VerificationFailed);
        }
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
}

fn commit_rows_dense<P: MultilinearPoly<Fr> + ?Sized>(
    poly: &P,
    sigma: usize,
    setup: &ArkworksProverSetup,
) -> Vec<ArkG1> {
    let num_cols = 1usize << sigma;
    let g1_bases = srs_prefix(&setup.g1_vec, num_cols);

    let mut rows: Vec<Vec<Fr>> = Vec::new();
    poly.for_each_row(sigma, &mut |_, row| rows.push(row.to_vec()));

    rows.par_iter()
        .map(|row| {
            let scalars: Vec<ArkFr> = row.iter().map(jolt_fr_to_ark).collect();
            JoltG1Routines::msm(srs_prefix(g1_bases, scalars.len()), &scalars)
        })
        .collect()
}

fn commit_rows_one_hot<P: MultilinearPoly<Fr> + ?Sized>(
    poly: &P,
    num_rows: usize,
    num_cols: usize,
    setup: &ArkworksProverSetup,
) -> Vec<ArkG1> {
    let g1_bases = srs_prefix(&setup.g1_vec, num_cols);

    let mut cols_per_row: Vec<Vec<usize>> = vec![Vec::new(); num_rows];
    poly.for_each_one(&mut |flat_idx| {
        let row = flat_idx / num_cols;
        let col = flat_idx % num_cols;
        debug_assert!(
            row < num_rows && col < num_cols,
            "for_each_one out-of-bounds flat_idx: row={row} num_rows={num_rows} col={col} num_cols={num_cols}",
        );
        #[expect(
            clippy::indexing_slicing,
            reason = "for_each_one yields flat_idx < num_rows * num_cols, so row = flat_idx / num_cols < num_rows"
        )]
        cols_per_row[row].push(col);
    });

    cols_per_row
        .par_iter()
        .map(|cols| {
            cols.iter()
                .fold(<InnerBN254 as PairingCurve>::G1::identity(), |acc, &col| {
                    #[expect(
                        clippy::indexing_slicing,
                        reason = "col = flat_idx % num_cols and g1_bases has exactly num_cols bases"
                    )]
                    let base = &g1_bases[col];
                    <InnerBN254 as PairingCurve>::G1::add(&acc, base)
                })
        })
        .collect()
}

fn compute_row_commitments<P: MultilinearPoly<Fr> + ?Sized>(
    poly: &P,
    setup: &DoryProverSetup,
) -> Result<Vec<ArkG1>, OpeningsError> {
    let num_vars = poly.num_vars();
    let (nu, sigma) = matrix_shape(num_vars);
    let max_cols = setup.0.g1_vec.len();
    let max_rows = setup.0.g2_vec.len();
    // An oversized polynomial must surface as the commit API's Err, not as an
    // out-of-bounds slice panic on the SRS below.
    let fits =
        sigma < usize::BITS as usize && (1usize << sigma) <= max_cols && (1usize << nu) <= max_rows;
    if !fits {
        return Err(OpeningsError::PolynomialTooLarge {
            poly_size: num_vars,
            setup_max: max_cols.ilog2() as usize + max_rows.ilog2() as usize,
        });
    }
    let num_cols = 1usize << sigma;
    let num_rows = 1usize << nu;

    Ok(if poly.is_one_hot() {
        commit_rows_one_hot(poly, num_rows, num_cols, &setup.0)
    } else {
        commit_rows_dense(poly, sigma, &setup.0)
    })
}

pub(crate) fn commit_rows_tier_2<M: Mode>(
    row_commitments: &[ArkG1],
    setup: &DoryProverSetup,
) -> (ArkGT, ArkFr) {
    let g2_bases = srs_prefix(&setup.0.g2_vec, row_commitments.len());
    let tier_2 = <InnerBN254 as PairingCurve>::multi_pair_g2_setup(row_commitments, g2_bases);
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
struct DorySourceAdapter<'a, S: MultilinearPoly<Fr> + ?Sized> {
    source: &'a S,
}

impl<'a, S: MultilinearPoly<Fr> + ?Sized> DorySourceAdapter<'a, S> {
    fn new(source: &'a S) -> Self {
        Self { source }
    }
}

impl<S: MultilinearPoly<Fr> + ?Sized> DoryPolynomial<ArkFr> for DorySourceAdapter<'_, S> {
    fn num_vars(&self) -> usize {
        self.source.num_vars()
    }

    fn evaluate(&self, point: &[ArkFr]) -> ArkFr {
        // Dory orders multilinear variables opposite to jolt's `MultilinearPoly`
        // (its `multilinear_lagrange_basis` is the bit-reverse of jolt's), which
        // is the order the VMV commitment and the non-ZK opening already use.
        // Reverse the dory point back to jolt order so this evaluation agrees
        // with the VMV. dory only calls this to compute the ZK `y_com` value;
        // without the reversal `y_com` commits to `source.evaluate(rev(point))`
        // instead of the opening claim, and the BlindFold eval-commitment check
        // (`y_com == claim·g + blind·h`) fails.
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

impl<S: MultilinearPoly<Fr> + ?Sized> MultilinearLagrange<ArkFr> for DorySourceAdapter<'_, S> {
    fn vector_matrix_product(&self, left_vec: &[ArkFr], _nu: usize, sigma: usize) -> Vec<ArkFr> {
        let native_left: Vec<Fr> = left_vec.iter().map(ark_to_jolt_fr).collect();
        let result = self.source.fold_rows(&native_left, sigma);
        result.iter().map(jolt_fr_to_ark).collect()
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests unwrap successful PCS operations")]

    use super::*;
    use jolt_crypto::{Pedersen, VectorCommitment};
    use jolt_field::{Field, Ring};
    use jolt_poly::Polynomial;
    use jolt_transcript::Blake2bTranscript;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    #[test]
    fn commit_rejects_polynomial_exceeding_setup_capacity() {
        let mut rng = ChaCha20Rng::seed_from_u64(700);
        let prover_setup = DoryScheme::setup_prover(2);
        let poly = Polynomial::<Fr>::random(6, &mut rng);
        let err = DoryScheme::commit(poly.evaluations(), &prover_setup).unwrap_err();
        assert!(
            matches!(err, OpeningsError::PolynomialTooLarge { .. }),
            "{err}"
        );
    }

    /// The square layout of a 4-variable polynomial is 4 rows of 4.
    const SQUARE_VARS: usize = 4;
    /// A wider layout of the same 4 variables: 2 rows of 8.
    const WIDE: (usize, usize) = (1, 3);

    /// Returns a polynomial whose square matrix has zero rows 2 and 3, and
    /// its wide twin: the same two nonzero rows, each padded with 4 zero
    /// columns. The twin is a different polynomial, but the square
    /// polynomial's commitment also commits to it under the wide layout.
    fn square_and_wide_twin(rng: &mut ChaCha20Rng) -> (Polynomial<Fr>, Polynomial<Fr>) {
        let zero = Fr::from_u64(0);
        let rows: Vec<[Fr; 4]> = (0..2)
            .map(|_| std::array::from_fn(|_| <Fr as Field>::random(rng)))
            .collect();
        let square = rows.iter().flatten().copied().chain([zero; 8]).collect();
        let wide = rows
            .iter()
            .flat_map(|row| row.iter().copied().chain([zero; 4]))
            .collect();
        (Polynomial::new(square), Polynomial::new(wide))
    }

    /// Proves `wide` at `point` under the wide layout, against the square
    /// commitment that `hint` belongs to.
    fn prove_wide<M: Mode>(
        wide: &[Fr],
        point: &[Fr],
        hint: DoryHint,
        setup: &DoryProverSetup,
    ) -> DoryProof {
        let (mut row_commitments, commit_blind) = hint.into_ark_parts();
        assert!(row_commitments
            .iter()
            .skip(1 << WIDE.0)
            .all(|row| *row == <ArkG1 as DoryGroup>::identity()));
        row_commitments.truncate(1 << WIDE.0);
        let ark_point: Vec<ArkFr> = point.iter().rev().map(jolt_fr_to_ark).collect();
        let mut transcript = Blake2bTranscript::new(b"matrix-shape");
        let (proof, _) = dory::prove::<ArkFr, InnerBN254, JoltG1Routines, JoltG2Routines, _, _, M>(
            &DorySourceAdapter::new(wide),
            &ark_point,
            row_commitments,
            commit_blind,
            WIDE.0,
            WIDE.1,
            &setup.0,
            &mut JoltToDoryTranscript::new(&mut transcript),
        )
        .unwrap();
        DoryProof(proof)
    }

    /// Runs dory's own verifier on `proof`, with no layout check.
    fn dory_accepts(
        commitment: &DoryCommitment,
        point: &[Fr],
        eval: Fr,
        proof: &DoryProof,
        setup: &DoryVerifierSetup,
    ) -> bool {
        let ark_point: Vec<ArkFr> = point.iter().rev().map(jolt_fr_to_ark).collect();
        let mut transcript = Blake2bTranscript::new(b"matrix-shape");
        dory::verify::<ArkFr, InnerBN254, JoltG1Routines, JoltG2Routines, _>(
            jolt_gt_to_ark(&commitment.0),
            jolt_fr_to_ark(&eval),
            &ark_point,
            &proof.0,
            setup.0.clone().into_inner(),
            &mut JoltToDoryTranscript::new(&mut transcript),
        )
        .is_ok()
    }

    /// dory accepts an opening of a square commitment under a wider layout,
    /// to the twin's evaluation. `verify` must reject it.
    #[test]
    fn verify_rejects_another_matrix_shape() {
        let mut rng = ChaCha20Rng::seed_from_u64(701);
        let prover_setup = DoryScheme::setup_prover(2 * WIDE.1);
        let verifier_setup = DoryScheme::setup_verifier(2 * WIDE.1);
        let (square, wide) = square_and_wide_twin(&mut rng);
        let point: Vec<Fr> = (0..SQUARE_VARS)
            .map(|_| <Fr as Field>::random(&mut rng))
            .collect();
        let (commitment, hint) = DoryScheme::commit(square.evaluations(), &prover_setup).unwrap();
        let wide_eval = wide.evaluate(&point);
        assert_ne!(wide_eval, square.evaluate(&point));

        let proof = prove_wide::<Transparent>(wide.evaluations(), &point, hint, &prover_setup);
        assert!(dory_accepts(
            &commitment,
            &point,
            wide_eval,
            &proof,
            &verifier_setup
        ));

        let mut transcript = Blake2bTranscript::new(b"matrix-shape");
        let result = DoryScheme::verify(
            &commitment,
            &point,
            wide_eval,
            &proof,
            &verifier_setup,
            &mut transcript,
        );
        assert!(
            matches!(result, Err(OpeningsError::VerificationFailed)),
            "{result:?}"
        );
    }

    /// The zero-knowledge version of [`verify_rejects_another_matrix_shape`].
    #[test]
    fn verify_zk_rejects_another_matrix_shape() {
        let mut rng = ChaCha20Rng::seed_from_u64(702);
        let prover_setup = DoryScheme::setup_prover(2 * WIDE.1);
        let verifier_setup = DoryScheme::setup_verifier(2 * WIDE.1);
        let (square, wide) = square_and_wide_twin(&mut rng);
        let point: Vec<Fr> = (0..SQUARE_VARS)
            .map(|_| <Fr as Field>::random(&mut rng))
            .collect();
        let (commitment, hint) =
            DoryScheme::commit_zk(square.evaluations(), &prover_setup).unwrap();

        let proof = prove_wide::<ZK>(wide.evaluations(), &point, hint, &prover_setup);
        // In ZK mode dory reads the evaluation from the proof's `y_com`.
        assert!(dory_accepts(
            &commitment,
            &point,
            Fr::from_u64(0),
            &proof,
            &verifier_setup
        ));

        let mut transcript = Blake2bTranscript::new(b"matrix-shape");
        let result = DoryScheme::verify_zk(
            &commitment,
            &point,
            &proof,
            &verifier_setup,
            &mut transcript,
        );
        assert!(
            matches!(result, Err(OpeningsError::VerificationFailed)),
            "{result:?}"
        );
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
            <Fr as Ring>::from_u64(1),
            <Fr as Ring>::from_u64(2),
            <Fr as Ring>::from_u64(3),
        ];
        let blinding = <Fr as Ring>::from_u64(42);
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
