//! Ring-relation prover for the Akita PCS (§4.2).
//!
//! Builds the stage-1 relation instance and witness (`M`, `y`, `z`, `v`) via
//! [`RingRelationProver`].
#[cfg(feature = "zk")]
use crate::protocol::masking::sample_blinding_digits;
use crate::validation::validate_i8_setup_log_basis;
use crate::{
    AkitaPolyOps, CyclicRowsComputeBackend, DecomposeFoldWitness, DigitRowsComputeBackend,
    RingSwitchComputeBackend, RingSwitchQuotientRowsPlan, RingSwitchRelationRowsPlan,
};
use akita_algebra::ring::cyclotomic::BalancedDecomposePow2I8Params;
use akita_algebra::CyclotomicRing;
use akita_challenges::{Challenges, IntegerChallenge, SparseChallenge};
use akita_field::parallel::*;
use akita_field::AkitaError;
use akita_field::{CanonicalField, FieldCore, FromPrimitiveInt, HalvingField};
use akita_transcript::labels::{ABSORB_PROVER_V, ABSORB_TERMINAL_E_HAT};
use akita_transcript::Transcript;
#[cfg(feature = "zk")]
use akita_types::terminal_e_hat_bytes_from_blocks;
use akita_types::{
    gadget_row_scalars, AkitaCommitmentHint, FlatDigitBlocks, MRowLayout, RingCommitment,
    RingSliceSerializer,
};
use akita_types::{LevelParams, OpeningBatch, RingRelationInstance};
use akita_types::{RingMultiplierOpeningPoint, RingOpeningPoint};

use super::fold_grind::{self, ProverTranscriptGrind};
use super::ring_relation_witness::RingRelationWitness;
use std::time::Instant;

mod relation_quotient;
mod repeated_b;

pub use akita_types::generate_y;
pub use relation_quotient::compute_relation_quotient;

#[cfg(feature = "zk")]
fn absorb_terminal_e_hat<F, T, const D: usize>(
    transcript: &mut T,
    e_hat: &FlatDigitBlocks<D>,
    planes_per_block: usize,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    let bytes = terminal_e_hat_bytes_from_blocks(e_hat, planes_per_block)?;
    transcript.absorb_and_record_bytes(ABSORB_TERMINAL_E_HAT, &bytes);
    Ok(())
}

#[cfg(not(feature = "zk"))]
fn absorb_terminal_e_folded_fields<F, T, const D: usize>(
    transcript: &mut T,
    e_folded: &[CyclotomicRing<F, D>],
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + akita_serialization::AkitaSerialize,
    T: Transcript<F>,
{
    let bytes = akita_types::e_folded_segment_bytes::<F, D>(e_folded)?;
    if bytes.is_empty() {
        return Err(AkitaError::InvalidInput(
            "terminal e_folded absorb cannot be empty".to_string(),
        ));
    }
    transcript.absorb_and_record_bytes(ABSORB_TERMINAL_E_HAT, &bytes);
    Ok(())
}

fn decompose_e_hat<F: FieldCore + CanonicalField, const D: usize>(
    pre_folded_e: &[Vec<CyclotomicRing<F, D>>],
    depth_open: usize,
    log_basis: u32,
) -> Result<FlatDigitBlocks<D>, AkitaError> {
    let q = (-F::one()).to_canonical_u128() + 1;
    let decompose_params = BalancedDecomposePow2I8Params::new(depth_open, log_basis, q);
    let total_rows: usize = pre_folded_e.iter().map(Vec::len).sum();
    let mut e_hat = FlatDigitBlocks::zeroed(vec![depth_open; total_rows])?;
    let mut offset = 0usize;
    for folded_rows in pre_folded_e {
        for w_i in folded_rows {
            w_i.balanced_decompose_pow2_i8_into_with_params(
                &mut e_hat.flat_digits_mut()[offset..offset + depth_open],
                &decompose_params,
            );
            offset += depth_open;
        }
    }
    Ok(e_hat)
}

fn flatten_commitment_hints_for_ring_relation<F, const D: usize>(
    hints: Vec<AkitaCommitmentHint<F, D>>,
    group_sizes: &[usize],
    num_digits_open: usize,
    log_basis: u32,
) -> Result<AkitaCommitmentHint<F, D>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    if hints.len() != group_sizes.len() {
        return Err(AkitaError::InvalidInput(
            "prover hint group count does not match commitment groups".to_string(),
        ));
    }

    let mut decomposed_inner_rows = Vec::new();
    let mut t_rows_by_poly = Vec::new();
    #[cfg(feature = "zk")]
    let mut b_blinding_digits = Vec::new();
    for (mut hint, &group_size) in hints.into_iter().zip(group_sizes.iter()) {
        if hint.decomposed_inner_rows.len() != group_size {
            return Err(AkitaError::InvalidInput(
                "prover hint group sizes do not match polynomial groups".to_string(),
            ));
        }
        hint.ensure_recomposed_inner_rows(num_digits_open, log_basis)?;
        #[cfg(feature = "zk")]
        let (digits_by_poly, rows_by_poly, mut blinding_by_group) = hint.into_parts();
        #[cfg(not(feature = "zk"))]
        let (digits_by_poly, rows_by_poly) = hint.into_parts();
        #[cfg(feature = "zk")]
        if blinding_by_group.len() != 1 {
            return Err(AkitaError::InvalidInput(
                "prover hint must carry exactly one blinding group per commitment".to_string(),
            ));
        }
        decomposed_inner_rows.extend(digits_by_poly);
        let rows_by_poly = rows_by_poly.ok_or_else(|| {
            AkitaError::InvalidInput("missing recomposed inner rows in prover hint".to_string())
        })?;
        t_rows_by_poly.extend(rows_by_poly);
        #[cfg(feature = "zk")]
        b_blinding_digits.append(&mut blinding_by_group);
    }

    #[cfg(feature = "zk")]
    {
        Ok(AkitaCommitmentHint::with_recomposed_inner_rows(
            decomposed_inner_rows,
            t_rows_by_poly,
            b_blinding_digits,
        ))
    }
    #[cfg(not(feature = "zk"))]
    {
        Ok(AkitaCommitmentHint::with_recomposed_inner_rows(
            decomposed_inner_rows,
            t_rows_by_poly,
        ))
    }
}

fn aggregate_decompose_fold_witnesses<F: FieldCore, const D: usize>(
    witnesses: Vec<DecomposeFoldWitness<F, D>>,
) -> Result<DecomposeFoldWitness<F, D>, AkitaError> {
    let Some((first, rest)) = witnesses.split_first() else {
        return Err(AkitaError::InvalidInput(
            "batched decompose_fold requires at least one witness".to_string(),
        ));
    };
    let z_len = first.z_folded_rings.len();
    let coeff_len = first.centered_coeffs.len();
    let mut z_folded_rings = first.z_folded_rings.clone();
    let mut centered_coeffs = first.centered_coeffs.clone();

    for witness in rest {
        if witness.z_folded_rings.len() != z_len || witness.centered_coeffs.len() != coeff_len {
            return Err(AkitaError::InvalidInput(
                "batched decompose_fold witness length mismatch".to_string(),
            ));
        }
        for (dst, src) in z_folded_rings.iter_mut().zip(witness.z_folded_rings.iter()) {
            *dst += *src;
        }
        for (dst, src) in centered_coeffs
            .iter_mut()
            .zip(witness.centered_coeffs.iter())
        {
            for k in 0..D {
                dst[k] = dst[k].checked_add(src[k]).ok_or_else(|| {
                    AkitaError::InvalidInput(
                        "batched decompose_fold centered coefficient overflow".to_string(),
                    )
                })?;
            }
        }
    }

    let centered_inf_norm = centered_coeffs
        .iter()
        .flat_map(|coeffs| coeffs.iter())
        .map(|coeff| coeff.unsigned_abs())
        .max()
        .unwrap_or(0);

    Ok(DecomposeFoldWitness {
        z_folded_rings,
        centered_coeffs,
        centered_inf_norm,
    })
}

pub(super) fn build_point_decompose_fold_witness<F, P, const D: usize>(
    challenges: &Challenges,
    point_polys: &[&P],
    point_indices: &[usize],
    lp: &LevelParams,
) -> Result<DecomposeFoldWitness<F, D>, AkitaError>
where
    F: FieldCore,
    P: AkitaPolyOps<F, D>,
{
    match challenges {
        Challenges::Sparse {
            challenges: sparse,
            num_blocks_per_claim,
            ..
        } => {
            let mut point_challenges =
                Vec::with_capacity(point_indices.len() * *num_blocks_per_claim);
            for &claim_idx in point_indices {
                let start = claim_idx
                    .checked_mul(*num_blocks_per_claim)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("batched challenge offset overflow".to_string())
                    })?;
                let end = start.checked_add(*num_blocks_per_claim).ok_or_else(|| {
                    AkitaError::InvalidSetup("batched challenge offset overflow".to_string())
                })?;
                point_challenges.extend_from_slice(sparse.get(start..end).ok_or(
                    AkitaError::InvalidSize {
                        expected: end,
                        actual: sparse.len(),
                    },
                )?);
            }
            if let Some(z_point) = P::decompose_fold_batched(
                point_polys,
                &point_challenges,
                lp.block_len,
                lp.num_digits_commit,
                lp.log_basis,
            ) {
                Ok(z_point)
            } else {
                let witnesses: Vec<DecomposeFoldWitness<F, D>> = point_polys
                    .iter()
                    .zip(point_challenges.chunks(*num_blocks_per_claim))
                    .map(|(poly, poly_challenges)| {
                        poly.decompose_fold(
                            poly_challenges,
                            lp.block_len,
                            lp.num_digits_commit,
                            lp.log_basis,
                        )
                    })
                    .collect();
                aggregate_decompose_fold_witnesses(witnesses)
            }
        }
        Challenges::Tensor { factored: _ } => {
            let selected = challenges.select_claims::<D>(point_indices)?;
            let point_factored = match selected {
                Challenges::Tensor { factored } => factored,
                Challenges::Sparse { .. } => {
                    return Err(AkitaError::InvalidSetup(
                        "tensor claim selection returned sparse challenges".to_string(),
                    ))
                }
            };
            match P::decompose_fold_tensor_batched(
                point_polys,
                &point_factored,
                lp.block_len,
                lp.num_digits_commit,
                lp.log_basis,
            )? {
                Some(witness) => Ok(witness),
                None => Err(AkitaError::InvalidSetup(
                    "polynomial backend has no tensor-shaped fold kernel".to_string(),
                )),
            }
        }
    }
}

/// Compute the D-side relation rows `v = D · e_hat` (plus ZK blinding when enabled).
fn compute_v_rows<F, B, const D: usize>(
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    row_len: usize,
    e_hat: &FlatDigitBlocks<D>,
    log_basis: u32,
    #[cfg(feature = "zk")] d_blinding_digits: &FlatDigitBlocks<D>,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + CanonicalField,
    B: DigitRowsComputeBackend<F>,
{
    #[cfg(feature = "zk")]
    {
        let mut rows =
            backend.digit_rows::<D>(prepared, row_len, e_hat.flat_digits(), log_basis)?;
        let blinding_rows = backend.zk_d_digit_rows::<D>(
            prepared,
            row_len,
            d_blinding_digits.flat_digits().len(),
            d_blinding_digits.flat_digits(),
        )?;
        for (row, blinding) in rows.iter_mut().zip(blinding_rows) {
            *row += blinding;
        }
        if rows.len() != row_len {
            return Err(AkitaError::InvalidProof);
        }
        Ok(rows)
    }
    #[cfg(not(feature = "zk"))]
    {
        let rows = backend.digit_rows::<D>(prepared, row_len, e_hat.flat_digits(), log_basis)?;
        if rows.len() != row_len {
            return Err(AkitaError::InvalidProof);
        }
        Ok(rows)
    }
}

fn compute_v_rows_for_layout<F, T, B, const D: usize>(
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    transcript: &mut T,
    lp: &LevelParams,
    e_hat: &FlatDigitBlocks<D>,
    m_row_layout: MRowLayout,
    #[cfg(feature = "zk")] d_blinding_digits: &FlatDigitBlocks<D>,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
    B: DigitRowsComputeBackend<F>,
{
    match m_row_layout {
        MRowLayout::WithDBlock => {
            let _span = tracing::info_span!(
                "compute_relation_v",
                e_hat_planes = e_hat.flat_digits().len()
            )
            .entered();
            let v = compute_v_rows(
                backend,
                prepared,
                lp.d_key.row_len(),
                e_hat,
                lp.log_basis,
                #[cfg(feature = "zk")]
                d_blinding_digits,
            )?;
            transcript.append_serde(ABSORB_PROVER_V, &RingSliceSerializer(&v));
            Ok(v)
        }
        MRowLayout::WithoutDBlock => Ok(Vec::new()),
    }
}

/// Prover-side builder for the ring relation $M(x) \cdot z = y(x) + (X^D + 1) \cdot r(x)$.
pub struct RingRelationProver;

impl RingRelationProver {
    /// Root-level constructor for one shared opening point with one or more
    /// polynomial slots.
    ///
    /// `opening_point` is the single ring-level opening point used by the
    /// batch.
    /// For the trivial single-claim case use `polys = &[poly]` and
    /// `gamma = vec![F::one()]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the batched hints, folded witnesses, or decomposed
    /// aggregate witness are malformed.
    ///
    /// # Panics
    ///
    /// Panics if the batched `e_hat` decomposition or flattened batched hints
    /// produced by the prover do not preserve the expected block sizes.  These
    /// invariants hold by construction for well-formed inputs accepted by the
    /// error checks above and are therefore treated as internal programming
    /// errors rather than recoverable failures.
    #[allow(clippy::too_many_arguments, clippy::new_ret_no_self)]
    #[tracing::instrument(skip_all, name = "RingRelationProver::new")]
    #[inline(never)]
    pub fn new<F, const D: usize, T, P, B>(
        backend: &B,
        prepared: &B::PreparedSetup<D>,
        opening_point: RingOpeningPoint<F>,
        ring_multiplier_point: RingMultiplierOpeningPoint<F, D>,
        polys: &[&P],
        pre_folded_e_by_poly: Vec<Vec<CyclotomicRing<F, D>>>,
        opening_batch: OpeningBatch,
        lp: LevelParams,
        hints: Vec<AkitaCommitmentHint<F, D>>,
        transcript: &mut T,
        commitments: &[RingCommitment<F, D>],
        row_coefficient_rings: Vec<CyclotomicRing<F, D>>,
        m_row_layout: MRowLayout,
    ) -> Result<(RingRelationInstance<F, D>, RingRelationWitness<F, D>), AkitaError>
    where
        F: FieldCore + CanonicalField,
        T: Transcript<F> + ProverTranscriptGrind<F>,
        P: AkitaPolyOps<F, D>,
        B: DigitRowsComputeBackend<F>,
    {
        {
            let x: u8 = 0;
            tracing::trace!(
                stack_ptr = format_args!("{:#x}", &x as *const u8 as usize),
                "RingRelationProver::new"
            );
        }
        validate_i8_setup_log_basis(lp.log_basis, "for i8 prover decomposition")?;
        if opening_point.a.len() < lp.block_len || opening_point.b.len() != lp.num_blocks {
            return Err(AkitaError::InvalidInput(
                "batched prover opening-point layout mismatch".to_string(),
            ));
        }
        if ring_multiplier_point.a_len() < lp.block_len
            || ring_multiplier_point.b_len() != lp.num_blocks
        {
            return Err(AkitaError::InvalidInput(
                "batched prover ring-multiplier opening-point layout mismatch".to_string(),
            ));
        }
        let num_claims = opening_batch.num_claims();
        if polys.is_empty() {
            return Err(AkitaError::InvalidInput(
                "batched prover requires at least one polynomial".to_string(),
            ));
        }
        if polys.len() != pre_folded_e_by_poly.len()
            || polys.len() != num_claims
            || opening_batch.claim_poly_indices().len() != num_claims
            || hints.len() != opening_batch.num_polys_per_commitment_group().len()
            || commitments.len() != opening_batch.num_polys_per_commitment_group().len()
        {
            return Err(AkitaError::InvalidInput(
                "batched prover input lengths do not match".to_string(),
            ));
        }
        for commitment in commitments {
            if commitment.u.len() != lp.effective_commit_rows() {
                return Err(AkitaError::InvalidInput(
                    "batched prover received a commitment with the wrong length".to_string(),
                ));
            }
        }
        if row_coefficient_rings.len() != num_claims {
            return Err(AkitaError::InvalidInput(
                "batched prover row coefficient length does not match claim count".to_string(),
            ));
        }
        let gamma = row_coefficient_rings
            .iter()
            .map(|ring| ring.coefficients()[0])
            .collect::<Vec<_>>();

        let e_hat = {
            let _span = tracing::info_span!("decompose_batched_e_hat").entered();
            decompose_e_hat::<F, D>(&pre_folded_e_by_poly, lp.num_digits_open, lp.log_basis)?
        };
        let flattened_hint = flatten_commitment_hints_for_ring_relation::<F, D>(
            hints,
            opening_batch.num_polys_per_commitment_group(),
            lp.num_digits_open,
            lp.log_basis,
        )?;

        // Terminal layout drops the D-block from the M-matrix entirely:
        // `v = D · e_hat` never travels on the wire, the verifier never
        // reconstructs it, and downstream prover paths (`ring_switch_build_w`,
        // `relation_claim_from_rows_extension`) consume an empty `v` slice.
        // Skip both the D-side blinding sample and the D-NTT under Terminal.
        #[cfg(feature = "zk")]
        let d_blinding_digits = match m_row_layout {
            MRowLayout::WithDBlock => {
                sample_blinding_digits::<F, D>(lp.d_key.row_len(), lp.log_basis)?
            }
            MRowLayout::WithoutDBlock => FlatDigitBlocks::<D>::empty(),
        };

        let v = compute_v_rows_for_layout::<F, T, B, D>(
            backend,
            prepared,
            transcript,
            &lp,
            &e_hat,
            m_row_layout,
            #[cfg(feature = "zk")]
            &d_blinding_digits,
        )?;

        if matches!(m_row_layout, MRowLayout::WithoutDBlock) {
            #[cfg(not(feature = "zk"))]
            {
                let e_folded_flat: Vec<CyclotomicRing<F, D>> = pre_folded_e_by_poly
                    .iter()
                    .flat_map(|block| block.iter().cloned())
                    .collect();
                absorb_terminal_e_folded_fields::<F, T, D>(transcript, &e_folded_flat)?;
            }
            #[cfg(feature = "zk")]
            absorb_terminal_e_hat::<F, T, D>(transcript, &e_hat, lp.num_digits_open)?;
        }
        let (z_folded_rings, challenges, fold_grind_nonce) =
            fold_grind::sample_fold_decompose_witness::<F, _, T, D>(
                transcript, polys, &lp, num_claims,
            )?;

        let commitment_rows = commitments
            .iter()
            .flat_map(|commitment| commitment.u.iter().copied())
            .collect::<Vec<_>>();
        // Terminal levels drop the D-block from M entirely, so `y` must
        // also drop the D-rows (the `v = D · ŵ` segment). Pass an empty
        // `v` slice with `n_d_active = 0` so `generate_y` emits
        // `[consistency | commitment_rows | A-zeros]` (no D-block).
        let (y_v_slice, n_d_active) = match m_row_layout {
            MRowLayout::WithDBlock => (v.as_slice(), lp.d_key.row_len()),
            MRowLayout::WithoutDBlock => (&[][..], 0usize),
        };
        let y = generate_y::<F, D>(
            y_v_slice,
            &commitment_rows,
            n_d_active,
            lp.effective_commit_rows(),
            lp.b_inner_rows_per_group(),
            lp.a_key.row_len(),
        )?;
        let e_folded = pre_folded_e_by_poly.into_iter().flatten().collect();

        let instance = RingRelationInstance::new(
            m_row_layout,
            challenges,
            opening_point,
            ring_multiplier_point,
            opening_batch,
            gamma,
            row_coefficient_rings,
            y,
            v,
        )?;
        instance.check_v_shape_for_level(&lp)?;
        let witness = RingRelationWitness {
            z_folded_rings,
            fold_grind_nonce,
            e_hat,
            e_folded,
            hint: flattened_hint,
            #[cfg(feature = "zk")]
            d_blinding_digits,
        };
        Ok((instance, witness))
    }
}
