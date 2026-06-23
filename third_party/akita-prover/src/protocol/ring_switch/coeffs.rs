use super::*;
use crate::validation::validate_i8_setup_log_basis;
use akita_serialization::AkitaSerialize;
#[cfg(feature = "zk")]
use std::marker::PhantomData;

/// Prover-side ring artifacts retained for segment-typed terminal encoding.
#[cfg(not(feature = "zk"))]
pub struct RingSwitchTerminalArtifacts<F: FieldCore, const D: usize> {
    pub e_folded: Vec<akita_algebra::CyclotomicRing<F, D>>,
    pub recomposed_inner_rows: Vec<Vec<akita_algebra::CyclotomicRing<F, D>>>,
    pub z_folded_centered: Vec<[i32; D]>,
    pub r: Vec<akita_algebra::CyclotomicRing<F, D>>,
    pub u_concat_planes: usize,
}

/// Output of [`ring_switch_build_w`].
pub struct RingSwitchBuildOutput<F: FieldCore, const D: usize> {
    pub w: RecursiveWitnessFlat,
    #[cfg(not(feature = "zk"))]
    pub terminal_artifacts: Option<RingSwitchTerminalArtifacts<F, D>>,
    #[cfg(feature = "zk")]
    _marker: PhantomData<F>,
}

/// Build the witness vector `w` from the ring-relation witness.
///
/// This is the first half of the ring switch: it computes `r` and assembles
/// `w` as a flat recursive witness. The resulting `w` is D-agnostic and can be
/// committed at any supported ring dimension by the recursive commitment path.
///
/// # Errors
///
/// Returns an error if the ring-relation witness is missing prover-side data.
///
/// # Panics
///
/// Panics with `feature = "zk"` enabled if the zero-length `FlatDigitBlocks`
/// constructor rejects an empty vector (an invariant of the type).
#[tracing::instrument(skip_all, name = "ring_switch_build_w")]
#[allow(clippy::too_many_arguments)]
#[cfg_attr(feature = "zk", allow(unused_variables))]
#[inline(never)]
pub fn ring_switch_build_w<F, B, const D: usize>(
    instance: &RingRelationInstance<F, D>,
    witness: RingRelationWitness<F, D>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    lp: &LevelParams,
    retain_terminal_artifacts: bool,
) -> Result<RingSwitchBuildOutput<F, D>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + FromPrimitiveInt
        + HalvingField
        + AkitaSerialize,
    B: RingSwitchComputeBackend<F>,
{
    let num_claims = instance.opening_batch().num_claims();
    {
        let x: u8 = 0;
        tracing::trace!(
            stack_ptr = format_args!("{:#x}", &x as *const u8 as usize),
            "ring_switch_build_w"
        );
    }
    let RingRelationWitness {
        z_folded_rings,
        fold_grind_nonce: _,
        e_hat,
        e_folded,
        mut hint,
        #[cfg(feature = "zk")]
        d_blinding_digits,
    } = witness;
    validate_i8_setup_log_basis(lp.log_basis, "for i8 prover decomposition")?;
    hint.ensure_recomposed_inner_rows(lp.num_digits_open, lp.log_basis)?;
    #[cfg(feature = "zk")]
    let (decomposed_inner_rows, recomposed_inner_rows, b_blinding_digits) = hint.into_flat_parts();
    #[cfg(not(feature = "zk"))]
    let (decomposed_inner_rows, recomposed_inner_rows) = hint.into_flat_parts();
    let recomposed_inner_rows = recomposed_inner_rows.ok_or_else(|| {
        AkitaError::InvalidInput("missing recomposed inner rows in prover hint".to_string())
    })?;
    let opening_batch = instance.opening_batch();

    let (r, u_concat_digits) = compute_relation_quotient::<F, B, D>(
        backend,
        prepared,
        lp,
        &instance.challenges,
        e_hat.flat_digits(),
        #[cfg(feature = "zk")]
        &d_blinding_digits,
        &decomposed_inner_rows,
        #[cfg(feature = "zk")]
        &b_blinding_digits,
        &recomposed_inner_rows,
        &e_folded,
        instance.ring_multiplier_point(),
        opening_batch.claim_to_commitment_group(),
        opening_batch.claim_poly_indices(),
        instance.row_coefficient_rings(),
        &z_folded_rings.centered_coeffs,
        z_folded_rings.centered_inf_norm,
        instance.y(),
        opening_batch.num_polys_per_commitment_group(),
        lp.num_blocks,
        lp.inner_width(),
        instance.m_row_layout(),
    )?;
    // Terminal layout drops the D-block from M and from the witness; the
    // d-blinding column segment must also disappear so the prover witness
    // matches the verifier's column offsets.
    #[cfg(feature = "zk")]
    let d_blinding_for_w: FlatDigitBlocks<D> = match instance.m_row_layout() {
        MRowLayout::WithDBlock => d_blinding_digits,
        MRowLayout::WithoutDBlock => {
            FlatDigitBlocks::zeroed(Vec::new()).expect("empty FlatDigitBlocks always valid")
        }
    };
    #[cfg(not(feature = "zk"))]
    let z_centered = z_folded_rings.centered_coeffs.clone();
    let w = {
        let _span = tracing::info_span!("build_w_coeffs").entered();
        build_w_coeffs::<F, D>(
            &e_hat,
            #[cfg(feature = "zk")]
            &d_blinding_for_w,
            &decomposed_inner_rows,
            #[cfg(feature = "zk")]
            &b_blinding_digits,
            &u_concat_digits,
            &z_folded_rings.centered_coeffs,
            &r,
            lp,
            num_claims,
        )
    };
    #[cfg(not(feature = "zk"))]
    let terminal_artifacts = if retain_terminal_artifacts {
        Some(RingSwitchTerminalArtifacts {
            e_folded,
            recomposed_inner_rows,
            z_folded_centered: z_centered,
            r,
            u_concat_planes: u_concat_digits.len(),
        })
    } else {
        None
    };
    Ok(RingSwitchBuildOutput {
        w,
        #[cfg(not(feature = "zk"))]
        terminal_artifacts,
        #[cfg(feature = "zk")]
        _marker: PhantomData,
    })
}

pub(super) fn balanced_decompose_centered_i32_i8_into<const D: usize>(
    centered: &[i32; D],
    out: &mut [[i8; D]],
    log_basis: u32,
) {
    let levels = out.len();
    assert!(
        log_basis > 0 && log_basis <= 6,
        "log_basis must be in 1..=6 for i8 output"
    );
    assert!(
        (levels as u32).saturating_mul(log_basis) <= 128 + log_basis,
        "levels * log_basis must be <= 128 + log_basis"
    );

    let half_b = 1i128 << (log_basis - 1);
    let b = half_b << 1;
    let mask = b - 1;

    for coeff_idx in 0..D {
        let mut c = centered[coeff_idx] as i128;
        for plane in out.iter_mut() {
            let d = c & mask;
            let balanced = if d >= half_b { d - b } else { d };
            c = (c - balanced) >> log_basis;
            plane[coeff_idx] = balanced as i8;
        }
    }
}

/// Emit flat digit planes contiguously (no block transpose). Used for the
/// tiered `û_concat` segment; a no-op for the single-tier path (empty slice).
fn emit_flat_planes<const D: usize>(out: &mut Vec<i8>, planes: &[[i8; D]]) {
    for plane in planes {
        out.extend_from_slice(plane);
    }
}

#[cfg(feature = "zk")]
fn emit_blinding_planes<const D: usize>(
    out: &mut Vec<i8>,
    blinding_by_group: &[FlatDigitBlocks<D>],
) {
    for blinding in blinding_by_group {
        for plane in blinding.flat_digits() {
            out.extend_from_slice(plane);
        }
    }
}

/// Decompose centered `z` fold response coeffs and emit digit-major planes.
fn emit_z_folded_block_inner<const D: usize>(
    out: &mut Vec<i8>,
    z_folded_centered: &[[i32; D]],
    block_len: usize,
    depth_commit: usize,
    num_digits_fold: usize,
    log_basis: u32,
) {
    let total_elems = z_folded_centered.len();
    let inner_width = block_len * depth_commit;
    debug_assert_eq!(
        total_elems % inner_width,
        0,
        "z_folded_rings length {total_elems} not divisible by inner_width {inner_width}",
    );

    let mut all_planes = vec![[0i8; D]; total_elems * num_digits_fold];
    for (k, z_j) in z_folded_centered.iter().enumerate() {
        balanced_decompose_centered_i32_i8_into(
            z_j,
            &mut all_planes[k * num_digits_fold..(k + 1) * num_digits_fold],
            log_basis,
        );
    }
    akita_types::emit_witness_z_folded_planes_inner::<D>(
        out,
        &all_planes,
        block_len,
        depth_commit,
        num_digits_fold,
        total_elems,
    );
}

/// Build the committed witness polynomial from ring-domain digit planes.
///
/// Emits field-domain coefficients in digit-major order (block index innermost)
/// with adaptive segment ordering: the segment whose block dimension is the
/// larger power of two comes first.
///
/// Segment ordering:
/// - If `m_vars >= r_vars`: z-hat (`2^m` blocks), e-hat + t-hat (`2^r` blocks), r-hat
/// - If `m_vars < r_vars`: e-hat + t-hat (`2^r` blocks), z-hat (`2^m` blocks), r-hat
///
/// Within each segment, the power-of-2 block index is the fastest-varying
/// (innermost) dimension.
///
/// `FlatDigitBlocks` stores ring-domain data in block-major order (all digit
/// planes for one block contiguously), which is natural for ring-domain matvec
/// and recomposition. This function transposes opening digits to digit-major at
/// the ring-to-field boundary; ZK blinding streams are already direct
/// digit-plane sources and are emitted in matrix-column order.
///
/// # Panics
///
/// Panics if the caller supplies digit blocks whose plane counts do not match
/// the fold layout in `lp`, or if ZK blinding digit counts do not match the
/// configured blinding columns.
#[allow(clippy::too_many_arguments)]
pub fn build_w_coeffs<F: CanonicalField, const D: usize>(
    e_hat: &FlatDigitBlocks<D>,
    #[cfg(feature = "zk")] d_blinding_digits: &FlatDigitBlocks<D>,
    t_hat: &FlatDigitBlocks<D>,
    #[cfg(feature = "zk")] b_blinding_digits: &[FlatDigitBlocks<D>],
    u_concat_digits: &[[i8; D]],
    z_folded_centered: &[[i32; D]],
    r: &[CyclotomicRing<F, D>],
    lp: &LevelParams,
    num_claims: usize,
) -> RecursiveWitnessFlat {
    let log_basis = lp.log_basis;
    let num_digits_fold = lp
        .num_digits_fold(num_claims, F::modulus_bits())
        .expect("build_w_coeffs: degenerate fold bound in validated level params");
    let depth_open = lp.num_digits_open;
    let depth_commit = lp.num_digits_commit;
    let block_len = lp.block_len;
    let levels = r_decomp_levels::<F>(log_basis);

    let e_hat_planes = e_hat.flat_digits().len();
    let t_hat_planes = t_hat.flat_digits().len();
    #[cfg(feature = "zk")]
    let d_blinding_planes = d_blinding_digits.flat_digits().len();
    #[cfg(not(feature = "zk"))]
    let d_blinding_planes = 0usize;
    #[cfg(feature = "zk")]
    let b_blinding_planes: usize = b_blinding_digits
        .iter()
        .map(|digits| digits.flat_digits().len())
        .sum();
    #[cfg(not(feature = "zk"))]
    let b_blinding_planes = 0usize;
    // Tiered: the hidden decomposed concatenated slice images `û_concat` are a
    // flat contiguous segment emitted immediately after `t̂` (at `offset_u`).
    let u_concat_planes = u_concat_digits.len();
    let z_count = e_hat_planes
        + d_blinding_planes
        + t_hat_planes
        + u_concat_planes
        + b_blinding_planes
        + z_folded_centered.len() * num_digits_fold;
    let r_hat_count = r.len() * levels;
    let z_first = akita_types::ring_column_z_first(lp);
    tracing::debug!(
        e_hat_planes,
        d_blinding_planes,
        t_hat_planes,
        b_blinding_planes,
        z_folded_elems = z_folded_centered.len(),
        z_folded_planes = z_folded_centered.len() * num_digits_fold,
        r_elems = r.len(),
        r_planes = r_hat_count,
        total_ring = z_count + r_hat_count,
        total_field = (z_count + r_hat_count) * D,
        z_first,
        "build_w_coeffs"
    );
    let total_planes = z_count + r_hat_count;
    let total_elems = total_planes * D;

    let mut out = Vec::with_capacity(total_elems);

    let w_block_count = e_hat.block_count();
    assert_eq!(
        e_hat_planes,
        w_block_count * depth_open,
        "build_w_coeffs: e_hat block layout does not match open digit depth"
    );
    let t_block_count = t_hat.block_count();
    let t_planes_per_block = if t_block_count == 0 {
        0
    } else {
        assert_eq!(
            t_hat_planes % t_block_count,
            0,
            "build_w_coeffs: t_hat block layout must be uniform"
        );
        t_hat_planes / t_block_count
    };

    if z_first {
        emit_z_folded_block_inner(
            &mut out,
            z_folded_centered,
            block_len,
            depth_commit,
            num_digits_fold,
            log_basis,
        );
        akita_types::emit_witness_planes_block_inner(
            &mut out,
            e_hat.flat_digits(),
            w_block_count,
            depth_open,
        );
        akita_types::emit_witness_planes_block_inner(
            &mut out,
            t_hat.flat_digits(),
            t_block_count,
            t_planes_per_block,
        );
        emit_flat_planes(&mut out, u_concat_digits);
        #[cfg(feature = "zk")]
        emit_blinding_planes(&mut out, b_blinding_digits);
        #[cfg(feature = "zk")]
        emit_blinding_planes(&mut out, std::slice::from_ref(d_blinding_digits));
    } else {
        akita_types::emit_witness_planes_block_inner(
            &mut out,
            e_hat.flat_digits(),
            w_block_count,
            depth_open,
        );
        akita_types::emit_witness_planes_block_inner(
            &mut out,
            t_hat.flat_digits(),
            t_block_count,
            t_planes_per_block,
        );
        emit_flat_planes(&mut out, u_concat_digits);
        #[cfg(feature = "zk")]
        emit_blinding_planes(&mut out, b_blinding_digits);
        #[cfg(feature = "zk")]
        emit_blinding_planes(&mut out, std::slice::from_ref(d_blinding_digits));
        emit_z_folded_block_inner(
            &mut out,
            z_folded_centered,
            block_len,
            depth_commit,
            num_digits_fold,
            log_basis,
        );
    }

    let mut r_planes = vec![[0i8; D]; levels];
    let q = (-F::one()).to_canonical_u128() + 1;
    let decompose_params = BalancedDecomposePow2I8Params::new(levels, log_basis, q);
    for ri in r {
        r_planes.fill([0i8; D]);
        ri.balanced_decompose_pow2_i8_into_with_params(&mut r_planes, &decompose_params);
        for plane in &r_planes {
            out.extend_from_slice(plane);
        }
    }
    RecursiveWitnessFlat::from_i8_digits(out)
}
