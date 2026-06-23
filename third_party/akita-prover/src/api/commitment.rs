//! Prover-owned commitment kernels.

#[cfg(feature = "zk")]
use crate::protocol::masking::sample_blinding_digits;
use crate::validation::validate_i8_setup_log_basis;
use crate::{AkitaPolyOps, CommitInnerWitness, CommitmentComputeBackend, FoldInputPoly};
use akita_algebra::CyclotomicRing;
use akita_config::CommitmentConfig;
use akita_field::parallel::*;
use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::{AkitaError, CanonicalField, FieldCore, FromPrimitiveInt, RandomSampling};
use akita_types::{
    root_tensor_projection_enabled, schedule_root_fold_step, AkitaCommitmentHint,
    AkitaExpandedSetup, FlatDigitBlocks, FpExtEncoding, LevelParams, OpeningBatch, RingCommitment,
};

pub(crate) fn commit_inner_block_digit_count(
    n_a: usize,
    num_digits_open: usize,
) -> Result<usize, AkitaError> {
    if num_digits_open == 0 {
        return Err(AkitaError::InvalidSetup(
            "num_digits_open must be nonzero for inner commitment digits".to_string(),
        ));
    }
    n_a.checked_mul(num_digits_open).ok_or_else(|| {
        AkitaError::InvalidSetup(
            "commit inner witness block digit count overflowed usize".to_string(),
        )
    })
}

pub(crate) fn commit_inner_flat_digit_count(
    num_blocks: usize,
    n_a: usize,
    num_digits_open: usize,
) -> Result<usize, AkitaError> {
    num_blocks
        .checked_mul(commit_inner_block_digit_count(n_a, num_digits_open)?)
        .ok_or_else(|| {
            AkitaError::InvalidSetup(
                "commit inner witness flat digit count overflowed usize".to_string(),
            )
        })
}

pub(crate) fn validate_commit_inner_shape<F, const D: usize>(
    inner: &CommitInnerWitness<F, D>,
    num_blocks: usize,
    n_a: usize,
    num_digits_open: usize,
    log_basis: u32,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField,
{
    let expected_block_digits = commit_inner_block_digit_count(n_a, num_digits_open)?;
    let expected_flat_digits = commit_inner_flat_digit_count(num_blocks, n_a, num_digits_open)?;
    validate_i8_setup_log_basis(log_basis, "when recomposing i8 inner commitment digits")?;

    if inner.recomposed_inner_rows.len() != num_blocks {
        return Err(AkitaError::InvalidSetup(format!(
            "backend returned {} inner commitment blocks, expected {}",
            inner.recomposed_inner_rows.len(),
            num_blocks
        )));
    }
    for (block_idx, block_rows) in inner.recomposed_inner_rows.iter().enumerate() {
        if block_rows.len() != n_a {
            return Err(AkitaError::InvalidSetup(format!(
                "backend returned {} A rows for inner commitment block {}, expected {}",
                block_rows.len(),
                block_idx,
                n_a
            )));
        }
    }

    if inner.decomposed_inner_rows.block_count() != num_blocks {
        return Err(AkitaError::InvalidSetup(format!(
            "backend returned {} decomposed inner commitment blocks, expected {}",
            inner.decomposed_inner_rows.block_count(),
            num_blocks
        )));
    }
    for (block_idx, &block_digits) in inner.decomposed_inner_rows.block_sizes().iter().enumerate() {
        if block_digits != expected_block_digits {
            return Err(AkitaError::InvalidSetup(format!(
                "backend returned {} decomposed digits for inner commitment block {}, expected {}",
                block_digits, block_idx, expected_block_digits
            )));
        }
    }
    if inner.decomposed_inner_rows.flat_digits().len() != expected_flat_digits {
        return Err(AkitaError::InvalidSetup(format!(
            "backend returned {} total decomposed inner commitment digits, expected {}",
            inner.decomposed_inner_rows.flat_digits().len(),
            expected_flat_digits
        )));
    }
    for (block_idx, block_digits) in inner.decomposed_inner_rows.iter_blocks().enumerate() {
        let recomposed_block = &inner.recomposed_inner_rows[block_idx];
        for (row_idx, row_digits) in block_digits.chunks(num_digits_open).enumerate() {
            let recomposed = CyclotomicRing::gadget_recompose_pow2_i8(row_digits, log_basis);
            if recomposed_block[row_idx] != recomposed {
                return Err(AkitaError::InvalidSetup(format!(
                    "backend returned recomposed row {row_idx} for inner commitment block {block_idx} that does not match its decomposed digits"
                )));
            }
        }
    }

    Ok(())
}

pub(crate) fn validate_commit_level_params<F, const D: usize>(
    params: &LevelParams,
    setup: &AkitaExpandedSetup<F>,
) -> Result<(), AkitaError>
where
    F: FieldCore,
{
    if params.ring_dimension != D {
        return Err(AkitaError::InvalidSetup(format!(
            "commit params ring dimension {} does not match static D={D}",
            params.ring_dimension
        )));
    }
    if params.num_blocks == 0 || params.block_len == 0 {
        return Err(AkitaError::InvalidSetup(
            "commit params require nonzero num_blocks and block_len".to_string(),
        ));
    }
    if params.num_digits_commit == 0 || params.num_digits_open == 0 {
        return Err(AkitaError::InvalidSetup(
            "commit params require nonzero digit depths".to_string(),
        ));
    }
    validate_i8_setup_log_basis(params.log_basis, "for i8 commitment decomposition")?;
    let expected_a_width = params
        .block_len
        .checked_mul(params.num_digits_commit)
        .ok_or_else(|| AkitaError::InvalidSetup("A commit width overflow".to_string()))?;
    if params.a_key.col_len() != expected_a_width {
        return Err(AkitaError::InvalidSetup(format!(
            "commit params A width {} does not match block_len * num_digits_commit = {expected_a_width}",
            params.a_key.col_len()
        )));
    }
    if params.b_key.col_len() == 0 {
        return Err(AkitaError::InvalidSetup(format!(
            "commit params require nonzero B width, got B={}",
            params.b_key.col_len()
        )));
    }
    // TODO: re-enable this D-side nonzero check (or scope it to non-root-direct
    // schedules) once root-direct commit params no longer carry a
    // zero-width D-key placeholder. Root-direct schedules don't run
    // the relation fold (which is what consumes D), so the planner
    // deliberately emits `d_key.col_len = 0`. This check should
    // eventually be gated on schedule shape (root-direct vs. fold-root)
    // rather than disabled outright.
    // if params.d_key.col_len() == 0 {
    //     return Err(AkitaError::InvalidSetup(format!(
    //         "commit params require nonzero D width, got D={}",
    //         params.d_key.col_len()
    //     )));
    // }
    let setup_len = setup.shared_matrix.total_ring_elements_at::<D>()?;
    let a_required = params
        .a_key
        .row_len()
        .checked_mul(params.a_key.col_len())
        .ok_or_else(|| AkitaError::InvalidSetup("A setup footprint overflow".to_string()))?;
    let b_required = params
        .b_key
        .row_len()
        .checked_mul(params.b_key.col_len())
        .ok_or_else(|| AkitaError::InvalidSetup("B setup footprint overflow".to_string()))?;
    let d_required = params
        .d_key
        .row_len()
        .checked_mul(params.d_key.col_len())
        .ok_or_else(|| AkitaError::InvalidSetup("D setup footprint overflow".to_string()))?;
    let required = a_required.max(b_required).max(d_required);
    if required > setup_len {
        return Err(AkitaError::InvalidSetup(format!(
            "commit params require {required} setup ring elements but setup has {setup_len}",
        )));
    }
    Ok(())
}

pub(crate) fn validate_commit_outer_input_nonempty(active_len: usize) -> Result<(), AkitaError> {
    if active_len == 0 {
        return Err(AkitaError::InvalidSetup(
            "commit B input must be nonempty".to_string(),
        ));
    }
    Ok(())
}

/// Validate a singleton commitment request against prover setup capacity.
///
/// # Errors
///
/// Returns an error if the request is empty, mixes polynomial dimensions, or
/// exceeds the prover setup capacity.
pub fn prepare_commit_inputs<F, const D: usize, P>(
    polys: &[P],
    setup: &AkitaExpandedSetup<F>,
) -> Result<OpeningBatch, AkitaError>
where
    F: FieldCore,
    P: AkitaPolyOps<F, D>,
{
    if polys.is_empty() {
        return Err(AkitaError::InvalidInput(
            "commit requires at least one polynomial".to_string(),
        ));
    }
    let num_vars = polys[0].num_vars();
    if polys.iter().any(|p| p.num_vars() != num_vars) {
        return Err(AkitaError::InvalidInput(
            "all polynomials in a batched commit must have the same num_vars".to_string(),
        ));
    }
    if polys.len() > setup.seed.max_num_batched_polys {
        return Err(AkitaError::InvalidInput(format!(
            "commit received {} polynomials but setup supports at most {}",
            polys.len(),
            setup.seed.max_num_batched_polys
        )));
    }
    if num_vars > setup.seed.max_num_vars {
        return Err(AkitaError::InvalidInput(format!(
            "commit received a polynomial with {} variables but setup supports at most {}",
            num_vars, setup.seed.max_num_vars
        )));
    }

    OpeningBatch::same_point(num_vars, polys.len())
}

pub(crate) fn validate_onehot_chunk_size_for_params<F, const D: usize, P>(
    polys: &[P],
    params: &LevelParams,
) -> Result<(), AkitaError>
where
    F: FieldCore,
    P: AkitaPolyOps<F, D>,
{
    let expected = params.onehot_chunk_size;
    if expected <= 1 {
        return Ok(());
    }
    for (poly_idx, poly) in polys.iter().enumerate() {
        if let Some(actual) = poly.onehot_chunk_size() {
            if actual != expected {
                return Err(AkitaError::InvalidInput(format!(
                    "one-hot polynomial {poly_idx} uses onehot_k={actual}, but this \
                     config/layout requires onehot_k={expected}"
                )));
            }
        }
    }
    Ok(())
}

pub(crate) fn validate_batched_onehot_chunk_size_for_params<F, const D: usize, P>(
    polys: &[P],
    params: &LevelParams,
) -> Result<(), AkitaError>
where
    F: FieldCore,
    P: AkitaPolyOps<F, D>,
{
    let expected = params.onehot_chunk_size;
    if expected <= 1 {
        return Ok(());
    }
    for (poly_idx, poly) in polys.iter().enumerate() {
        if let Some(actual) = poly.onehot_chunk_size() {
            if actual != expected {
                return Err(AkitaError::InvalidInput(format!(
                    "one-hot polynomial {poly_idx} uses onehot_k={actual}, but this \
                     config/layout requires onehot_k={expected}"
                )));
            }
        }
    }
    Ok(())
}

fn checked_commit_b_input_len(total_polys: usize, per_poly: usize) -> Result<usize, AkitaError> {
    total_polys.checked_mul(per_poly).ok_or_else(|| {
        AkitaError::InvalidInput(format!(
            "commit B digit input length overflow for {total_polys} polynomials with {per_poly} digits each"
        ))
    })
}

/// Tiered second-tier commitment: `u_final = F · decompose(blockdiag(B')·t̂)`.
///
/// `b_input_digits` is the full first-tier opening-digit input `t̂` (the same
/// `[[i8; D]]` the single-tier path feeds to `B`). The first-tier matrix `B'`
/// (`params.b_key`) is reused across `tier_split` equal column-slices, the
/// concatenated images are decomposed at `num_digits_open`, and the second-tier
/// matrix `F` (`params.f_key`) commits them. Reads the `B'` and `F` prefixes of
/// the shared setup matrix from the origin (overlapping, like A/B/D).
///
/// Returns the sent commitment `u_final` (length `f_key.row_len()`).
///
/// # Errors
///
/// Returns an error when `params.f_key` is absent, when the `B'` width does not
/// divide `b_input_digits`, or when a matvec fails.
pub(crate) fn tiered_commit_u_final<F, const D: usize, B>(
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    params: &LevelParams,
    b_input_digits: &[[i8; D]],
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + CanonicalField,
    B: CommitmentComputeBackend<F>,
{
    let f_key = params.f_key.as_ref().ok_or_else(|| {
        AkitaError::InvalidSetup("tiered_commit_u_final requires a second-tier F key".to_string())
    })?;
    let n_b_small = params.b_key.row_len();
    let width_small = params.b_key.col_len();
    let delta_open = params.num_digits_open;
    let log_basis = params.log_basis;
    if width_small == 0 || !b_input_digits.len().is_multiple_of(width_small) {
        return Err(AkitaError::InvalidSetup(
            "tiered commit: first-tier B' width does not divide the opening input".to_string(),
        ));
    }
    // u_concat = (B'·t̂_slice_0 ‖ … ‖ B'·t̂_slice_{f-1}), negacyclic.
    let mut u_concat: Vec<CyclotomicRing<F, D>> = Vec::new();
    for chunk in b_input_digits.chunks(width_small) {
        let rows = backend.digit_rows::<D>(prepared, n_b_small, chunk, log_basis)?;
        u_concat.extend(rows);
    }
    // û_concat = decompose(u_concat) at the opening digit depth, ordered
    // [slice][b'_row][digit].
    let mut u_hat = vec![[0i8; D]; u_concat.len() * delta_open];
    for (dst, ring) in u_hat.chunks_mut(delta_open).zip(u_concat.iter()) {
        ring.balanced_decompose_pow2_i8_into(dst, log_basis);
    }
    // u_final = F · û_concat (reads the F prefix of the shared matrix).
    let u_final = backend.digit_rows::<D>(prepared, f_key.row_len(), &u_hat, log_basis)?;
    if u_final.len() != f_key.row_len() {
        return Err(AkitaError::InvalidProof);
    }
    Ok(u_final)
}

fn commit_with_validated_params<F, const D: usize, P, B>(
    polys: &[P],
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    params: &LevelParams,
) -> Result<(RingCommitment<F, D>, AkitaCommitmentHint<F, D>), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    P: AkitaPolyOps<F, D>,
    B: CommitmentComputeBackend<F>,
{
    let b_input_len_per_poly = commit_inner_flat_digit_count(
        params.num_blocks,
        params.a_key.row_len(),
        params.num_digits_open,
    )?;
    let total_b_input_len = checked_commit_b_input_len(polys.len(), b_input_len_per_poly)?;
    let mut b_input_digits = vec![[0i8; D]; total_b_input_len];
    let mut decomposed_inner_rows: Vec<FlatDigitBlocks<D>> = (0..polys.len())
        .map(|_| FlatDigitBlocks::new(Vec::new(), Vec::new()))
        .collect::<Result<_, _>>()?;
    let mut recomposed_inner_rows: Vec<Vec<Vec<CyclotomicRing<F, D>>>> =
        vec![Vec::new(); polys.len()];
    cfg_chunks_mut!(b_input_digits, b_input_len_per_poly)
        .zip(cfg_iter!(polys))
        .zip(cfg_iter_mut!(decomposed_inner_rows))
        .zip(cfg_iter_mut!(recomposed_inner_rows))
        .try_for_each(
            |(((dst, poly), decomposed), recomposed)| -> Result<(), AkitaError> {
                let inner = poly.commit_inner(
                    backend,
                    prepared,
                    params.a_key.row_len(),
                    params.block_len,
                    params.num_blocks,
                    params.num_digits_commit,
                    params.num_digits_open,
                    params.log_basis,
                )?;
                validate_commit_inner_shape(
                    &inner,
                    params.num_blocks,
                    params.a_key.row_len(),
                    params.num_digits_open,
                    params.log_basis,
                )?;
                dst.copy_from_slice(inner.decomposed_inner_rows.flat_digits());
                *decomposed = inner.decomposed_inner_rows;
                *recomposed = inner.recomposed_inner_rows;
                Ok(())
            },
        )?;
    #[cfg(feature = "zk")]
    let b_blinding_digits =
        sample_blinding_digits::<F, D>(params.b_key.row_len(), params.log_basis)?;
    validate_commit_outer_input_nonempty(b_input_digits.len())?;
    let u: Vec<CyclotomicRing<F, D>> = if params.f_key.is_some() {
        // Tiered: the sent commitment is the second-tier image
        // `u_final = F·decompose(blockdiag(B')·t̂)`. ZK blinding of the F tier
        // is a non-goal; tiered proofs are exercised non-zk.
        tiered_commit_u_final::<F, D, B>(backend, prepared, params, &b_input_digits)?
    } else {
        #[cfg(feature = "zk")]
        let mut u: Vec<CyclotomicRing<F, D>> = backend.digit_rows::<D>(
            prepared,
            params.b_key.row_len(),
            &b_input_digits,
            params.log_basis,
        )?;
        #[cfg(not(feature = "zk"))]
        let u: Vec<CyclotomicRing<F, D>> = backend.digit_rows::<D>(
            prepared,
            params.b_key.row_len(),
            &b_input_digits,
            params.log_basis,
        )?;
        #[cfg(feature = "zk")]
        {
            let blinding_rows = backend.zk_b_digit_rows::<D>(
                prepared,
                params.b_key.row_len(),
                b_blinding_digits.flat_digits().len(),
                b_blinding_digits.flat_digits(),
            )?;
            for (row, blinding) in u.iter_mut().zip(blinding_rows) {
                *row += blinding;
            }
        }
        if u.len() != params.b_key.row_len() {
            return Err(AkitaError::InvalidSetup(format!(
                "backend returned {} B commitment rows, expected {}",
                u.len(),
                params.b_key.row_len()
            )));
        }
        u
    };
    let hint = AkitaCommitmentHint::with_recomposed_inner_rows(
        decomposed_inner_rows,
        recomposed_inner_rows,
        #[cfg(feature = "zk")]
        vec![b_blinding_digits],
    );
    Ok((RingCommitment { u }, hint))
}

/// Commit a group of polynomials using already-selected level parameters.
///
/// Config/schedule policy chooses `params`; this function owns only the
/// prover-side matrix work for the supplied concrete layout.
///
/// # Errors
///
/// Returns an error if input validation, inner witness commitment, or hint
/// allocation fails.
pub fn commit_with_params<F, const D: usize, P, B>(
    polys: &[P],
    expanded: &AkitaExpandedSetup<F>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    params: &LevelParams,
) -> Result<(RingCommitment<F, D>, AkitaCommitmentHint<F, D>), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    P: AkitaPolyOps<F, D>,
    B: CommitmentComputeBackend<F>,
{
    backend.validate_prepared_setup::<D>(prepared, expanded)?;
    prepare_commit_inputs::<F, D, P>(polys, expanded)?;
    validate_commit_level_params::<F, D>(params, expanded)?;
    validate_onehot_chunk_size_for_params::<F, D, P>(polys, params)?;
    commit_with_validated_params::<F, D, P, B>(polys, backend, prepared, params)
}

/// Decide whether a root commitment must be tensor-projected before commit.
///
/// Root tensor projection only applies when the field tower admits it and the
/// config-selected schedule starts with a fold. This is the prover-owned
/// analogue of the former scheme-local `should_transform_root_commitment`.
///
/// # Errors
///
/// Propagates [`CommitmentConfig::get_params_for_prove`].
fn should_transform_root_commitment<Cfg, const D: usize>(
    opening_batch: &OpeningBatch,
) -> Result<bool, AkitaError>
where
    Cfg: CommitmentConfig,
{
    if !root_tensor_projection_enabled::<Cfg::Field, Cfg::ExtField, Cfg::ExtField, D>(
        opening_batch.num_vars(),
    ) {
        return Ok(false);
    }
    let schedule = Cfg::get_params_for_prove(opening_batch)?;
    Ok(schedule_root_fold_step(&schedule).is_some())
}

/// Commit a group of polynomials under config `Cfg`.
///
/// The prover crate owns input validation, the root tensor-projection
/// transform decision, config-driven layout selection, and commitment
/// execution.
///
/// # Errors
///
/// Returns an error if input validation, parameter selection, or commitment
/// execution fails.
#[allow(clippy::type_complexity)]
pub fn commit<Cfg, const D: usize, P, B>(
    polys: &[P],
    expanded: &AkitaExpandedSetup<Cfg::Field>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
) -> Result<
    (
        RingCommitment<Cfg::Field, D>,
        AkitaCommitmentHint<Cfg::Field, D>,
    ),
    AkitaError,
>
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore + CanonicalField + RandomSampling + FromPrimitiveInt + HasWide,
    <Cfg::Field as HasWide>::Wide: From<Cfg::Field> + ReduceTo<Cfg::Field>,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>,
    P: AkitaPolyOps<Cfg::Field, D>,
    B: CommitmentComputeBackend<Cfg::Field>,
{
    backend.validate_prepared_setup::<D>(prepared, expanded)?;
    let opening_batch = prepare_commit_inputs::<Cfg::Field, D, P>(polys, expanded)?;
    let params = Cfg::get_params_for_batched_commitment(&opening_batch)?;
    validate_onehot_chunk_size_for_params::<Cfg::Field, D, P>(polys, &params)?;
    if should_transform_root_commitment::<Cfg, D>(&opening_batch)? {
        let transformed = polys
            .iter()
            .map(|poly| poly.tensor_packed_extension_fold_input::<Cfg::ExtField>())
            .collect::<Result<Vec<FoldInputPoly<'_, Cfg::Field, P, D>>, _>>()?;
        validate_commit_level_params::<Cfg::Field, D>(&params, expanded)?;
        return commit_with_validated_params::<Cfg::Field, D, FoldInputPoly<'_, Cfg::Field, P, D>, B>(
            &transformed,
            backend,
            prepared,
            &params,
        );
    }
    validate_commit_level_params::<Cfg::Field, D>(&params, expanded)?;
    commit_with_validated_params::<Cfg::Field, D, P, B>(polys, backend, prepared, &params)
}

/// Validate a batched commitment request and derive its `OpeningBatch`.
///
/// Each slice is one commitment group at the shared opening point. Polynomials
/// may have smaller natural arity than the shared padded batch domain; the
/// largest arity selects the root layout.
///
/// # Errors
///
/// Returns an error if the groups are empty, exceed the prover setup capacity,
/// or have a variable count exceeding the prover setup capacity.
pub fn prepare_batched_commit_inputs<F, const D: usize, P>(
    polys_per_commitment_group: &[&[P]],
    setup: &AkitaExpandedSetup<F>,
) -> Result<OpeningBatch, AkitaError>
where
    F: FieldCore,
    P: AkitaPolyOps<F, D>,
{
    if polys_per_commitment_group.is_empty() {
        return Err(AkitaError::InvalidInput(
            "batched_commit requires at least one commitment group".to_string(),
        ));
    }
    if polys_per_commitment_group
        .iter()
        .any(|group| group.is_empty())
    {
        return Err(AkitaError::InvalidInput(
            "batched_commit commitment groups must be nonempty".to_string(),
        ));
    }
    let padded_num_vars = polys_per_commitment_group
        .iter()
        .flat_map(|group| group.iter())
        .map(AkitaPolyOps::num_vars)
        .max()
        .ok_or_else(|| {
            AkitaError::InvalidInput("batched_commit bundles must be nonempty".to_string())
        })?;
    if padded_num_vars > setup.seed.max_num_vars {
        return Err(AkitaError::InvalidInput(format!(
            "batched_commit received a polynomial with {} variables but setup supports at most {}",
            padded_num_vars, setup.seed.max_num_vars
        )));
    }

    let mut group_counts = Vec::with_capacity(polys_per_commitment_group.len());
    let total_polys = polys_per_commitment_group
        .iter()
        .try_fold(0usize, |acc, group| {
            group_counts.push(group.len());
            acc.checked_add(group.len())
        })
        .ok_or_else(|| {
            AkitaError::InvalidInput("batched_commit total polynomial count overflow".to_string())
        })?;
    if total_polys > setup.seed.max_num_batched_polys {
        return Err(AkitaError::InvalidInput(format!(
            "batched_commit received {total_polys} polynomials but setup supports at most {}",
            setup.seed.max_num_batched_polys
        )));
    }

    OpeningBatch::from_commitment_groups(padded_num_vars, &group_counts)
}

/// Commit one polynomial bundle under config `Cfg`.
///
/// The config-selected schedule supplies the shared root commitment layout.
/// The root tensor-projection transform is applied internally when the field
/// tower and schedule call for it.
///
/// # Errors
///
/// Returns an error if input validation, parameter selection, or commitment
/// execution fails.
#[allow(clippy::type_complexity)]
pub fn batched_commit<Cfg, const D: usize, P, B>(
    polys_per_commitment_group: &[&[P]],
    expanded: &AkitaExpandedSetup<Cfg::Field>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
) -> Result<
    Vec<(
        RingCommitment<Cfg::Field, D>,
        AkitaCommitmentHint<Cfg::Field, D>,
    )>,
    AkitaError,
>
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore + CanonicalField + RandomSampling + FromPrimitiveInt + HasWide,
    <Cfg::Field as HasWide>::Wide: From<Cfg::Field> + ReduceTo<Cfg::Field>,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>,
    P: AkitaPolyOps<Cfg::Field, D>,
    B: CommitmentComputeBackend<Cfg::Field>,
{
    backend.validate_prepared_setup::<D>(prepared, expanded)?;
    let opening_batch =
        prepare_batched_commit_inputs::<Cfg::Field, D, P>(polys_per_commitment_group, expanded)?;
    let params = Cfg::get_params_for_batched_commitment(&opening_batch)?;
    for group in polys_per_commitment_group {
        validate_batched_onehot_chunk_size_for_params::<Cfg::Field, D, P>(group, &params)?;
    }
    if should_transform_root_commitment::<Cfg, D>(&opening_batch)? {
        let transformed: Vec<Vec<FoldInputPoly<'_, Cfg::Field, P, D>>> = polys_per_commitment_group
            .iter()
            .map(|group| {
                group
                    .iter()
                    .map(|poly| poly.tensor_packed_extension_fold_input::<Cfg::ExtField>())
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<_, _>>()?;
        validate_commit_level_params::<Cfg::Field, D>(&params, expanded)?;
        return transformed
            .iter()
            .map(|group| {
                commit_with_validated_params::<
                    Cfg::Field,
                    D,
                    FoldInputPoly<'_, Cfg::Field, P, D>,
                    B,
                >(group, backend, prepared, &params)
            })
            .collect();
    }
    validate_commit_level_params::<Cfg::Field, D>(&params, expanded)?;
    polys_per_commitment_group
        .iter()
        .map(|group| {
            commit_with_validated_params::<Cfg::Field, D, P, B>(group, backend, prepared, &params)
        })
        .collect()
}

/// Commit one polynomial bundle per opening point using already-selected
/// level parameters.
///
/// The caller has already resolved the shared root commitment layout (e.g.
/// via [`batched_commit`]); this function owns only the prover-
/// side matrix work for the supplied concrete layout.
///
/// # Errors
///
/// Returns an error if batched input validation fails or any per-point
/// commitment fails.
#[allow(clippy::type_complexity)]
pub fn batched_commit_with_params<F, const D: usize, P, B>(
    polys_per_commitment_group: &[&[P]],
    expanded: &AkitaExpandedSetup<F>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    params: &LevelParams,
) -> Result<Vec<(RingCommitment<F, D>, AkitaCommitmentHint<F, D>)>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    P: AkitaPolyOps<F, D>,
    B: CommitmentComputeBackend<F>,
{
    backend.validate_prepared_setup::<D>(prepared, expanded)?;
    prepare_batched_commit_inputs::<F, D, P>(polys_per_commitment_group, expanded)?;
    validate_commit_level_params::<F, D>(params, expanded)?;
    for group in polys_per_commitment_group {
        validate_batched_onehot_chunk_size_for_params::<F, D, P>(group, params)?;
    }
    polys_per_commitment_group
        .iter()
        .map(|group| commit_with_validated_params::<F, D, P, B>(group, backend, prepared, params))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AkitaProverSetup, OneHotPoly};
    use akita_challenges::SparseChallengeConfig;
    use akita_field::Fp64;
    use akita_types::{SetupMatrixEnvelope, SisModulusFamily};

    type F = Fp64<4294967197>;
    const D: usize = 32;

    fn inner_witness(
        recomposed_blocks: usize,
        rows_per_block: usize,
        block_sizes: Vec<usize>,
    ) -> CommitInnerWitness<F, D> {
        let total_digits = block_sizes.iter().sum();
        CommitInnerWitness {
            recomposed_inner_rows: vec![
                vec![CyclotomicRing::<F, D>::zero(); rows_per_block];
                recomposed_blocks
            ],
            decomposed_inner_rows: FlatDigitBlocks::new(vec![[0i8; D]; total_digits], block_sizes)
                .expect("valid flat digit blocks"),
        }
    }

    #[test]
    fn commit_inner_shape_accepts_expected_layout() {
        let inner = inner_witness(2, 3, vec![6, 6]);
        validate_commit_inner_shape(&inner, 2, 3, 2, 4).expect("shape should match");
    }

    #[test]
    fn commit_inner_shape_rejects_bad_block_count() {
        let inner = inner_witness(1, 3, vec![6, 6]);
        assert!(validate_commit_inner_shape(&inner, 2, 3, 2, 4).is_err());
    }

    #[test]
    fn commit_inner_shape_rejects_bad_digit_block_size() {
        let inner = inner_witness(2, 3, vec![6, 5]);
        assert!(validate_commit_inner_shape(&inner, 2, 3, 2, 4).is_err());
    }

    #[test]
    fn commit_inner_shape_rejects_recomposition_mismatch() {
        let mut inner = inner_witness(1, 1, vec![2]);
        inner.decomposed_inner_rows.flat_digits_mut()[0][0] = 1;
        assert!(validate_commit_inner_shape(&inner, 1, 1, 2, 4).is_err());
    }

    #[test]
    fn commit_inner_shape_rejects_log_basis_above_i8_range() {
        let inner = inner_witness(1, 1, vec![2]);
        assert!(matches!(
            validate_commit_inner_shape(&inner, 1, 1, 2, 7),
            Err(AkitaError::InvalidSetup(_))
        ));
    }

    #[test]
    fn commit_level_params_reject_log_basis_above_i8_range() {
        let expanded = AkitaProverSetup::<F, D>::generate_with_capacity(
            5,
            1,
            SetupMatrixEnvelope {
                max_setup_len: 8,
                #[cfg(feature = "zk")]
                max_zk_b_len: 1,
                #[cfg(feature = "zk")]
                max_zk_d_len: 1,
            },
        )
        .unwrap()
        .expanded;
        let params = LevelParams::params_only(
            SisModulusFamily::Q32,
            D,
            7,
            1,
            1,
            1,
            SparseChallengeConfig::Uniform {
                weight: 1,
                nonzero_coeffs: vec![-1, 1],
            },
        )
        .with_decomp(1, 1, 2, 2, 0)
        .unwrap();

        assert!(matches!(
            validate_commit_level_params::<F, D>(&params, &expanded),
            Err(AkitaError::InvalidSetup(_))
        ));
    }

    #[test]
    fn commit_b_input_len_rejects_overflow() {
        assert_eq!(checked_commit_b_input_len(3, 5).expect("fits"), 15);
        assert!(matches!(
            checked_commit_b_input_len(usize::MAX, 2),
            Err(AkitaError::InvalidInput(_))
        ));
    }

    #[test]
    fn onehot_chunk_size_validator_rejects_mismatched_k() {
        let params = LevelParams::params_only(
            SisModulusFamily::Q32,
            D,
            2,
            1,
            1,
            1,
            SparseChallengeConfig::Uniform {
                weight: 1,
                nonzero_coeffs: vec![-1, 1],
            },
        )
        .with_onehot_chunk_size(256);
        let wrong = OneHotPoly::<F, D, u16>::new(64, vec![Some(1), None]).unwrap();
        let ok = OneHotPoly::<F, D, u16>::new(256, vec![Some(1), None]).unwrap();

        assert!(matches!(
            validate_onehot_chunk_size_for_params::<F, D, _>(&[wrong], &params),
            Err(AkitaError::InvalidInput(_))
        ));
        validate_onehot_chunk_size_for_params::<F, D, _>(&[ok], &params)
            .expect("matching onehot_k should be accepted");
    }

    #[test]
    fn commit_outer_input_validation_allows_logical_input_longer_than_setup_stride() {
        validate_commit_outer_input_nonempty(9).expect("logical B input may exceed row stride");
        assert!(matches!(
            validate_commit_outer_input_nonempty(0),
            Err(AkitaError::InvalidSetup(_))
        ));
    }
}
