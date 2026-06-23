use super::*;
use crate::AkitaPolyOps;

/// Result of committing the next logical recursive witness.
pub struct NextWitnessCommitment<F: FieldCore> {
    /// Physical witness representation when extension packing changes the logical witness.
    pub witness: Option<RecursiveWitnessFlat>,
    /// Commitment to the physical next-level witness.
    pub commitment: FlatRingVec<F>,
    /// Prover hint for `commitment`.
    pub hint: RecursiveCommitmentHintCache<F>,
}

/// Commit the D-agnostic ring-switch witness `w` at the caller-selected ring
/// dimension.
///
/// This is the D-boundary in the protocol: ring switching produces a flat
/// witness using the current level's ring dimension, then this function
/// re-chunks that witness into `D`-sized ring elements and commits it with the
/// recursive commitment layout supplied by the root scheduler.
///
/// # Errors
///
/// Returns an error if the witness length is not divisible by `D` or if the
/// recursive inner commitment fails.
#[tracing::instrument(skip_all, name = "commit_w")]
#[inline(never)]
pub fn commit_w<F, B, const D: usize>(
    w: &RecursiveWitnessFlat,
    expanded: &AkitaExpandedSetup<F>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    commit_layout: &LevelParams,
) -> Result<(RingCommitment<F, D>, AkitaCommitmentHint<F, D>), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    B: CommitmentComputeBackend<F>,
{
    if commit_layout.ring_dimension != D {
        return Err(AkitaError::InvalidInput(format!(
            "commit_w layout D={} does not match target D={D}",
            commit_layout.ring_dimension
        )));
    }
    if !w.len().is_multiple_of(D) {
        return Err(AkitaError::InvalidSize {
            expected: D,
            actual: w.len(),
        });
    }
    backend.validate_prepared_setup::<D>(prepared, expanded)?;
    validate_commit_level_params::<F, D>(commit_layout, expanded)?;

    let num_ring_elems = w.len() / D;
    tracing::debug!(
        num_ring_elems,
        num_blocks = commit_layout.num_blocks,
        block_len = commit_layout.block_len,
        depth_commit = commit_layout.num_digits_commit,
        depth_open = commit_layout.num_digits_open,
        m_vars = commit_layout.m_vars,
        r_vars = commit_layout.r_vars,
        inner_width = commit_layout.inner_width(),
        pow2_block = 1usize << commit_layout.m_vars,
        "commit_w layout"
    );

    let w_view = w.view::<F, D>()?;
    let inner = w_view.commit_inner(
        backend,
        prepared,
        commit_layout.a_key.row_len(),
        commit_layout.block_len,
        commit_layout.num_blocks,
        commit_layout.num_digits_commit,
        commit_layout.num_digits_open,
        commit_layout.log_basis,
    )?;
    validate_commit_inner_shape(
        &inner,
        commit_layout.num_blocks,
        commit_layout.a_key.row_len(),
        commit_layout.num_digits_open,
        commit_layout.log_basis,
    )?;

    #[cfg(feature = "zk")]
    let b_blinding_digits =
        sample_blinding_digits::<F, D>(commit_layout.b_key.row_len(), commit_layout.log_basis)?;
    let outer_input = inner.decomposed_inner_rows.flat_digits().to_vec();
    validate_commit_outer_input_nonempty(outer_input.len())?;
    let u: Vec<CyclotomicRing<F, D>> = if commit_layout.f_key.is_some() {
        // Tiered: u_final = F·decompose(blockdiag(B')·t̂). ZK blinding of the F
        // tier is a non-goal; tiered proofs are exercised non-zk.
        crate::api::commitment::tiered_commit_u_final::<F, D, B>(
            backend,
            prepared,
            commit_layout,
            &outer_input,
        )?
    } else {
        #[cfg(feature = "zk")]
        let mut u: Vec<CyclotomicRing<F, D>> = backend.digit_rows::<D>(
            prepared,
            commit_layout.b_key.row_len(),
            &outer_input,
            commit_layout.log_basis,
        )?;
        #[cfg(not(feature = "zk"))]
        let u: Vec<CyclotomicRing<F, D>> = backend.digit_rows::<D>(
            prepared,
            commit_layout.b_key.row_len(),
            &outer_input,
            commit_layout.log_basis,
        )?;
        #[cfg(feature = "zk")]
        {
            let blinding_rows = backend.zk_b_digit_rows::<D>(
                prepared,
                commit_layout.b_key.row_len(),
                b_blinding_digits.flat_digits().len(),
                b_blinding_digits.flat_digits(),
            )?;
            for (row, blinding) in u.iter_mut().zip(blinding_rows) {
                *row += blinding;
            }
        }
        if u.len() != commit_layout.b_key.row_len() {
            return Err(AkitaError::InvalidProof);
        }
        u
    };
    #[cfg(feature = "zk")]
    let hint = AkitaCommitmentHint::singleton_with_recomposed_inner_rows(
        inner.decomposed_inner_rows,
        inner.recomposed_inner_rows,
        b_blinding_digits,
    );
    #[cfg(not(feature = "zk"))]
    let hint = {
        AkitaCommitmentHint::singleton_with_recomposed_inner_rows(
            inner.decomposed_inner_rows,
            inner.recomposed_inner_rows,
        )
    };
    Ok((RingCommitment { u }, hint))
}

/// Dispatch a recursive `w` commitment to the selected ring dimension under
/// config `Cfg`.
///
/// The prover crate owns typed backend preparation and `commit_w` execution;
/// the recursive layout is derived from `Cfg`.
///
/// # Errors
///
/// Returns an error if layout selection, backend preparation, commitment, or
/// D-erased hint conversion fails.
#[inline(never)]
fn dispatch_commit_w_with_layout_policy<Cfg, B>(
    backend: &B,
    commit_params: LevelParams,
    expanded: &std::sync::Arc<AkitaExpandedSetup<Cfg::Field>>,
    logical_w: &RecursiveWitnessFlat,
) -> Result<NextWitnessCommitment<Cfg::Field>, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore + CanonicalField + RandomSampling,
    B: CommitmentComputeBackend<Cfg::Field>,
{
    let commit_d = commit_params.ring_dimension;
    dispatch_ring_dim_result!(commit_d, |D_COMMIT| {
        let prepared_commit = backend.prepare_expanded::<D_COMMIT>(expanded.clone())?;
        if <Cfg::ExtField as ExtField<Cfg::Field>>::EXT_DEGREE == 1 {
            let (wc, wh) = commit_w::<Cfg::Field, B, { D_COMMIT }>(
                logical_w,
                expanded.as_ref(),
                backend,
                &prepared_commit,
                &commit_params,
            )?;
            Ok(NextWitnessCommitment {
                witness: None,
                commitment: FlatRingVec::from_commitment(&wc),
                hint: RecursiveCommitmentHintCache::from_typed(wh)?,
            })
        } else {
            // The tensor pack is length-preserving (it redistributes the same
            // digit count), so the committed witness fits the schedule's
            // recursive commit params directly — no per-length re-derivation.
            let committed_w =
                tensor_pack_recursive_witness::<Cfg::Field, Cfg::ExtField, { D_COMMIT }>(
                    logical_w,
                )?;
            let (wc, wh) = commit_w::<Cfg::Field, B, { D_COMMIT }>(
                &committed_w,
                expanded.as_ref(),
                backend,
                &prepared_commit,
                &commit_params,
            )?;
            Ok(NextWitnessCommitment {
                witness: Some(committed_w),
                commitment: FlatRingVec::from_commitment(&wc),
                hint: RecursiveCommitmentHintCache::from_typed(wh)?,
            })
        }
    })
}

/// Commit the next recursive witness under config `Cfg`.
///
/// The same-D fast path reuses the caller's prepared backend context. Cross-D
/// commitments prepare a typed backend context for the target ring dimension.
/// The recursive commitment layout is derived from `Cfg::decomposition()` and
/// `Cfg::ring_subfield_embedding_norm_bound()`.
///
/// # Errors
///
/// Returns an error if layout selection, commitment, backend preparation, or
/// D-erased hint conversion fails.
#[inline(never)]
pub fn commit_next_w<Cfg, B, const D: usize>(
    commit_params: &LevelParams,
    expanded: &std::sync::Arc<AkitaExpandedSetup<Cfg::Field>>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    logical_w: &RecursiveWitnessFlat,
) -> Result<NextWitnessCommitment<Cfg::Field>, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore + CanonicalField + RandomSampling,
    B: CommitmentComputeBackend<Cfg::Field>,
{
    if commit_params.ring_dimension == D {
        if <Cfg::ExtField as ExtField<Cfg::Field>>::EXT_DEGREE == 1 {
            let (wc, wh) = commit_w::<Cfg::Field, B, D>(
                logical_w,
                expanded.as_ref(),
                backend,
                prepared,
                commit_params,
            )?;
            Ok(NextWitnessCommitment {
                witness: None,
                commitment: FlatRingVec::from_commitment(&wc),
                hint: RecursiveCommitmentHintCache::from_typed(wh)?,
            })
        } else {
            // The tensor pack is length-preserving, so the committed witness
            // fits the schedule's recursive commit params directly.
            let committed_w =
                tensor_pack_recursive_witness::<Cfg::Field, Cfg::ExtField, D>(logical_w)?;
            let (wc, wh) = commit_w::<Cfg::Field, B, D>(
                &committed_w,
                expanded.as_ref(),
                backend,
                prepared,
                commit_params,
            )?;
            Ok(NextWitnessCommitment {
                witness: Some(committed_w),
                commitment: FlatRingVec::from_commitment(&wc),
                hint: RecursiveCommitmentHintCache::from_typed(wh)?,
            })
        }
    } else {
        dispatch_commit_w_with_layout_policy::<Cfg, B>(
            backend,
            commit_params.clone(),
            expanded,
            logical_w,
        )
    }
}
