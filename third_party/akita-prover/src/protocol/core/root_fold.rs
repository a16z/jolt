use super::*;
#[cfg(not(feature = "zk"))]
use akita_types::CleartextWitnessShape;

fn append_shared_opening_point_to_transcript<F, E, T>(
    shared_opening_point: &[E],
    transcript: &mut T,
) where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
    T: Transcript<F>,
{
    for coord in shared_opening_point {
        append_ext_field::<F, E, T>(transcript, ABSORB_EVALUATION_CLAIMS, coord);
    }
}

fn validate_non_eor_root_opening_shape<F, E, const D: usize>(
    alpha_bits: usize,
) -> Result<(), AkitaError>
where
    F: FieldCore,
    E: FpExtEncoding<F>,
{
    if !D.is_multiple_of(<E as ExtField<F>>::EXT_DEGREE)
        || !(D / <E as ExtField<F>>::EXT_DEGREE).is_power_of_two()
    {
        return Err(AkitaError::InvalidInput(
            "extension-field degree must divide the ring dimension into power-of-two slots"
                .to_string(),
        ));
    }

    let packed_slots = D / <E as ExtField<F>>::EXT_DEGREE;
    let packed_inner_bits = packed_slots.trailing_zeros() as usize;
    if packed_inner_bits > alpha_bits {
        return Err(AkitaError::InvalidPointDimension {
            expected: packed_inner_bits,
            actual: alpha_bits,
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn prepare_root<F, E, T, P, B, const D: usize>(
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    transcript: &mut T,
    polys: &[&P],
    opening_batch: OpeningBatch,
    shared_opening_point: &[E],
    commitments: &[RingCommitment<F, D>],
    commitment_hints: Vec<AkitaCommitmentHint<F, D>>,
    root_params: &LevelParams,
    m_row_layout: MRowLayout,
    #[cfg(feature = "zk")] zk_hiding: ZkHidingProverState<F>,
    basis: BasisMode,
) -> Result<PreparedFold<F, E, D>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HasWide + HalvingField,
    E: FpExtEncoding<F>
        + ExtField<F>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + MulBaseUnreduced<F>
        + AkitaSerialize,
    T: Transcript<F> + ProverTranscriptGrind<F>,
    P: AkitaPolyOps<F, D>,
    B: ProverComputeBackend<F>,
{
    let num_claims = opening_batch.num_claims();
    let opening_num_vars = opening_batch.num_vars();
    let alpha_bits = root_params.ring_dimension.trailing_zeros() as usize;
    let needs_extension_reduction = root_tensor_projection_enabled::<F, E, E, D>(opening_num_vars);

    if shared_opening_point.len() > opening_num_vars {
        return Err(AkitaError::InvalidPointDimension {
            expected: opening_num_vars,
            actual: shared_opening_point.len(),
        });
    }

    let expected_openings = if needs_extension_reduction {
        Some({
            let _span = tracing::info_span!("root_extension_check_openings", num_claims).entered();
            let mut padded_point = shared_opening_point.to_vec();
            padded_point.resize(opening_num_vars, E::zero());
            cfg_iter!(polys)
                .map(|poly| poly.evaluate_extension(&padded_point))
                .collect::<Result<Vec<_>, _>>()?
        })
    } else {
        None
    };
    let commitment_rows = flatten_batched_commitment_rows(commitments);
    prepare_fold_inner::<F, E, T, P, P, _, B, D>(
        backend,
        prepared,
        needs_extension_reduction,
        polys,
        polys,
        &opening_batch,
        opening_batch.clone(),
        &opening_batch,
        shared_opening_point,
        #[cfg(feature = "zk")]
        None,
        #[cfg(feature = "zk")]
        None,
        false,
        transcript,
        #[cfg(feature = "zk")]
        zk_hiding,
        expected_openings,
        shared_opening_point.to_vec(),
        || validate_non_eor_root_opening_shape::<F, E, D>(alpha_bits),
        root_params,
        alpha_bits,
        basis,
        BlockOrder::RowMajor,
        commitment_hints,
        commitments,
        m_row_layout,
        FlatRingVec::from_ring_elems(&commitment_rows),
    )
}

/// Prove the folded-root proof payload for an intermediate root.
///
/// The caller owns schedule/config selection and passes the validated schedule
/// execution for level 0. This function owns root polynomial folding, public
/// root transcript setup, root ring-relation construction, and the folded-root
/// prover mechanics.
///
/// # Errors
///
/// Returns an error if root inputs are malformed, polynomial folding or
/// ring-relation construction fails, or the folded-root prover fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_root<F, E, T, P, B, Cfg, const D: usize>(
    expanded: &Arc<AkitaExpandedSetup<F>>,
    prefix_slots: &SetupPrefixProverRegistry<F, D>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    transcript: &mut T,
    polys: &[&P],
    opening_batch: OpeningBatch,
    shared_opening_point: &[E],
    commitments: &[RingCommitment<F, D>],
    commitment_hints: Vec<AkitaCommitmentHint<F, D>>,
    scheduled: &ExecutionSchedule,
    #[cfg(feature = "zk")] zk_hiding: ZkHidingProverState<F>,
    basis: BasisMode,
    setup_contribution_mode: SetupContributionMode,
) -> Result<ProveLevelOutput<F, E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HasWide + HalvingField + PseudoMersenneField,
    E: FpExtEncoding<F>
        + ExtField<F>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + MulBaseUnreduced<F>
        + AkitaSerialize,
    T: Transcript<F> + ProverTranscriptGrind<F>,
    P: AkitaPolyOps<F, D>,
    B: ProverComputeBackend<F>,
    Cfg: CommitmentConfig<Field = F, ExtField = E>,
{
    let num_claims = opening_batch.num_claims();
    let root_params = &scheduled.params;

    if polys.len() != num_claims {
        return Err(AkitaError::InvalidInput(
            "invalid root-level inputs".to_string(),
        ));
    }

    {
        let x: u8 = 0;
        tracing::trace!(
            stack_ptr = format_args!("{:#x}", &x as *const u8 as usize),
            level = 0usize,
            num_claims,
            "prove_root"
        );
    }

    append_opening_batch_shape_to_transcript::<F, T>(&opening_batch, transcript)?;
    append_batched_commitments_to_transcript(commitments, transcript);
    append_shared_opening_point_to_transcript::<F, E, T>(shared_opening_point, transcript);

    let prepared_fold = prepare_root::<F, E, T, P, B, D>(
        backend,
        prepared,
        transcript,
        polys,
        opening_batch,
        shared_opening_point,
        commitments,
        commitment_hints,
        root_params,
        MRowLayout::WithDBlock,
        #[cfg(feature = "zk")]
        zk_hiding,
        basis,
    )?;

    prove_fold::<F, E, T, B, Cfg, D>(
        expanded,
        prefix_slots,
        backend,
        prepared,
        transcript,
        0,
        scheduled,
        prepared_fold,
        setup_contribution_mode,
        false,
        #[cfg(not(feature = "zk"))]
        None,
    )?
    .get_intermediate()
}

/// Terminal-root analogue of [`prove_root`] used when the
/// schedule has exactly one fold level (the root is itself the terminal).
///
/// Mirrors the intermediate-root path through opening-batch absorbs,
/// optional extension-opening reduction, and ring-relation setup, then
/// emits a [`TerminalLevelProof`] through the shared fold prover instead of a
/// [`ProveLevelOutput`].
///
/// # Errors
///
/// Returns an error if opening-batch setup, EOR construction, or the inner
/// terminal-root prover fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_terminal_root_fold_with_params<Cfg, F, E, T, P, B, const D: usize>(
    expanded: &Arc<AkitaExpandedSetup<F>>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    transcript: &mut T,
    polys: &[&P],
    opening_batch: OpeningBatch,
    shared_opening_point: &[E],
    commitments: &[RingCommitment<F, D>],
    commitment_hints: Vec<AkitaCommitmentHint<F, D>>,
    scheduled: &ExecutionSchedule,
    #[cfg(not(feature = "zk"))] terminal_direct_witness_shape: &CleartextWitnessShape,
    basis: BasisMode,
    setup_contribution_mode: SetupContributionMode,
    #[cfg(feature = "zk")] zk_hiding: &mut ZkHidingProverState<F>,
) -> Result<TerminalLevelProof<F, E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HasWide + HalvingField + PseudoMersenneField,
    E: FpExtEncoding<F>
        + ExtField<F>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + MulBaseUnreduced<F>
        + AkitaSerialize,
    T: Transcript<F> + ProverTranscriptGrind<F>,
    P: AkitaPolyOps<F, D>,
    B: ProverComputeBackend<F>,
    Cfg: CommitmentConfig<Field = F, ExtField = E>,
{
    let num_claims = opening_batch.num_claims();
    let root_params = &scheduled.params;

    if polys.len() != num_claims {
        return Err(AkitaError::InvalidInput(
            "invalid root-level inputs".to_string(),
        ));
    }

    {
        let x: u8 = 0;
        tracing::trace!(
            stack_ptr = format_args!("{:#x}", &x as *const u8 as usize),
            level = 0usize,
            num_claims,
            "prove_terminal_root_fold_with_params"
        );
    }

    append_opening_batch_shape_to_transcript::<F, T>(&opening_batch, transcript)?;
    append_batched_commitments_to_transcript(commitments, transcript);
    append_shared_opening_point_to_transcript::<F, E, T>(shared_opening_point, transcript);

    #[cfg(feature = "zk")]
    let owned_zk_hiding = std::mem::replace(zk_hiding, ZkHidingProverState::new(Vec::new()));
    let prepared_fold = prepare_root::<F, E, T, P, B, D>(
        backend,
        prepared,
        transcript,
        polys,
        opening_batch,
        shared_opening_point,
        commitments,
        commitment_hints,
        root_params,
        MRowLayout::WithoutDBlock,
        #[cfg(feature = "zk")]
        owned_zk_hiding,
        basis,
    )?;
    let prefix_slots = SetupPrefixProverRegistry::new();
    let terminal_result = prove_fold::<F, E, T, B, Cfg, D>(
        expanded,
        &prefix_slots,
        backend,
        prepared,
        transcript,
        0,
        scheduled,
        prepared_fold,
        setup_contribution_mode,
        true,
        #[cfg(not(feature = "zk"))]
        Some(terminal_direct_witness_shape),
    )?
    .get_terminal()?;

    #[cfg(not(feature = "zk"))]
    {
        Ok(terminal_result)
    }
    #[cfg(feature = "zk")]
    {
        let (terminal, returned_zk_hiding) = terminal_result;
        *zk_hiding = returned_zk_hiding;
        Ok(terminal)
    }
}
