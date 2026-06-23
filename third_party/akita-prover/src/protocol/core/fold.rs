use super::*;
#[cfg(feature = "zk")]
use akita_types::terminal_witness_segment_layout;
use cfg_if::cfg_if;

#[cfg(not(feature = "zk"))]
use crate::protocol::ring_switch::RingSwitchTerminalArtifacts;
#[cfg(not(feature = "zk"))]
use akita_types::build_segment_typed_witness;
#[cfg(not(feature = "zk"))]
use akita_types::validate_segment_typed_z_payload;
#[cfg(not(feature = "zk"))]
use akita_types::CleartextWitnessShape;

fn trace_layout_for_instance<F: FieldCore + CanonicalField, const D: usize>(
    lp: &LevelParams,
    instance: &RingRelationInstance<F, D>,
    col_bits: usize,
    ring_bits: usize,
    num_trace_blocks: usize,
) -> Result<(RingRelationSegmentLayout, akita_types::TraceWeightLayout), AkitaError> {
    let segment = instance.segment_layout(lp)?;
    let layout =
        trace_weight_layout_from_segment(lp, &segment, col_bits, ring_bits, num_trace_blocks)?;
    Ok((segment, layout))
}

#[allow(clippy::too_many_arguments)]
fn build_recursive_stage2_trace_table<F, E, const D: usize>(
    lp: &LevelParams,
    instance: &RingRelationInstance<F, D>,
    prepared: &PreparedOpeningPoint<F, E, D>,
    trace_scale: E,
    output_scale: E,
    col_bits: usize,
    ring_bits: usize,
    live_x_cols: usize,
) -> Result<TraceTable<E>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + Invertible,
    E: FpExtEncoding<F> + ExtField<F> + FromPrimitiveInt,
{
    let (_, layout) = trace_layout_for_instance(lp, instance, col_bits, ring_bits, lp.num_blocks)?;
    let public_weights = trace_public_weights_recursive::<F, E, D>(prepared, trace_scale)?;
    build_trace_table_scaled(&layout, &public_weights, live_x_cols, output_scale)
}

#[allow(clippy::too_many_arguments)]
fn build_root_stage2_trace_table<F, E, const D: usize>(
    lp: &LevelParams,
    instance: &RingRelationInstance<F, D>,
    prepared_point: &PreparedOpeningPoint<F, E, D>,
    row_coefficients: &[E],
    trace_claim_scales: Option<&[E]>,
    output_scale: E,
    col_bits: usize,
    ring_bits: usize,
    live_x_cols: usize,
) -> Result<TraceTable<E>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + Invertible,
    E: FpExtEncoding<F> + ExtField<F> + FromPrimitiveInt,
{
    let num_trace_blocks = instance
        .opening_batch()
        .num_claims()
        .checked_mul(lp.num_blocks)
        .ok_or_else(|| AkitaError::InvalidSetup("trace block count overflow".to_string()))?;
    let (_, layout) =
        trace_layout_for_instance(lp, instance, col_bits, ring_bits, num_trace_blocks)?;
    let public_weights = trace_public_weights_root_terms::<F, E, D>(
        lp,
        instance.opening_batch(),
        prepared_point,
        row_coefficients,
        trace_claim_scales,
    )?;
    build_trace_table_scaled(&layout, &public_weights, live_x_cols, output_scale)
}

pub(in crate::protocol::core) struct TraceTarget<L: FieldCore> {
    pub(in crate::protocol::core) trace_eval_target: L,
    #[cfg(feature = "zk")]
    pub(in crate::protocol::core) trace_eval_target_public: L,
    pub(in crate::protocol::core) trace_claim_scales: Option<Vec<L>>,
    pub(in crate::protocol::core) trace_scale: L,
}

pub(in crate::protocol::core) struct PreparedFold<F: FieldCore, L: FieldCore, const D: usize> {
    pub(in crate::protocol::core) commitment: FlatRingVec<F>,
    pub(in crate::protocol::core) instance: RingRelationInstance<F, D>,
    pub(in crate::protocol::core) witness: RingRelationWitness<F, D>,
    pub(in crate::protocol::core) extension_opening_reduction:
        Option<ExtensionOpeningReductionProof<L>>,
    pub(in crate::protocol::core) trace_eval_target: L,
    #[cfg(feature = "zk")]
    pub(in crate::protocol::core) trace_eval_target_public: L,
    pub(in crate::protocol::core) trace_prepared_point: Option<PreparedOpeningPoint<F, L, D>>,
    pub(in crate::protocol::core) trace_claim_scales: Option<Vec<L>>,
    pub(in crate::protocol::core) trace_scale: L,
    #[cfg(feature = "zk")]
    pub(in crate::protocol::core) zk_hiding: ZkHidingProverState<F>,
    pub(in crate::protocol::core) row_coefficients: Option<Vec<L>>,
}

fn multiplier_ring_weights<F: FieldCore, const D: usize>(
    point: &RingMultiplierOpeningPoint<F, D>,
) -> Result<MultiplierWeightSlices<'_, F, D>, AkitaError> {
    let b = point.b_rings().ok_or_else(|| {
        AkitaError::InvalidInput("ring multiplier must carry ring b weights".to_string())
    })?;
    let a = point.a_rings().ok_or_else(|| {
        AkitaError::InvalidInput("ring multiplier must carry ring a weights".to_string())
    })?;
    Ok((b, a))
}

fn evaluate_poly_at_multiplier_point<F, P, const D: usize>(
    poly: &P,
    point: &RingMultiplierOpeningPoint<F, D>,
    block_len: usize,
) -> Result<(CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>), AkitaError>
where
    F: FieldCore,
    P: AkitaPolyOps<F, D>,
{
    if let Some(base_point) = point.as_base() {
        return Ok(poly.evaluate_and_fold(&base_point.b, &base_point.a, block_len));
    }

    let (b, a) = multiplier_ring_weights(point)?;
    Ok(poly.evaluate_and_fold_ring(b, a, block_len))
}

pub(in crate::protocol::core) fn evaluate_claims_at_prepared_point<F, C, P, const D: usize>(
    polys: &[&P],
    prepared_point: &PreparedOpeningPoint<F, C, D>,
    block_len: usize,
) -> Result<FoldedClaimEvals<F, D>, AkitaError>
where
    F: FieldCore,
    C: FieldCore,
    P: AkitaPolyOps<F, D>,
{
    let _span = tracing::info_span!("fold_evaluate_claims", num_claims = polys.len()).entered();
    let mut folded_rings = Vec::with_capacity(polys.len());
    let mut folded_blocks = Vec::with_capacity(polys.len());
    for poly in polys {
        let (folded_ring, folded_block) = evaluate_poly_at_multiplier_point(
            *poly,
            &prepared_point.ring_multiplier_point,
            block_len,
        )?;
        folded_rings.push(folded_ring);
        folded_blocks.push(folded_block);
    }
    Ok((folded_rings, folded_blocks))
}

#[allow(clippy::too_many_arguments)]
pub(in crate::protocol::core) fn compute_trace_target<F, E, T, const D: usize>(
    reduction: &Option<ExtensionOpeningReduction<E>>,
    folded_rings: &[CyclotomicRing<F, D>],
    prepared_point: &PreparedOpeningPoint<F, E, D>,
    protocol_point: &[E],
    alpha_bits: usize,
    basis: BasisMode,
    opening_batch: &OpeningBatch,
    row_coefficients: Option<Vec<E>>,
    transcript: &mut T,
) -> Result<(TraceTarget<E>, Vec<E>), AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    E: FpExtEncoding<F> + ExtField<F>,
    T: Transcript<F>,
{
    let inner_claim_point = &protocol_point[..protocol_point.len().min(alpha_bits)];
    let openings = folded_rings
        .iter()
        .map(|folded_ring| {
            scalar_opening_from_folded_ring::<F, E, E, D>(
                folded_ring,
                prepared_point,
                inner_claim_point,
                basis,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let row_coefficients = if let Some(row_coefficients) = row_coefficients {
        row_coefficients
    } else {
        append_claim_values_to_transcript::<F, E, T>(&openings, transcript);
        if opening_batch.num_claims() == 1 {
            vec![E::one()]
        } else {
            sample_public_row_coefficients::<F, E, T>(opening_batch, transcript)?
        }
    };
    let ordinary_trace_eval_target =
        batched_eval_target_from_opening_batch(opening_batch, &row_coefficients, &openings)?;
    let trace_eval_target =
        reduction
            .as_ref()
            .map_or(Ok(ordinary_trace_eval_target), |reduction| {
                check_extension_opening_reduction_output(
                    reduction.final_claim,
                    ordinary_trace_eval_target,
                    reduction.final_factor,
                )?;
                Ok(reduction.final_claim)
            })?;
    #[cfg(feature = "zk")]
    let trace_eval_target_public = reduction
        .as_ref()
        .map_or(trace_eval_target, |reduction| reduction.final_claim_public);
    let trace_claim_scales = reduction
        .as_ref()
        .map(|reduction| vec![reduction.final_factor; opening_batch.num_claims()]);
    let trace_scale = reduction
        .as_ref()
        .map_or(E::one(), |reduction| reduction.final_factor);

    Ok((
        TraceTarget {
            trace_eval_target,
            #[cfg(feature = "zk")]
            trace_eval_target_public,
            trace_claim_scales,
            trace_scale,
        },
        row_coefficients,
    ))
}

#[allow(clippy::too_many_arguments)]
pub(in crate::protocol::core) fn prepare_fold_inner<
    'a,
    F,
    E,
    T,
    EorP,
    FoldP,
    V,
    B,
    const D: usize,
>(
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    needs_extension_reduction: bool,
    eor_polys: &[&EorP],
    fold_polys: &[&'a FoldP],
    opening_batch: &OpeningBatch,
    relation_opening_batch: OpeningBatch,
    trace_opening_batch: &OpeningBatch,
    opening_point: &[E],
    #[cfg(feature = "zk")] public_openings: Option<&[E]>,
    #[cfg(feature = "zk")] no_eor_trace_eval_target_public: Option<E>,
    pad_base_evals: bool,
    transcript: &mut T,
    #[cfg(feature = "zk")] mut zk_hiding: ZkHidingProverState<F>,
    expected_openings: Option<Vec<E>>,
    non_eor_protocol_point: Vec<E>,
    validate_non_eor: V,
    level_params: &LevelParams,
    alpha_bits: usize,
    basis: BasisMode,
    block_order: BlockOrder,
    commitment_hints: Vec<AkitaCommitmentHint<F, D>>,
    commitments: &[RingCommitment<F, D>],
    m_row_layout: MRowLayout,
    commitment: FlatRingVec<F>,
) -> Result<PreparedFold<F, E, D>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide,
    E: FpExtEncoding<F>
        + ExtField<F>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + MulBaseUnreduced<F>
        + AkitaSerialize,
    T: Transcript<F> + ProverTranscriptGrind<F>,
    EorP: AkitaPolyOps<F, D>,
    FoldP: AkitaPolyOps<F, D>,
    V: FnOnce() -> Result<(), AkitaError>,
    B: ProverComputeBackend<F>,
{
    let (fold_inputs, protocol_point, row_coefficients, reduction) = if needs_extension_reduction {
        let proved = prove_extension_opening_reduction::<F, E, T, EorP, D>(
            eor_polys,
            opening_batch,
            opening_point,
            #[cfg(feature = "zk")]
            public_openings,
            pad_base_evals,
            transcript,
            if pad_base_evals { "recursive" } else { "root" },
            #[cfg(feature = "zk")]
            &mut zk_hiding,
        )?;
        if let Some(expected_openings) = expected_openings.as_ref() {
            if proved.openings != *expected_openings {
                return Err(AkitaError::InvalidProof);
            }
        }
        let fold_inputs = {
            let _span =
                tracing::info_span!("extension_transform_polys", num_claims = fold_polys.len())
                    .entered();
            cfg_iter!(fold_polys)
                .map(|poly| {
                    <FoldP as AkitaPolyOps<F, D>>::tensor_packed_extension_fold_input::<E>(*poly)
                })
                .collect::<Result<Vec<FoldInputPoly<'a, F, FoldP, D>>, _>>()?
        };
        (
            fold_inputs,
            proved.protocol_point,
            Some(proved.row_coefficients),
            Some(proved.reduction),
        )
    } else {
        validate_non_eor()?;
        let fold_inputs = fold_polys
            .iter()
            .map(|poly| FoldInputPoly::Original(*poly))
            .collect::<Vec<_>>();
        let row_coefficients = if pad_base_evals {
            Some(vec![E::one()])
        } else {
            None
        };
        (fold_inputs, non_eor_protocol_point, row_coefficients, None)
    };
    let prepared_point = prepare_opening_point::<F, E, D>(
        &protocol_point,
        basis,
        level_params,
        alpha_bits,
        block_order,
    )?;
    let fold_refs = fold_inputs.iter().collect::<Vec<_>>();
    let (folded_rings, e_folded_by_claim) =
        evaluate_claims_at_prepared_point(&fold_refs, &prepared_point, level_params.block_len)?;
    for pt in &prepared_point.padded_point {
        append_ext_field::<F, E, T>(transcript, ABSORB_EVALUATION_CLAIMS, pt);
    }
    let (trace_target, row_coefficients) = compute_trace_target::<F, E, T, D>(
        &reduction,
        &folded_rings,
        &prepared_point,
        &protocol_point,
        alpha_bits,
        basis,
        trace_opening_batch,
        row_coefficients,
        transcript,
    )?;
    #[cfg(feature = "zk")]
    let mut trace_target = trace_target;
    #[cfg(feature = "zk")]
    if reduction.is_none() {
        if let Some(public_target) = no_eor_trace_eval_target_public {
            trace_target.trace_eval_target_public = public_target;
        }
    }
    let row_coefficient_rings = row_coefficient_rings::<F, E, D>(&row_coefficients)?;
    let (instance, witness) = RingRelationProver::new::<F, D, _, _, _>(
        backend,
        prepared,
        prepared_point.ring_opening_point.clone(),
        prepared_point.ring_multiplier_point.clone(),
        &fold_refs,
        e_folded_by_claim,
        relation_opening_batch,
        level_params.clone(),
        commitment_hints,
        transcript,
        commitments,
        row_coefficient_rings,
        m_row_layout,
    )?;

    let extension_opening_reduction = reduction.map(|reduction| reduction.proof);
    let row_coefficients = if pad_base_evals {
        None
    } else {
        Some(row_coefficients)
    };
    let trace_claim_scales = if pad_base_evals {
        None
    } else {
        trace_target.trace_claim_scales
    };

    Ok(PreparedFold {
        commitment,
        instance,
        witness,
        extension_opening_reduction,
        trace_eval_target: trace_target.trace_eval_target,
        trace_scale: trace_target.trace_scale,
        trace_prepared_point: Some(prepared_point),
        trace_claim_scales,
        #[cfg(feature = "zk")]
        trace_eval_target_public: trace_target.trace_eval_target_public,
        #[cfg(feature = "zk")]
        zk_hiding,
        row_coefficients,
    })
}

#[cfg(not(feature = "zk"))]
pub(in crate::protocol::core) type TerminalFoldResult<F, L> = TerminalLevelProof<F, L>;
#[cfg(feature = "zk")]
pub(in crate::protocol::core) type TerminalFoldResult<F, L> =
    (TerminalLevelProof<F, L>, ZkHidingProverState<F>);

pub(in crate::protocol::core) enum FoldProveOutput<F: FieldCore, L: FieldCore> {
    Intermediate(Box<ProveLevelOutput<F, L>>),
    Terminal(Box<TerminalFoldResult<F, L>>),
}

impl<F: FieldCore, L: FieldCore> FoldProveOutput<F, L> {
    pub(in crate::protocol::core) fn get_intermediate(
        self,
    ) -> Result<ProveLevelOutput<F, L>, AkitaError> {
        match self {
            Self::Intermediate(out) => Ok(*out),
            Self::Terminal(_) => Err(AkitaError::InvalidInput(
                "intermediate fold unexpectedly returned terminal proof".to_string(),
            )),
        }
    }

    pub(in crate::protocol::core) fn get_terminal(
        self,
    ) -> Result<TerminalFoldResult<F, L>, AkitaError> {
        match self {
            Self::Terminal(terminal) => Ok(*terminal),
            Self::Intermediate(_) => Err(AkitaError::InvalidInput(
                "terminal fold unexpectedly returned intermediate proof".to_string(),
            )),
        }
    }
}
type BoundNextWitness<F> = (
    Option<NextWitnessCommitment<F>>,
    Option<CleartextWitnessProof<F>>,
);
/// Prove one recursive fold level after the caller has built its ring-relation
/// equation and selected the commitment policy for the next `w`.
///
/// This function owns prover mechanics: build `w`, commit it, finish ring
/// switching, run stage-1/stage-2 sumchecks, and produce the next recursive
/// state.
///
/// # Errors
///
/// Returns an error if ring switching, recursive commitment, or either
/// sumcheck prover fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(in crate::protocol::core) fn prove_fold<F, L, T, B, Cfg, const D: usize>(
    expanded: &Arc<AkitaExpandedSetup<F>>,
    prefix_slots: &SetupPrefixProverRegistry<F, D>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    transcript: &mut T,
    level: usize,
    scheduled: &ExecutionSchedule,
    prepared_fold: PreparedFold<F, L, D>,
    setup_contribution_mode: SetupContributionMode,
    is_terminal_fold: bool,
    #[cfg(not(feature = "zk"))] terminal_direct_witness_shape: Option<&CleartextWitnessShape>,
) -> Result<FoldProveOutput<F, L>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + HasWide
        + HalvingField
        + Invertible
        + PseudoMersenneField
        + AkitaSerialize,
    L: ExtField<F>
        + FpExtEncoding<F>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<F> + ProverTranscriptGrind<F>,
    B: ProverComputeBackend<F>,
    Cfg: CommitmentConfig<Field = F, ExtField = L>,
{
    #[cfg(feature = "zk")]
    let mut zk_hiding = prepared_fold.zk_hiding;
    let lp = &scheduled.params;
    let fold_grind_nonce = prepared_fold.witness.fold_grind_nonce;
    let commitment_u = prepared_fold.commitment.as_ring_slice::<D>()?;
    let build_output = ring_switch_build_w::<F, B, D>(
        &prepared_fold.instance,
        prepared_fold.witness,
        backend,
        prepared,
        lp,
        is_terminal_fold,
    )?;
    let logical_w = build_output.w;
    scheduled.validate_next_w_len(logical_w.len())?;
    let next_commitment = if is_terminal_fold {
        None
    } else {
        let _span = tracing::info_span!("commit_w_level", level).entered();
        Some(crate::commit_next_w::<Cfg, B, D>(
            &scheduled.next_params,
            expanded,
            backend,
            prepared,
            &logical_w,
        )?)
    };
    let (next_commitment, final_witness) = bind_next_witness_for_ring_switch::<F, T, D>(
        transcript,
        is_terminal_fold,
        lp,
        &prepared_fold.instance,
        &logical_w,
        next_commitment,
        if is_terminal_fold {
            Some(scheduled.next_params.log_basis)
        } else {
            None
        },
        #[cfg(not(feature = "zk"))]
        build_output.terminal_artifacts,
        #[cfg(not(feature = "zk"))]
        terminal_direct_witness_shape,
    )?;
    let m_row_layout = if is_terminal_fold {
        MRowLayout::WithoutDBlock
    } else {
        MRowLayout::WithDBlock
    };
    let rs = ring_switch_finalize::<F, L, T, D>(
        &prepared_fold.instance,
        expanded.as_ref(),
        transcript,
        &logical_w,
        lp,
        prepared_fold.row_coefficients.as_deref(),
        m_row_layout,
    )?;

    let relation_rows = if is_terminal_fold {
        &[][..]
    } else {
        prepared_fold.instance.v.as_slice()
    };
    let relation_claim = relation_claim_from_rows_extension::<F, L, D>(
        &rs.tau1,
        rs.alpha,
        relation_rows,
        commitment_u,
    )?;
    #[cfg(feature = "zk")]
    let relation_claim_public = relation_claim;
    #[cfg(feature = "zk")]
    let stage2_round_pads;
    let (stage1_proof, stage1_point, s_claim) = if is_terminal_fold {
        #[cfg(feature = "zk")]
        {
            stage2_round_pads =
                zk_hiding.take_compressed_rounds::<L>(rs.col_bits + rs.ring_bits, 3)?;
        }
        (None, vec![L::zero(); rs.col_bits + rs.ring_bits], L::zero())
    } else {
        #[cfg(feature = "zk")]
        let (stage1_round_pads, stage1_child_claim_masks, next_stage2_round_pads) =
            zk_hiding.take_current_level_pads::<L>(rs.col_bits + rs.ring_bits, rs.b)?;
        #[cfg(feature = "zk")]
        {
            stage2_round_pads = next_stage2_round_pads;
        }
        let (stage1_proof, stage1_point, s_claim) = prove_stage1::<F, L, T>(
            transcript,
            &rs,
            #[cfg(feature = "zk")]
            stage1_round_pads,
            #[cfg(feature = "zk")]
            stage1_child_claim_masks,
        )?;
        transcript.append_serde(ABSORB_SUMCHECK_S_CLAIM, &stage1_proof.s_claim);
        (Some(stage1_proof), stage1_point, s_claim)
    };
    let batching_coeff: L = if is_terminal_fold {
        L::zero()
    } else {
        sample_ext_challenge::<F, L, T>(transcript, CHALLENGE_SUMCHECK_BATCH)
    };
    let trace_coeff = {
        let trace_gamma = if is_terminal_fold {
            sample_ext_challenge::<F, L, T>(transcript, CHALLENGE_SUMCHECK_BATCH)
        } else {
            batching_coeff
        };
        stage2_trace_coeff(batching_coeff, trace_gamma, is_terminal_fold)
    };
    let trace_opening_claim = trace_coeff * prepared_fold.trace_eval_target;
    #[cfg(feature = "zk")]
    let trace_eval_target_public_claim = trace_coeff * prepared_fold.trace_eval_target_public;
    ensure_trace_stage2_supported(L::EXT_DEGREE)?;
    let trace_compact = if let Some(row_coefficients) = prepared_fold.row_coefficients.as_ref() {
        Some(build_root_stage2_trace_table::<F, L, D>(
            lp,
            &prepared_fold.instance,
            prepared_fold
                .trace_prepared_point
                .as_ref()
                .ok_or(AkitaError::InvalidProof)?,
            row_coefficients,
            prepared_fold.trace_claim_scales.as_deref(),
            trace_coeff,
            rs.col_bits,
            rs.ring_bits,
            rs.live_x_cols,
        )?)
    } else if let Some(prepared) = prepared_fold.trace_prepared_point.as_ref() {
        Some(build_recursive_stage2_trace_table::<F, L, D>(
            lp,
            &prepared_fold.instance,
            prepared,
            prepared_fold.trace_scale,
            trace_coeff,
            rs.col_bits,
            rs.ring_bits,
            rs.live_x_cols,
        )?)
    } else {
        None
    };
    let ring_bits = rs.ring_bits;
    let tau1 = rs.tau1.clone();
    let alpha = rs.alpha;
    #[cfg(feature = "zk")]
    let stage1_s_claim = stage1_proof
        .as_ref()
        .map(|proof| proof.s_claim)
        .unwrap_or_else(L::zero);
    let (stage2_sumcheck_proof, sumcheck_challenges, stage2_prover) = prove_stage2::<F, L, T>(
        transcript,
        batching_coeff,
        rs,
        &stage1_point,
        s_claim,
        relation_claim,
        #[cfg(feature = "zk")]
        relation_claim_public,
        #[cfg(feature = "zk")]
        stage1_s_claim,
        trace_compact,
        trace_opening_claim,
        #[cfg(feature = "zk")]
        trace_eval_target_public_claim,
        #[cfg(feature = "zk")]
        stage2_round_pads,
    )?;
    if is_terminal_fold {
        let final_witness = final_witness.ok_or_else(|| {
            AkitaError::InvalidInput("terminal fold did not bind a final witness".to_string())
        })?;
        let proof = TerminalLevelProof::new_with_extension_opening_reduction(
            prepared_fold.extension_opening_reduction,
            #[cfg(not(feature = "zk"))]
            stage2_sumcheck_proof,
            #[cfg(feature = "zk")]
            stage2_sumcheck_proof,
            final_witness,
            fold_grind_nonce,
        );
        cfg_if! {
            if #[cfg(feature = "zk")] {
                Ok(FoldProveOutput::Terminal(Box::new((proof, zk_hiding))))
            } else {
                Ok(FoldProveOutput::Terminal(Box::new(proof)))
            }
        }
    } else {
        let w_eval = {
            let _span = tracing::info_span!("multilinear_eval", level).entered();
            stage2_prover.final_w_eval()
        };
        #[cfg(feature = "zk")]
        let proof_w_eval = w_eval + zk_hiding.take_next_w_eval_mask::<L>()?;
        #[cfg(not(feature = "zk"))]
        let proof_w_eval = w_eval;
        transcript.append_serde(ABSORB_STAGE2_NEXT_W_EVAL, &proof_w_eval);
        let stage3_sumcheck_proof = prove_stage3::<F, L, T, D>(
            setup_contribution_mode,
            expanded.as_ref(),
            prefix_slots,
            lp,
            &scheduled.next_params,
            &prepared_fold.instance,
            &tau1,
            alpha,
            &sumcheck_challenges,
            ring_bits,
            transcript,
        )?;
        let stage1_proof = stage1_proof.ok_or_else(|| {
            AkitaError::InvalidInput("intermediate fold missing stage-1 proof".to_string())
        })?;
        let NextWitnessCommitment {
            witness: packed_witness,
            commitment: committed_commitment,
            hint: committed_hint,
        } = next_commitment.ok_or_else(|| {
            AkitaError::InvalidInput("intermediate fold did not bind a next commitment".to_string())
        })?;
        let w_commitment_proof = committed_commitment.clone();
        let level_proof = AkitaLevelProof::Intermediate {
            extension_opening_reduction: prepared_fold.extension_opening_reduction,
            v: FlatRingVec::from_ring_elems(&prepared_fold.instance.v).into_compact(),
            fold_grind_nonce,
            stage1: stage1_proof,
            stage2: AkitaStage2Proof::Intermediate(AkitaIntermediateStage2Proof {
                #[cfg(not(feature = "zk"))]
                sumcheck_proof: stage2_sumcheck_proof,
                #[cfg(feature = "zk")]
                sumcheck_proof_masked: stage2_sumcheck_proof,
                next_w_commitment: w_commitment_proof.into_compact(),
                #[cfg(not(feature = "zk"))]
                next_w_eval: proof_w_eval,
                #[cfg(feature = "zk")]
                next_w_eval_masked: proof_w_eval,
            }),
            stage3_sumcheck_proof,
        };

        let (committed_witness, logical_w) = match packed_witness {
            Some(packed_witness) => (packed_witness, Some(logical_w)),
            None => (logical_w, None),
        };

        Ok(FoldProveOutput::Intermediate(Box::new(ProveLevelOutput {
            level_proof,
            next_state: SuffixProverState {
                w: committed_witness,
                logical_w,
                commitment: committed_commitment,
                hint: committed_hint,
                log_basis: scheduled.next_params.log_basis,
                sumcheck_challenges,
                opening: w_eval,
                #[cfg(feature = "zk")]
                opening_public: proof_w_eval,
                #[cfg(feature = "zk")]
                zk_hiding,
            },
        })))
    }
}

#[allow(clippy::too_many_arguments)]
pub(in crate::protocol::core) fn bind_next_witness_for_ring_switch<F, T, const D: usize>(
    transcript: &mut T,
    is_terminal_fold: bool,
    lp: &LevelParams,
    instance: &RingRelationInstance<F, D>,
    #[cfg_attr(not(feature = "zk"), allow(unused_variables))] logical_w: &RecursiveWitnessFlat,
    next_commitment: Option<NextWitnessCommitment<F>>,
    final_log_basis: Option<u32>,
    #[cfg(not(feature = "zk"))] terminal_artifacts: Option<RingSwitchTerminalArtifacts<F, D>>,
    #[cfg(not(feature = "zk"))] terminal_direct_witness_shape: Option<&CleartextWitnessShape>,
) -> Result<BoundNextWitness<F>, AkitaError>
where
    F: FieldCore + CanonicalField + HalvingField + AkitaSerialize,
    T: Transcript<F>,
{
    if is_terminal_fold {
        #[cfg(feature = "zk")]
        let final_log_basis = final_log_basis.ok_or_else(|| {
            AkitaError::InvalidInput("terminal fold missing final witness basis".to_string())
        })?;
        #[cfg(not(feature = "zk"))]
        final_log_basis.ok_or_else(|| {
            AkitaError::InvalidInput("terminal fold missing final witness basis".to_string())
        })?;
        #[cfg(not(feature = "zk"))]
        {
            if let Some(artifacts) = terminal_artifacts {
                if artifacts.u_concat_planes != 0 {
                    return Err(AkitaError::InvalidInput(
                        "segment-typed terminal witness does not support tiered u_concat"
                            .to_string(),
                    ));
                }
                let num_commitment_groups = instance
                    .opening_batch()
                    .num_polys_per_commitment_group()
                    .len();
                let CleartextWitnessShape::SegmentTyped(scheduled_shape) =
                    terminal_direct_witness_shape.ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "terminal fold missing scheduled segment-typed witness shape"
                                .to_string(),
                        )
                    })?
                else {
                    return Err(AkitaError::InvalidSetup(
                        "terminal fold expected segment-typed witness shape".to_string(),
                    ));
                };
                let (num_w_vectors, num_t_vectors, num_public_rows) =
                    akita_types::tail_segment_multiplicities_from_layout(
                        lp,
                        &scheduled_shape.layout,
                    )?;
                let segment = build_segment_typed_witness::<D, F>(
                    &artifacts.e_folded,
                    &artifacts.recomposed_inner_rows,
                    &artifacts.z_folded_centered,
                    &artifacts.r,
                    lp,
                    num_w_vectors,
                    num_t_vectors,
                    num_public_rows,
                    num_commitment_groups,
                )?;
                if segment.layout != scheduled_shape.layout {
                    return Err(AkitaError::InvalidSetup(
                        "segment-typed witness layout does not match schedule".to_string(),
                    ));
                }
                validate_segment_typed_z_payload(&segment, scheduled_shape.z_payload_bytes)?;
                let parts = segment.terminal_transcript_parts()?;
                transcript.absorb_and_record_bytes(ABSORB_TERMINAL_W_REMAINDER, &parts.remainder);
                return Ok((None, Some(CleartextWitnessProof::SegmentTyped(segment))));
            }
            return Err(AkitaError::InvalidSetup(
                "terminal fold missing segment-typed witness artifacts".to_string(),
            ));
        }
        #[cfg(feature = "zk")]
        {
            let final_witness =
                CleartextWitnessProof::PackedDigits(PackedDigits::from_i8_digits_with_min_bits(
                    logical_w.as_i8_digits(),
                    final_log_basis,
                ));
            let terminal_layout = terminal_witness_segment_layout(
                lp,
                instance.opening_batch().num_claims(),
                1,
                F::modulus_bits(),
            )?;
            let parts = final_witness.terminal_transcript_parts(terminal_layout)?;
            if final_witness.packed_i8_digits()?.as_slice() != logical_w.as_i8_digits() {
                return Err(AkitaError::InvalidInput(
                    "terminal final witness does not match ring-switch witness".to_string(),
                ));
            }
            transcript.absorb_and_record_bytes(ABSORB_TERMINAL_W_REMAINDER, &parts.remainder);
            return Ok((None, Some(final_witness)));
        }
    }

    let next_commitment = next_commitment.ok_or_else(|| {
        AkitaError::InvalidInput("intermediate fold missing next commitment".to_string())
    })?;
    transcript.append_serde(
        ABSORB_NEXT_LEVEL_WITNESS_BINDING,
        &next_commitment.commitment,
    );
    Ok((Some(next_commitment), None))
}

#[allow(clippy::too_many_arguments)]
pub(in crate::protocol::core) fn prove_stage1<F, L, T>(
    transcript: &mut T,
    rs: &RingSwitchOutput<L>,
    #[cfg(feature = "zk")] stage1_round_pads: Vec<Vec<akita_sumcheck::EqFactoredUniPoly<L>>>,
    #[cfg(feature = "zk")] stage1_child_claim_masks: Vec<Vec<L>>,
) -> Result<(AkitaStage1Proof<L>, Vec<L>, L), AkitaError>
where
    F: FieldCore + CanonicalField,
    L: ExtField<F> + HasUnreducedOps + HasOptimizedFold + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
    let tau0_reordered = reorder_stage1_coords(&rs.tau0, rs.col_bits, rs.ring_bits);
    let stage1_prover = AkitaStage1Prover::new(
        &rs.w_evals_compact,
        &tau0_reordered,
        rs.b,
        rs.live_x_cols,
        rs.col_bits,
        rs.ring_bits,
    )?;
    cfg_if! {
        if #[cfg(feature = "zk")] {
            stage1_prover.prove::<F, T>(transcript, stage1_round_pads, stage1_child_claim_masks)
        } else {
            let (stage1_proof, stage1_point) = stage1_prover.prove::<F, T>(transcript)?;
            let s_claim = stage1_proof.s_claim;
            Ok((stage1_proof, stage1_point, s_claim))
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn prove_stage2<F, L, T>(
    transcript: &mut T,
    batching_coeff: L,
    rs: RingSwitchOutput<L>,
    stage1_point: &[L],
    s_claim: L,
    relation_claim: L,
    #[cfg(feature = "zk")] relation_claim_public: L,
    #[cfg(feature = "zk")] stage1_s_claim: L,
    trace_compact: Option<TraceTable<L>>,
    trace_opening_claim: L,
    #[cfg(feature = "zk")] trace_eval_target_public_claim: L,
    #[cfg(feature = "zk")] stage2_round_pads: Vec<CompressedUniPoly<L>>,
) -> Result<Stage2ProveResult<L>, AkitaError>
where
    F: FieldCore + CanonicalField,
    L: ExtField<F> + HasUnreducedOps + HasOptimizedFold + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    let _sumcheck_span = tracing::info_span!("stage2_sumcheck").entered();
    let mut stage2_prover = AkitaStage2Prover::new(
        batching_coeff,
        rs.w_evals_compact,
        stage1_point,
        s_claim,
        rs.b,
        rs.alpha_evals_y,
        rs.m_evals_x,
        rs.live_x_cols,
        rs.col_bits,
        rs.ring_bits,
        relation_claim,
        trace_compact.clone(),
        trace_opening_claim,
    )?;
    cfg_if! {
        if #[cfg(feature = "zk")] {
            let mut stage2_public_input = batching_coeff * stage1_s_claim + relation_claim_public;
            if trace_compact.is_some() {
                stage2_public_input += trace_eval_target_public_claim;
            }
            let (stage2_sumcheck_proof_masked, sumcheck_challenges) = stage2_prover
                .prove_zk::<F, T, _>(
                    stage2_public_input,
                    transcript,
                    |tr| sample_ext_challenge::<F, L, T>(tr, CHALLENGE_SUMCHECK_ROUND),
                    stage2_round_pads,
                )?;
            Ok((
                stage2_sumcheck_proof_masked,
                sumcheck_challenges,
                stage2_prover,
            ))
        } else {
            let (stage2_sumcheck_proof, sumcheck_challenges, _) = stage2_prover
                .prove::<F, T, _>(transcript, |tr| {
                    sample_ext_challenge::<F, L, T>(tr, CHALLENGE_SUMCHECK_ROUND)
                })?;
            Ok((stage2_sumcheck_proof, sumcheck_challenges, stage2_prover))
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(in crate::protocol::core) fn prove_stage3<F, L, T, const D: usize>(
    setup_contribution_mode: SetupContributionMode,
    expanded: &AkitaExpandedSetup<F>,
    prefix_slots: &SetupPrefixProverRegistry<F, D>,
    lp: &LevelParams,
    next_level_params: &LevelParams,
    instance: &RingRelationInstance<F, D>,
    tau1: &[L],
    alpha: L,
    sumcheck_challenges: &[L],
    ring_bits: usize,
    transcript: &mut T,
) -> Result<Option<SetupSumcheckProof<L>>, AkitaError>
where
    F: FieldCore + CanonicalField,
    L: FpExtEncoding<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    match setup_contribution_mode {
        SetupContributionMode::Recursive => {
            let output = SetupSumcheckProver::prove::<F, T, _, D>(
                expanded,
                prefix_slots,
                lp,
                next_level_params,
                instance,
                tau1,
                alpha,
                &sumcheck_challenges[ring_bits..],
                transcript,
                |tr| sample_ext_challenge::<F, L, T>(tr, CHALLENGE_SUMCHECK_ROUND),
            )?;
            Ok(Some(SetupSumcheckProof {
                claim: output.claim,
                sumcheck: output.sumcheck,
            }))
        }
        SetupContributionMode::Direct => Ok(None),
    }
}
