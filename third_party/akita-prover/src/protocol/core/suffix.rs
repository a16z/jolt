use super::*;
#[cfg(not(feature = "zk"))]
use akita_types::schedule_terminal_direct_witness_shape;

/// Prover state carried between suffix fold levels.
pub struct SuffixProverState<F: FieldCore, L: FieldCore> {
    /// Current committed suffix witness representation.
    pub w: RecursiveWitnessFlat,
    /// Logical suffix witness when it differs from the committed representation.
    pub logical_w: Option<RecursiveWitnessFlat>,
    /// Current suffix witness commitment.
    pub commitment: FlatRingVec<F>,
    /// D-erased suffix commitment hint cache.
    pub hint: RecursiveCommitmentHintCache<F>,
    /// Current digit basis, as `log2(b)`.
    pub log_basis: u32,
    /// Sumcheck challenges that become the next suffix opening point.
    pub sumcheck_challenges: Vec<L>,
    /// Claimed logical opening of `logical_w` at `sumcheck_challenges`.
    pub opening: L,
    /// Transcript-visible masked handle for `opening`.
    #[cfg(feature = "zk")]
    pub opening_public: L,
    /// Proof-level ZK hiding material fixed at batched-prove startup.
    #[cfg(feature = "zk")]
    pub zk_hiding: ZkHidingProverState<F>,
}

impl<F: FieldCore, L: FieldCore> SuffixProverState<F, L> {
    /// Logical witness represented by the carried opening claim.
    #[inline]
    pub fn logical_w(&self) -> &RecursiveWitnessFlat {
        self.logical_w.as_ref().unwrap_or(&self.w)
    }
}

/// Drive the recursive fold suffix (after the root) under config `Cfg`.
///
/// The selected planner `schedule` is authoritative: it determines the fold
/// count, per-level `LevelParams`, successor params, and the terminal direct
/// witness basis. Earlier suffix levels run intermediate folds; the last
/// suffix level runs the terminal fold which ships the cleartext
/// `final_witness`.
///
/// # Errors
///
/// Returns an error if level proving fails, or an invalid-setup error when the
/// schedule's recursive suffix is empty (root-terminal proofs do not run this
/// helper).
#[allow(clippy::too_many_arguments)]
pub fn prove_suffix<Cfg, T, B, const D: usize>(
    expanded: &Arc<AkitaExpandedSetup<Cfg::Field>>,
    prefix_slots: &SetupPrefixProverRegistry<Cfg::Field, D>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    transcript: &mut T,
    starting_state: SuffixProverState<Cfg::Field, Cfg::ExtField>,
    schedule: &Schedule,
    setup_contribution_mode: SetupContributionMode,
) -> Result<RecursiveSuffixOutcome<Cfg::Field, Cfg::ExtField>, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore
        + CanonicalField
        + RandomSampling
        + HasWide
        + HalvingField
        + Invertible
        + PseudoMersenneField,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>
        + FrobeniusExtField<Cfg::Field>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + AkitaSerialize
        + MulBaseUnreduced<Cfg::Field>,
    T: Transcript<Cfg::Field> + ProverTranscriptGrind<Cfg::Field>,
    B: ProverComputeBackend<Cfg::Field>,
{
    let planned_num_levels = schedule_num_fold_levels(schedule);
    if planned_num_levels < 2 {
        return Err(AkitaError::InvalidSetup(
            "prove_suffix expects a non-empty recursive suffix".to_string(),
        ));
    }
    let mut intermediate_levels = Vec::new();
    let mut current_state = starting_state;
    let mut level = 1usize;

    #[cfg(not(feature = "zk"))]
    let terminal_direct_witness_shape = schedule_terminal_direct_witness_shape(schedule)?;
    let terminal_result = loop {
        let scheduled = schedule.get_execution_schedule(level)?;
        scheduled.validate_current_w_len(current_state.w.len())?;
        let level_params = &scheduled.params;
        let level_d = level_params.ring_dimension;
        let is_terminal_level = scheduled.is_terminal;
        let m_row_layout = if is_terminal_level {
            MRowLayout::WithoutDBlock
        } else {
            MRowLayout::WithDBlock
        };
        let out = if level_d == D {
            let prepared_fold = prepare_suffix::<Cfg::Field, Cfg::ExtField, T, B, D>(
                backend,
                prepared,
                transcript,
                current_state,
                level,
                level_params,
                m_row_layout,
            )
            .map_err(|err| {
                AkitaError::InvalidInput(format!("suffix prepare level {level} failed: {err:?}"))
            })?;
            prove_fold::<Cfg::Field, Cfg::ExtField, T, B, Cfg, D>(
                expanded,
                prefix_slots,
                backend,
                prepared,
                transcript,
                level,
                &scheduled,
                prepared_fold,
                setup_contribution_mode,
                is_terminal_level,
                #[cfg(not(feature = "zk"))]
                if is_terminal_level {
                    Some(terminal_direct_witness_shape)
                } else {
                    None
                },
            )
            .map_err(|err| {
                AkitaError::InvalidInput(format!("suffix prove_fold level {level} failed: {err:?}"))
            })
        } else {
            dispatch_ring_dim_result!(level_d, |D_LEVEL| {
                let level_prepared = backend.prepare_expanded::<D_LEVEL>(expanded.clone())?;
                let level_prefix_slots = SetupPrefixProverRegistry::new();
                let prepared_fold = prepare_suffix::<Cfg::Field, Cfg::ExtField, T, B, { D_LEVEL }>(
                    backend,
                    &level_prepared,
                    transcript,
                    current_state,
                    level,
                    level_params,
                    m_row_layout,
                )
                .map_err(|err| {
                    AkitaError::InvalidInput(format!(
                        "suffix prepare level {level} D{D_LEVEL} failed: {err:?}"
                    ))
                })?;
                prove_fold::<Cfg::Field, Cfg::ExtField, T, B, Cfg, { D_LEVEL }>(
                    expanded,
                    &level_prefix_slots,
                    backend,
                    &level_prepared,
                    transcript,
                    level,
                    &scheduled,
                    prepared_fold,
                    setup_contribution_mode,
                    is_terminal_level,
                    #[cfg(not(feature = "zk"))]
                    if is_terminal_level {
                        Some(terminal_direct_witness_shape)
                    } else {
                        None
                    },
                )
                .map_err(|err| {
                    AkitaError::InvalidInput(format!(
                        "suffix prove_fold level {level} D{D_LEVEL} failed: {err:?}"
                    ))
                })
            })
        }?;
        if is_terminal_level {
            break out.get_terminal()?;
        }

        let out = out.get_intermediate()?;
        intermediate_levels.push(out.level_proof);
        current_state = out.next_state;
        level += 1;
    };
    #[cfg(not(feature = "zk"))]
    let terminal = terminal_result;
    #[cfg(feature = "zk")]
    let (terminal, zk_hiding) = terminal_result;

    let mut steps = intermediate_levels;
    let final_w_len = terminal.final_witness().num_elems();
    steps.push(AkitaLevelProof::Terminal {
        extension_opening_reduction: terminal.extension_opening_reduction,
        fold_grind_nonce: terminal.fold_grind_nonce,
        stage2: terminal.stage2,
        final_w_len,
    });

    Ok(RecursiveSuffixOutcome {
        steps,
        #[cfg(feature = "zk")]
        zk_hiding,
        num_levels: planned_num_levels,
    })
}
/// Prove one recursive fold level using already-selected current and next
/// level parameters.
///
/// The caller owns schedule/config selection and passes the next-level
/// commitment params. This function owns recursive opening-point reduction,
/// witness folding, public recursive transcript absorbs, recursive
/// ring-relation construction, and the folded-level prover mechanics.
///
/// # Errors
///
/// Returns an error if the recursive opening point has the wrong dimension,
/// witness folding or ring-relation construction fails, or the folded
/// prover fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn prepare_suffix<F, L, T, B, const D: usize>(
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    transcript: &mut T,
    current_state: SuffixProverState<F, L>,
    level: usize,
    level_params: &LevelParams,
    m_row_layout: MRowLayout,
) -> Result<PreparedFold<F, L, D>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + HasWide
        + HalvingField
        + Invertible
        + PseudoMersenneField,
    L: FpExtEncoding<F>
        + FrobeniusExtField<F>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + AkitaSerialize
        + MulBaseUnreduced<F>,
    T: Transcript<F> + ProverTranscriptGrind<F>,
    B: ProverComputeBackend<F>,
{
    {
        let x: u8 = 0;
        tracing::trace!(
            stack_ptr = format_args!("{:#x}", &x as *const u8 as usize),
            level,
            "prepare_suffix"
        );
    }

    let witness_view = current_state.w.view::<F, D>()?;
    let logical_w = current_state.logical_w.as_ref().unwrap_or(&current_state.w);
    let typed_hint = current_state.hint.to_typed::<D>()?;
    let opening_point = &current_state.sumcheck_challenges;
    #[cfg(feature = "zk")]
    let zk_hiding = current_state.zk_hiding;

    current_state
        .commitment
        .append_as_ring_commitment::<T, D>(ABSORB_COMMITMENT, transcript)?;

    let alpha = level_params.ring_dimension.trailing_zeros() as usize;
    let needs_extension_reduction = <L as ExtField<F>>::EXT_DEGREE != 1;
    let logical_view = logical_w.view::<F, D>()?;
    let logical_polys = [&logical_view];
    let fold_polys = [&witness_view];
    let eor_opening_batch = OpeningBatch::same_point(opening_point.len(), 1)?;
    let expected_openings = needs_extension_reduction.then(|| vec![current_state.opening]);
    let recursive_num_vars = level_params.recursive_opening_num_vars()?;
    let opening_batch = OpeningBatch::same_point(recursive_num_vars, 1)?;
    let commitment_u = current_state.commitment.as_ring_slice::<D>()?;
    let recursive_commitment = RingCommitment {
        u: commitment_u.to_vec(),
    };
    prepare_fold_inner::<F, L, T, _, _, _, B, D>(
        backend,
        prepared,
        needs_extension_reduction,
        &logical_polys,
        &fold_polys,
        &eor_opening_batch,
        opening_batch.clone(),
        &opening_batch,
        opening_point,
        #[cfg(feature = "zk")]
        None,
        #[cfg(feature = "zk")]
        Some(current_state.opening_public),
        true,
        transcript,
        #[cfg(feature = "zk")]
        zk_hiding,
        expected_openings,
        opening_point.to_vec(),
        || Ok(()),
        level_params,
        alpha,
        BasisMode::Lagrange,
        BlockOrder::ColumnMajor,
        vec![typed_hint],
        std::slice::from_ref(&recursive_commitment),
        m_row_layout,
        current_state.commitment,
    )
}

#[cfg(all(test, not(feature = "zk")))]
mod tests {
    use super::*;
    use crate::protocol::core::fold::compute_trace_target;
    use akita_field::Fp32;
    use akita_transcript::AkitaTranscript;
    use akita_types::RingOpeningPoint;

    type TestF = Fp32<251>;
    const D: usize = 4;

    #[test]
    fn non_zk_eor_mismatch_is_rejected() {
        let prepared_point: PreparedOpeningPoint<TestF, TestF, D> = PreparedOpeningPoint {
            padded_point: Vec::new(),
            ring_opening_point: RingOpeningPoint {
                a: vec![TestF::one()],
                b: vec![TestF::one()],
            },
            ring_multiplier_point: RingMultiplierOpeningPoint::from_base(&RingOpeningPoint {
                a: vec![TestF::one()],
                b: vec![TestF::one()],
            }),
            packed_inner_point: CyclotomicRing::<TestF, D>::zero(),
        };
        let folded_rings = [CyclotomicRing::<TestF, D>::zero()];
        let reduction = Some(ExtensionOpeningReduction {
            proof: ExtensionOpeningReductionProof {
                partials: Vec::new(),
                sumcheck: SumcheckProof {
                    round_polys: Vec::new(),
                },
            },
            final_claim: TestF::one(),
            final_factor: TestF::one(),
        });

        let opening_batch = OpeningBatch::same_point(0, 1).expect("singleton opening batch");
        let mut transcript = AkitaTranscript::<TestF>::new(b"test/suffix-shared-trace-target");
        let err = match compute_trace_target::<TestF, TestF, _, D>(
            &reduction,
            &folded_rings,
            &prepared_point,
            &[],
            0,
            BasisMode::Lagrange,
            &opening_batch,
            Some(vec![TestF::one()]),
            &mut transcript,
        ) {
            Ok(_) => panic!("non-zk EOR mismatch should reject"),
            Err(err) => err,
        };

        assert!(matches!(err, AkitaError::InvalidProof));
    }
}
