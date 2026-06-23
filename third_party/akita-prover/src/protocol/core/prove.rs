use super::*;
use crate::api::commitment::validate_onehot_chunk_size_for_params;
#[cfg(not(feature = "zk"))]
use akita_types::schedule_terminal_direct_witness_shape;

struct ProverPreparedOpeningBatch<'a, F: FieldCore, E: FieldCore, P, const D: usize> {
    point: &'a [E],
    payloads: Vec<CommittedPolynomials<'a, P, RingCommitment<F, D>, AkitaCommitmentHint<F, D>>>,
    summary: OpeningBatch,
}

fn prover_claims_to_opening_batch<'a, F, E, P, const D: usize>(
    expanded: &AkitaExpandedSetup<F>,
    claims: ProverClaims<'a, E, P, RingCommitment<F, D>, AkitaCommitmentHint<F, D>>,
) -> Result<ProverPreparedOpeningBatch<'a, F, E, P, D>, AkitaError>
where
    F: FieldCore,
    E: FieldCore,
    P: AkitaPolyOps<F, D>,
{
    let (point, payloads) = claims;
    let slots = payloads
        .iter()
        .enumerate()
        .flat_map(|(commitment_group, payload)| {
            payload
                .polynomials
                .iter()
                .enumerate()
                .map(move |(poly_idx, poly)| OpeningClaimSlot {
                    commitment_group,
                    poly_idx,
                    // Prover inputs do not contain claimed evaluations. The shared
                    // validator ignores this field, so zero is only a structural
                    // placeholder.
                    claimed_eval: E::zero(),
                    natural_num_vars: poly.num_vars(),
                    kind: OpeningClaimKind::Polynomial,
                })
        })
        .collect::<Vec<_>>();
    let total_polys = payloads
        .iter()
        .try_fold(0usize, |acc, payload| acc.checked_add(payload.poly_count()))
        .ok_or_else(|| {
            AkitaError::InvalidInput("batched_prove polynomial count overflow".to_string())
        })?;
    if slots.len() != total_polys {
        return Err(AkitaError::InvalidInput(
            "batched_prove polynomial slot count mismatch".to_string(),
        ));
    }

    let batch = OpeningBatchInput { point, slots };
    let summary = batch.validate(OpeningBatchLimits {
        max_num_vars: expanded.seed.max_num_vars,
        max_num_claims: expanded.seed.max_num_batched_polys,
    })?;

    Ok(ProverPreparedOpeningBatch {
        point,
        payloads,
        summary,
    })
}
/// Validate and flatten batched prover claims into the root proof shape.
///
/// # Errors
///
/// Returns an error if the claim shape exceeds setup capacity, mixes
/// incompatible dimensions, or has malformed batch counts.
pub fn prepare_batched_prove_inputs<'a, F, E, P, const D: usize>(
    expanded: &AkitaExpandedSetup<F>,
    claims: ProverClaims<'a, E, P, RingCommitment<F, D>, AkitaCommitmentHint<F, D>>,
) -> Result<PreparedBatchedProveInputs<'a, F, E, P, D>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
    P: AkitaPolyOps<F, D>,
{
    validate_batched_inputs(
        expanded,
        &claims,
        |payloads| {
            payloads
                .iter()
                .map(|payload| payload.polynomials.len())
                .sum()
        },
        true,
    )?;

    let prepared_batch = prover_claims_to_opening_batch(expanded, claims)?;
    let opening_point = prepared_batch.point;
    let commitments = prepared_batch
        .payloads
        .iter()
        .map(|payload| payload.commitment.clone())
        .collect::<Vec<_>>();
    let opening_batch = prepared_batch.summary;
    let flat_polys: Vec<&P> = opening_batch
        .claim_to_commitment_group()
        .iter()
        .zip(opening_batch.claim_poly_indices().iter())
        .map(|(&group_idx, &poly_idx)| &prepared_batch.payloads[group_idx].polynomials[poly_idx])
        .collect();
    let commitment_hints = prepared_batch
        .payloads
        .into_iter()
        .map(|payload| payload.hint)
        .collect();

    Ok(PreparedBatchedProveInputs {
        opening_point,
        commitments,
        opening_batch,
        flat_polys,
        commitment_hints,
    })
}

/// Build a root-direct batched proof from flattened polynomial references and
/// their commitment-group hints.
///
/// # Errors
///
/// Returns an error if any polynomial cannot produce a direct root witness.
pub fn prove_root_direct<F, L, const D: usize, P>(
    polys: &[&P],
    hints: &[AkitaCommitmentHint<F, D>],
) -> Result<AkitaBatchedProof<F, L>, AkitaError>
where
    F: FieldCore,
    L: ExtField<F>,
    P: AkitaPolyOps<F, D>,
{
    let witnesses = polys
        .iter()
        .map(|poly| poly.direct_root_witness())
        .collect::<Result<Vec<_>, _>>()?;
    #[cfg(feature = "zk")]
    {
        let b_blinding_digits = hints
            .iter()
            .flat_map(|hint| hint.b_blinding_digits())
            .map(|digits| {
                let mut flat_digits = Vec::with_capacity(digits.flat_digits().len() * D);
                for plane in digits.flat_digits() {
                    flat_digits.extend_from_slice(plane);
                }
                flat_digits
            })
            .collect();
        Ok(AkitaBatchedProof {
            zk_hiding: ZkHidingProof::default(),
            root: AkitaBatchedRootProof::new_zero_fold(witnesses, b_blinding_digits),
            steps: Vec::new(),
        })
    }
    #[cfg(not(feature = "zk"))]
    {
        let _ = hints;
        Ok(AkitaBatchedProof {
            root: AkitaBatchedRootProof::new_zero_fold(witnesses),
            steps: Vec::new(),
        })
    }
}

/// Drive batched proving end-to-end under config `Cfg`.
///
/// This owns the full top-level prover work: validate/flatten public prover
/// claims, select the schedule from `Cfg`, apply the root-direct shortcut when
/// the selected schedule says no fold is needed, bind the transcript instance
/// descriptor, and either emit a root-direct proof or run the folded-root
/// prover.
///
/// # Errors
///
/// Returns an error if claim preparation, schedule selection, root-direct
/// witness construction, transcript binding, or folded-root proving fails.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn batched_prove<'a, Cfg, T, P, B, const D: usize>(
    expanded: &Arc<AkitaExpandedSetup<Cfg::Field>>,
    prefix_slots: &SetupPrefixProverRegistry<Cfg::Field, D>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    claims: ProverClaims<
        'a,
        Cfg::ExtField,
        P,
        RingCommitment<Cfg::Field, D>,
        AkitaCommitmentHint<Cfg::Field, D>,
    >,
    transcript: &mut T,
    basis: BasisMode,
    setup_contribution_mode: SetupContributionMode,
) -> Result<AkitaBatchedProof<Cfg::Field, Cfg::ExtField>, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore
        + CanonicalField
        + RandomSampling
        + HasWide
        + HalvingField
        + Invertible
        + PseudoMersenneField,
    Cfg::ExtField: FpExtEncoding<Cfg::Field> + MulBaseUnreduced<Cfg::Field>,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>
        + ExtField<Cfg::Field>
        + FrobeniusExtField<Cfg::Field>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<Cfg::Field> + ProverTranscriptGrind<Cfg::Field>,
    P: AkitaPolyOps<Cfg::Field, D>,
    B: ProverComputeBackend<Cfg::Field>,
{
    backend.validate_prepared_setup::<D>(prepared, expanded.as_ref())?;
    let prepared_claims = {
        let _span = tracing::info_span!("prepare_batched_prove_inputs").entered();
        prepare_batched_prove_inputs::<Cfg::Field, Cfg::ExtField, P, D>(expanded.as_ref(), claims)?
    };
    let num_vars = prepared_claims.opening_batch.num_vars();
    let mut schedule = Cfg::get_params_for_prove(&prepared_claims.opening_batch)?;
    if let Some(root_step) = schedule_root_fold_step(&schedule) {
        let alpha_bits = root_step.params.ring_dimension.trailing_zeros() as usize;
        if !folded_root_supports_opening_shape::<Cfg::Field, Cfg::ExtField, Cfg::ExtField, D>(
            std::slice::from_ref(&prepared_claims.opening_point),
            &root_step.params,
            alpha_bits,
        ) && !root_tensor_projection_enabled::<Cfg::Field, Cfg::ExtField, Cfg::ExtField, D>(
            num_vars,
        ) {
            let commit_params =
                Cfg::get_params_for_batched_commitment(&prepared_claims.opening_batch)?;
            schedule = root_direct_schedule(num_vars, commit_params)?;
        }
    }
    let root_commit_params = match schedule.steps.first() {
        Some(Step::Fold(root)) => Some(&root.params),
        Some(Step::Direct(root)) => root.params.as_ref(),
        None => None,
    }
    .ok_or_else(|| AkitaError::InvalidSetup("root schedule is empty".to_string()))?;
    validate_onehot_chunk_size_for_params::<Cfg::Field, D, &P>(
        &prepared_claims.flat_polys,
        root_commit_params,
    )?;

    bind_transcript_instance_descriptor::<Cfg::Field, T, D, Cfg>(
        expanded.as_ref(),
        &prepared_claims.opening_batch,
        &schedule,
        basis,
        transcript,
    )?;

    if schedule_is_root_direct(&schedule) {
        return prove_root_direct::<Cfg::Field, Cfg::ExtField, D, P>(
            &prepared_claims.flat_polys,
            &prepared_claims.commitment_hints,
        );
    }

    if schedule_root_fold_step(&schedule).is_none() {
        return Err(AkitaError::InvalidSetup(
            "root schedule does not start with a fold".to_string(),
        ));
    }
    prove::<Cfg, T, P, B, D>(
        expanded,
        prefix_slots,
        backend,
        prepared,
        transcript,
        prepared_claims,
        &schedule,
        basis,
        setup_contribution_mode,
    )
    .map(|(proof, _total_levels)| proof)
}

/// Prove a folded batched root and assemble the recursive suffix under config
/// `Cfg`.
///
/// The prover crate owns folded-root preparation (root schedule shape checks,
/// opening-point reduction, commitment row shape validation), root fold
/// proving, the next-`w` commitment, recursive suffix proving, and final proof
/// assembly. All policy facts are obtained directly from `Cfg`.
///
/// # Errors
///
/// Returns an error if the schedule is not folded, root inputs are malformed,
/// root proving fails, or suffix construction fails.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[inline(never)]
pub fn prove<'a, Cfg, T, P, B, const D: usize>(
    expanded: &Arc<AkitaExpandedSetup<Cfg::Field>>,
    prefix_slots: &SetupPrefixProverRegistry<Cfg::Field, D>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    transcript: &mut T,
    prepared_claims: PreparedBatchedProveInputs<'a, Cfg::Field, Cfg::ExtField, P, D>,
    schedule: &Schedule,
    basis: BasisMode,
    setup_contribution_mode: SetupContributionMode,
) -> Result<(AkitaBatchedProof<Cfg::Field, Cfg::ExtField>, usize), AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore
        + CanonicalField
        + RandomSampling
        + HasWide
        + HalvingField
        + Invertible
        + PseudoMersenneField,
    Cfg::ExtField: FpExtEncoding<Cfg::Field> + MulBaseUnreduced<Cfg::Field>,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>
        + ExtField<Cfg::Field>
        + FrobeniusExtField<Cfg::Field>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<Cfg::Field> + ProverTranscriptGrind<Cfg::Field>,
    P: AkitaPolyOps<Cfg::Field, D>,
    B: ProverComputeBackend<Cfg::Field>,
{
    backend.validate_prepared_setup::<D>(prepared, expanded.as_ref())?;

    let root_scheduled = schedule.get_execution_schedule(0)?;

    if prepared_claims
        .commitments
        .iter()
        .any(|commitment| commitment.u.len() != root_scheduled.params.effective_commit_rows())
    {
        return Err(AkitaError::InvalidInput(
            "batched_prove received a commitment with the wrong length".to_string(),
        ));
    }

    let root_packed_w_len = root_current_w_len(&root_scheduled.params);
    root_scheduled.validate_current_w_len(root_packed_w_len)?;

    #[cfg(feature = "zk")]
    let (zk_hiding_commitment, mut zk_hiding_state) =
        build_zk_hiding_context::<Cfg::Field, Cfg::ExtField, Cfg::ExtField, B, D>(
            backend,
            prepared,
            schedule,
            &root_scheduled.params,
            prepared_claims.opening_batch.num_vars(),
            prepared_claims.opening_batch.num_claims(),
            1,
        )?;
    #[cfg(feature = "zk")]
    transcript.append_serde(ABSORB_ZK_HIDING_COMMITMENT, &zk_hiding_commitment.u_blind);

    if root_scheduled.is_terminal {
        // Root is itself the terminal fold: no recursive suffix.
        #[cfg(not(feature = "zk"))]
        let terminal_shape = schedule_terminal_direct_witness_shape(schedule)?;
        let terminal =
            prove_terminal_root_fold_with_params::<Cfg, Cfg::Field, Cfg::ExtField, T, P, B, D>(
                expanded,
                backend,
                prepared,
                transcript,
                &prepared_claims.flat_polys,
                prepared_claims.opening_batch,
                prepared_claims.opening_point,
                &prepared_claims.commitments,
                prepared_claims.commitment_hints,
                &root_scheduled,
                #[cfg(not(feature = "zk"))]
                terminal_shape,
                basis,
                setup_contribution_mode,
                #[cfg(feature = "zk")]
                &mut zk_hiding_state,
            )?;
        #[cfg(feature = "zk")]
        let zk_hiding_proof = zk_hiding_state.into_proof(zk_hiding_commitment)?;
        return Ok((
            AkitaBatchedProof {
                #[cfg(feature = "zk")]
                zk_hiding: zk_hiding_proof,
                root: AkitaBatchedRootProof::new_terminal(terminal),
                steps: Vec::new(),
            },
            1,
        ));
    }

    let root = prove_root::<Cfg::Field, Cfg::ExtField, T, P, B, Cfg, D>(
        expanded,
        prefix_slots,
        backend,
        prepared,
        transcript,
        &prepared_claims.flat_polys,
        prepared_claims.opening_batch,
        prepared_claims.opening_point,
        &prepared_claims.commitments,
        prepared_claims.commitment_hints,
        &root_scheduled,
        #[cfg(feature = "zk")]
        zk_hiding_state,
        basis,
        setup_contribution_mode,
    )?;
    let next_state = root.next_state;
    let root = AkitaBatchedRootProof::new(root.level_proof);

    let suffix = crate::prove_suffix::<Cfg, T, B, D>(
        expanded,
        prefix_slots,
        backend,
        prepared,
        transcript,
        next_state,
        schedule,
        setup_contribution_mode,
    )?;
    #[cfg(feature = "zk")]
    let zk_hiding = suffix.zk_hiding.into_proof(zk_hiding_commitment)?;
    Ok((
        AkitaBatchedProof {
            #[cfg(feature = "zk")]
            zk_hiding,
            root,
            steps: suffix.steps,
        },
        suffix.num_levels,
    ))
}
