//! Adapts Akita's native batched opening protocols to Jolt.
//!
//! Two kinds of batching meet at this seam:
//!
//! - **Jolt-side batching** happens upstream in the PIOP: the opening
//!   accumulator reduces the claims produced by the sumcheck stages (via RLC
//!   combination, claim reductions, or prefix packing) down to evaluation
//!   claims about committed polynomials at a common point.
//! - **Akita-native batching** is what this module delegates to: the Akita
//!   backend proves one group at a common point, or a heterogeneous sequence
//!   of independently committed groups at their group-local points, in one
//!   backend proof.
//!
//! This adapter performs no claim combination of its own — it validates the
//! statement shape, bridges Jolt's Fiat-Shamir transcript into Akita's, and
//! embeds the backend proof bytes wholesale.

use akita_config::CommitmentConfig;
use akita_pcs::AkitaTranscript;
use std::sync::Arc;

use akita_prover::{
    CpuBackend, PreparedGroupProveOps, PreparedProverGroup, ReleaseRootNttAfterFold,
    SelectedProverOpeningData,
};
use akita_types::{
    BasisMode, GroupBatchStatement, OpeningClaims, OpeningScheduleSelection, PolynomialGroupClaims,
};
use jolt_openings::{
    BatchOpeningScheme, GroupOpeningClaim, OpeningsError, PrecommittedClaim, PrecommittedOpening,
    VerifierOpeningClaim,
};
use jolt_poly::MultilinearPoly;
use jolt_transcript::{AppendToTranscript, Label, LabelWithCount, Transcript, U64Word};
use tracing::info_span;

use crate::adapters::{
    akita_error, append_batch_statement, append_verifier_setup, backend_stack,
    bridge_jolt_statement_challenge, invalid_batch, prove_failed, reverse_point, serialize_akita,
    with_backend_pool, AkitaBackendCommitment, AkitaBackendExtField, AkitaBackendFlavor,
    AkitaBackendHint, AkitaBackendOneHotPoly, AkitaBackendProof, AkitaBackendScheme,
    AkitaBatchProof, AkitaCommitment, AkitaConfig, AkitaField, AkitaHintPolynomials,
    AkitaOneHotK16BackendScheme, AkitaOneHotK16Config, AkitaOneHotK256BackendScheme,
    AkitaOneHotK256Config, AkitaProverHint, AkitaProverSetup, AkitaVerifierSetup,
    AKITA_ONE_HOT_K16, AKITA_ONE_HOT_K256,
};
use crate::scheme::validate_precommitted_order;
#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::scheme::TraceCommitmentBackend;
use crate::trace_onehot::GroupedRootSource;

type GroupedTraceOpening<'a> = SelectedProverOpeningData<
    'a,
    AkitaField,
    PreparedProverGroup<'a, GroupedRootSource>,
    AkitaField,
>;

/// Marker adapter selecting Akita's native batched opening as the Jolt batch
/// opening protocol.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AkitaNativeBatching;

fn validate_grouped_claim(
    role: &'static str,
    claim: &GroupOpeningClaim<AkitaField, AkitaCommitment>,
) -> Result<(), OpeningsError> {
    if claim.point.len() != claim.commitment.num_vars {
        return Err(invalid_batch(format!(
            "Akita {role} point has {} variables but commitment has {}",
            claim.point.len(),
            claim.commitment.num_vars
        )));
    }
    if claim.evaluations.len() != claim.commitment.poly_count {
        return Err(invalid_batch(format!(
            "Akita {role} group has {} evaluations but commitment covers {} polynomials",
            claim.evaluations.len(),
            claim.commitment.poly_count
        )));
    }
    Ok(())
}

fn validate_grouped_hint(
    role: &'static str,
    claim: &GroupOpeningClaim<AkitaField, AkitaCommitment>,
    hint: &AkitaProverHint,
) -> Result<(), OpeningsError> {
    if hint.commitment != claim.commitment {
        return Err(invalid_batch(format!(
            "Akita {role} hint does not match its public commitment"
        )));
    }
    Ok(())
}

fn validate_trace_batch_statement(
    setup: &AkitaVerifierSetup,
    precommitted: &[PrecommittedClaim<AkitaField, AkitaCommitment>],
    main: &GroupOpeningClaim<AkitaField, AkitaCommitment>,
) -> Result<(), OpeningsError> {
    validate_precommitted_order(precommitted.iter().map(|entry| entry.role))?;
    for entry in precommitted {
        validate_grouped_claim(entry.role.diagnostic_name(), &entry.claim)?;
        if entry.claim.commitment.backend_flavor != AkitaBackendFlavor::Dense
            || entry.claim.commitment.one_hot_k != 0
            || entry.claim.commitment.poly_count != 1
        {
            return Err(invalid_batch(format!(
                "Akita {} group must be one dense polynomial",
                entry.role.diagnostic_name()
            )));
        }
        if entry.claim.commitment.num_vars > setup.max_num_vars {
            return Err(invalid_batch(format!(
                "Akita {} arity exceeds grouped setup capacity",
                entry.role.diagnostic_name()
            )));
        }
    }
    validate_grouped_claim("main-trace", main)?;
    if main.commitment.backend_flavor != AkitaBackendFlavor::OneHot
        || main.commitment.one_hot_k != setup.one_hot_k
        || main.commitment.poly_count != 1
    {
        return Err(invalid_batch(
            "Akita final trace group must be one setup-matched one-hot polynomial",
        ));
    }
    let supported_grouped_config =
        setup.one_hot_k == AKITA_ONE_HOT_K256 || setup.one_hot_k == AKITA_ONE_HOT_K16;
    if !supported_grouped_config
        || main.commitment.num_vars != setup.max_num_vars
        || main.commitment.layout_digest != setup.default_layout_digest
    {
        return Err(invalid_batch(
            "Akita final trace commitment does not match the grouped final setup",
        ));
    }
    if main.commitment.poly_count > setup.max_num_polys_per_commitment_group
        || precommitted.iter().any(|entry| {
            entry.claim.commitment.poly_count > setup.max_num_polys_per_commitment_group
        })
    {
        return Err(invalid_batch(
            "Akita grouped commitment exceeds the group-local polynomial capacity",
        ));
    }
    let total = main
        .commitment
        .poly_count
        .checked_add(precommitted.len())
        .ok_or_else(|| invalid_batch("Akita grouped polynomial count overflows"))?;
    if total > setup.max_total_batch_polys {
        return Err(invalid_batch(format!(
            "Akita grouped opening has {total} polynomials but setup supports {}",
            setup.max_total_batch_polys
        )));
    }
    Ok(())
}

fn bind_grouped_statement_transcripts<T>(
    transcript: &mut T,
    setup: &AkitaVerifierSetup,
    selection: OpeningScheduleSelection,
    precommitted: &[PrecommittedClaim<AkitaField, AkitaCommitment>],
    main: &GroupOpeningClaim<AkitaField, AkitaCommitment>,
) -> Result<(AkitaTranscript<AkitaField>, Vec<u8>), OpeningsError>
where
    T: Transcript<Challenge = AkitaField>,
{
    append_verifier_setup(transcript, setup, AkitaBackendFlavor::OneHot);
    transcript.append(&Label(b"akita_precommit_batch_v3"));
    transcript.append_bytes(&serialize_akita(&selection)?);
    let group_count = precommitted
        .len()
        .checked_add(1)
        .ok_or_else(|| invalid_batch("Akita grouped statement group count overflows"))?;
    transcript.append(&LabelWithCount(b"akita_groups", group_count as u64));
    let groups = precommitted
        .iter()
        .map(|entry| (Some(entry.role), &entry.claim))
        .chain(std::iter::once((None, main)));
    for (index, (role, claim)) in groups.enumerate() {
        transcript.append(&U64Word(index as u64));
        if let Some(role) = role {
            transcript.append_bytes(role.transcript_label());
            if let Some(role_index) = role.transcript_index() {
                transcript.append(&U64Word(role_index));
            }
        } else {
            transcript.append_bytes(b"main_trace");
        }
        transcript.append(&U64Word(u64::from(role.is_some())));
        claim.commitment.append_to_transcript(transcript);
        transcript.append_values(b"akita_group_point", &claim.point);
        transcript.append(&LabelWithCount(
            b"akita_group_evaluations",
            claim.evaluations.len() as u64,
        ));
        for evaluation in &claim.evaluations {
            evaluation.append_to_transcript(transcript);
        }
    }
    let mut akita_transcript =
        AkitaTranscript::<AkitaField>::new(b"jolt-akita/precommitted-group-batch/v3");
    let bridge = bridge_jolt_statement_challenge(transcript, &mut akita_transcript);
    Ok((akita_transcript, bridge))
}

fn prove_grouped_trace_cpu(
    setup: &AkitaProverSetup,
    opening: GroupedTraceOpening<'_>,
    akita_transcript: &mut AkitaTranscript<AkitaField>,
) -> Result<AkitaBackendProof, OpeningsError> {
    let (backend_prover_setup, prepared_backend_setup) = setup.one_hot_backend()?;
    let stack = backend_stack(backend_prover_setup, prepared_backend_setup)?;
    let releasing_stack = ReleaseRootNttAfterFold::new(&stack);
    with_backend_pool(|| match setup.one_hot_k() {
        AKITA_ONE_HOT_K256 => AkitaOneHotK256BackendScheme::batched_prove(
            backend_prover_setup,
            opening,
            &releasing_stack,
            akita_transcript,
            BasisMode::Lagrange,
        ),
        AKITA_ONE_HOT_K16 => AkitaOneHotK16BackendScheme::batched_prove(
            backend_prover_setup,
            opening,
            &releasing_stack,
            akita_transcript,
            BasisMode::Lagrange,
        ),
        _ => unreachable!("one-hot K was validated by setup"),
    })
    .map_err(prove_failed)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn prove_grouped_trace_metal(
    setup: &AkitaProverSetup,
    opening: GroupedTraceOpening<'_>,
    metal: &crate::scheme::RequiredMetalTraceCommitment,
    akita_transcript: &mut AkitaTranscript<AkitaField>,
) -> Result<AkitaBackendProof, OpeningsError> {
    let (backend_prover_setup, cpu_prepared) = setup.one_hot_backend()?;
    let setup_owner = setup
        .one_hot_backend_prover_setup
        .as_ref()
        .ok_or_else(|| invalid_batch("Akita setup has no one-hot backend"))?;
    let prepared = metal.prepared_setup(setup_owner)?;
    let stack = akita_prover::ProverComputeStack::new(
        (&CpuBackend::DEFAULT, cpu_prepared),
        (&metal.backend, prepared.as_ref()),
        (&CpuBackend::DEFAULT, cpu_prepared),
        (&CpuBackend::DEFAULT, cpu_prepared),
        backend_prover_setup.expanded.as_ref(),
    )
    .map_err(akita_error)?;
    let releasing_stack = ReleaseRootNttAfterFold::new(&stack);
    metal
        .backend
        .begin_opening_metrics()
        .map_err(|error| OpeningsError::InvalidSetup(error.to_string()))?;
    with_backend_pool(|| match setup.one_hot_k() {
        AKITA_ONE_HOT_K256 => AkitaOneHotK256BackendScheme::batched_prove_with_stack(
            backend_prover_setup,
            opening,
            &releasing_stack,
            akita_transcript,
            BasisMode::Lagrange,
        ),
        AKITA_ONE_HOT_K16 => AkitaOneHotK16BackendScheme::batched_prove_with_stack(
            backend_prover_setup,
            opening,
            &releasing_stack,
            akita_transcript,
            BasisMode::Lagrange,
        ),
        _ => unreachable!("one-hot K was validated by setup"),
    })
    .map_err(prove_failed)
}

impl AkitaNativeBatching {
    pub(crate) fn prove_trace_batch<T>(
        setup: &AkitaProverSetup,
        precommitted: Vec<PrecommittedOpening<AkitaField, AkitaCommitment, AkitaProverHint>>,
        main: GroupOpeningClaim<AkitaField, AkitaCommitment>,
        main_hint: AkitaProverHint,
        transcript: &mut T,
    ) -> Result<AkitaBatchProof, OpeningsError>
    where
        T: Transcript<Challenge = AkitaField>,
    {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let trace_backend = main_hint.trace_backend.clone();
        let precommitted_claims = precommitted
            .iter()
            .map(|(entry, _)| entry.clone())
            .collect::<Vec<_>>();
        validate_trace_batch_statement(&setup.verifier, &precommitted_claims, &main)?;
        for (entry, hint) in &precommitted {
            validate_grouped_hint(entry.role.diagnostic_name(), &entry.claim, hint)?;
        }
        validate_grouped_hint("main-trace", &main, &main_hint)?;

        let mut precommitted_sources = Vec::with_capacity(precommitted.len());
        let mut precommitted_backend = Vec::with_capacity(precommitted.len());
        for (entry, hint) in &precommitted {
            let polys = match &hint.polynomials {
                AkitaHintPolynomials::Dense(polys) if polys.len() == 1 => Arc::clone(polys),
                AkitaHintPolynomials::Dense(_)
                | AkitaHintPolynomials::OneHot(_)
                | AkitaHintPolynomials::TraceOneHot(_) => {
                    return Err(invalid_batch(format!(
                        "Akita {} hint must retain one dense source",
                        entry.role.diagnostic_name()
                    )))
                }
            };
            let backend = hint.backend.clone().ok_or_else(|| {
                invalid_batch(format!(
                    "Akita {} hint has no backend payload",
                    entry.role.diagnostic_name()
                ))
            })?;
            precommitted_sources.push(GroupedRootSource::Dense(polys));
            precommitted_backend.push(backend);
        }
        let main_source = match &main_hint.polynomials {
            AkitaHintPolynomials::TraceOneHot(poly) => {
                GroupedRootSource::Trace(vec![poly.clone()].into())
            }
            AkitaHintPolynomials::OneHot(polys) if polys.len() == 1 => {
                GroupedRootSource::OneHot(Arc::clone(polys))
            }
            AkitaHintPolynomials::Dense(_) | AkitaHintPolynomials::OneHot(_) => {
                return Err(invalid_batch(
                    "Akita main-trace hint must retain one one-hot source",
                ))
            }
        };
        let (main_backend_commitment, main_backend_hint) = main_hint
            .backend
            .clone()
            .ok_or_else(|| invalid_batch("Akita main-trace hint has no backend payload"))?;

        // Group order is canonical: every precommitted group, then the final
        // trace group. Claims, backend hints, and sources stay index-aligned.
        let mut group_refs: Vec<[&GroupedRootSource; 1]> = precommitted_sources
            .iter()
            .map(|source| [source])
            .collect::<Vec<_>>();
        group_refs.push([&main_source]);
        let group_slices = group_refs
            .iter()
            .map(|refs| refs.as_slice())
            .collect::<Vec<_>>();

        let backend_main_point = reverse_point(&main.point);
        let mut group_claims = Vec::with_capacity(group_refs.len());
        let mut backend_hints = Vec::with_capacity(group_refs.len());
        for ((entry, _), (backend_commitment, backend_hint)) in
            precommitted.iter().zip(precommitted_backend)
        {
            group_claims.push(
                PolynomialGroupClaims::new(
                    entry.claim.point.clone(),
                    entry.claim.evaluations.clone(),
                    backend_commitment,
                )
                .map_err(akita_error)?,
            );
            backend_hints.push(backend_hint);
        }
        group_claims.push(
            PolynomialGroupClaims::new(
                backend_main_point,
                main.evaluations.clone(),
                main_backend_commitment,
            )
            .map_err(akita_error)?,
        );
        backend_hints.push(main_backend_hint);
        let claims = OpeningClaims::from_groups(group_claims).map_err(akita_error)?;
        let opening = match setup.one_hot_k() {
            AKITA_ONE_HOT_K256 => SelectedProverOpeningData::from_committed_claims::<
                AkitaOneHotK256Config,
            >(claims, backend_hints, group_slices),
            AKITA_ONE_HOT_K16 => SelectedProverOpeningData::from_committed_claims::<
                AkitaOneHotK16Config,
            >(claims, backend_hints, group_slices),
            _ => unreachable!("one-hot K was validated by setup"),
        }
        .map_err(akita_error)?;
        let selection = opening.selection();
        let (mut akita_transcript, statement_bridge) = bind_grouped_statement_transcripts(
            transcript,
            &setup.verifier,
            selection,
            &precommitted_claims,
            &main,
        )?;
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let backend_proof = match trace_backend.as_ref().and_then(|backend| {
            TraceCommitmentBackend::opening_shape_is_metal_qualified(
                setup.one_hot_k(),
                main.commitment.num_vars,
            )
            .then(|| backend.required_metal())
            .flatten()
        }) {
            Some(metal) => prove_grouped_trace_metal(setup, opening, metal, &mut akita_transcript)?,
            None => prove_grouped_trace_cpu(setup, opening, &mut akita_transcript)?,
        };
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        let backend_proof = prove_grouped_trace_cpu(setup, opening, &mut akita_transcript)?;
        let proof = AkitaBatchProof {
            statement_bridge,
            serialized_schedule_selection: serialize_akita(&selection)?,
            serialized_akita_proof_shape: serialize_akita(&backend_proof.shape())?,
            serialized_akita_proof: serialize_akita(&backend_proof)?,
        };
        transcript.append(&proof);
        Ok(proof)
    }

    pub(crate) fn verify_trace_batch<T>(
        setup: &AkitaVerifierSetup,
        precommitted: &[PrecommittedClaim<AkitaField, AkitaCommitment>],
        main: &GroupOpeningClaim<AkitaField, AkitaCommitment>,
        proof: &AkitaBatchProof,
        transcript: &mut T,
    ) -> Result<(), OpeningsError>
    where
        T: Transcript<Challenge = AkitaField>,
    {
        setup.ensure_schedule_rows()?;
        validate_trace_batch_statement(setup, precommitted, main)?;
        let backend_main_point = reverse_point(&main.point);
        let precommitted_commitments = precommitted
            .iter()
            .map(|entry| &entry.claim.commitment)
            .collect::<Vec<_>>();
        let (selection, precommitted_backend, main_backend, backend_proof) =
            crate::shape_guard::deserialize_checked_grouped_backend_payload(
                &precommitted_commitments,
                &main.commitment,
                proof,
                &backend_main_point,
                setup.one_hot_k,
            )?;
        let (mut akita_transcript, bridge) =
            bind_grouped_statement_transcripts(transcript, setup, selection, precommitted, main)?;
        if proof.statement_bridge != bridge {
            return Err(OpeningsError::VerificationFailed);
        }
        transcript.append(proof);
        let mut group_claims = Vec::with_capacity(precommitted.len() + 1);
        for (entry, backend) in precommitted.iter().zip(&precommitted_backend) {
            group_claims.push(
                PolynomialGroupClaims::new(
                    entry.claim.point.clone(),
                    entry.claim.evaluations.clone(),
                    backend,
                )
                .map_err(akita_error)?,
            );
        }
        group_claims.push(
            PolynomialGroupClaims::new(backend_main_point, main.evaluations.clone(), &main_backend)
                .map_err(akita_error)?,
        );
        let claims = OpeningClaims::from_groups(group_claims).map_err(akita_error)?;
        let batch_statement = GroupBatchStatement::new(selection, claims).map_err(akita_error)?;
        let backend_verifier = setup.backend_verifier(AkitaBackendFlavor::OneHot)?;
        with_backend_pool(|| match setup.one_hot_k {
            AKITA_ONE_HOT_K256 => AkitaOneHotK256BackendScheme::batched_verify(
                &backend_proof,
                backend_verifier,
                &mut akita_transcript,
                batch_statement,
                BasisMode::Lagrange,
            ),
            AKITA_ONE_HOT_K16 => AkitaOneHotK16BackendScheme::batched_verify(
                &backend_proof,
                backend_verifier,
                &mut akita_transcript,
                batch_statement,
                BasisMode::Lagrange,
            ),
            _ => unreachable!("one-hot K was validated by setup"),
        })
        .map_err(|_| OpeningsError::VerificationFailed)
    }
}

pub type AkitaNativeBatchStatement = Vec<VerifierOpeningClaim<AkitaField, AkitaCommitment>>;

pub type AkitaNativeBatchPolynomials<'a> = Vec<&'a (dyn MultilinearPoly<AkitaField> + 'a)>;

struct ValidatedStatement<'a> {
    commitment: &'a AkitaCommitment,
    point: &'a [AkitaField],
}

/// Checks that the statement is a same-point batch over exactly one
/// commitment group whose shape matches the setup.
fn validate_statement(
    statement: &[VerifierOpeningClaim<AkitaField, AkitaCommitment>],
    max_num_vars: usize,
    max_num_polys_per_commitment_group: usize,
    one_hot_k: usize,
) -> Result<ValidatedStatement<'_>, OpeningsError> {
    let first = statement
        .first()
        .ok_or_else(|| invalid_batch("Akita native batching requires at least one claim"))?;
    let commitment = &first.commitment;
    let point = first.evaluation.point.as_slice();

    if point.len() != commitment.num_vars {
        return Err(invalid_batch(format!(
            "Akita opening point has {} variables but commitment has {}",
            point.len(),
            commitment.num_vars
        )));
    }
    for claim in statement {
        if claim.commitment != *commitment {
            return Err(invalid_batch(
                "Akita batch statement must use exactly one commitment group",
            ));
        }
        if claim.evaluation.point.as_slice() != point {
            return Err(invalid_batch(
                "Akita native batching claims must use one common point",
            ));
        }
    }
    if commitment.poly_count != statement.len() {
        return Err(invalid_batch(format!(
            "Akita commitment covers {} polynomials but statement has {} claims",
            commitment.poly_count,
            statement.len()
        )));
    }
    if commitment.num_vars != max_num_vars {
        return Err(invalid_batch(format!(
            "Akita commitment dimension {} does not match exact setup dimension {max_num_vars}",
            commitment.num_vars
        )));
    }
    if commitment.poly_count > max_num_polys_per_commitment_group {
        return Err(invalid_batch(format!(
            "Akita commitment covers {} polynomials but setup supports {}",
            commitment.poly_count, max_num_polys_per_commitment_group
        )));
    }
    match commitment.backend_flavor {
        AkitaBackendFlavor::Dense if commitment.one_hot_k != 0 => {
            return Err(invalid_batch(
                "Akita dense commitment has invalid one-hot metadata",
            ));
        }
        AkitaBackendFlavor::OneHot if commitment.one_hot_k != one_hot_k => {
            return Err(invalid_batch(format!(
                "Akita commitment one-hot K={} does not match setup K={one_hot_k}",
                commitment.one_hot_k
            )));
        }
        AkitaBackendFlavor::Dense | AkitaBackendFlavor::OneHot => {}
    }
    Ok(ValidatedStatement { commitment, point })
}

/// Checks that the prover hint and witness polynomials match the statement's
/// commitment group. The hint's backend polynomials need no shape checks:
/// hints are only constructible by this crate's commit paths, which derive the
/// commitment's shape from those same polynomials.
fn validate_witness(
    hint: &AkitaProverHint,
    commitment: &AkitaCommitment,
    polynomials: &[&(dyn MultilinearPoly<AkitaField> + '_)],
) -> Result<(), OpeningsError> {
    if hint.commitment != *commitment {
        return Err(invalid_batch(
            "Akita prover hint does not match the statement commitment",
        ));
    }
    if polynomials.len() != commitment.poly_count {
        return Err(invalid_batch(format!(
            "Akita prover received {} polynomials for {} commitment slots",
            polynomials.len(),
            commitment.poly_count
        )));
    }
    for polynomial in polynomials {
        if polynomial.num_vars() != commitment.num_vars {
            return Err(invalid_batch(format!(
                "Akita witness polynomial has {} variables but commitment has {}",
                polynomial.num_vars(),
                commitment.num_vars
            )));
        }
    }
    if matches!(
        hint.polynomials,
        AkitaHintPolynomials::OneHot(_) | AkitaHintPolynomials::TraceOneHot(_)
    ) && !polynomials.iter().all(|polynomial| polynomial.is_one_hot())
    {
        return Err(invalid_batch(format!(
            "Akita {} prover hint requires one-hot witness polynomials",
            hint.polynomials.kind()
        )));
    }
    Ok(())
}

/// Binds the verifier setup and statement into Jolt's transcript, then bridges
/// a Jolt challenge into a fresh Akita transcript so the backend proof is
/// bound to everything Jolt observed.
fn bind_statement_transcripts<T>(
    transcript: &mut T,
    verifier_setup: &AkitaVerifierSetup,
    statement: &[VerifierOpeningClaim<AkitaField, AkitaCommitment>],
    commitment: &AkitaCommitment,
    point: &[AkitaField],
) -> (AkitaTranscript<AkitaField>, Vec<u8>)
where
    T: Transcript<Challenge = AkitaField>,
{
    {
        let _span = info_span!("AkitaNativeBatching::append_setup_and_statement").entered();
        append_verifier_setup(transcript, verifier_setup, commitment.backend_flavor);
        append_batch_statement(transcript, statement, commitment, point);
    }
    let mut akita_transcript = AkitaTranscript::<AkitaField>::new(b"jolt-akita/batch");
    let statement_bridge = {
        let _span = info_span!("AkitaNativeBatching::bridge_transcripts").entered();
        bridge_jolt_statement_challenge(transcript, &mut akita_transcript)
    };
    (akita_transcript, statement_bridge)
}

/// Assembles the single-group opening data handed to Akita's native batched
/// prover: the shared point, per-polynomial claimed values, the group
/// commitment, and the commit-time hint.
fn single_group_batch<'a, Cfg, P>(
    point: &[AkitaField],
    evaluations: &[AkitaField],
    polynomials: &'a [&'a P],
    backend_commitment: AkitaBackendCommitment,
    backend_hint: AkitaBackendHint,
) -> Result<
    SelectedProverOpeningData<'a, AkitaField, PreparedProverGroup<'a, P>, AkitaField>,
    akita_pcs::AkitaError,
>
where
    Cfg: CommitmentConfig<Field = AkitaField, ExtField = AkitaField>,
    P: akita_prover::RootPolyMeta<AkitaField>,
{
    let group =
        PolynomialGroupClaims::new(point.to_vec(), evaluations.to_vec(), backend_commitment)?;
    let claims = OpeningClaims::from_groups(vec![group])?;
    SelectedProverOpeningData::from_committed_claims::<Cfg>(
        claims,
        vec![backend_hint],
        vec![polynomials],
    )
}

/// The one-hot backend consumes the point in reversed variable order and uses
/// the dedicated one-hot setup pair.
fn prove_one_hot<'a, P>(
    setup: &AkitaProverSetup,
    point: &[AkitaField],
    evaluations: &[AkitaField],
    polynomials: &'a [&'a P],
    backend_commitment: AkitaBackendCommitment,
    backend_hint: AkitaBackendHint,
    akita_transcript: &mut AkitaTranscript<AkitaField>,
) -> Result<(OpeningScheduleSelection, AkitaBackendProof), OpeningsError>
where
    P: akita_prover::RootPolyMeta<AkitaField>,
    PreparedProverGroup<'a, P>: PreparedGroupProveOps<AkitaField, AkitaBackendExtField, CpuBackend>,
{
    let (backend_prover_setup, prepared_backend_setup) = setup.one_hot_backend()?;
    let backend_point = reverse_point(point);
    let opening = match setup.one_hot_k() {
        AKITA_ONE_HOT_K16 => single_group_batch::<AkitaOneHotK16Config, _>(
            &backend_point,
            evaluations,
            polynomials,
            backend_commitment,
            backend_hint,
        )
        .map_err(akita_error)?,
        AKITA_ONE_HOT_K256 => single_group_batch::<AkitaOneHotK256Config, _>(
            &backend_point,
            evaluations,
            polynomials,
            backend_commitment,
            backend_hint,
        )
        .map_err(akita_error)?,
        _ => unreachable!("the one-hot setup geometry was validated during setup"),
    };
    let selection = opening.selection();
    let stack = backend_stack(backend_prover_setup, prepared_backend_setup)?;
    let releasing_stack = ReleaseRootNttAfterFold::new(&stack);
    let _span = info_span!("AkitaNativeBatching::backend_batched_prove").entered();
    let proof = with_backend_pool(|| match setup.one_hot_k() {
        AKITA_ONE_HOT_K16 => AkitaOneHotK16BackendScheme::batched_prove(
            backend_prover_setup,
            opening,
            &releasing_stack,
            akita_transcript,
            BasisMode::Lagrange,
        ),
        AKITA_ONE_HOT_K256 => AkitaOneHotK256BackendScheme::batched_prove(
            backend_prover_setup,
            opening,
            &releasing_stack,
            akita_transcript,
            BasisMode::Lagrange,
        ),
        _ => unreachable!("the one-hot setup geometry was validated during setup"),
    })
    .map_err(prove_failed)?;
    Ok((selection, proof))
}

impl BatchOpeningScheme for AkitaNativeBatching {
    type Field = AkitaField;
    type ProverSetup = AkitaProverSetup;
    type VerifierSetup = AkitaVerifierSetup;
    type Statement = AkitaNativeBatchStatement;
    type Polynomials<'a>
        = AkitaNativeBatchPolynomials<'a>
    where
        Self: 'a;
    type Hints = AkitaProverHint;
    type Proof = AkitaBatchProof;

    fn prove_batch<'a, T>(
        setup: &Self::ProverSetup,
        statement: Self::Statement,
        polynomials: Self::Polynomials<'a>,
        hint: Self::Hints,
        transcript: &mut T,
    ) -> Result<Self::Proof, OpeningsError>
    where
        Self: 'a,
        T: Transcript<Challenge = Self::Field>,
    {
        let ValidatedStatement { commitment, point } = validate_statement(
            &statement,
            setup.max_num_vars(),
            setup.max_num_polys_per_commitment_group(),
            setup.one_hot_k(),
        )?;
        let _span = info_span!(
            "AkitaNativeBatching::prove_batch",
            source_kind = hint.polynomials.kind(),
            num_vars = point.len(),
            num_claims = statement.len(),
            poly_count = commitment.poly_count,
        )
        .entered();
        validate_witness(&hint, commitment, &polynomials)?;
        let (backend_commitment, backend_hint) = hint
            .backend
            .ok_or_else(|| invalid_batch("Akita prover hint is missing backend opening data"))?;

        let (mut akita_transcript, statement_bridge) =
            bind_statement_transcripts(transcript, &setup.verifier, &statement, commitment, point);

        let evaluations: Vec<AkitaField> = statement
            .iter()
            .map(|claim| claim.evaluation.value)
            .collect();
        let (selection, backend_proof) = match &hint.polynomials {
            AkitaHintPolynomials::Dense(dense) => {
                let refs = dense.iter().collect::<Vec<_>>();
                let opening = single_group_batch::<AkitaConfig, _>(
                    point,
                    &evaluations,
                    &refs,
                    backend_commitment,
                    backend_hint,
                )
                .map_err(akita_error)?;
                let selection = opening.selection();
                let (backend_prover_setup, prepared_backend_setup) = setup.dense_backend()?;
                let stack = backend_stack(backend_prover_setup, prepared_backend_setup)?;
                let releasing_stack = ReleaseRootNttAfterFold::new(&stack);
                let _span = info_span!("AkitaNativeBatching::backend_batched_prove").entered();
                let proof = with_backend_pool(|| {
                    AkitaBackendScheme::batched_prove(
                        backend_prover_setup,
                        opening,
                        &releasing_stack,
                        &mut akita_transcript,
                        BasisMode::Lagrange,
                    )
                })
                .map_err(prove_failed)?;
                (selection, proof)
            }
            AkitaHintPolynomials::OneHot(one_hot) => {
                let refs = one_hot.iter().collect::<Vec<_>>();
                prove_one_hot::<AkitaBackendOneHotPoly>(
                    setup,
                    point,
                    &evaluations,
                    &refs,
                    backend_commitment,
                    backend_hint,
                    &mut akita_transcript,
                )?
            }
            AkitaHintPolynomials::TraceOneHot(one_hot) => {
                let refs = [one_hot];
                prove_one_hot::<crate::trace_onehot::TracePackedOneHot>(
                    setup,
                    point,
                    &evaluations,
                    &refs,
                    backend_commitment,
                    backend_hint,
                    &mut akita_transcript,
                )?
            }
        };

        let proof = {
            let _span = info_span!("AkitaNativeBatching::serialize_backend_proof").entered();
            let proof_shape = backend_proof.shape();
            AkitaBatchProof {
                statement_bridge,
                serialized_schedule_selection: serialize_akita(&selection)?,
                serialized_akita_proof_shape: serialize_akita(&proof_shape)?,
                serialized_akita_proof: serialize_akita(&backend_proof)?,
            }
        };
        {
            let _span = info_span!("AkitaNativeBatching::append_proof").entered();
            transcript.append(&proof);
        }
        Ok(proof)
    }

    fn verify_batch<T>(
        setup: &Self::VerifierSetup,
        statement: &Self::Statement,
        proof: &Self::Proof,
        transcript: &mut T,
    ) -> Result<(), OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>,
    {
        let ValidatedStatement { commitment, point } = validate_statement(
            statement,
            setup.max_num_vars,
            setup.max_num_polys_per_commitment_group,
            setup.one_hot_k,
        )?;
        let backend_point = match commitment.backend_flavor {
            AkitaBackendFlavor::Dense => point.to_vec(),
            AkitaBackendFlavor::OneHot => reverse_point(point),
        };
        // Deserializes the proof-controlled backend payloads only after their
        // shapes are validated against the trusted schedule, so a malformed
        // proof cannot drive shape-backed allocations (see `shape_guard`).
        let (selection, backend_commitment, backend_proof) =
            crate::shape_guard::deserialize_checked_backend_payload(
                commitment,
                proof,
                statement.len(),
                &backend_point,
            )?;

        let (mut akita_transcript, statement_bridge) =
            bind_statement_transcripts(transcript, setup, statement, commitment, point);
        if proof.statement_bridge != statement_bridge {
            return Err(OpeningsError::VerificationFailed);
        }
        transcript.append(proof);

        let backend_verifier = setup.backend_verifier(commitment.backend_flavor)?;
        let openings: Vec<AkitaField> = statement
            .iter()
            .map(|claim| claim.evaluation.value)
            .collect();
        let group = PolynomialGroupClaims::new(backend_point, openings, &backend_commitment)
            .map_err(akita_error)?;
        let claims = OpeningClaims::from_groups(vec![group]).map_err(akita_error)?;
        let batch_statement = GroupBatchStatement::new(selection, claims).map_err(akita_error)?;
        with_backend_pool(|| match commitment.backend_flavor {
            AkitaBackendFlavor::Dense => AkitaBackendScheme::batched_verify(
                &backend_proof,
                backend_verifier,
                &mut akita_transcript,
                batch_statement,
                BasisMode::Lagrange,
            ),
            AkitaBackendFlavor::OneHot => match setup.one_hot_k {
                AKITA_ONE_HOT_K16 => AkitaOneHotK16BackendScheme::batched_verify(
                    &backend_proof,
                    backend_verifier,
                    &mut akita_transcript,
                    batch_statement,
                    BasisMode::Lagrange,
                ),
                AKITA_ONE_HOT_K256 => AkitaOneHotK256BackendScheme::batched_verify(
                    &backend_proof,
                    backend_verifier,
                    &mut akita_transcript,
                    batch_statement,
                    BasisMode::Lagrange,
                ),
                _ => unreachable!("the one-hot setup geometry was validated during setup"),
            },
        })
        .map_err(|_| OpeningsError::VerificationFailed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Zero;
    use jolt_openings::PrecommittedRole;

    fn commitment(
        backend_flavor: AkitaBackendFlavor,
        num_vars: usize,
        layout_digest: [u8; 32],
        one_hot_k: usize,
    ) -> AkitaCommitment {
        AkitaCommitment {
            backend_flavor,
            layout_digest,
            num_vars,
            poly_count: 1,
            one_hot_k,
            backend_coeff_len: 0,
            serialized_backend_bytes: Vec::new(),
        }
    }

    fn claim(commitment: AkitaCommitment) -> GroupOpeningClaim<AkitaField, AkitaCommitment> {
        GroupOpeningClaim::new(
            commitment.clone(),
            vec![AkitaField::zero(); commitment.num_vars],
            vec![AkitaField::zero()],
        )
    }

    #[test]
    fn verifier_shape_enforces_the_260_group_limit() {
        let layout_digest = [9; 32];
        let mut setup = AkitaVerifierSetup {
            max_num_vars: 34,
            max_num_polys_per_commitment_group: 1,
            max_total_batch_polys: 260,
            default_layout_digest: layout_digest,
            one_hot_k: AKITA_ONE_HOT_K256,
            precommitted_schedule: None,
            backend_cache: Default::default(),
        };
        let dense = || commitment(AkitaBackendFlavor::Dense, 14, [7; 32], 0);
        let precommitted = (0_u64..259)
            .map(|order| {
                PrecommittedClaim::new(
                    PrecommittedRole::new(order, b"precommitted", "precommitted"),
                    claim(dense()),
                )
            })
            .collect::<Vec<_>>();
        let main = claim(commitment(
            AkitaBackendFlavor::OneHot,
            34,
            layout_digest,
            AKITA_ONE_HOT_K256,
        ));

        assert!(validate_trace_batch_statement(&setup, &precommitted, &main).is_ok());
        setup.max_total_batch_polys = 259;
        assert!(validate_trace_batch_statement(&setup, &precommitted, &main).is_err());
    }
}
