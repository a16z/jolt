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
//! statement shape, bridges Jolt's Fiat-Shamir transcript into Akita's
//! session, and embeds the backend argument bytes wholesale.

use akita_config::{CommitmentConfig, TrustedScheduleCatalog};
use akita_params::{BasisMode, OpeningScheduleSelection};
use akita_pcs::{AkitaError, SelectedProverOpeningData};
use akita_types::{GroupBatchStatement, OpeningClaims, PolynomialGroupClaims};
use jolt_openings::{
    BatchOpeningScheme, GroupOpeningClaim, GroupOpeningWithHint, OpeningsError,
    TaggedGroupOpeningClaim, VerifierOpeningClaim,
};
use jolt_poly::MultilinearPoly;
use jolt_transcript::{AppendToTranscript, Label, LabelWithCount, Transcript, U64Word};
use tracing::info_span;

use crate::adapters::{
    akita_error, append_batch_statement, append_verifier_setup, bridged_akita_session,
    invalid_batch, prove_failed, reverse_point, serialize_akita, validate_one_hot_k,
    with_backend_pool, AkitaBackendCommitment, AkitaBackendExtField, AkitaBackendFlavor,
    AkitaBackendHint, AkitaBatchProof, AkitaCommitment, AkitaConfig, AkitaField, AkitaHintSource,
    AkitaProverHint, AkitaProverSetup, AkitaVerifierSetup, AKITA_ONE_HOT_K16, AKITA_ONE_HOT_K256,
};
use crate::one_hot_family::with_one_hot_family;
use crate::scheme::validate_group_order;

/// Marker adapter selecting Akita's native batched opening as the Jolt batch
/// opening protocol.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AkitaNativeBatching;

type AkitaOpening<'a> =
    SelectedProverOpeningData<'a, AkitaBackendExtField, AkitaBackendHint, AkitaField>;

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
    auxiliary_groups: &[TaggedGroupOpeningClaim<AkitaField, AkitaCommitment>],
    main: &GroupOpeningClaim<AkitaField, AkitaCommitment>,
) -> Result<(), OpeningsError> {
    validate_group_order(auxiliary_groups.iter().map(|entry| entry.role))?;
    for entry in auxiliary_groups {
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
        // Only an object above the final arity needs the catalog-derived
        // capacity; the common case stays a pure shape check.
        if entry.claim.commitment.num_vars > setup.max_num_vars
            && entry.claim.commitment.num_vars > setup.one_hot_backend_num_vars()?
        {
            return Err(invalid_batch(format!(
                "Akita {} arity exceeds grouped setup capacity",
                entry.role.diagnostic_name()
            )));
        }
    }
    validate_grouped_claim("main-trace", main)?;
    if main.commitment.backend_flavor != AkitaBackendFlavor::OneHot
        || main.commitment.one_hot_k != setup.one_hot_k
        || main.commitment.poly_count != setup.max_num_polys_per_commitment_group
    {
        return Err(invalid_batch(
            "Akita final trace group must be a setup-matched one-hot batch",
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
        || auxiliary_groups.iter().any(|entry| {
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
        .checked_add(auxiliary_groups.len())
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
    auxiliary_groups: &[TaggedGroupOpeningClaim<AkitaField, AkitaCommitment>],
    main: &GroupOpeningClaim<AkitaField, AkitaCommitment>,
) -> Result<Vec<u8>, OpeningsError>
where
    T: Transcript<Challenge = AkitaField>,
{
    append_verifier_setup(transcript, setup, AkitaBackendFlavor::OneHot)?;
    transcript.append(&Label(b"akita_precommit_batch_v4"));
    transcript.append_bytes(&serialize_akita(&selection)?);
    let group_count = auxiliary_groups
        .len()
        .checked_add(1)
        .ok_or_else(|| invalid_batch("Akita grouped statement group count overflows"))?;
    transcript.append(&LabelWithCount(b"akita_groups", group_count as u64));
    let groups = auxiliary_groups
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
    Ok(bridged_akita_session(
        transcript,
        b"jolt-akita/precommitted-group-batch/v4",
    ))
}

fn prove_one_hot_opening(
    setup: &AkitaProverSetup,
    opening: AkitaOpening<'_>,
    session: &[u8],
) -> Result<Vec<u8>, OpeningsError> {
    let (backend_prover_setup, backend) = setup.one_hot_backend()?;
    let _span = info_span!("AkitaNativeBatching::backend_batched_prove").entered();
    let scheme = setup.verifier.one_hot_scheme()?;
    with_backend_pool(|| {
        let proof = with_one_hot_family!(scheme scheme, |scheme| scheme.batched_prove(
            backend_prover_setup,
            opening,
            backend,
            session,
            BasisMode::Lagrange,
        ))?;
        let _ = backend.trim_caches()?;
        Ok::<_, AkitaError>(proof)
    })
    .map_err(prove_failed)
}

fn verify_one_hot_statement(
    setup: &AkitaVerifierSetup,
    proof: &AkitaBatchProof,
    session: &[u8],
    statement: GroupBatchStatement<'_, AkitaBackendExtField, AkitaField>,
) -> Result<(), OpeningsError> {
    let verifier = setup.one_hot_verifier()?;
    let verified = with_backend_pool(|| {
        with_one_hot_family!(verifier verifier, |verifier| verifier.batched_verify(
            &proof.backend_proof,
            session,
            statement,
            BasisMode::Lagrange,
        ))
    });
    verified.map_err(|_| OpeningsError::VerificationFailed)
}

impl AkitaNativeBatching {
    pub(crate) fn prove_trace_batch<T>(
        setup: &AkitaProverSetup,
        auxiliary_groups: Vec<GroupOpeningWithHint<AkitaField, AkitaCommitment, AkitaProverHint>>,
        main: GroupOpeningClaim<AkitaField, AkitaCommitment>,
        main_hint: AkitaProverHint,
        transcript: &mut T,
    ) -> Result<AkitaBatchProof, OpeningsError>
    where
        T: Transcript<Challenge = AkitaField>,
    {
        let auxiliary_claims = auxiliary_groups
            .iter()
            .map(|(entry, _)| entry.clone())
            .collect::<Vec<_>>();
        validate_trace_batch_statement(&setup.verifier, &auxiliary_claims, &main)?;
        for (entry, hint) in &auxiliary_groups {
            validate_grouped_hint(entry.role.diagnostic_name(), &entry.claim, hint)?;
        }
        validate_grouped_hint("main-trace", &main, &main_hint)?;

        // Group order is canonical: every auxiliary group, then the final
        // trace group. Claims and backend handles stay index-aligned.
        let mut group_claims = Vec::with_capacity(auxiliary_groups.len() + 1);
        let mut handles = Vec::with_capacity(auxiliary_groups.len() + 1);
        for (entry, hint) in auxiliary_groups {
            if !matches!(hint.source, AkitaHintSource::Dense { poly_count: 1 }) {
                return Err(invalid_batch(format!(
                    "Akita {} hint must retain one dense source",
                    entry.role.diagnostic_name()
                )));
            }
            let (backend_commitment, backend_hint) = hint.backend.ok_or_else(|| {
                invalid_batch(format!(
                    "Akita {} hint has no backend payload",
                    entry.role.diagnostic_name()
                ))
            })?;
            group_claims.push(
                PolynomialGroupClaims::new(
                    entry.claim.point.clone(),
                    entry.claim.evaluations.clone(),
                    backend_commitment,
                )
                .map_err(akita_error)?,
            );
            handles.push(backend_hint);
        }
        if !matches!(
            main_hint.source,
            AkitaHintSource::TraceOneHot { .. } | AkitaHintSource::OneHot { .. }
        ) {
            return Err(invalid_batch(
                "Akita main-trace hint must retain the one-hot batch",
            ));
        }
        let (main_backend_commitment, main_backend_hint) = main_hint
            .backend
            .ok_or_else(|| invalid_batch("Akita main-trace hint has no backend payload"))?;
        group_claims.push(
            PolynomialGroupClaims::new(
                reverse_point(&main.point),
                main.evaluations.clone(),
                main_backend_commitment,
            )
            .map_err(akita_error)?,
        );
        // Auxiliary objects were committed on their own setups' dense
        // backends, but a handle only proves on the backend that owns it and the
        // grouped argument runs on the trace backend. Akita re-derives each
        // imported handle from its retained source and checks the recomputed
        // commitment against this setup's public matrix (shared by every Jolt
        // setup through the deterministic setup seed).
        let (_, backend) = setup.one_hot_backend()?;
        let mut handles = with_backend_pool(|| {
            let _span = info_span!("AkitaNativeBatching::import_auxiliary_handles").entered();
            handles
                .iter()
                .map(|handle| backend.import_commitment(handle))
                .collect::<Result<Vec<_>, _>>()
        })
        .map_err(prove_failed)?;
        handles.push(main_backend_hint);
        let claims = OpeningClaims::from_groups(group_claims).map_err(akita_error)?;
        let opening = with_one_hot_family!(scheme setup.verifier.one_hot_scheme()?, |scheme, Cfg| {
            SelectedProverOpeningData::from_committed_claims::<Cfg>(
                claims,
                handles,
                scheme.schedules(),
            )
        })
        .map_err(akita_error)?;
        let selection = opening.selection();
        let session = bind_grouped_statement_transcripts(
            transcript,
            &setup.verifier,
            selection,
            &auxiliary_claims,
            &main,
        )?;
        let backend_proof = prove_one_hot_opening(setup, opening, &session)?;
        Ok(AkitaBatchProof::new(selection, backend_proof))
    }

    pub(crate) fn verify_trace_batch<T>(
        setup: &AkitaVerifierSetup,
        auxiliary_groups: &[TaggedGroupOpeningClaim<AkitaField, AkitaCommitment>],
        main: &GroupOpeningClaim<AkitaField, AkitaCommitment>,
        proof: &AkitaBatchProof,
        transcript: &mut T,
    ) -> Result<(), OpeningsError>
    where
        T: Transcript<Challenge = AkitaField>,
    {
        validate_trace_batch_statement(setup, auxiliary_groups, main)?;
        let backend_main_point = reverse_point(&main.point);
        let auxiliary_commitments = auxiliary_groups
            .iter()
            .map(|entry| &entry.claim.commitment)
            .collect::<Vec<_>>();
        let (auxiliary_backend, main_backend) = with_one_hot_family!(scheme setup.one_hot_scheme()?, |scheme| {
            crate::shape_guard::deserialize_checked_grouped_backend_payload(
                scheme.schedules(),
                &auxiliary_commitments,
                &main.commitment,
                proof.selection(),
            )
        })?;
        let selection = proof.selection();
        let session = bind_grouped_statement_transcripts(
            transcript,
            setup,
            selection,
            auxiliary_groups,
            main,
        )?;
        let mut group_claims = Vec::with_capacity(auxiliary_groups.len() + 1);
        for (entry, backend) in auxiliary_groups.iter().zip(&auxiliary_backend) {
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
        verify_one_hot_statement(setup, proof, &session, batch_statement)
    }
}

pub type AkitaNativeBatchStatement = Vec<VerifierOpeningClaim<AkitaField, AkitaCommitment>>;

pub type AkitaNativeBatchPolynomials<'a> = Vec<&'a (dyn MultilinearPoly<AkitaField> + 'a)>;

struct ValidatedStatement<'a> {
    commitment: &'a AkitaCommitment,
    point: &'a [AkitaField],
}

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
        AkitaBackendFlavor::OneHot => {
            let _ = validate_one_hot_k(one_hot_k)?;
            if commitment.one_hot_k != one_hot_k {
                return Err(invalid_batch(format!(
                    "Akita commitment one-hot K={} does not match setup K={one_hot_k}",
                    commitment.one_hot_k
                )));
            }
        }
        AkitaBackendFlavor::Dense => {}
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
    Ok(ValidatedStatement { commitment, point })
}

/// Checks that the prover hint and witness polynomials match the statement's
/// commitment group. The hint's backend handle needs no shape checks: hints
/// are only constructible by this crate's commit paths, which derive the
/// commitment's shape from the source the handle retains.
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
        hint.source,
        AkitaHintSource::OneHot { .. } | AkitaHintSource::TraceOneHot { .. }
    ) && !polynomials.iter().all(|polynomial| polynomial.is_one_hot())
    {
        return Err(invalid_batch(format!(
            "Akita {} prover hint requires one-hot witness polynomials",
            hint.source.kind()
        )));
    }
    Ok(())
}

/// Binds the verifier setup and statement into Jolt's transcript, then bridges
/// a Jolt challenge into the Akita session bytes so the backend argument is
/// bound to everything Jolt observed.
fn bind_statement_transcripts<T>(
    transcript: &mut T,
    verifier_setup: &AkitaVerifierSetup,
    statement: &[VerifierOpeningClaim<AkitaField, AkitaCommitment>],
    commitment: &AkitaCommitment,
    point: &[AkitaField],
) -> Result<Vec<u8>, OpeningsError>
where
    T: Transcript<Challenge = AkitaField>,
{
    {
        let _span = info_span!("AkitaNativeBatching::append_setup_and_statement").entered();
        append_verifier_setup(transcript, verifier_setup, commitment.backend_flavor)?;
        append_batch_statement(transcript, statement, commitment, point);
    }
    let _span = info_span!("AkitaNativeBatching::bridge_transcripts").entered();
    Ok(bridged_akita_session(transcript, b"jolt-akita/batch"))
}

fn single_group_batch<'a, Cfg>(
    schedules: &TrustedScheduleCatalog<Cfg>,
    point: &[AkitaField],
    evaluations: &[AkitaField],
    backend_commitment: AkitaBackendCommitment,
    backend_hint: AkitaBackendHint,
) -> Result<AkitaOpening<'a>, AkitaError>
where
    Cfg: CommitmentConfig<Field = AkitaField, ExtField = AkitaField>,
{
    let group =
        PolynomialGroupClaims::new(point.to_vec(), evaluations.to_vec(), backend_commitment)?;
    let claims = OpeningClaims::from_groups(vec![group])?;
    SelectedProverOpeningData::from_committed_claims::<Cfg>(claims, vec![backend_hint], schedules)
}

/// The one-hot backend consumes the point in reversed variable order and uses
/// the dedicated one-hot setup pair.
fn prove_one_hot(
    setup: &AkitaProverSetup,
    point: &[AkitaField],
    evaluations: &[AkitaField],
    backend_commitment: AkitaBackendCommitment,
    backend_hint: AkitaBackendHint,
    session: &[u8],
) -> Result<(OpeningScheduleSelection, Vec<u8>), OpeningsError> {
    let backend_point = reverse_point(point);
    let opening = with_one_hot_family!(scheme setup.verifier.one_hot_scheme()?, |scheme, Cfg| {
        single_group_batch::<Cfg>(
            scheme.schedules(),
            &backend_point,
            evaluations,
            backend_commitment,
            backend_hint,
        )
    })
    .map_err(akita_error)?;
    let selection = opening.selection();
    let proof = prove_one_hot_opening(setup, opening, session)?;
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
            source_kind = hint.source.kind(),
            num_vars = point.len(),
            num_claims = statement.len(),
            poly_count = commitment.poly_count,
        )
        .entered();
        validate_witness(&hint, commitment, &polynomials)?;
        let (backend_commitment, backend_hint) = hint
            .backend
            .ok_or_else(|| invalid_batch("Akita prover hint is missing backend opening data"))?;

        let session =
            bind_statement_transcripts(transcript, &setup.verifier, &statement, commitment, point)?;

        let evaluations: Vec<AkitaField> = statement
            .iter()
            .map(|claim| claim.evaluation.value)
            .collect();
        let (selection, backend_proof) = match hint.source {
            AkitaHintSource::Dense { .. } => {
                let scheme = setup.verifier.dense_scheme()?;
                let opening = single_group_batch::<AkitaConfig>(
                    scheme.schedules(),
                    point,
                    &evaluations,
                    backend_commitment,
                    backend_hint,
                )
                .map_err(akita_error)?;
                let selection = opening.selection();
                let (backend_prover_setup, backend) = setup.dense_backend()?;
                let _span = info_span!("AkitaNativeBatching::backend_batched_prove").entered();
                let proof = with_backend_pool(|| {
                    let proof = scheme.batched_prove(
                        backend_prover_setup,
                        opening,
                        backend,
                        &session,
                        BasisMode::Lagrange,
                    )?;
                    let _ = backend.trim_caches()?;
                    Ok::<_, AkitaError>(proof)
                })
                .map_err(prove_failed)?;
                (selection, proof)
            }
            AkitaHintSource::OneHot { .. } | AkitaHintSource::TraceOneHot { .. } => prove_one_hot(
                setup,
                point,
                &evaluations,
                backend_commitment,
                backend_hint,
                &session,
            )?,
        };
        Ok(AkitaBatchProof::new(selection, backend_proof))
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
        // Deserializes the proof-controlled backend commitment only after its
        // shape is validated against the trusted schedule, so a malformed
        // proof cannot drive shape-backed allocations (see `shape_guard`).
        let backend_commitment = match commitment.backend_flavor {
            AkitaBackendFlavor::Dense => crate::shape_guard::deserialize_checked_backend_payload(
                setup.dense_scheme()?.schedules(),
                commitment,
                proof.selection(),
                statement.len(),
                &backend_point,
            ),
            AkitaBackendFlavor::OneHot => {
                with_one_hot_family!(scheme setup.one_hot_scheme()?, |scheme| {
                    crate::shape_guard::deserialize_checked_backend_payload(
                        scheme.schedules(),
                        commitment,
                        proof.selection(),
                        statement.len(),
                        &backend_point,
                    )
                })
            }
        }?;

        let session = bind_statement_transcripts(transcript, setup, statement, commitment, point)?;

        let openings: Vec<AkitaField> = statement
            .iter()
            .map(|claim| claim.evaluation.value)
            .collect();
        let group = PolynomialGroupClaims::new(backend_point, openings, &backend_commitment)
            .map_err(akita_error)?;
        let claims = OpeningClaims::from_groups(vec![group]).map_err(akita_error)?;
        let batch_statement =
            GroupBatchStatement::new(proof.selection(), claims).map_err(akita_error)?;
        match commitment.backend_flavor {
            AkitaBackendFlavor::Dense => {
                let verifier = setup.dense_verifier()?;
                with_backend_pool(|| {
                    verifier.batched_verify(
                        &proof.backend_proof,
                        &session,
                        batch_statement,
                        BasisMode::Lagrange,
                    )
                })
                .map_err(|_| OpeningsError::VerificationFailed)
            }
            AkitaBackendFlavor::OneHot => {
                verify_one_hot_statement(setup, proof, &session, batch_statement)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::unwrap_used,
        clippy::expect_used,
        reason = "tests assert successful fixture construction and verifier rejection"
    )]

    use super::*;
    use jolt_field::Zero;
    use jolt_openings::{CommitmentGroupRole, EvaluationClaim};
    use jolt_transcript::Blake2bTranscript;

    use crate::adapters::AkitaVerifierScheduleArtifacts;

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
    fn verifier_rejects_transported_unsupported_one_hot_setup() {
        let setup = AkitaVerifierSetup {
            max_num_vars: 4,
            max_num_polys_per_commitment_group: 1,
            max_total_batch_polys: 1,
            default_layout_digest: [9; 32],
            one_hot_k: AKITA_ONE_HOT_K16,
            schedule_artifacts: AkitaVerifierScheduleArtifacts::OneHot {
                one_hot: Vec::new(),
            },
            backend_cache: Default::default(),
        };
        let mut serialized = serde_json::to_value(setup).unwrap();
        *serialized.get_mut("one_hot_k").unwrap() = serde_json::json!(8);
        let transported: AkitaVerifierSetup = serde_json::from_value(serialized).unwrap();
        let statement = vec![VerifierOpeningClaim {
            commitment: commitment(AkitaBackendFlavor::OneHot, 4, [9; 32], 8),
            evaluation: EvaluationClaim::new(vec![AkitaField::zero(); 4], AkitaField::zero()),
        }];
        let proof = AkitaBatchProof {
            schedule_selection: [0; 32],
            backend_proof: Vec::new(),
        };
        let mut transcript = Blake2bTranscript::new(b"invalid-transported-setup");
        let error = <AkitaNativeBatching as BatchOpeningScheme>::verify_batch(
            &transported,
            &statement,
            &proof,
            &mut transcript,
        )
        .expect_err("unsupported setup K must reject before backend dispatch");
        assert!(matches!(error, OpeningsError::InvalidBatch(_)));
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
            schedule_artifacts: AkitaVerifierScheduleArtifacts::Both {
                dense: Vec::new(),
                one_hot: Vec::new(),
            },
            backend_cache: Default::default(),
        };
        let dense = || commitment(AkitaBackendFlavor::Dense, 14, [7; 32], 0);
        let auxiliary_groups = (0_u64..259)
            .map(|order| {
                TaggedGroupOpeningClaim::new(
                    CommitmentGroupRole::new(order, b"precommitted", "auxiliary group"),
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

        assert!(validate_trace_batch_statement(&setup, &auxiliary_groups, &main).is_ok());
        setup.max_total_batch_polys = 259;
        assert!(validate_trace_batch_statement(&setup, &auxiliary_groups, &main).is_err());
    }
}
