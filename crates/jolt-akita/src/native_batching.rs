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
//! This adapter performs no claim combination of its own. It validates the
//! statement shape and runs Akita's argument. Jolt's trace batch runs inside
//! Jolt's own Fiat-Shamir sponge: Akita absorbs its statement and writes its
//! messages into Jolt's argument string. The generic [`BatchOpeningScheme`]
//! and single-polynomial paths run Akita standalone (see
//! [`STANDALONE_SESSION`]).

use akita_config::{CommitmentConfig, TrustedScheduleCatalog};
use akita_params::{BasisMode, OpeningScheduleSelection};
use akita_pcs::{AkitaError, SelectedProverOpeningData};
use akita_transcript::{ProverChannel, VerifierChannel};
use akita_types::{GroupBatchStatement, OpeningClaims, PendingEof, PolynomialGroupClaims};
use jolt_openings::{
    BatchOpeningScheme, GroupOpeningClaim, GroupOpeningWithHint, OpeningsError,
    TaggedGroupOpeningClaim, VerifierOpeningClaim,
};
use jolt_poly::MultilinearPoly;
use jolt_transcript::FieldTranscript;
use tracing::info_span;

use crate::adapters::{
    akita_error, invalid_batch, prove_failed, reverse_point, validate_one_hot_k, with_backend_pool,
    AkitaBackendCommitment, AkitaBackendExtField, AkitaBackendFlavor, AkitaBackendHint,
    AkitaBatchProof, AkitaCommitment, AkitaConfig, AkitaField, AkitaHintSource, AkitaProverHint,
    AkitaProverSetup, AkitaVerifierSetup, AKITA_ONE_HOT_K16, AKITA_ONE_HOT_K256,
};
use crate::one_hot_family::with_one_hot_family;
use crate::scheme::validate_group_order;

/// Marker adapter selecting Akita's native batched opening as the Jolt batch
/// opening protocol.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AkitaNativeBatching;

/// Akita session for an opening that runs outside Jolt's sponge.
///
/// Such an opening is a standalone argument: Akita's own transcript binds its
/// setup, schedule selection, points, evaluations and commitment digests. It
/// neither reads nor advances the caller's transcript. Jolt's trace batch
/// never takes this path.
const STANDALONE_SESSION: &[u8] = b"jolt-akita/batch";

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

/// Returns the backend handle of a hint whose commitment is the claim's.
fn grouped_backend_hint(
    role: &'static str,
    claim: &GroupOpeningClaim<AkitaField, AkitaCommitment>,
    hint: AkitaProverHint,
) -> Result<AkitaBackendHint, OpeningsError> {
    match hint.opening {
        Some((commitment, backend_hint)) if commitment == claim.commitment => Ok(backend_hint),
        Some(_) => Err(invalid_batch(format!(
            "Akita {role} hint does not match its public commitment"
        ))),
        None => Err(invalid_batch(format!(
            "Akita {role} hint has no backend opening data"
        ))),
    }
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

fn prove_one_hot_opening(
    setup: &AkitaProverSetup,
    opening: AkitaOpening<'_>,
) -> Result<Vec<u8>, OpeningsError> {
    let (backend_prover_setup, backend) = setup.one_hot_backend()?;
    let _span = info_span!("AkitaNativeBatching::backend_batched_prove").entered();
    let scheme = setup.verifier.one_hot_scheme()?;
    with_backend_pool(|| {
        let proof = with_one_hot_family!(scheme scheme, |scheme| scheme.batched_prove(
            backend_prover_setup,
            opening,
            backend,
            STANDALONE_SESSION,
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
    statement: GroupBatchStatement<'_, AkitaBackendExtField, AkitaField>,
) -> Result<(), OpeningsError> {
    let verifier = setup.one_hot_verifier()?;
    let verified = with_backend_pool(|| {
        with_one_hot_family!(verifier verifier, |verifier| verifier.batched_verify(
            &proof.backend_proof,
            STANDALONE_SESSION,
            statement,
            BasisMode::Lagrange,
        ))
    });
    verified.map_err(|_| OpeningsError::VerificationFailed)
}

impl AkitaNativeBatching {
    /// Proves Jolt's trace batch inside Jolt's sponge. Akita writes its whole
    /// argument into `transcript`, so the returned proof carries only the
    /// schedule selection.
    pub(crate) fn prove_trace_batch(
        setup: &AkitaProverSetup,
        auxiliary_groups: Vec<GroupOpeningWithHint<AkitaField, AkitaCommitment, AkitaProverHint>>,
        main: GroupOpeningClaim<AkitaField, AkitaCommitment>,
        main_hint: AkitaProverHint,
        transcript: &mut ProverChannel,
    ) -> Result<AkitaBatchProof, OpeningsError> {
        let auxiliary_claims = auxiliary_groups
            .iter()
            .map(|(entry, _)| entry.clone())
            .collect::<Vec<_>>();
        validate_trace_batch_statement(&setup.verifier, &auxiliary_claims, &main)?;

        // Group order is canonical: every auxiliary group, then the final
        // trace group. Claims and backend handles stay index-aligned.
        let mut group_claims = Vec::with_capacity(auxiliary_groups.len() + 1);
        let mut handles = Vec::with_capacity(auxiliary_groups.len() + 1);
        for (entry, hint) in auxiliary_groups {
            let role = entry.role.diagnostic_name();
            if !matches!(hint.source, AkitaHintSource::Dense { poly_count: 1 }) {
                return Err(invalid_batch(format!(
                    "Akita {role} hint must retain one dense source"
                )));
            }
            handles.push(grouped_backend_hint(role, &entry.claim, hint)?);
            group_claims.push(
                PolynomialGroupClaims::new(
                    entry.claim.point,
                    entry.claim.evaluations,
                    entry.claim.commitment.group,
                )
                .map_err(akita_error)?,
            );
        }
        if !matches!(
            main_hint.source,
            AkitaHintSource::TraceOneHot { .. } | AkitaHintSource::OneHot { .. }
        ) {
            return Err(invalid_batch(
                "Akita main-trace hint must retain the one-hot batch",
            ));
        }
        let main_backend_hint = grouped_backend_hint("main-trace", &main, main_hint)?;
        group_claims.push(
            PolynomialGroupClaims::new(
                reverse_point(&main.point),
                main.evaluations,
                main.commitment.group,
            )
            .map_err(akita_error)?,
        );
        // Auxiliary objects were committed on their own setups' dense
        // backends, but a handle only proves on the backend that owns it and the
        // grouped argument runs on the trace backend. Akita re-derives each
        // imported handle from its retained source and checks the recomputed
        // commitment against this setup's public matrix (shared by every Jolt
        // setup through the deterministic setup seed).
        let (backend_prover_setup, backend) = setup.one_hot_backend()?;
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
        let scheme = setup.verifier.one_hot_scheme()?;
        let opening = with_one_hot_family!(scheme scheme, |scheme, Cfg| {
            SelectedProverOpeningData::from_committed_claims::<Cfg>(
                claims,
                handles,
                scheme.schedules(),
            )
        })
        .map_err(akita_error)?;
        let selection = opening.selection();
        let _span = info_span!("AkitaNativeBatching::backend_batched_prove_embedded").entered();
        with_backend_pool(|| {
            with_one_hot_family!(scheme scheme, |scheme| scheme.batched_prove_embedded(
                backend_prover_setup,
                opening,
                backend,
                transcript,
                BasisMode::Lagrange,
            ))?;
            let _ = backend.trim_caches()?;
            Ok::<_, AkitaError>(())
        })
        .map_err(prove_failed)?;
        Ok(AkitaBatchProof::new(selection, Vec::new()))
    }

    /// Verifies Jolt's trace batch inside Jolt's sponge. The caller completes
    /// the verification with [`PendingEof::check_eof`] after its own last
    /// message.
    pub(crate) fn verify_trace_batch(
        setup: &AkitaVerifierSetup,
        auxiliary_groups: &[TaggedGroupOpeningClaim<AkitaField, AkitaCommitment>],
        main: &GroupOpeningClaim<AkitaField, AkitaCommitment>,
        proof: &AkitaBatchProof,
        transcript: &mut VerifierChannel<'_>,
    ) -> Result<PendingEof, OpeningsError> {
        validate_trace_batch_statement(setup, auxiliary_groups, main)?;
        // Akita's argument lives in Jolt's argument string. Bytes here would be
        // an unbound second encoding of the same proof.
        if !proof.backend_proof.is_empty() {
            return Err(invalid_batch(
                "Akita embedded trace batch carries a standalone argument",
            ));
        }
        let mut group_claims = Vec::with_capacity(auxiliary_groups.len() + 1);
        for entry in auxiliary_groups {
            group_claims.push(
                PolynomialGroupClaims::new(
                    entry.claim.point.clone(),
                    entry.claim.evaluations.clone(),
                    &entry.claim.commitment.group,
                )
                .map_err(akita_error)?,
            );
        }
        group_claims.push(
            PolynomialGroupClaims::new(
                reverse_point(&main.point),
                main.evaluations.clone(),
                &main.commitment.group,
            )
            .map_err(akita_error)?,
        );
        let claims = OpeningClaims::from_groups(group_claims).map_err(akita_error)?;
        let statement = GroupBatchStatement::new(proof.selection(), claims).map_err(akita_error)?;
        let verifier = setup.one_hot_verifier()?;
        with_backend_pool(|| {
            with_one_hot_family!(verifier verifier, |verifier| verifier.batched_verify_embedded(
                transcript,
                statement,
                BasisMode::Lagrange,
            ))
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
    if hint.opening.as_ref().map(|(hinted, _)| hinted) != Some(commitment) {
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
    let proof = prove_one_hot_opening(setup, opening)?;
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

    /// Runs Akita standalone under [`STANDALONE_SESSION`]; `_transcript` is
    /// untouched.
    fn prove_batch<'a, T>(
        setup: &Self::ProverSetup,
        statement: Self::Statement,
        polynomials: Self::Polynomials<'a>,
        hint: Self::Hints,
        _transcript: &mut T,
    ) -> Result<Self::Proof, OpeningsError>
    where
        Self: 'a,
        T: FieldTranscript<Self::Field>,
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
        let backend_commitment = commitment.group.clone();
        let source = hint.source;
        let (_, backend_hint) = hint
            .opening
            .ok_or_else(|| invalid_batch("Akita prover hint is missing backend opening data"))?;

        let evaluations: Vec<AkitaField> = statement
            .iter()
            .map(|claim| claim.evaluation.value)
            .collect();
        let (selection, backend_proof) = match source {
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
                        STANDALONE_SESSION,
                        BasisMode::Lagrange,
                    )?;
                    let _ = backend.trim_caches()?;
                    Ok::<_, AkitaError>(proof)
                })
                .map_err(prove_failed)?;
                (selection, proof)
            }
            AkitaHintSource::OneHot { .. } | AkitaHintSource::TraceOneHot { .. } => {
                prove_one_hot(setup, point, &evaluations, backend_commitment, backend_hint)?
            }
        };
        Ok(AkitaBatchProof::new(selection, backend_proof))
    }

    /// Verifies a standalone Akita argument under [`STANDALONE_SESSION`];
    /// `_transcript` is untouched.
    fn verify_batch<T>(
        setup: &Self::VerifierSetup,
        statement: &Self::Statement,
        proof: &Self::Proof,
        _transcript: &mut T,
    ) -> Result<(), OpeningsError>
    where
        T: FieldTranscript<Self::Field>,
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
        let openings: Vec<AkitaField> = statement
            .iter()
            .map(|claim| claim.evaluation.value)
            .collect();
        let group = PolynomialGroupClaims::new(backend_point, openings, &commitment.group)
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
                        STANDALONE_SESSION,
                        batch_statement,
                        BasisMode::Lagrange,
                    )
                })
                .map_err(|_| OpeningsError::VerificationFailed)
            }
            AkitaBackendFlavor::OneHot => verify_one_hot_statement(setup, proof, batch_statement),
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
    use jolt_transcript::{prover_transcript, Blake2b512};

    use crate::adapters::AkitaVerifierScheduleArtifacts;
    use crate::configs::AkitaChunkProfile;
    use crate::AkitaScheduleArtifacts;
    use akita_params::{ChunkedWitnessCfg, PolynomialGroupLayout, ScheduleLookupKey};
    use akita_types::{Commitment, RingVec};
    use std::sync::OnceLock;

    /// A committed group under an admitted dense profile, with zero rows. The
    /// checks under test read only Jolt's metadata.
    fn placeholder_group() -> AkitaBackendCommitment {
        static GROUP: OnceLock<AkitaBackendCommitment> = OnceLock::new();
        GROUP.get_or_init(build_placeholder_group).clone()
    }

    fn build_placeholder_group() -> AkitaBackendCommitment {
        let artifacts = AkitaScheduleArtifacts::from_directory(
            AkitaScheduleArtifacts::packaged_directory(),
            AkitaChunkProfile::new(8, 2),
        )
        .unwrap();
        let schedules = TrustedScheduleCatalog::<AkitaConfig>::new(
            artifacts.dense_catalog().unwrap(),
            ChunkedWitnessCfg {
                num_chunks: 8,
                num_activated_levels: 2,
            },
        )
        .unwrap();
        let profile = schedules
            .resolve_key(&ScheduleLookupKey::single(PolynomialGroupLayout::new(
                14, 1,
            )))
            .unwrap()
            .profiles()
            .final_group;
        let coefficients = profile
            .outer_compression_plan()
            .unwrap()
            .terminal_coefficients();
        AkitaBackendCommitment::new(
            profile,
            Commitment::new(RingVec::from_coeffs(vec![AkitaField::zero(); coefficients])),
        )
        .unwrap()
    }

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
            group: placeholder_group(),
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
                profile: AkitaChunkProfile::default(),
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
        let mut transcript = prover_transcript::<Blake2b512>(b"invalid-transported-setup", b"");
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
                dense_profile: AkitaChunkProfile::new(8, 2),
                profile: AkitaChunkProfile::default(),
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
