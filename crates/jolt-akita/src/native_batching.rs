//! Adapts Akita's native same-point batched opening protocol to Jolt's
//! [`BatchOpeningScheme`] trait.
//!
//! Two kinds of batching meet at this seam:
//!
//! - **Jolt-side batching** happens upstream in the PIOP: the opening
//!   accumulator reduces the claims produced by the sumcheck stages (via RLC
//!   combination, claim reductions, or prefix packing) down to evaluation
//!   claims about committed polynomials at a common point.
//! - **Akita-native batching** is what this module delegates to: the Akita
//!   backend proves every polynomial of a commitment group at that common
//!   point in a single backend proof, using its own transcript and batching
//!   rules.
//!
//! This adapter performs no claim combination of its own — it validates the
//! statement shape, bridges Jolt's Fiat-Shamir transcript into Akita's, and
//! embeds the backend proof bytes wholesale.

use akita_pcs::AkitaTranscript;
use akita_prover::ProverOpeningData;
use akita_types::{
    BasisMode, OpeningClaims, PointVariableSelection, PolynomialGroupClaims, SetupContributionMode,
};
use jolt_openings::{BatchOpeningScheme, OpeningsError, VerifierOpeningClaim};
use jolt_poly::MultilinearPoly;
use jolt_transcript::Transcript;
use tracing::info_span;

use crate::adapters::{
    akita_error, append_batch_statement, append_verifier_setup, backend_stack,
    bridge_jolt_statement_challenge, invalid_batch, prove_failed, reverse_point, serialize_akita,
    AkitaBackendCommitment, AkitaBackendFlavor, AkitaBackendHint, AkitaBackendOneHotPoly,
    AkitaBackendProof, AkitaBackendScheme, AkitaBatchProof, AkitaCommitment, AkitaField,
    AkitaHintPolynomials, AkitaOneHotK16BackendScheme, AkitaOneHotK256BackendScheme,
    AkitaProverHint, AkitaProverSetup, AkitaVerifierSetup, AKITA_ONE_HOT_K16, AKITA_ONE_HOT_K256,
};

/// Marker adapter selecting Akita's native batched opening as the Jolt batch
/// opening protocol.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AkitaNativeBatching;

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
                "Akita dense commitment must not declare a one-hot chunk size",
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
        AkitaHintPolynomials::OneHot(_) | AkitaHintPolynomials::SparseUnit(_)
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
fn single_group_batch<'a, P>(
    point: &[AkitaField],
    evaluations: &[AkitaField],
    polynomials: &'a [&'a P],
    backend_commitment: AkitaBackendCommitment,
    backend_hint: AkitaBackendHint,
) -> Result<ProverOpeningData<'a, AkitaField, P, AkitaField>, OpeningsError> {
    let group = PolynomialGroupClaims::new(
        PointVariableSelection::prefix(point.len(), point.len()).map_err(akita_error)?,
        evaluations.to_vec(),
        backend_commitment,
    )
    .map_err(akita_error)?;
    let claims = OpeningClaims::from_groups(point.to_vec(), vec![group]).map_err(akita_error)?;
    ProverOpeningData::new(claims, vec![backend_hint], vec![polynomials]).map_err(akita_error)
}

/// Dense-flavor batched prove shared by the dense and sparse-unit paths —
/// they differ only in the backend polynomial type, whose opening-view trait
/// chain is too deep to name generically, hence a macro.
macro_rules! prove_dense_backend {
    ($setup:expr, $point:expr, $evaluations:expr, $polynomials:expr, $commitment:expr, $hint:expr, $transcript:expr) => {{
        let claims = single_group_batch($point, $evaluations, $polynomials, $commitment, $hint)?;
        let (backend_prover_setup, prepared_backend_setup) = $setup.dense_backend()?;
        let stack = backend_stack(backend_prover_setup, prepared_backend_setup)?;
        let _span = info_span!("AkitaNativeBatching::backend_batched_prove").entered();
        AkitaBackendScheme::batched_prove(
            backend_prover_setup,
            claims,
            &stack,
            $transcript,
            BasisMode::Lagrange,
            SetupContributionMode::Direct,
        )
        .map_err(prove_failed)?
    }};
}

/// The one-hot backend consumes the point in reversed variable order and uses
/// the dedicated one-hot setup pair.
fn prove_one_hot(
    setup: &AkitaProverSetup,
    point: &[AkitaField],
    evaluations: &[AkitaField],
    polynomials: &[&AkitaBackendOneHotPoly],
    backend_commitment: AkitaBackendCommitment,
    backend_hint: AkitaBackendHint,
    akita_transcript: &mut AkitaTranscript<AkitaField>,
) -> Result<AkitaBackendProof, OpeningsError> {
    let (backend_prover_setup, prepared_backend_setup) = setup.one_hot_backend()?;
    let backend_point = reverse_point(point);
    let claims = single_group_batch(
        &backend_point,
        evaluations,
        polynomials,
        backend_commitment,
        backend_hint,
    )?;
    let stack = backend_stack(backend_prover_setup, prepared_backend_setup)?;
    let _span = info_span!("AkitaNativeBatching::backend_batched_prove").entered();
    match setup.one_hot_k() {
        AKITA_ONE_HOT_K16 => AkitaOneHotK16BackendScheme::batched_prove(
            backend_prover_setup,
            claims,
            &stack,
            akita_transcript,
            BasisMode::Lagrange,
            SetupContributionMode::Direct,
        ),
        AKITA_ONE_HOT_K256 => AkitaOneHotK256BackendScheme::batched_prove(
            backend_prover_setup,
            claims,
            &stack,
            akita_transcript,
            BasisMode::Lagrange,
            SetupContributionMode::Direct,
        ),
        _ => unreachable!("one-hot K is validated during setup"),
    }
    .map_err(prove_failed)
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
        let backend_proof = match &hint.polynomials {
            AkitaHintPolynomials::Dense(dense) => {
                let refs = dense.iter().collect::<Vec<_>>();
                prove_dense_backend!(
                    setup,
                    point,
                    &evaluations,
                    &refs,
                    backend_commitment,
                    backend_hint,
                    &mut akita_transcript
                )
            }
            AkitaHintPolynomials::OneHot(one_hot) => {
                let refs = one_hot.iter().collect::<Vec<_>>();
                prove_one_hot(
                    setup,
                    point,
                    &evaluations,
                    &refs,
                    backend_commitment,
                    backend_hint,
                    &mut akita_transcript,
                )?
            }
            AkitaHintPolynomials::SparseUnit(sparse) => {
                let refs = sparse.iter().collect::<Vec<_>>();
                prove_dense_backend!(
                    setup,
                    point,
                    &evaluations,
                    &refs,
                    backend_commitment,
                    backend_hint,
                    &mut akita_transcript
                )
            }
        };

        let proof = {
            let _span = info_span!("AkitaNativeBatching::serialize_backend_proof").entered();
            let proof_shape = backend_proof.shape();
            AkitaBatchProof {
                statement_bridge,
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
        let (backend_commitment, backend_proof) =
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
        let group = PolynomialGroupClaims::new(
            PointVariableSelection::prefix(backend_point.len(), backend_point.len())
                .map_err(akita_error)?,
            openings,
            &backend_commitment,
        )
        .map_err(akita_error)?;
        let claims = OpeningClaims::from_groups(backend_point, vec![group]).map_err(akita_error)?;
        match commitment.backend_flavor {
            AkitaBackendFlavor::Dense => AkitaBackendScheme::batched_verify(
                &backend_proof,
                backend_verifier,
                &mut akita_transcript,
                claims,
                BasisMode::Lagrange,
                SetupContributionMode::Direct,
            ),
            AkitaBackendFlavor::OneHot => match setup.one_hot_k {
                AKITA_ONE_HOT_K16 => AkitaOneHotK16BackendScheme::batched_verify(
                    &backend_proof,
                    backend_verifier,
                    &mut akita_transcript,
                    claims,
                    BasisMode::Lagrange,
                    SetupContributionMode::Direct,
                ),
                AKITA_ONE_HOT_K256 => AkitaOneHotK256BackendScheme::batched_verify(
                    &backend_proof,
                    backend_verifier,
                    &mut akita_transcript,
                    claims,
                    BasisMode::Lagrange,
                    SetupContributionMode::Direct,
                ),
                _ => unreachable!("one-hot K is validated during setup"),
            },
        }
        .map_err(|_| OpeningsError::VerificationFailed)
    }
}
