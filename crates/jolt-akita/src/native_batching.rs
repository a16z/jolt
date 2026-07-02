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

use akita_pcs::{AkitaTranscript, CommitmentProver, ProverCommitmentGroup, ProverOpeningBatch};
use akita_types::{
    BasisMode, CommitmentGroup, CommitmentVerifier, PointVariableSelection, SetupContributionMode,
    VerifierOpeningBatch,
};
use jolt_openings::{BatchOpeningScheme, OpeningsError, VerifierOpeningClaim};
use jolt_poly::MultilinearPoly;
use jolt_transcript::Transcript;
use tracing::info_span;

use crate::adapters::{
    akita_error, append_batch_statement, append_verifier_setup, backend_stack,
    bridge_jolt_statement_challenge, deserialize_akita, invalid_batch, prove_failed, reverse_point,
    serialize_akita, AkitaBackendCommitment, AkitaBackendDensePoly, AkitaBackendFlavor,
    AkitaBackendHint, AkitaBackendOneHotPoly, AkitaBackendProof, AkitaBackendProofShape,
    AkitaBackendScheme, AkitaBackendSparsePoly, AkitaBatchProof, AkitaCommitment, AkitaField,
    AkitaHintPolynomials, AkitaOneHotBackendScheme, AkitaProverHint, AkitaProverSetup,
    AkitaVerifierSetup, AKITA_D,
};

/// Marker adapter selecting Akita's native batched opening as the Jolt batch
/// opening protocol.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AkitaNativeBatching;

pub type AkitaNativeBatchStatement = Vec<VerifierOpeningClaim<AkitaField, AkitaCommitment>>;

pub type AkitaNativeBatchWitness<'a> = (
    Vec<&'a (dyn MultilinearPoly<AkitaField> + 'a)>,
    AkitaProverHint,
);

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

/// Assembles the single-group opening batch handed to Akita's native batched
/// prover.
fn single_group_batch<'a, P>(
    point: &'a [AkitaField],
    polynomials: &'a [&'a P],
    backend_commitment: AkitaBackendCommitment,
    backend_hint: AkitaBackendHint,
) -> Result<ProverOpeningBatch<'a, AkitaField, P, AkitaField, AKITA_D>, OpeningsError> {
    Ok(ProverOpeningBatch {
        point: point.into(),
        groups: vec![ProverCommitmentGroup {
            point_vars: PointVariableSelection::prefix(point.len(), point.len())
                .map_err(akita_error)?,
            polynomials,
            commitment: (backend_commitment, backend_hint),
        }],
    })
}

fn prove_dense(
    setup: &AkitaProverSetup,
    point: &[AkitaField],
    polynomials: &[&AkitaBackendDensePoly],
    backend_commitment: AkitaBackendCommitment,
    backend_hint: AkitaBackendHint,
    akita_transcript: &mut AkitaTranscript<AkitaField>,
) -> Result<AkitaBackendProof, OpeningsError> {
    let claims = single_group_batch(point, polynomials, backend_commitment, backend_hint)?;
    let stack = backend_stack(&setup.backend_prover_setup, &setup.prepared_backend_setup)?;
    let _span = info_span!("AkitaNativeBatching::backend_batched_prove").entered();
    AkitaBackendScheme::batched_prove(
        &setup.backend_prover_setup,
        claims,
        &stack,
        akita_transcript,
        BasisMode::Lagrange,
        SetupContributionMode::Direct,
    )
    .map_err(prove_failed)
}

/// The one-hot backend consumes the point in reversed variable order and uses
/// the dedicated one-hot setup pair.
fn prove_one_hot(
    setup: &AkitaProverSetup,
    point: &[AkitaField],
    polynomials: &[&AkitaBackendOneHotPoly],
    backend_commitment: AkitaBackendCommitment,
    backend_hint: AkitaBackendHint,
    akita_transcript: &mut AkitaTranscript<AkitaField>,
) -> Result<AkitaBackendProof, OpeningsError> {
    let (backend_prover_setup, prepared_backend_setup) = setup.one_hot_backend()?;
    let backend_point = reverse_point(point);
    let claims = single_group_batch(
        &backend_point,
        polynomials,
        backend_commitment,
        backend_hint,
    )?;
    let stack = backend_stack(backend_prover_setup, prepared_backend_setup)?;
    let _span = info_span!("AkitaNativeBatching::backend_batched_prove").entered();
    AkitaOneHotBackendScheme::batched_prove(
        backend_prover_setup,
        claims,
        &stack,
        akita_transcript,
        BasisMode::Lagrange,
        SetupContributionMode::Direct,
    )
    .map_err(prove_failed)
}

fn prove_sparse_unit(
    setup: &AkitaProverSetup,
    point: &[AkitaField],
    polynomials: &[&AkitaBackendSparsePoly],
    backend_commitment: AkitaBackendCommitment,
    backend_hint: AkitaBackendHint,
    akita_transcript: &mut AkitaTranscript<AkitaField>,
) -> Result<AkitaBackendProof, OpeningsError> {
    let claims = single_group_batch(point, polynomials, backend_commitment, backend_hint)?;
    let stack = backend_stack(&setup.backend_prover_setup, &setup.prepared_backend_setup)?;
    let _span = info_span!("AkitaNativeBatching::backend_batched_prove").entered();
    AkitaBackendScheme::batched_prove(
        &setup.backend_prover_setup,
        claims,
        &stack,
        akita_transcript,
        BasisMode::Lagrange,
        SetupContributionMode::Direct,
    )
    .map_err(prove_failed)
}

impl BatchOpeningScheme for AkitaNativeBatching {
    type Field = AkitaField;
    type ProverSetup = AkitaProverSetup;
    type VerifierSetup = AkitaVerifierSetup;
    type Statement = AkitaNativeBatchStatement;
    type BatchingWitness<'a>
        = AkitaNativeBatchWitness<'a>
    where
        Self: 'a;
    type Proof = AkitaBatchProof;

    fn prove_batch<'a, T>(
        setup: &Self::ProverSetup,
        statement: Self::Statement,
        witness: Self::BatchingWitness<'a>,
        transcript: &mut T,
    ) -> Result<Self::Proof, OpeningsError>
    where
        Self: 'a,
        T: Transcript<Challenge = Self::Field>,
    {
        let (polynomials, hint) = witness;
        let ValidatedStatement { commitment, point } = validate_statement(
            &statement,
            setup.max_num_vars(),
            setup.max_num_polys_per_commitment_group(),
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

        let backend_proof = match &hint.polynomials {
            AkitaHintPolynomials::Dense(dense) => {
                let refs = dense.iter().collect::<Vec<_>>();
                prove_dense(
                    setup,
                    point,
                    &refs,
                    backend_commitment,
                    backend_hint,
                    &mut akita_transcript,
                )?
            }
            AkitaHintPolynomials::OneHot(one_hot) => {
                let refs = one_hot.iter().collect::<Vec<_>>();
                prove_one_hot(
                    setup,
                    point,
                    &refs,
                    backend_commitment,
                    backend_hint,
                    &mut akita_transcript,
                )?
            }
            AkitaHintPolynomials::SparseUnit(sparse) => {
                let refs = sparse.iter().collect::<Vec<_>>();
                prove_sparse_unit(
                    setup,
                    point,
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
                commitment: commitment.clone(),
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
        statement: Self::Statement,
        proof: &Self::Proof,
        transcript: &mut T,
    ) -> Result<(), OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>,
    {
        let ValidatedStatement { commitment, point } = validate_statement(
            &statement,
            setup.max_num_vars,
            setup.max_num_polys_per_commitment_group,
        )?;
        if proof.commitment != *commitment {
            return Err(OpeningsError::VerificationFailed);
        }

        let (mut akita_transcript, statement_bridge) =
            bind_statement_transcripts(transcript, setup, &statement, commitment, point);
        if proof.statement_bridge != statement_bridge {
            return Err(OpeningsError::VerificationFailed);
        }
        transcript.append(proof);

        let backend_verifier = setup.backend_verifier(commitment.backend_flavor)?;
        let backend_commitment = deserialize_akita::<AkitaBackendCommitment>(
            &proof.commitment.serialized_backend_bytes,
            &(),
        )?;
        let proof_shape =
            deserialize_akita::<AkitaBackendProofShape>(&proof.serialized_akita_proof_shape, &())?;
        let backend_proof =
            deserialize_akita::<AkitaBackendProof>(&proof.serialized_akita_proof, &proof_shape)?;
        let backend_point = match commitment.backend_flavor {
            AkitaBackendFlavor::Full => point.to_vec(),
            AkitaBackendFlavor::OneHot => reverse_point(point),
        };
        let openings = statement
            .iter()
            .map(|claim| claim.evaluation.value)
            .collect();
        let claims = VerifierOpeningBatch::from_groups(
            backend_point,
            vec![CommitmentGroup {
                claims: openings,
                commitment: &backend_commitment,
            }],
        )
        .map_err(akita_error)?;
        match commitment.backend_flavor {
            AkitaBackendFlavor::Full => AkitaBackendScheme::batched_verify(
                &backend_proof,
                backend_verifier,
                &mut akita_transcript,
                claims,
                BasisMode::Lagrange,
                SetupContributionMode::Direct,
            ),
            AkitaBackendFlavor::OneHot => AkitaOneHotBackendScheme::batched_verify(
                &backend_proof,
                backend_verifier,
                &mut akita_transcript,
                claims,
                BasisMode::Lagrange,
                SetupContributionMode::Direct,
            ),
        }
        .map_err(|_| OpeningsError::VerificationFailed)
    }
}
