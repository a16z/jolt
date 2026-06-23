use akita_pcs::{AkitaTranscript, CommitmentProver, CpuBackend};
use akita_prover::{AkitaPolyOps, CommittedPolynomials, ProverClaims};
use akita_types::{
    BasisMode, CommitmentVerifier, CommittedOpenings, SetupContributionMode, VerifierClaims,
};
use jolt_openings::{
    BatchOpeningResult, BatchOpeningScheme, BatchOpeningStatement, OpeningsError, PhysicalView,
};
use jolt_transcript::Transcript;

use crate::{
    native::{akita_error, dense_polynomials, deserialize_akita, invalid_batch, serialize_akita},
    scheme::AkitaScheme,
    transcript::{
        bind_batch_statement, bind_jolt_transcript_bridge, bind_proof_bytes,
        bind_verifier_setup_key,
    },
    types::{
        AkitaBatchProof, AkitaCommitment, AkitaField, AkitaProverHint, AkitaProverSetup,
        AkitaVerifierSetup, NativeCommitment, NativeHint, NativeProof, NativeProofShape,
        NativeScheme, NativeVerifier, AKITA_D,
    },
};

impl BatchOpeningScheme for AkitaScheme {
    fn prove_batch<T, OpeningId, RelationId>(
        setup: &Self::ProverSetup,
        transcript: &mut T,
        statement: &BatchOpeningStatement<Self::Field, Self::Output, OpeningId, RelationId>,
        polynomials: &[Self::Polynomial],
        hints: Vec<Self::OpeningHint>,
    ) -> Result<Self::Proof, OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>,
    {
        let dense = dense_polynomials(polynomials)?;
        let dense_refs = dense.iter().collect::<Vec<_>>();
        prove_batch_with_native_polynomials(
            setup,
            transcript,
            statement,
            dense_refs.as_slice(),
            hints,
        )
    }

    fn verify_batch<T, OpeningId, RelationId>(
        setup: &Self::VerifierSetup,
        transcript: &mut T,
        statement: &BatchOpeningStatement<Self::Field, Self::Output, OpeningId, RelationId>,
        proof: &Self::Proof,
    ) -> Result<BatchOpeningResult<Self::Field, Self::Output>, OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>,
    {
        let normalized = normalize_clear_batch(statement)?;
        if proof.commitment != normalized.commitment {
            return Err(OpeningsError::VerificationFailed);
        }
        validate_verifier_setup(setup, &normalized.commitment)?;
        bind_verifier_setup_key(setup, transcript);
        bind_batch_statement(
            statement,
            &normalized.commitment,
            &normalized.coefficients,
            normalized.reduced_opening,
            transcript,
        );
        let mut akita_transcript = AkitaTranscript::<AkitaField>::new(b"jolt-akita/batch");
        let statement_bridge = bind_jolt_transcript_bridge(transcript, &mut akita_transcript);
        if proof.statement_bridge != statement_bridge {
            return Err(OpeningsError::VerificationFailed);
        }
        bind_proof_bytes(proof, transcript);

        let native_verifier = deserialize_akita::<NativeVerifier>(&setup.native, &())?;
        let native_commitment =
            deserialize_akita::<NativeCommitment>(&proof.commitment.native, &())?;
        let proof_shape = deserialize_akita::<NativeProofShape>(&proof.proof_shape, &())?;
        let native_proof = deserialize_akita::<NativeProof>(&proof.proof, &proof_shape)?;
        let openings = statement
            .claims
            .iter()
            .map(|claim| claim.claim)
            .collect::<Vec<_>>();
        let claims: VerifierClaims<'_, AkitaField, _> = (
            statement.pcs_point.as_slice(),
            vec![CommittedOpenings {
                openings: openings.as_slice(),
                commitment: &native_commitment,
            }],
        );
        NativeScheme::batched_verify(
            &native_proof,
            &native_verifier,
            &mut akita_transcript,
            claims,
            BasisMode::Lagrange,
            SetupContributionMode::Direct,
        )
        .map_err(|_| OpeningsError::VerificationFailed)?;

        Ok(BatchOpeningResult {
            coefficients: normalized.coefficients,
            joint_commitment: normalized.commitment,
            reduced_opening: normalized.reduced_opening,
        })
    }
}

pub(crate) fn prove_batch_with_native_polynomials<T, P, OpeningId, RelationId>(
    setup: &AkitaProverSetup,
    transcript: &mut T,
    statement: &BatchOpeningStatement<AkitaField, AkitaCommitment, OpeningId, RelationId>,
    polynomials: &[P],
    hints: Vec<AkitaProverHint>,
) -> Result<AkitaBatchProof, OpeningsError>
where
    T: Transcript<Challenge = AkitaField>,
    P: AkitaPolyOps<AkitaField, AKITA_D>,
{
    let normalized = normalize_clear_batch(statement)?;
    validate_native_prover_inputs(setup, &normalized.commitment, polynomials, &hints)?;
    bind_verifier_setup_key(&setup.verifier, transcript);
    bind_batch_statement(
        statement,
        &normalized.commitment,
        &normalized.coefficients,
        normalized.reduced_opening,
        transcript,
    );
    let mut akita_transcript = AkitaTranscript::<AkitaField>::new(b"jolt-akita/batch");
    let statement_bridge = bind_jolt_transcript_bridge(transcript, &mut akita_transcript);

    let native_commitment =
        deserialize_akita::<NativeCommitment>(&normalized.commitment.native, &())?;
    let native_hint = hints
        .into_iter()
        .next()
        .and_then(|hint| hint.native)
        .ok_or_else(|| invalid_batch("Akita prover hint is missing native opening data"))?;
    let claims: ProverClaims<'_, AkitaField, P, _, NativeHint> = (
        statement.pcs_point.as_slice(),
        vec![CommittedPolynomials {
            polynomials,
            commitment: &native_commitment,
            hint: native_hint,
        }],
    );

    let native_proof = NativeScheme::batched_prove(
        &setup.native,
        &CpuBackend,
        &setup.prepared,
        claims,
        &mut akita_transcript,
        BasisMode::Lagrange,
        SetupContributionMode::Direct,
    )
    .map_err(akita_error)?;
    let proof_shape = native_proof.shape();
    let proof = AkitaBatchProof {
        commitment: normalized.commitment,
        statement_bridge,
        proof_shape: serialize_akita(&proof_shape)?,
        proof: serialize_akita(&native_proof)?,
    };
    bind_proof_bytes(&proof, transcript);
    Ok(proof)
}

struct NormalizedBatch {
    commitment: AkitaCommitment,
    coefficients: Vec<AkitaField>,
    reduced_opening: AkitaField,
}

fn normalize_clear_batch<OpeningId, RelationId>(
    statement: &BatchOpeningStatement<AkitaField, AkitaCommitment, OpeningId, RelationId>,
) -> Result<NormalizedBatch, OpeningsError> {
    if statement.claims.is_empty() {
        return Err(invalid_batch(
            "Akita batch opening requires at least one claim",
        ));
    }
    let commitment = statement.claims[0].commitment.clone();
    if statement.pcs_point.len() != commitment.num_vars {
        return Err(invalid_batch(format!(
            "Akita opening point has {} variables but commitment has {}",
            statement.pcs_point.len(),
            commitment.num_vars
        )));
    }
    if statement.layout_digest != commitment.layout_digest {
        return Err(invalid_batch(
            "Akita direct batch statement layout digest does not match commitment layout digest",
        ));
    }
    if statement.logical_point != statement.pcs_point {
        return Err(invalid_batch(
            "Akita direct batch requires logical point and PCS point to match",
        ));
    }
    let mut coefficients = Vec::with_capacity(statement.claims.len());
    let mut reduced_opening = AkitaField::zero();
    for claim in &statement.claims {
        if claim.commitment != commitment {
            return Err(invalid_batch(
                "Akita batch statement must use exactly one commitment group",
            ));
        }
        if !matches!(claim.view, PhysicalView::Direct) {
            return Err(invalid_batch(
                "Akita native adapter expects direct physical views; lower packed views before calling it",
            ));
        }
        coefficients.push(claim.scale);
        reduced_opening += claim.scale * claim.claim;
    }
    if commitment.poly_count != statement.claims.len() {
        return Err(invalid_batch(format!(
            "Akita commitment covers {} polynomials but statement has {} claims",
            commitment.poly_count,
            statement.claims.len()
        )));
    }
    Ok(NormalizedBatch {
        commitment,
        coefficients,
        reduced_opening,
    })
}

fn validate_native_prover_inputs<P>(
    setup: &AkitaProverSetup,
    commitment: &AkitaCommitment,
    polynomials: &[P],
    hints: &[AkitaProverHint],
) -> Result<(), OpeningsError>
where
    P: AkitaPolyOps<AkitaField, AKITA_D>,
{
    validate_setup_shape(
        setup.max_num_vars,
        setup.max_num_polys_per_commitment_group,
        setup.default_layout_digest,
        commitment,
    )?;
    if polynomials.len() != commitment.poly_count {
        return Err(invalid_batch(format!(
            "Akita prover received {} polynomials for {} commitment slots",
            polynomials.len(),
            commitment.poly_count
        )));
    }
    if hints.len() != 1 {
        return Err(invalid_batch(format!(
            "Akita prover expects one grouped commitment hint, got {}",
            hints.len()
        )));
    }
    if !hints[0].matches_commitment(commitment) {
        return Err(invalid_batch(
            "Akita prover hint does not match the statement commitment",
        ));
    }
    for polynomial in polynomials {
        if polynomial.num_vars() != commitment.num_vars {
            return Err(invalid_batch(format!(
                "Akita polynomial has {} variables but commitment has {}",
                polynomial.num_vars(),
                commitment.num_vars
            )));
        }
    }
    Ok(())
}

fn validate_verifier_setup(
    setup: &AkitaVerifierSetup,
    commitment: &AkitaCommitment,
) -> Result<(), OpeningsError> {
    validate_setup_shape(
        setup.max_num_vars,
        setup.max_num_polys_per_commitment_group,
        setup.default_layout_digest,
        commitment,
    )
}

fn validate_setup_shape(
    max_num_vars: usize,
    max_num_polys_per_commitment_group: usize,
    _default_layout_digest: [u8; 32],
    commitment: &AkitaCommitment,
) -> Result<(), OpeningsError> {
    if commitment.num_vars > max_num_vars {
        return Err(OpeningsError::PolynomialTooLarge {
            poly_size: commitment.num_vars,
            setup_max: max_num_vars,
        });
    }
    if commitment.poly_count > max_num_polys_per_commitment_group {
        return Err(invalid_batch(format!(
            "Akita commitment covers {} polynomials but setup supports {}",
            commitment.poly_count, max_num_polys_per_commitment_group
        )));
    }
    Ok(())
}
