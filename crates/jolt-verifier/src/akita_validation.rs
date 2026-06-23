use crate::{
    akita::{
        AkitaPackingBatchProof, AkitaPackingVerifierSetup, AkitaPackingWitnessArtifacts,
        AkitaVerifierPreprocessing,
    },
    config::{validate_protocol_config, JoltProtocolConfig, PcsFamily},
    proof::CommitmentPayload,
    stages::stage8::validate_lattice_packed_witness_layout_config,
    VerifierError,
};
use jolt_akita::{AkitaBatchProof, AkitaCommitment, AkitaField, AKITA_FIELD_MODULUS};
use jolt_field::FixedByteSize;
use jolt_openings::PackingWitnessLayout;

pub(crate) fn validate_akita_artifacts_for_proof(
    setup: &AkitaPackingVerifierSetup,
    proof_protocol: &JoltProtocolConfig,
    proof_commitments: &CommitmentPayload<AkitaCommitment>,
    artifacts: &AkitaPackingWitnessArtifacts,
) -> Result<(), VerifierError> {
    validate_akita_verifier_setup_layout(setup, &artifacts.layout)?;
    validate_lattice_packed_witness_layout_config(&artifacts.protocol, &artifacts.layout)?;
    if proof_protocol != &artifacts.protocol {
        return Err(VerifierError::ProtocolConfigMismatch {
            expected: artifacts.protocol,
            got: *proof_protocol,
        });
    }
    let artifact_payload =
        artifacts
            .payload()
            .ok_or_else(|| VerifierError::InvalidProtocolConfig {
                reason: "lattice proof assembly requires lattice packing witness artifacts"
                    .to_string(),
            })?;
    let proof_payload =
        proof_commitments
            .as_lattice()
            .ok_or(VerifierError::CommitmentPayloadFamilyMismatch {
                expected: PcsFamily::Lattice,
                got: proof_commitments.family(),
            })?;
    if proof_payload != artifact_payload {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "lattice proof commitments do not match packed witness artifacts".to_string(),
        });
    }
    Ok(())
}

pub(crate) fn validate_akita_verifier_setup_config(
    setup: &AkitaPackingVerifierSetup,
    config: &JoltProtocolConfig,
) -> Result<(), VerifierError> {
    if validate_protocol_config(config)? != PcsFamily::Lattice {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "Akita verifier setup requires lattice PCS mode".to_string(),
        });
    }

    let packed_witness = config.lattice.packed_witness;
    let expected_digest =
        packed_witness
            .layout_digest
            .ok_or_else(|| VerifierError::InvalidProtocolConfig {
                reason: "Akita verifier setup requires a packed witness layout digest".to_string(),
            })?;
    let expected_dimension =
        packed_witness
            .d_pack
            .ok_or_else(|| VerifierError::InvalidProtocolConfig {
                reason: "Akita verifier setup requires D_pack".to_string(),
            })?;
    validate_akita_verifier_setup_shape(setup, expected_digest, expected_dimension)?;

    if setup.layout.digest != expected_digest {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "Akita verifier setup layout digest does not match protocol config".to_string(),
        });
    }
    if setup.layout.dimension != expected_dimension {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "Akita verifier setup layout dimension does not match protocol D_pack"
                .to_string(),
        });
    }

    Ok(())
}

pub(crate) fn validate_akita_proof_payload_shape(
    setup: &AkitaPackingVerifierSetup,
    proof_commitments: &CommitmentPayload<AkitaCommitment>,
) -> Result<(), VerifierError> {
    let payload =
        proof_commitments
            .as_lattice()
            .ok_or(VerifierError::CommitmentPayloadFamilyMismatch {
                expected: PcsFamily::Lattice,
                got: proof_commitments.family(),
            })?;
    validate_akita_verifier_setup_shape(setup, payload.layout_digest, payload.d_pack)?;
    if payload.packed_witness.layout_digest != payload.layout_digest {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "lattice packing witness commitment layout digest does not match proof payload"
                .to_string(),
        });
    }
    if payload.packed_witness.num_vars != payload.d_pack {
        return Err(VerifierError::InvalidProtocolConfig {
            reason:
                "lattice packing witness commitment dimension does not match proof payload D_pack"
                    .to_string(),
        });
    }
    if payload.packed_witness.poly_count != 1 {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "lattice packing witness commitment must contain exactly one polynomial"
                .to_string(),
        });
    }
    validate_akita_commitment_bytes(&payload.packed_witness)?;
    Ok(())
}

pub(crate) fn validate_akita_opening_proof_payload_shape(
    proof_commitments: &CommitmentPayload<AkitaCommitment>,
    opening_proof: &AkitaPackingBatchProof,
) -> Result<(), VerifierError> {
    let payload =
        proof_commitments
            .as_lattice()
            .ok_or(VerifierError::CommitmentPayloadFamilyMismatch {
                expected: PcsFamily::Lattice,
                got: proof_commitments.family(),
            })?;
    if opening_proof.native.commitment != payload.packed_witness {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "Akita opening proof commitment does not match packed witness payload"
                .to_string(),
        });
    }
    validate_akita_commitment_bytes(&opening_proof.native.commitment)?;
    if opening_proof.native.statement_bridge.is_empty() {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "Akita opening proof is missing statement bridge bytes".to_string(),
        });
    }
    if opening_proof.native.proof_shape.is_empty() {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "Akita opening proof is missing native proof shape bytes".to_string(),
        });
    }
    if opening_proof.native.proof.is_empty() {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "Akita opening proof is missing native proof bytes".to_string(),
        });
    }
    if let Some(reduction) = &opening_proof.reduction {
        validate_akita_field_bytes(
            "lattice packing reduction opening eval",
            &reduction.opening_eval,
        )?;
        for round in &reduction.rounds {
            for eval in round {
                validate_akita_field_bytes("lattice packing reduction round eval", eval)?;
            }
        }
    }
    Ok(())
}

pub(crate) fn validate_akita_packing_opening_proof_payload_shape(
    proof_commitments: &CommitmentPayload<AkitaCommitment>,
    opening_proof: &AkitaPackingBatchProof,
    field: &'static str,
) -> Result<(), VerifierError> {
    validate_akita_opening_proof_payload_shape(proof_commitments, opening_proof)?;
    if opening_proof.reduction.is_none() {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: format!("{field} must include a packed reduction"),
        });
    }
    Ok(())
}

pub(crate) fn validate_akita_precommitted_opening_proof_payload_shapes(
    proof_commitments: &CommitmentPayload<AkitaCommitment>,
    opening_proofs: &[AkitaPackingBatchProof],
) -> Result<(), VerifierError> {
    for opening_proof in opening_proofs {
        validate_akita_precommitted_opening_proof_payload_shape(proof_commitments, opening_proof)?;
    }
    Ok(())
}

fn validate_akita_precommitted_opening_proof_payload_shape(
    proof_commitments: &CommitmentPayload<AkitaCommitment>,
    opening_proof: &AkitaPackingBatchProof,
) -> Result<(), VerifierError> {
    let payload =
        proof_commitments
            .as_lattice()
            .ok_or(VerifierError::CommitmentPayloadFamilyMismatch {
                expected: PcsFamily::Lattice,
                got: proof_commitments.family(),
            })?;
    if opening_proof.native.commitment == payload.packed_witness {
        return Err(VerifierError::InvalidProtocolConfig {
            reason:
                "Akita precommitted opening proof must target a separate precommitted commitment"
                    .to_string(),
        });
    }
    if opening_proof.reduction.is_some() {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "Akita precommitted opening proof must not include a packed reduction"
                .to_string(),
        });
    }
    validate_akita_native_opening_proof_payload_shape(&opening_proof.native)
}

fn validate_akita_native_opening_proof_payload_shape(
    opening_proof: &AkitaBatchProof,
) -> Result<(), VerifierError> {
    validate_akita_commitment_bytes(&opening_proof.commitment)?;
    if opening_proof.statement_bridge.is_empty() {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "Akita opening proof is missing statement bridge bytes".to_string(),
        });
    }
    if opening_proof.proof_shape.is_empty() {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "Akita opening proof is missing native proof shape bytes".to_string(),
        });
    }
    if opening_proof.proof.is_empty() {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "Akita opening proof is missing native proof bytes".to_string(),
        });
    }
    Ok(())
}

fn validate_akita_commitment_bytes(commitment: &AkitaCommitment) -> Result<(), VerifierError> {
    if commitment.native.is_empty() {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "Akita commitment is missing native commitment bytes".to_string(),
        });
    }
    Ok(())
}

fn validate_akita_field_bytes(label: &'static str, bytes: &[u8]) -> Result<(), VerifierError> {
    if bytes.len() != AkitaField::NUM_BYTES {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: format!(
                "{label} has {} bytes but expected {}",
                bytes.len(),
                AkitaField::NUM_BYTES
            ),
        });
    }
    let value = u128::from_le_bytes(bytes.try_into().map_err(|_| {
        VerifierError::InvalidProtocolConfig {
            reason: format!("{label} must be exactly {} bytes", AkitaField::NUM_BYTES),
        }
    })?);
    if value >= AKITA_FIELD_MODULUS {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: format!("{label} is not a canonical Akita field encoding"),
        });
    }
    Ok(())
}

pub(crate) fn validate_akita_advice_commitment_aliases(
    proof_commitments: &CommitmentPayload<AkitaCommitment>,
    untrusted_advice_commitment: Option<&AkitaCommitment>,
    trusted_advice_commitment: Option<&AkitaCommitment>,
) -> Result<(), VerifierError> {
    let payload =
        proof_commitments
            .as_lattice()
            .ok_or(VerifierError::CommitmentPayloadFamilyMismatch {
                expected: PcsFamily::Lattice,
                got: proof_commitments.family(),
            })?;
    if untrusted_advice_commitment.is_some_and(|commitment| commitment != &payload.packed_witness) {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "Akita untrusted advice commitment must alias the packed witness commitment"
                .to_string(),
        });
    }
    if let Some(commitment) = trusted_advice_commitment {
        validate_akita_precommitted_commitment_is_separate(
            &payload.packed_witness,
            commitment,
            "trusted advice",
        )?;
    }
    Ok(())
}

pub(crate) fn validate_akita_precommitted_commitment_aliases(
    preprocessing: &AkitaVerifierPreprocessing,
    proof_commitments: &CommitmentPayload<AkitaCommitment>,
    trusted_advice_commitment: Option<&AkitaCommitment>,
) -> Result<(), VerifierError> {
    let payload =
        proof_commitments
            .as_lattice()
            .ok_or(VerifierError::CommitmentPayloadFamilyMismatch {
                expected: PcsFamily::Lattice,
                got: proof_commitments.family(),
            })?;
    if let Some(commitment) = trusted_advice_commitment {
        validate_akita_precommitted_commitment_is_separate(
            &payload.packed_witness,
            commitment,
            "trusted advice",
        )?;
    }
    if let Some(committed) = preprocessing.program.committed() {
        for commitment in &committed.bytecode_chunk_commitments {
            validate_akita_precommitted_commitment_is_separate(
                &payload.packed_witness,
                commitment,
                "bytecode chunk",
            )?;
        }
        validate_akita_precommitted_commitment_is_separate(
            &payload.packed_witness,
            &committed.program_image_commitment,
            "program image",
        )?;
    }
    Ok(())
}

pub(crate) fn validate_akita_precommitted_commitment_is_separate(
    packed_witness: &AkitaCommitment,
    precommitted: &AkitaCommitment,
    label: &'static str,
) -> Result<(), VerifierError> {
    if precommitted == packed_witness {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: format!("Akita {label} commitment must be separate from packed witness"),
        });
    }
    validate_akita_commitment_bytes(precommitted)
}

pub(crate) fn validate_akita_verifier_setup_layout(
    setup: &AkitaPackingVerifierSetup,
    layout: &PackingWitnessLayout,
) -> Result<(), VerifierError> {
    validate_akita_verifier_setup_shape(setup, layout.digest, layout.dimension)?;
    if setup.layout != *layout {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "Akita verifier setup layout does not match packed witness artifact layout"
                .to_string(),
        });
    }

    Ok(())
}

fn validate_akita_verifier_setup_shape(
    setup: &AkitaPackingVerifierSetup,
    expected_digest: [u8; 32],
    expected_dimension: usize,
) -> Result<(), VerifierError> {
    if setup.pcs.default_layout_digest != expected_digest {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "Akita verifier setup layout digest does not match packed witness layout"
                .to_string(),
        });
    }
    if setup.pcs.max_num_vars != expected_dimension {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "Akita verifier setup max_num_vars does not match packed witness dimension"
                .to_string(),
        });
    }
    if setup.pcs.max_num_polys_per_commitment_group == 0 {
        return Err(VerifierError::InvalidProtocolConfig {
            reason:
                "Akita verifier setup must support at least one polynomial per commitment group"
                    .to_string(),
        });
    }
    if setup.pcs.native.is_empty() {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "Akita verifier setup is missing native setup bytes".to_string(),
        });
    }

    Ok(())
}
