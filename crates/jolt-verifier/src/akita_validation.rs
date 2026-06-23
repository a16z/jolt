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
            reason: "lattice packing verifier setup requires lattice PCS mode".to_string(),
        });
    }

    let packed_witness = config.lattice.packed_witness;
    let expected_digest =
        packed_witness
            .layout_digest
            .ok_or_else(|| VerifierError::InvalidProtocolConfig {
                reason: "lattice packing verifier setup requires a packed witness layout digest"
                    .to_string(),
            })?;
    let expected_dimension =
        packed_witness
            .d_pack
            .ok_or_else(|| VerifierError::InvalidProtocolConfig {
                reason: "lattice packing verifier setup requires D_pack".to_string(),
            })?;
    validate_akita_verifier_setup_shape(setup, expected_digest, expected_dimension)?;

    if setup.layout.digest != expected_digest {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "lattice packing verifier setup layout digest does not match protocol config"
                .to_string(),
        });
    }
    if setup.layout.dimension != expected_dimension {
        return Err(VerifierError::InvalidProtocolConfig {
            reason:
                "lattice packing verifier setup layout dimension does not match protocol D_pack"
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
            reason: "lattice packed opening proof commitment does not match packed witness payload"
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
                "lattice precommitted opening proof must target a separate precommitted commitment"
                    .to_string(),
        });
    }
    if opening_proof.reduction.is_some() {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "lattice precommitted opening proof must not include a packed reduction"
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
            reason: "lattice untrusted advice commitment must alias the packed witness commitment"
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
            reason: format!("lattice {label} commitment must be separate from packed witness"),
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
            reason: "lattice packing verifier setup layout does not match packed witness artifact layout"
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
            reason:
                "lattice packing verifier setup layout digest does not match packed witness layout"
                    .to_string(),
        });
    }
    if setup.pcs.max_num_vars != expected_dimension {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "lattice packing verifier setup max_num_vars does not match packed witness dimension"
                .to_string(),
        });
    }
    if setup.pcs.max_num_polys_per_commitment_group == 0 {
        return Err(VerifierError::InvalidProtocolConfig {
            reason:
                "lattice packing verifier setup must support at least one polynomial per commitment group"
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

#[cfg(test)]
mod tests {
    #![expect(
        clippy::expect_used,
        reason = "tests assert successful validation fixture construction"
    )]

    use super::*;
    use crate::{
        akita::{commit_akita_packing_witness, AkitaPackingProverSetup},
        akita_packing::AkitaPackingScheme,
        proof::LatticeCommitmentPayload,
        stages::stage8::lattice_protocol_config_for_packed_witness_layout,
    };
    use jolt_openings::{
        CommitmentScheme, PackingAlphabet, PackingCellAddress, PackingFactDomain, PackingFamilyId,
        PackingFamilySpec, PackingSetupParams, SparsePackingWitness,
    };

    fn tiny_layout() -> PackingWitnessLayout {
        PackingWitnessLayout::new([
            PackingFamilySpec::direct(
                PackingFamilyId::InstructionRa { index: 0 },
                PackingFactDomain::TraceRows { log_t: 0 },
                1,
                PackingAlphabet::Byte,
            ),
            PackingFamilySpec::direct(
                PackingFamilyId::UnsignedIncMsb,
                PackingFactDomain::TraceRows { log_t: 0 },
                1,
                PackingAlphabet::Bit,
            ),
        ])
        .expect("layout should build")
    }

    fn packed_cell(family: PackingFamilyId, symbol: usize) -> PackingCellAddress {
        PackingCellAddress {
            family,
            row: 0,
            limb: 0,
            symbol,
        }
    }

    fn akita_packing_setup(
        layout: &PackingWitnessLayout,
        max_num_polys_per_commitment_group: usize,
    ) -> (AkitaPackingProverSetup, AkitaPackingVerifierSetup) {
        AkitaPackingScheme::setup(PackingSetupParams {
            pcs: jolt_akita::AkitaSetupParams::new(
                layout.dimension,
                max_num_polys_per_commitment_group,
                layout.digest,
            ),
            layout: layout.clone(),
        })
    }

    fn empty_artifacts(
        layout: PackingWitnessLayout,
    ) -> (AkitaPackingVerifierSetup, AkitaPackingWitnessArtifacts) {
        let (prover_setup, verifier_setup) = akita_packing_setup(&layout, 1);
        let source = SparsePackingWitness::try_new(layout, Vec::new())
            .expect("empty sparse source should build");
        let artifacts = commit_akita_packing_witness(&prover_setup, &source)
            .expect("packed witness should commit");
        (verifier_setup, artifacts)
    }

    fn lattice_payload(
        artifacts: &AkitaPackingWitnessArtifacts,
    ) -> LatticeCommitmentPayload<AkitaCommitment> {
        artifacts
            .commitments
            .as_lattice()
            .expect("artifact should carry lattice payload")
            .clone()
    }

    #[test]
    fn akita_verifier_setup_binds_protocol_config() {
        let layout = tiny_layout();
        let (_, verifier_setup) = akita_packing_setup(&layout, 1);
        let config = lattice_protocol_config_for_packed_witness_layout(&layout);

        validate_akita_verifier_setup_config(&verifier_setup, &config)
            .expect("setup should match generated Akita protocol config");

        let mut wrong_digest = config;
        let mut digest = layout.digest;
        digest[0] ^= 1;
        wrong_digest.lattice.packed_witness.layout_digest = Some(digest);
        assert!(matches!(
            validate_akita_verifier_setup_config(&verifier_setup, &wrong_digest),
            Err(VerifierError::InvalidProtocolConfig { .. })
        ));

        let mut wrong_dimension = config;
        wrong_dimension.lattice.packed_witness.d_pack = Some(layout.dimension + 1);
        assert!(matches!(
            validate_akita_verifier_setup_config(&verifier_setup, &wrong_dimension),
            Err(VerifierError::InvalidProtocolConfig { .. })
        ));

        let mut wrong_setup_layout = verifier_setup.clone();
        wrong_setup_layout.layout.digest[0] ^= 1;
        assert!(matches!(
            validate_akita_verifier_setup_config(&wrong_setup_layout, &config),
            Err(VerifierError::InvalidProtocolConfig { .. })
        ));

        let mut missing_native = verifier_setup;
        missing_native.pcs.native.clear();
        assert!(matches!(
            validate_akita_verifier_setup_config(&missing_native, &config),
            Err(VerifierError::InvalidProtocolConfig { reason })
                if reason.contains("native setup bytes")
        ));
    }

    #[test]
    fn akita_verifier_setup_binds_artifact_layout() {
        let layout = tiny_layout();
        let (_, verifier_setup) = akita_packing_setup(&layout, 1);

        validate_akita_verifier_setup_layout(&verifier_setup, &layout)
            .expect("setup should match generated Akita packing layout");

        let other_layout = PackingWitnessLayout::new([PackingFamilySpec::direct(
            PackingFamilyId::InstructionRa { index: 1 },
            PackingFactDomain::TraceRows { log_t: 0 },
            1,
            PackingAlphabet::Byte,
        )])
        .expect("layout should build");
        assert!(matches!(
            validate_akita_verifier_setup_layout(&verifier_setup, &other_layout),
            Err(VerifierError::InvalidProtocolConfig { .. })
        ));

        let mut zero_group_setup = verifier_setup;
        zero_group_setup.pcs.max_num_polys_per_commitment_group = 0;
        assert!(matches!(
            validate_akita_verifier_setup_layout(&zero_group_setup, &layout),
            Err(VerifierError::InvalidProtocolConfig { .. })
        ));
    }

    #[test]
    fn akita_verifier_payload_shape_binds_inner_commitment_metadata() {
        let layout = tiny_layout();
        let (verifier_setup, artifacts) = empty_artifacts(layout.clone());
        validate_akita_proof_payload_shape(&verifier_setup, &artifacts.commitments)
            .expect("matching payload shape should pass");
        let payload = lattice_payload(&artifacts);

        let mut wrong_commitment_digest = payload.clone();
        wrong_commitment_digest.packed_witness.layout_digest = [9; 32];
        assert!(matches!(
            validate_akita_proof_payload_shape(
                &verifier_setup,
                &CommitmentPayload::Lattice(wrong_commitment_digest),
            ),
            Err(VerifierError::InvalidProtocolConfig { reason })
                if reason.contains("commitment layout digest")
        ));

        let mut wrong_commitment_dimension = payload.clone();
        wrong_commitment_dimension.packed_witness.num_vars = layout.dimension + 1;
        assert!(matches!(
            validate_akita_proof_payload_shape(
                &verifier_setup,
                &CommitmentPayload::Lattice(wrong_commitment_dimension),
            ),
            Err(VerifierError::InvalidProtocolConfig { reason })
                if reason.contains("commitment dimension")
        ));

        let mut wrong_poly_count = payload.clone();
        wrong_poly_count.packed_witness.poly_count = 2;
        assert!(matches!(
            validate_akita_proof_payload_shape(
                &verifier_setup,
                &CommitmentPayload::Lattice(wrong_poly_count),
            ),
            Err(VerifierError::InvalidProtocolConfig { reason })
                if reason.contains("exactly one polynomial")
        ));

        let mut missing_native_commitment = payload;
        missing_native_commitment.packed_witness.native.clear();
        assert!(matches!(
            validate_akita_proof_payload_shape(
                &verifier_setup,
                &CommitmentPayload::Lattice(missing_native_commitment),
            ),
            Err(VerifierError::InvalidProtocolConfig { reason })
                if reason.contains("native commitment bytes")
        ));
    }

    #[test]
    fn akita_untrusted_advice_aliases_packed_witness_but_trusted_must_be_separate() {
        let (_, artifacts) = empty_artifacts(tiny_layout());
        let payload = lattice_payload(&artifacts);
        let packed_witness = &payload.packed_witness;
        validate_akita_advice_commitment_aliases(&artifacts.commitments, None, None)
            .expect("absent advice commitments should pass");
        validate_akita_advice_commitment_aliases(
            &artifacts.commitments,
            Some(packed_witness),
            None,
        )
        .expect("packed-witness untrusted advice alias should pass");
        assert!(matches!(
            validate_akita_advice_commitment_aliases(
                &artifacts.commitments,
                None,
                Some(packed_witness),
            ),
            Err(VerifierError::InvalidProtocolConfig { reason })
                if reason.contains("trusted advice commitment must be separate")
        ));

        let mut other_commitment = packed_witness.clone();
        other_commitment.layout_digest[0] ^= 1;
        assert!(matches!(
            validate_akita_advice_commitment_aliases(
                &artifacts.commitments,
                Some(&other_commitment),
                None,
            ),
            Err(VerifierError::InvalidProtocolConfig { reason })
                if reason.contains("untrusted advice commitment")
        ));
        validate_akita_advice_commitment_aliases(
            &artifacts.commitments,
            None,
            Some(&other_commitment),
        )
        .expect("trusted advice may use a separate precommitted commitment");
    }

    #[test]
    fn akita_precommitted_commitments_must_not_alias_packed_witness() {
        let (_, artifacts) = empty_artifacts(tiny_layout());
        let payload = lattice_payload(&artifacts);
        let packed_witness = &payload.packed_witness;

        assert!(matches!(
            validate_akita_precommitted_commitment_is_separate(
                packed_witness,
                packed_witness,
                "bytecode chunk",
            ),
            Err(VerifierError::InvalidProtocolConfig { reason })
                if reason.contains("bytecode chunk commitment must be separate")
        ));

        let mut separate_commitment = packed_witness.clone();
        separate_commitment.layout_digest[0] ^= 1;
        validate_akita_precommitted_commitment_is_separate(
            packed_witness,
            &separate_commitment,
            "program image",
        )
        .expect("separate precommitted commitment should pass");
    }

    #[test]
    fn akita_artifact_preflight_rejects_stale_protocol_and_commitments() {
        let layout = tiny_layout();
        let (prover_setup, verifier_setup) = akita_packing_setup(&layout, 1);
        let source = SparsePackingWitness::try_from_cells(
            layout.clone(),
            [
                (
                    packed_cell(PackingFamilyId::InstructionRa { index: 0 }, 7),
                    AkitaField::one(),
                ),
                (
                    packed_cell(PackingFamilyId::UnsignedIncMsb, 1),
                    AkitaField::one(),
                ),
            ],
        )
        .expect("source should build");
        let other_source = SparsePackingWitness::try_from_cells(
            layout.clone(),
            [
                (
                    packed_cell(PackingFamilyId::InstructionRa { index: 0 }, 8),
                    AkitaField::one(),
                ),
                (
                    packed_cell(PackingFamilyId::UnsignedIncMsb, 0),
                    AkitaField::one(),
                ),
            ],
        )
        .expect("other source should build");
        let artifacts = commit_akita_packing_witness(&prover_setup, &source)
            .expect("packed witness should commit");
        let other_artifacts = commit_akita_packing_witness(&prover_setup, &other_source)
            .expect("other packed witness should commit");

        validate_akita_artifacts_for_proof(
            &verifier_setup,
            &artifacts.protocol,
            &artifacts.commitments,
            &artifacts,
        )
        .expect("matching artifacts should pass preflight");

        let mut stale_protocol = artifacts.protocol;
        stale_protocol.lattice.packed_witness.d_pack = Some(layout.dimension + 1);
        assert!(matches!(
            validate_akita_artifacts_for_proof(
                &verifier_setup,
                &stale_protocol,
                &artifacts.commitments,
                &artifacts,
            ),
            Err(VerifierError::ProtocolConfigMismatch { expected, got })
                if expected == artifacts.protocol && got == stale_protocol
        ));

        assert!(matches!(
            validate_akita_artifacts_for_proof(
                &verifier_setup,
                &artifacts.protocol,
                &other_artifacts.commitments,
                &artifacts,
            ),
            Err(VerifierError::InvalidProtocolConfig { reason })
                if reason.contains("do not match packed witness artifacts")
        ));
    }
}
