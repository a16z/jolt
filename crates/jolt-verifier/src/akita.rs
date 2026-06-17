//! Prover-facing helpers for assembling Akita verifier artifacts.

use crate::{
    config::{
        AdviceLatticeConfig, FieldInlineLatticeConfig, IncrementCommitmentMode, JoltProtocolConfig,
        LatticeConfig, PackedWitnessConfig, PcsFamily, ProgramMode,
    },
    proof::{AkitaCommitmentPayload, CommitmentPayload},
    stages::stage8::validate_akita_packed_witness_layout_config,
    VerifierError,
};
use jolt_akita::{
    AkitaCommitment, AkitaField, AkitaPackedScheme, AkitaProverHint, AkitaProverSetup,
    PackedAdviceKind, PackedFamilyId, PackedWitnessLayout, PackedWitnessSource,
};

#[derive(Clone, Debug)]
pub struct AkitaPackedWitnessArtifacts {
    pub protocol: JoltProtocolConfig,
    pub layout: PackedWitnessLayout,
    pub commitments: CommitmentPayload<AkitaCommitment>,
    pub hint: AkitaProverHint,
}

impl AkitaPackedWitnessArtifacts {
    pub fn payload(&self) -> Option<&AkitaCommitmentPayload<AkitaCommitment>> {
        self.commitments.as_akita()
    }
}

pub fn akita_lattice_protocol_config_for_layout(
    layout: &PackedWitnessLayout,
) -> JoltProtocolConfig {
    let mut config = JoltProtocolConfig::for_zk(false).with_pcs_family(PcsFamily::Lattice);
    config.lattice = LatticeConfig {
        program_mode: ProgramMode::Committed,
        increment_mode: IncrementCommitmentMode::FusedOneHot,
        packed_witness: PackedWitnessConfig {
            layout_digest: Some(layout.digest),
            d_pack: Some(layout.dimension),
            field_rd_inc_family: layout_has_field_rd_inc(layout),
            trusted_advice_family: layout_has_advice(layout, PackedAdviceKind::Trusted),
            untrusted_advice_family: layout_has_advice(layout, PackedAdviceKind::Untrusted),
        },
        field_inline: FieldInlineLatticeConfig {
            enabled: layout_has_field_rd_inc(layout),
        },
        advice: AdviceLatticeConfig {
            trusted: layout_has_advice(layout, PackedAdviceKind::Trusted),
            untrusted: layout_has_advice(layout, PackedAdviceKind::Untrusted),
        },
        zk: false,
    };
    config
}

pub fn commit_akita_packed_witness<S>(
    setup: &AkitaProverSetup,
    source: &S,
) -> Result<AkitaPackedWitnessArtifacts, VerifierError>
where
    S: PackedWitnessSource<AkitaField>,
{
    let protocol = akita_lattice_protocol_config_for_layout(source.layout());
    commit_akita_packed_witness_with_config(protocol, setup, source)
}

pub fn commit_akita_packed_witness_with_config<S>(
    protocol: JoltProtocolConfig,
    setup: &AkitaProverSetup,
    source: &S,
) -> Result<AkitaPackedWitnessArtifacts, VerifierError>
where
    S: PackedWitnessSource<AkitaField>,
{
    let layout = source.layout().clone();
    validate_akita_packed_witness_layout_config(&protocol, &layout)?;
    let (commitment, hint) =
        AkitaPackedScheme::commit_packed_source(setup, source).map_err(|error| {
            VerifierError::AkitaCommitmentFailed {
                reason: error.to_string(),
            }
        })?;
    let payload = AkitaCommitmentPayload::new(commitment, layout.digest, layout.dimension);
    crate::proof::validate_akita_commitment_payload_config(&protocol, &payload)?;

    Ok(AkitaPackedWitnessArtifacts {
        protocol,
        layout,
        commitments: CommitmentPayload::Akita(payload),
        hint,
    })
}

fn layout_has_field_rd_inc(layout: &PackedWitnessLayout) -> bool {
    layout.families.iter().any(|family| {
        matches!(
            family.id,
            PackedFamilyId::FieldRdIncByte { .. } | PackedFamilyId::FieldRdIncSign
        )
    })
}

fn layout_has_advice(layout: &PackedWitnessLayout, kind: PackedAdviceKind) -> bool {
    layout.families.iter().any(|family| {
        matches!(
            family.id,
            PackedFamilyId::AdviceBytes {
                kind: family_kind,
                ..
            } if family_kind == kind
        )
    })
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::expect_used,
        reason = "tests assert successful artifact construction"
    )]

    use super::*;
    use jolt_akita::{
        AkitaSetupParams, PackedAlphabet, PackedFactDomain, PackedFamilySpec, SparsePackedWitness,
    };
    use jolt_openings::CommitmentScheme;

    fn tiny_layout() -> PackedWitnessLayout {
        PackedWitnessLayout::new([
            PackedFamilySpec::direct(
                PackedFamilyId::InstructionRa { index: 0 },
                PackedFactDomain::TraceRows { log_t: 0 },
                1,
                PackedAlphabet::Byte,
            ),
            PackedFamilySpec::direct(
                PackedFamilyId::IncSign,
                PackedFactDomain::TraceRows { log_t: 0 },
                1,
                PackedAlphabet::Bit,
            ),
        ])
        .expect("layout should build")
    }

    #[test]
    fn protocol_config_binds_layout_digest_and_dimension() {
        let layout = tiny_layout();

        let config = akita_lattice_protocol_config_for_layout(&layout);

        assert_eq!(
            config.lattice.packed_witness.layout_digest,
            Some(layout.digest)
        );
        assert_eq!(config.lattice.packed_witness.d_pack, Some(layout.dimension));
        assert_eq!(config.lattice.program_mode, ProgramMode::Committed);
        assert_eq!(
            config.lattice.increment_mode,
            IncrementCommitmentMode::FusedOneHot
        );
    }

    #[test]
    fn commits_packed_witness_and_returns_verifier_payload() {
        let layout = tiny_layout();
        let params = AkitaSetupParams::from_packed_layout(&layout, 1);
        let (prover_setup, _) = AkitaPackedScheme::setup(params);
        let source = SparsePackedWitness::try_new(
            layout.clone(),
            vec![(0, AkitaField::from_u64(1)), (256, AkitaField::from_u64(1))],
        )
        .expect("source should build");

        let artifact = commit_akita_packed_witness(&prover_setup, &source)
            .expect("packed witness should commit");

        assert_eq!(artifact.layout, layout);
        let payload = artifact
            .payload()
            .expect("artifact should carry Akita payload");
        assert_eq!(payload.layout_digest, layout.digest);
        assert_eq!(payload.d_pack, layout.dimension);
        assert_eq!(payload.packed_witness.layout_digest, layout.digest);
        assert_eq!(payload.packed_witness.num_vars, layout.dimension);
        assert_eq!(
            artifact.protocol.lattice.packed_witness.layout_digest,
            Some(layout.digest)
        );
    }

    #[test]
    fn configured_layout_mismatch_rejects_before_commit() {
        let layout = tiny_layout();
        let params = AkitaSetupParams::from_packed_layout(&layout, 1);
        let (prover_setup, _) = AkitaPackedScheme::setup(params);
        let source = SparsePackedWitness::try_new(layout.clone(), Vec::new())
            .expect("empty sparse source should build");
        let mut config = akita_lattice_protocol_config_for_layout(&layout);
        config.lattice.packed_witness.layout_digest = Some([9; 32]);

        let error = commit_akita_packed_witness_with_config(config, &prover_setup, &source)
            .expect_err("layout mismatch should reject");

        assert!(matches!(error, VerifierError::InvalidProtocolConfig { .. }));
    }
}
