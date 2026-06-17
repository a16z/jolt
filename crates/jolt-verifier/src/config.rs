//! Verifier-selected protocol configuration.

use jolt_claims::protocols::field_inline::FieldInlineConfig;
use serde::{Deserialize, Serialize};

use crate::{
    proof::{validate_commitment_payload_config, JoltProof},
    VerifierError,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZkConfig {
    Transparent,
    BlindFold,
}

impl ZkConfig {
    pub const fn from_bool(zk: bool) -> Self {
        if zk {
            Self::BlindFold
        } else {
            Self::Transparent
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PcsFamily {
    Curve,
    Lattice,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PcsFamilyFlags {
    pub curve: bool,
    pub lattice: bool,
}

impl Default for PcsFamilyFlags {
    fn default() -> Self {
        Self {
            curve: true,
            lattice: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgramMode {
    #[default]
    Full,
    Committed,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum IncrementCommitmentMode {
    #[default]
    Dense,
    SeparateOneHot,
    FusedOneHot,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackedWitnessConfig {
    pub layout_digest: Option<[u8; 32]>,
    pub d_pack: Option<usize>,
    pub field_rd_inc_family: bool,
    pub trusted_advice_family: bool,
    pub untrusted_advice_family: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldInlineLatticeConfig {
    pub enabled: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdviceLatticeConfig {
    pub trusted: bool,
    pub untrusted: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LatticeConfig {
    pub program_mode: ProgramMode,
    pub increment_mode: IncrementCommitmentMode,
    pub packed_witness: PackedWitnessConfig,
    pub field_inline: FieldInlineLatticeConfig,
    pub advice: AdviceLatticeConfig,
    pub zk: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JoltProtocolConfig {
    pub zk: ZkConfig,
    pub field_inline: FieldInlineConfig,
    #[serde(default)]
    pub pcs: PcsFamilyFlags,
    #[serde(default)]
    pub lattice: LatticeConfig,
}

impl JoltProtocolConfig {
    pub const fn for_zk(zk: bool) -> Self {
        Self {
            zk: ZkConfig::from_bool(zk),
            field_inline: SELECTED_FIELD_INLINE_CONFIG,
            pcs: PcsFamilyFlags {
                curve: true,
                lattice: false,
            },
            lattice: LatticeConfig {
                program_mode: ProgramMode::Full,
                increment_mode: IncrementCommitmentMode::Dense,
                packed_witness: PackedWitnessConfig {
                    layout_digest: None,
                    d_pack: None,
                    field_rd_inc_family: false,
                    trusted_advice_family: false,
                    untrusted_advice_family: false,
                },
                field_inline: FieldInlineLatticeConfig { enabled: false },
                advice: AdviceLatticeConfig {
                    trusted: false,
                    untrusted: false,
                },
                zk,
            },
        }
    }

    pub const fn with_pcs_family(mut self, family: PcsFamily) -> Self {
        self.pcs = match family {
            PcsFamily::Curve => PcsFamilyFlags {
                curve: true,
                lattice: false,
            },
            PcsFamily::Lattice => PcsFamilyFlags {
                curve: false,
                lattice: true,
            },
        };
        self
    }
}

#[cfg(feature = "field-inline")]
pub const SELECTED_FIELD_INLINE_CONFIG: FieldInlineConfig = FieldInlineConfig::native_v1();

#[cfg(not(feature = "field-inline"))]
pub const SELECTED_FIELD_INLINE_CONFIG: FieldInlineConfig = FieldInlineConfig::disabled();

#[cfg(feature = "zk")]
pub const SELECTED_ZK_CONFIG: ZkConfig = ZkConfig::BlindFold;

#[cfg(not(feature = "zk"))]
pub const SELECTED_ZK_CONFIG: ZkConfig = ZkConfig::Transparent;

pub const JOLT_VERIFIER_CONFIG: JoltProtocolConfig = JoltProtocolConfig {
    zk: SELECTED_ZK_CONFIG,
    field_inline: SELECTED_FIELD_INLINE_CONFIG,
    pcs: PcsFamilyFlags {
        curve: true,
        lattice: false,
    },
    lattice: LatticeConfig {
        program_mode: ProgramMode::Full,
        increment_mode: IncrementCommitmentMode::Dense,
        packed_witness: PackedWitnessConfig {
            layout_digest: None,
            d_pack: None,
            field_rd_inc_family: false,
            trusted_advice_family: false,
            untrusted_advice_family: false,
        },
        field_inline: FieldInlineLatticeConfig { enabled: false },
        advice: AdviceLatticeConfig {
            trusted: false,
            untrusted: false,
        },
        zk: cfg!(feature = "zk"),
    },
};

pub fn validate_protocol_config(config: &JoltProtocolConfig) -> Result<PcsFamily, VerifierError> {
    let family = match (config.pcs.curve, config.pcs.lattice) {
        (true, false) => PcsFamily::Curve,
        (false, true) => PcsFamily::Lattice,
        (true, true) => {
            return Err(invalid_config(
                "PCS family flags are mutually exclusive; choose curve or lattice",
            ));
        }
        (false, false) => {
            return Err(invalid_config(
                "at least one PCS family flag must be selected",
            ));
        }
    };

    if family == PcsFamily::Lattice {
        validate_lattice_config(config)?;
    }

    Ok(family)
}

fn validate_lattice_config(config: &JoltProtocolConfig) -> Result<(), VerifierError> {
    if config.zk != ZkConfig::Transparent || config.lattice.zk {
        return Err(invalid_config(
            "lattice PCS mode is transparent-only until Akita ZK openings are specified",
        ));
    }
    if config.lattice.program_mode != ProgramMode::Committed {
        return Err(invalid_config(
            "lattice PCS mode requires ProgramMode::Committed",
        ));
    }
    if config.lattice.increment_mode != IncrementCommitmentMode::FusedOneHot {
        return Err(invalid_config(
            "lattice PCS mode requires fused one-hot base increments",
        ));
    }
    if config.lattice.field_inline.enabled && !config.lattice.packed_witness.field_rd_inc_family {
        return Err(invalid_config(
            "field-inline lattice mode requires FieldRdInc packed witness families",
        ));
    }
    if config.lattice.advice.trusted && !config.lattice.packed_witness.trusted_advice_family {
        return Err(invalid_config(
            "trusted advice lattice mode requires trusted advice packed witness families",
        ));
    }
    if config.lattice.advice.untrusted && !config.lattice.packed_witness.untrusted_advice_family {
        return Err(invalid_config(
            "untrusted advice lattice mode requires untrusted advice packed witness families",
        ));
    }
    if config.lattice.packed_witness.layout_digest.is_none()
        || config.lattice.packed_witness.d_pack.is_none()
    {
        return Err(invalid_config(
            "lattice PCS mode requires a packed witness layout digest and D_pack",
        ));
    }
    Ok(())
}

fn invalid_config(reason: impl Into<String>) -> VerifierError {
    VerifierError::InvalidProtocolConfig {
        reason: reason.into(),
    }
}

pub fn validate_proof_config<PCS, VC, ZkProof>(
    config: &JoltProtocolConfig,
    proof: &JoltProof<PCS, VC, ZkProof>,
) -> Result<(), VerifierError>
where
    PCS: jolt_openings::CommitmentScheme,
    VC: jolt_crypto::VectorCommitment<Field = PCS::Field>,
{
    let _ = validate_protocol_config(config)?;
    if proof.protocol != *config {
        return Err(VerifierError::ProtocolConfigMismatch {
            expected: *config,
            got: proof.protocol,
        });
    }
    validate_commitment_payload_config(config, &proof.commitments)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_lattice_config() -> JoltProtocolConfig {
        let mut config = JoltProtocolConfig::for_zk(false).with_pcs_family(PcsFamily::Lattice);
        config.lattice.program_mode = ProgramMode::Committed;
        config.lattice.increment_mode = IncrementCommitmentMode::FusedOneHot;
        config.lattice.packed_witness.layout_digest = Some([7; 32]);
        config.lattice.packed_witness.d_pack = Some(43);
        config
    }

    fn invalid(config: &JoltProtocolConfig) -> bool {
        matches!(
            validate_protocol_config(config),
            Err(VerifierError::InvalidProtocolConfig { .. })
        )
    }

    #[test]
    fn pcs_family_flags_default_to_curve() {
        let config = JoltProtocolConfig::for_zk(false);

        assert_eq!(
            validate_protocol_config(&config).ok(),
            Some(PcsFamily::Curve)
        );
    }

    #[test]
    fn pcs_family_flags_are_mutually_exclusive() {
        let mut both = JoltProtocolConfig::for_zk(false);
        both.pcs = PcsFamilyFlags {
            curve: true,
            lattice: true,
        };
        let mut neither = JoltProtocolConfig::for_zk(false);
        neither.pcs = PcsFamilyFlags {
            curve: false,
            lattice: false,
        };

        assert!(invalid(&both));
        assert!(invalid(&neither));
    }

    #[test]
    fn akita_requires_committed_program() {
        let mut config = valid_lattice_config();
        config.lattice.program_mode = ProgramMode::Full;

        assert!(invalid(&config));
    }

    #[test]
    fn akita_requires_fused_increments() {
        let mut dense = valid_lattice_config();
        dense.lattice.increment_mode = IncrementCommitmentMode::Dense;
        let mut separate = valid_lattice_config();
        separate.lattice.increment_mode = IncrementCommitmentMode::SeparateOneHot;

        assert!(invalid(&dense));
        assert!(invalid(&separate));
    }

    #[test]
    fn akita_zk_rejects() {
        let mut config = valid_lattice_config();
        config.zk = ZkConfig::BlindFold;

        assert!(invalid(&config));

        let mut lattice_zk = valid_lattice_config();
        lattice_zk.lattice.zk = true;

        assert!(invalid(&lattice_zk));
    }

    #[test]
    fn akita_field_inline_requires_layout_families() {
        let mut config = valid_lattice_config();
        config.lattice.field_inline.enabled = true;

        assert!(invalid(&config));

        config.lattice.packed_witness.field_rd_inc_family = true;
        assert_eq!(
            validate_protocol_config(&config).ok(),
            Some(PcsFamily::Lattice)
        );
    }

    #[test]
    fn akita_advice_requires_layout_families() {
        let mut trusted = valid_lattice_config();
        trusted.lattice.advice.trusted = true;
        assert!(invalid(&trusted));

        trusted.lattice.packed_witness.trusted_advice_family = true;
        assert_eq!(
            validate_protocol_config(&trusted).ok(),
            Some(PcsFamily::Lattice)
        );

        let mut untrusted = valid_lattice_config();
        untrusted.lattice.advice.untrusted = true;
        assert!(invalid(&untrusted));

        untrusted.lattice.packed_witness.untrusted_advice_family = true;
        assert_eq!(
            validate_protocol_config(&untrusted).ok(),
            Some(PcsFamily::Lattice)
        );
    }

    #[test]
    fn akita_requires_layout_digest_and_dimension() {
        let mut no_digest = valid_lattice_config();
        no_digest.lattice.packed_witness.layout_digest = None;
        let mut no_dimension = valid_lattice_config();
        no_dimension.lattice.packed_witness.d_pack = None;

        assert!(invalid(&no_digest));
        assert!(invalid(&no_dimension));
    }
}
