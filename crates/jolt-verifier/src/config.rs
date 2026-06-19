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
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
pub struct PackedWitnessConfig {
    pub layout_digest: Option<[u8; 32]>,
    pub d_pack: Option<usize>,
    #[serde(default)]
    pub validity_digest: Option<[u8; 32]>,
    pub field_rd_inc_family: bool,
    pub trusted_advice_family: bool,
    pub untrusted_advice_family: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FieldInlineLatticeConfig {
    pub enabled: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdviceLatticeConfig {
    pub trusted: bool,
    pub untrusted: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LatticeConfig {
    pub program_mode: ProgramMode,
    pub increment_mode: IncrementCommitmentMode,
    pub packed_witness: PackedWitnessConfig,
    pub field_inline: FieldInlineLatticeConfig,
    pub advice: AdviceLatticeConfig,
    pub zk: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
                    validity_digest: None,
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
            validity_digest: None,
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
    #[cfg(feature = "field-inline")]
    if !config.field_inline.enabled {
        return Err(invalid_config(
            "field-inline verifier builds require field-inline lattice protocols",
        ));
    }
    #[cfg(not(feature = "field-inline"))]
    if config.field_inline.enabled {
        return Err(invalid_config(
            "field-inline lattice protocols require the jolt-verifier field-inline feature",
        ));
    }
    if config.field_inline.enabled != config.lattice.field_inline.enabled {
        return Err(invalid_config(
            "lattice field-inline mode must match the selected field-inline protocol",
        ));
    }
    if config.lattice.field_inline.enabled && !config.lattice.packed_witness.field_rd_inc_family {
        return Err(invalid_config(
            "field-inline lattice mode requires FieldRdInc packed witness families",
        ));
    }
    if !config.lattice.field_inline.enabled && config.lattice.packed_witness.field_rd_inc_family {
        return Err(invalid_config(
            "FieldRdInc packed witness families require field-inline lattice mode",
        ));
    }
    if config.lattice.packed_witness.trusted_advice_family {
        return Err(invalid_config(
            "trusted advice uses separate precommitted openings and cannot be a packed witness family",
        ));
    }
    if config.lattice.advice.untrusted && !config.lattice.packed_witness.untrusted_advice_family {
        return Err(invalid_config(
            "untrusted advice lattice mode requires untrusted advice packed witness families",
        ));
    }
    if !config.lattice.advice.untrusted && config.lattice.packed_witness.untrusted_advice_family {
        return Err(invalid_config(
            "untrusted advice packed witness families require untrusted advice lattice mode",
        ));
    }
    if config.lattice.packed_witness.layout_digest.is_none()
        || config.lattice.packed_witness.d_pack.is_none()
        || config.lattice.packed_witness.validity_digest.is_none()
    {
        return Err(invalid_config(
            "lattice PCS mode requires packed witness layout, validity, and D_pack bindings",
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
        config.lattice.packed_witness.validity_digest = Some([11; 32]);
        #[cfg(feature = "field-inline")]
        {
            config.lattice.field_inline.enabled = true;
            config.lattice.packed_witness.field_rd_inc_family = true;
        }
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
    #[cfg(feature = "field-inline")]
    fn akita_field_inline_requires_layout_families() {
        let mut config = valid_lattice_config();
        config.lattice.packed_witness.field_rd_inc_family = false;

        assert!(invalid(&config));

        config.lattice.packed_witness.field_rd_inc_family = true;
        assert_eq!(
            validate_protocol_config(&config).ok(),
            Some(PcsFamily::Lattice)
        );
    }

    #[test]
    #[cfg(feature = "field-inline")]
    fn akita_field_inline_must_match_protocol_config() {
        let mut config = valid_lattice_config();
        config.lattice.field_inline.enabled = false;

        assert!(invalid(&config));
    }

    #[test]
    #[cfg(not(feature = "field-inline"))]
    fn akita_rejects_field_inline_when_protocol_disabled() {
        let mut config = valid_lattice_config();
        config.lattice.field_inline.enabled = true;
        config.lattice.packed_witness.field_rd_inc_family = true;

        assert!(invalid(&config));
    }

    #[test]
    fn akita_advice_config_splits_trusted_precommitted_from_untrusted_packed() {
        let mut trusted = valid_lattice_config();
        trusted.lattice.advice.trusted = true;
        assert_eq!(
            validate_protocol_config(&trusted).ok(),
            Some(PcsFamily::Lattice)
        );

        trusted.lattice.packed_witness.trusted_advice_family = true;
        assert!(invalid(&trusted));

        let mut untrusted = valid_lattice_config();
        untrusted.lattice.advice.untrusted = true;
        assert!(invalid(&untrusted));

        untrusted.lattice.packed_witness.untrusted_advice_family = true;
        assert_eq!(
            validate_protocol_config(&untrusted).ok(),
            Some(PcsFamily::Lattice)
        );

        let mut extra_trusted_family = valid_lattice_config();
        extra_trusted_family
            .lattice
            .packed_witness
            .trusted_advice_family = true;
        assert!(invalid(&extra_trusted_family));

        let mut extra_untrusted_family = valid_lattice_config();
        extra_untrusted_family
            .lattice
            .packed_witness
            .untrusted_advice_family = true;
        assert!(invalid(&extra_untrusted_family));
    }

    #[test]
    fn akita_requires_layout_digest_and_dimension() {
        let mut no_digest = valid_lattice_config();
        no_digest.lattice.packed_witness.layout_digest = None;
        let mut no_dimension = valid_lattice_config();
        no_dimension.lattice.packed_witness.d_pack = None;
        let mut no_validity = valid_lattice_config();
        no_validity.lattice.packed_witness.validity_digest = None;

        assert!(invalid(&no_digest));
        assert!(invalid(&no_dimension));
        assert!(invalid(&no_validity));
    }

    #[test]
    #[expect(
        clippy::expect_used,
        reason = "test fixture mutation should fail loudly if the serialized shape changes"
    )]
    fn protocol_config_rejects_unknown_serialized_fields() {
        fn with_extra_field(path: &[&str], field: &str) -> serde_json::Value {
            let mut value =
                serde_json::to_value(valid_lattice_config()).expect("config should serialize");
            let mut cursor = &mut value;
            for segment in path {
                cursor = cursor
                    .as_object_mut()
                    .expect("config segment should be an object")
                    .get_mut(*segment)
                    .expect("config segment should exist");
            }
            let previous = cursor
                .as_object_mut()
                .expect("target config segment should be an object")
                .insert(field.to_string(), serde_json::Value::Bool(true));
            assert!(previous.is_none());
            value
        }

        for (path, field) in [
            (&[][..], "extra_root"),
            (&["pcs"][..], "extra_pcs_family"),
            (&["lattice"][..], "extra_lattice_mode"),
            (&["lattice", "packed_witness"][..], "extra_packed_witness"),
            (&["lattice", "field_inline"][..], "extra_field_inline"),
            (&["lattice", "advice"][..], "extra_advice"),
        ] {
            assert!(
                serde_json::from_value::<JoltProtocolConfig>(with_extra_field(path, field))
                    .is_err(),
                "unknown config field {field} at path {path:?} must reject"
            );
        }
    }
}
