use jolt_verifier::{JoltProtocolConfig, JOLT_VERIFIER_CONFIG};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProverFeatureSet {
    pub zk: bool,
    pub field_inline: bool,
}

impl ProverFeatureSet {
    pub const COMPILED: Self = Self {
        zk: cfg!(feature = "zk"),
        field_inline: cfg!(feature = "field-inline"),
    };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProverConfig {
    pub protocol: JoltProtocolConfig,
    pub features: ProverFeatureSet,
}

impl Default for ProverConfig {
    fn default() -> Self {
        Self {
            protocol: JOLT_VERIFIER_CONFIG,
            features: ProverFeatureSet::COMPILED,
        }
    }
}
