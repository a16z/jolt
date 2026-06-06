use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltReadWriteConfig, TracePolynomialOrder};
use jolt_verifier::{JoltProtocolConfig, ZkConfig, JOLT_VERIFIER_CONFIG};

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

    pub fn from_protocol(protocol: &JoltProtocolConfig) -> Self {
        Self {
            zk: matches!(protocol.zk, ZkConfig::BlindFold),
            field_inline: protocol.field_inline.enabled,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProverConfig {
    pub protocol: JoltProtocolConfig,
    pub features: ProverFeatureSet,
    pub proof_shape: Option<ProverProofShape>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProverProofShape {
    pub trace_length: usize,
    pub ram_k: usize,
    pub rw_config: JoltReadWriteConfig,
    pub one_hot_config: JoltOneHotConfig,
    pub trace_polynomial_order: TracePolynomialOrder,
}

impl ProverProofShape {
    pub const fn new(
        trace_length: usize,
        ram_k: usize,
        rw_config: JoltReadWriteConfig,
        one_hot_config: JoltOneHotConfig,
        trace_polynomial_order: TracePolynomialOrder,
    ) -> Self {
        Self {
            trace_length,
            ram_k,
            rw_config,
            one_hot_config,
            trace_polynomial_order,
        }
    }
}

impl Default for ProverConfig {
    fn default() -> Self {
        Self {
            protocol: JOLT_VERIFIER_CONFIG,
            features: ProverFeatureSet::COMPILED,
            proof_shape: None,
        }
    }
}

impl ProverConfig {
    pub const fn with_proof_shape(mut self, proof_shape: ProverProofShape) -> Self {
        self.proof_shape = Some(proof_shape);
        self
    }
}
