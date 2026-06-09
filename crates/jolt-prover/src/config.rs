use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltReadWriteConfig, TracePolynomialOrder};
use jolt_verifier::{JoltProtocolConfig, ZkConfig, JOLT_VERIFIER_CONFIG};

use crate::error::ProverError;

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
    pub proof_parameters: Option<ProofParameters>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProofParameters {
    pub trace_length: usize,
    pub ram_k: usize,
    pub rw_config: JoltReadWriteConfig,
    pub one_hot_config: JoltOneHotConfig,
    pub trace_polynomial_order: TracePolynomialOrder,
}

impl ProofParameters {
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
            proof_parameters: None,
        }
    }
}

impl ProverConfig {
    pub const fn with_proof_parameters(mut self, proof_parameters: ProofParameters) -> Self {
        self.proof_parameters = Some(proof_parameters);
        self
    }

    pub(crate) fn validate_for_proving(&self) -> Result<(), ProverError> {
        if self.features != ProverFeatureSet::COMPILED {
            return Err(ProverError::InvalidProverConfig {
                reason: format!(
                    "requested features {:?} do not match compiled features {:?}",
                    self.features,
                    ProverFeatureSet::COMPILED
                ),
            });
        }

        let protocol_features = ProverFeatureSet::from_protocol(&self.protocol);
        if protocol_features != self.features {
            return Err(ProverError::InvalidProverConfig {
                reason: format!(
                    "protocol {:?} implies features {:?}, but requested features are {:?}",
                    self.protocol, protocol_features, self.features
                ),
            });
        }

        if self.protocol != JOLT_VERIFIER_CONFIG {
            return Err(ProverError::InvalidProverConfig {
                reason: format!(
                    "requested protocol {:?} does not match compiled verifier protocol {:?}",
                    self.protocol, JOLT_VERIFIER_CONFIG
                ),
            });
        }

        if let Some(proof_parameters) = self.proof_parameters {
            proof_parameters.validate_for_proving()?;
        }

        Ok(())
    }
}

impl ProofParameters {
    pub(crate) fn validate_for_proving(self) -> Result<(), ProverError> {
        if self.trace_length == 0 || !self.trace_length.is_power_of_two() {
            return Err(ProverError::InvalidProverConfig {
                reason: format!(
                    "proof trace_length must be a nonzero power of two, got {}",
                    self.trace_length
                ),
            });
        }

        if self.ram_k == 0 || !self.ram_k.is_power_of_two() {
            return Err(ProverError::InvalidProverConfig {
                reason: format!(
                    "proof ram_k must be a nonzero power of two, got {}",
                    self.ram_k
                ),
            });
        }

        Ok(())
    }
}
