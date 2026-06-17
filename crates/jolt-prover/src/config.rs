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

/// The per-run prover inputs: the workload shape for a single proof.
///
/// The protocol is *not* a field — a prover binary always proves against its
/// compiled [`JOLT_VERIFIER_CONFIG`], so it is referenced as that constant
/// wherever needed rather than carried (and varied) per config.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProverConfig {
    pub trace_length: usize,
    pub ram_k: usize,
    pub rw_config: JoltReadWriteConfig,
    pub one_hot_config: JoltOneHotConfig,
    pub trace_polynomial_order: TracePolynomialOrder,
}

impl ProverConfig {
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

    /// The compile-relevant feature subset (`zk`, `field_inline`) this binary
    /// proves with, derived from the pinned [`JOLT_VERIFIER_CONFIG`].
    pub fn features() -> ProverFeatureSet {
        ProverFeatureSet::from_protocol(&JOLT_VERIFIER_CONFIG)
    }

    pub(crate) fn validate_for_proving(&self) -> Result<(), ProverError> {
        let features = Self::features();
        if features != ProverFeatureSet::COMPILED {
            return Err(ProverError::InvalidProverConfig {
                reason: format!(
                    "the verifier protocol implies prover features {features:?}, but this binary was compiled with {:?}",
                    ProverFeatureSet::COMPILED
                ),
            });
        }

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
