use common::jolt_device::JoltDevice;
use jolt_claims::protocols::jolt::JoltFormulaDimensions;

use super::prove::CommitmentStageConfig;
use crate::ProofParameters;

pub(crate) fn prover_config(
    public_io: &JoltDevice,
    proof_parameters: ProofParameters,
    formula_dimensions: JoltFormulaDimensions,
) -> CommitmentStageConfig {
    let log_t = proof_parameters.trace_length.trailing_zeros() as usize;

    CommitmentStageConfig::new(
        formula_dimensions.ra_layout,
        !public_io.trusted_advice.is_empty(),
        !public_io.untrusted_advice.is_empty(),
    )
    .with_final_opening_trace_embedding(
        log_t,
        proof_parameters.one_hot_config.committed_chunk_bits(),
        proof_parameters.trace_polynomial_order,
    )
}
