use common::jolt_device::JoltDevice;
use jolt_claims::protocols::jolt::{
    formulas::claim_reductions::hamming_weight::HammingWeightClaimReductionDimensions,
    JoltFormulaDimensions,
};

use super::prove::Stage7ProverConfig;
use crate::stages::advice::advice_layouts;
use crate::{ProverConfig, ProverError};

pub(crate) fn prover_config(
    public_io: &JoltDevice,
    proof_parameters: ProverConfig,
    formula_dimensions: JoltFormulaDimensions,
) -> Result<Stage7ProverConfig, ProverError> {
    let log_t = proof_parameters.trace_length.trailing_zeros() as usize;
    let committed_chunk_bits = proof_parameters.one_hot_config.committed_chunk_bits();
    let hamming_dimensions = HammingWeightClaimReductionDimensions::new(
        formula_dimensions.ra_layout,
        committed_chunk_bits,
    );
    let (trusted_advice_layout, untrusted_advice_layout) =
        advice_layouts(public_io, proof_parameters, log_t, committed_chunk_bits)?;

    Ok(Stage7ProverConfig::new(
        hamming_dimensions,
        trusted_advice_layout,
        untrusted_advice_layout,
    ))
}
