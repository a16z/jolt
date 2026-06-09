use common::jolt_device::JoltDevice;
use jolt_claims::protocols::jolt::{
    formulas::claim_reductions::hamming_weight::HammingWeightClaimReductionDimensions,
    AdviceClaimReductionLayout, JoltFormulaDimensions,
};

use super::prove::Stage7ProverConfig;
use crate::ProofParameters;

pub(crate) fn prover_config(
    public_io: &JoltDevice,
    proof_parameters: ProofParameters,
    formula_dimensions: JoltFormulaDimensions,
) -> Stage7ProverConfig {
    let log_t = proof_parameters.trace_length.trailing_zeros() as usize;
    let committed_chunk_bits = proof_parameters.one_hot_config.committed_chunk_bits();
    let hamming_dimensions = HammingWeightClaimReductionDimensions::new(
        formula_dimensions.ra_layout,
        committed_chunk_bits,
    );
    let (trusted_advice_layout, untrusted_advice_layout) =
        advice_layouts(public_io, proof_parameters, log_t, committed_chunk_bits);

    Stage7ProverConfig::new(
        hamming_dimensions,
        trusted_advice_layout,
        untrusted_advice_layout,
    )
}

fn advice_layouts(
    public_io: &JoltDevice,
    proof_parameters: ProofParameters,
    log_t: usize,
    committed_chunk_bits: usize,
) -> (
    Option<AdviceClaimReductionLayout>,
    Option<AdviceClaimReductionLayout>,
) {
    let trusted = (!public_io.trusted_advice.is_empty()).then(|| {
        AdviceClaimReductionLayout::balanced(
            proof_parameters.trace_polynomial_order,
            log_t,
            committed_chunk_bits,
            public_io.memory_layout.max_trusted_advice_size as usize,
        )
    });
    let untrusted = (!public_io.untrusted_advice.is_empty()).then(|| {
        AdviceClaimReductionLayout::balanced(
            proof_parameters.trace_polynomial_order,
            log_t,
            committed_chunk_bits,
            public_io.memory_layout.max_untrusted_advice_size as usize,
        )
    });

    (trusted, untrusted)
}
