use common::jolt_device::JoltDevice;
use jolt_claims::protocols::jolt::formulas::claim_reductions::advice::candidate_total_vars;
use jolt_claims::protocols::jolt::{AdviceClaimReductionLayout, PrecommittedClaimReduction};

use crate::{ProverConfig, ProverError};

/// Builds the trusted/untrusted advice claim-reduction layouts exactly as the
/// verifier's `PrecommittedSchedule::new` does.
pub(crate) fn advice_layouts(
    public_io: &JoltDevice,
    proof_parameters: ProverConfig,
    log_t: usize,
    committed_chunk_bits: usize,
) -> Result<
    (
        Option<AdviceClaimReductionLayout>,
        Option<AdviceClaimReductionLayout>,
    ),
    ProverError,
> {
    let trusted_max_bytes = (!public_io.trusted_advice.is_empty())
        .then_some(public_io.memory_layout.max_trusted_advice_size as usize);
    let untrusted_max_bytes = (!public_io.untrusted_advice.is_empty())
        .then_some(public_io.memory_layout.max_untrusted_advice_size as usize);

    let candidates = candidate_total_vars(trusted_max_bytes, untrusted_max_bytes);
    let scheduling_reference = PrecommittedClaimReduction::scheduling_reference(
        log_t + committed_chunk_bits,
        &candidates,
        committed_chunk_bits,
    );

    let layout =
        |max_bytes: Option<usize>| -> Result<Option<AdviceClaimReductionLayout>, ProverError> {
            max_bytes
                .map(|max_bytes| {
                    AdviceClaimReductionLayout::balanced(
                        proof_parameters.trace_polynomial_order,
                        log_t,
                        scheduling_reference,
                        max_bytes,
                    )
                    .map_err(|error| ProverError::InvalidProverConfig {
                        reason: format!(
                            "advice claim-reduction layout could not be built: {error}"
                        ),
                    })
                })
                .transpose()
        };

    Ok((layout(trusted_max_bytes)?, layout(untrusted_max_bytes)?))
}
