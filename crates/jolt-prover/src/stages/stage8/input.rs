use jolt_claims::protocols::jolt::{
    formulas::dimensions::TracePolynomialOrder, formulas::ra::JoltRaPolynomialLayout,
    AdviceClaimReductionLayout,
};

use {
    jolt_field::Field,
    jolt_verifier::{
        stages::{stage6::Stage6ClearOutput, stage7::outputs::Stage7ClearOutput},
        CheckedInputs,
    },
};

/// Canonical Stage 8 prover configuration.
///
/// Carries the verifier-equivalent dimensions, RA layout, trace polynomial order,
/// and advice layouts needed to build the final batched-opening order exactly as
/// `jolt-verifier/src/stages/stage8/verify.rs` does.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage8ProverConfig {
    pub log_t: usize,
    pub committed_chunk_bits: usize,
    pub layout: JoltRaPolynomialLayout,
    pub trace_polynomial_order: TracePolynomialOrder,
    pub trusted_advice_layout: Option<AdviceClaimReductionLayout>,
    pub untrusted_advice_layout: Option<AdviceClaimReductionLayout>,
}

impl Stage8ProverConfig {
    pub const fn new(
        log_t: usize,
        committed_chunk_bits: usize,
        layout: JoltRaPolynomialLayout,
        trace_polynomial_order: TracePolynomialOrder,
        trusted_advice_layout: Option<AdviceClaimReductionLayout>,
        untrusted_advice_layout: Option<AdviceClaimReductionLayout>,
    ) -> Self {
        Self {
            log_t,
            committed_chunk_bits,
            layout,
            trace_polynomial_order,
            trusted_advice_layout,
            untrusted_advice_layout,
        }
    }
}

/// Canonical Stage 8 prover input (clear path, deterministic-structure slice).
///
/// Stage 8 consumes Stage 6 (increment claim-reduction openings, advice
/// cycle-phase openings) and Stage 7 (hamming-weight RA-family reduced openings
/// and the common opening point) to build the final batched PCS opening.
#[derive(Clone, Copy, Debug)]
pub struct Stage8ProverInput<'a, F: Field> {
    pub config: &'a Stage8ProverConfig,
    pub checked: &'a CheckedInputs,
    pub stage6: &'a Stage6ClearOutput<F>,
    pub stage7: &'a Stage7ClearOutput<F>,
}

impl<'a, F: Field> Stage8ProverInput<'a, F> {
    pub const fn new(
        config: &'a Stage8ProverConfig,
        checked: &'a CheckedInputs,
        stage6: &'a Stage6ClearOutput<F>,
        stage7: &'a Stage7ClearOutput<F>,
    ) -> Self {
        Self {
            config,
            checked,
            stage6,
            stage7,
        }
    }
}
