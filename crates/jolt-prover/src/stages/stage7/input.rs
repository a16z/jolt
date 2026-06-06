use jolt_claims::protocols::jolt::{
    formulas::claim_reductions::hamming_weight::HammingWeightClaimReductionDimensions,
    AdviceClaimReductionLayout,
};

use {
    jolt_field::Field,
    jolt_verifier::{
        stages::{stage4::Stage4ClearOutput, stage6::Stage6ClearOutput},
        CheckedInputs,
    },
};

/// Canonical Stage 7 prover configuration.
///
/// Carries the verifier-equivalent dimensions and advice layouts needed to build
/// the hamming-weight claim reduction and (optional) advice address-phase
/// statements, mirroring `jolt-verifier/src/stages/stage7/verify.rs`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage7ProverConfig {
    pub log_t: usize,
    pub hamming_dimensions: HammingWeightClaimReductionDimensions,
    pub trusted_advice_layout: Option<AdviceClaimReductionLayout>,
    pub untrusted_advice_layout: Option<AdviceClaimReductionLayout>,
}

impl Stage7ProverConfig {
    pub const fn new(
        log_t: usize,
        hamming_dimensions: HammingWeightClaimReductionDimensions,
        trusted_advice_layout: Option<AdviceClaimReductionLayout>,
        untrusted_advice_layout: Option<AdviceClaimReductionLayout>,
    ) -> Self {
        Self {
            log_t,
            hamming_dimensions,
            trusted_advice_layout,
            untrusted_advice_layout,
        }
    }
}

/// Canonical Stage 7 prover input (transparent/clear path).
///
/// Bridges Stage 4 (clear RAM value-check advice contributions, used for advice
/// address-phase final scaling) and Stage 6 (hamming-weight RA-family input
/// claims and advice cycle-phase output claims) into the Stage 7 batched
/// sumcheck. Self-contained: the prefix is derived purely from prior-stage
/// outputs and the transcript.
#[derive(Clone, Copy, Debug)]
pub struct Stage7ProverInput<'a, F: Field, W> {
    pub config: &'a Stage7ProverConfig,
    pub checked: &'a CheckedInputs,
    pub stage4: &'a Stage4ClearOutput<F>,
    pub stage6: &'a Stage6ClearOutput<F>,
    pub witness: &'a W,
}

impl<'a, F: Field, W> Stage7ProverInput<'a, F, W> {
    pub const fn new(
        config: &'a Stage7ProverConfig,
        checked: &'a CheckedInputs,
        stage4: &'a Stage4ClearOutput<F>,
        stage6: &'a Stage6ClearOutput<F>,
        witness: &'a W,
    ) -> Self {
        Self {
            config,
            checked,
            stage4,
            stage6,
            witness,
        }
    }
}
