use jolt_field::Field;
use jolt_verifier::stages::{stage1::Stage1ClearOutput, stage2::Stage2ClearOutput};
use jolt_verifier::CheckedInputs;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage3ProverConfig {
    pub log_t: usize,
}

impl Stage3ProverConfig {
    pub const fn new(log_t: usize) -> Self {
        Self { log_t }
    }
}

/// Canonical Stage 3 prover input.
///
/// Stage 3 bridges Stage 1/2 outputs into the regular batched sumcheck over the
/// Spartan shift, instruction-input virtualization, and register claim-reduction
/// statements. It consumes prior-stage verifier outputs directly so it never
/// re-derives verifier reductions. Stage 3 has no field-inline-specific relation,
/// so the input bundle is identical across feature modes.
#[derive(Clone, Copy, Debug)]
pub struct Stage3ProverInput<'a, F: Field, W> {
    pub config: Stage3ProverConfig,
    pub checked: &'a CheckedInputs,
    pub stage1: &'a Stage1ClearOutput<F>,
    pub stage2: &'a Stage2ClearOutput<F>,
    pub witness: &'a W,
}

impl<'a, F: Field, W> Stage3ProverInput<'a, F, W> {
    pub const fn new(
        config: Stage3ProverConfig,
        checked: &'a CheckedInputs,
        stage1: &'a Stage1ClearOutput<F>,
        stage2: &'a Stage2ClearOutput<F>,
        witness: &'a W,
    ) -> Self {
        Self {
            config,
            checked,
            stage1,
            stage2,
            witness,
        }
    }
}
