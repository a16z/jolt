#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::FieldInlineConfig;
use jolt_claims::protocols::jolt::JoltReadWriteConfig;
use jolt_field::Field;
use jolt_verifier::stages::{stage2::Stage2ClearOutput, stage3::Stage3ClearOutput};
use jolt_verifier::CheckedInputs;
#[cfg(not(feature = "field-inline"))]
use std::marker::PhantomData;

use super::output::Stage4RamValCheckInitialEvaluation;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage4ProverConfig {
    pub log_t: usize,
    pub log_k: usize,
    pub rw_config: JoltReadWriteConfig,
    #[cfg(feature = "field-inline")]
    pub field_inline: FieldInlineConfig,
}

impl Stage4ProverConfig {
    pub const fn new(log_t: usize, log_k: usize, rw_config: JoltReadWriteConfig) -> Self {
        Self {
            log_t,
            log_k,
            rw_config,
            #[cfg(feature = "field-inline")]
            field_inline: FieldInlineConfig::native_v1(),
        }
    }
}

/// Canonical Stage 4 prover input (transparent path).
///
/// Bridges Stage 2/3 clear outputs into the registers read-write, RAM
/// value-check (and field-inline registers) batched sumcheck. `ram_val_check_init`
/// is the decomposed initial RAM evaluation (public initial RAM plus advice
/// contributions); the prover-side computation of it from preprocessing and the
/// advice witness is a tracked self-containment follow-up.
#[derive(Clone, Debug)]
pub struct Stage4ProverInput<'a, F: Field, W, FI = ()> {
    pub config: Stage4ProverConfig,
    pub checked: &'a CheckedInputs,
    pub stage2: &'a Stage2ClearOutput<F>,
    pub stage3: &'a Stage3ClearOutput<F>,
    pub ram_val_check_init: Stage4RamValCheckInitialEvaluation<F>,
    pub witness: &'a W,
    #[cfg(feature = "field-inline")]
    pub field_inline_witness: &'a FI,
    #[cfg(not(feature = "field-inline"))]
    _field_inline: PhantomData<FI>,
}

#[cfg(not(feature = "field-inline"))]
impl<'a, F: Field, W> Stage4ProverInput<'a, F, W> {
    pub const fn new(
        config: Stage4ProverConfig,
        checked: &'a CheckedInputs,
        stage2: &'a Stage2ClearOutput<F>,
        stage3: &'a Stage3ClearOutput<F>,
        ram_val_check_init: Stage4RamValCheckInitialEvaluation<F>,
        witness: &'a W,
    ) -> Self {
        Self {
            config,
            checked,
            stage2,
            stage3,
            ram_val_check_init,
            witness,
            _field_inline: PhantomData,
        }
    }
}

#[cfg(feature = "field-inline")]
impl<'a, F: Field, W, FI> Stage4ProverInput<'a, F, W, FI> {
    pub const fn new(
        config: Stage4ProverConfig,
        checked: &'a CheckedInputs,
        stage2: &'a Stage2ClearOutput<F>,
        stage3: &'a Stage3ClearOutput<F>,
        ram_val_check_init: Stage4RamValCheckInitialEvaluation<F>,
        witness: &'a W,
        field_inline_witness: &'a FI,
    ) -> Self {
        Self {
            config,
            checked,
            stage2,
            stage3,
            ram_val_check_init,
            witness,
            field_inline_witness,
        }
    }
}
