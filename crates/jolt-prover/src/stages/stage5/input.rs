#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::FieldInlineConfig;
use jolt_claims::protocols::jolt::formulas::instruction::InstructionReadRafDimensions;
use jolt_field::Field;
use jolt_verifier::stages::{stage2::Stage2ClearOutput, stage4::Stage4ClearOutput};
use jolt_verifier::CheckedInputs;
#[cfg(not(feature = "field-inline"))]
use std::marker::PhantomData;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage5ProverConfig {
    pub log_t: usize,
    pub log_k: usize,
    pub instruction_read_raf_dimensions: InstructionReadRafDimensions,
    #[cfg(feature = "field-inline")]
    pub field_inline: FieldInlineConfig,
}

impl Stage5ProverConfig {
    pub const fn new(
        log_t: usize,
        log_k: usize,
        instruction_read_raf_dimensions: InstructionReadRafDimensions,
    ) -> Self {
        Self {
            log_t,
            log_k,
            instruction_read_raf_dimensions,
            #[cfg(feature = "field-inline")]
            field_inline: FieldInlineConfig::native_v1(),
        }
    }
}

/// Canonical Stage 5 prover input (transparent path).
///
/// Bridges Stage 2/4 clear outputs into the instruction read-RAF, RAM-RA
/// reduction, and register value-evaluation batched sumcheck. Self-contained:
/// the prefix is derived purely from prior-stage outputs and the transcript.
#[derive(Clone, Debug)]
pub struct Stage5ProverInput<'a, F: Field, W, FI = ()> {
    pub config: Stage5ProverConfig,
    pub checked: &'a CheckedInputs,
    pub stage2: &'a Stage2ClearOutput<F>,
    pub stage4: &'a Stage4ClearOutput<F>,
    pub witness: &'a W,
    #[cfg(feature = "field-inline")]
    pub field_inline_witness: &'a FI,
    #[cfg(not(feature = "field-inline"))]
    _field_inline: PhantomData<FI>,
}

#[cfg(not(feature = "field-inline"))]
impl<'a, F: Field, W> Stage5ProverInput<'a, F, W> {
    pub const fn new(
        config: Stage5ProverConfig,
        checked: &'a CheckedInputs,
        stage2: &'a Stage2ClearOutput<F>,
        stage4: &'a Stage4ClearOutput<F>,
        witness: &'a W,
    ) -> Self {
        Self {
            config,
            checked,
            stage2,
            stage4,
            witness,
            _field_inline: PhantomData,
        }
    }
}

#[cfg(feature = "field-inline")]
impl<'a, F: Field, W, FI> Stage5ProverInput<'a, F, W, FI> {
    pub const fn new(
        config: Stage5ProverConfig,
        checked: &'a CheckedInputs,
        stage2: &'a Stage2ClearOutput<F>,
        stage4: &'a Stage4ClearOutput<F>,
        witness: &'a W,
        field_inline_witness: &'a FI,
    ) -> Self {
        Self {
            config,
            checked,
            stage2,
            stage4,
            witness,
            field_inline_witness,
        }
    }
}
