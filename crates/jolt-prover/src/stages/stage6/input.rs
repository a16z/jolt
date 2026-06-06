#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::FieldInlineConfig;
use jolt_claims::protocols::jolt::{
    formulas::{
        booleanity::BooleanityDimensions, bytecode::BytecodeReadRafDimensions,
        dimensions::TraceDimensions, instruction::InstructionRaVirtualizationDimensions,
        ram::RamRaVirtualizationDimensions,
    },
    AdviceClaimReductionLayout,
};
use jolt_riscv::JoltInstructionRow;
#[cfg(not(feature = "field-inline"))]
use std::marker::PhantomData;
use {
    jolt_field::Field,
    jolt_verifier::{
        stages::{
            stage1::Stage1ClearOutput, stage2::Stage2ClearOutput, stage3::Stage3ClearOutput,
            stage4::Stage4ClearOutput, stage5::Stage5ClearOutput,
        },
        CheckedInputs,
    },
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6BytecodeContext {
    pub rows: Vec<JoltInstructionRow>,
    pub entry_bytecode_index: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6ProverConfig {
    pub log_t: usize,
    pub log_k: usize,
    pub committed_chunk_bits: usize,
    pub bytecode_read_raf_dimensions: BytecodeReadRafDimensions,
    pub booleanity_dimensions: BooleanityDimensions,
    pub ram_ra_virtualization_dimensions: RamRaVirtualizationDimensions,
    pub instruction_ra_virtualization_dimensions: InstructionRaVirtualizationDimensions,
    pub trusted_advice_layout: Option<AdviceClaimReductionLayout>,
    pub untrusted_advice_layout: Option<AdviceClaimReductionLayout>,
    pub bytecode_context: Option<Stage6BytecodeContext>,
    #[cfg(feature = "field-inline")]
    pub field_inline: FieldInlineConfig,
}

impl Stage6ProverConfig {
    #[expect(
        clippy::too_many_arguments,
        reason = "Stage 6 dimensions are derived by distinct protocol subsystems."
    )]
    pub const fn new(
        log_t: usize,
        log_k: usize,
        committed_chunk_bits: usize,
        bytecode_read_raf_dimensions: BytecodeReadRafDimensions,
        booleanity_dimensions: BooleanityDimensions,
        ram_ra_virtualization_dimensions: RamRaVirtualizationDimensions,
        instruction_ra_virtualization_dimensions: InstructionRaVirtualizationDimensions,
        trusted_advice_layout: Option<AdviceClaimReductionLayout>,
        untrusted_advice_layout: Option<AdviceClaimReductionLayout>,
    ) -> Self {
        Self {
            log_t,
            log_k,
            committed_chunk_bits,
            bytecode_read_raf_dimensions,
            booleanity_dimensions,
            ram_ra_virtualization_dimensions,
            instruction_ra_virtualization_dimensions,
            trusted_advice_layout,
            untrusted_advice_layout,
            bytecode_context: None,
            #[cfg(feature = "field-inline")]
            field_inline: FieldInlineConfig::native_v1(),
        }
    }

    pub fn with_bytecode_context(
        mut self,
        rows: Vec<JoltInstructionRow>,
        entry_bytecode_index: usize,
    ) -> Self {
        self.bytecode_context = Some(Stage6BytecodeContext {
            rows,
            entry_bytecode_index,
        });
        self
    }

    pub const fn trace_dimensions(&self) -> TraceDimensions {
        TraceDimensions::new(self.log_t)
    }
}

/// Canonical Stage 6 prover input (transparent path).
///
/// Bridges Stage 1–5 clear outputs into the bytecode read-RAF, booleanity,
/// RAM-Hamming booleanity, RAM/instruction RA-virtualization, increment
/// claim-reduction, and advice cycle-phase batched sumcheck.
#[derive(Clone, Debug)]
pub struct Stage6ProverInput<'a, F: Field, W, FI = ()> {
    pub config: &'a Stage6ProverConfig,
    pub checked: &'a CheckedInputs,
    pub stage1: &'a Stage1ClearOutput<F>,
    pub stage2: &'a Stage2ClearOutput<F>,
    pub stage3: &'a Stage3ClearOutput<F>,
    pub stage4: &'a Stage4ClearOutput<F>,
    pub stage5: &'a Stage5ClearOutput<F>,
    pub witness: &'a W,
    #[cfg(feature = "field-inline")]
    pub field_inline_witness: &'a FI,
    #[cfg(not(feature = "field-inline"))]
    _field_inline: PhantomData<FI>,
}

#[cfg(not(feature = "field-inline"))]
impl<'a, F: Field, W> Stage6ProverInput<'a, F, W> {
    #[expect(
        clippy::too_many_arguments,
        reason = "Stage 6 consumes every prior clear stage output."
    )]
    pub const fn new(
        config: &'a Stage6ProverConfig,
        checked: &'a CheckedInputs,
        stage1: &'a Stage1ClearOutput<F>,
        stage2: &'a Stage2ClearOutput<F>,
        stage3: &'a Stage3ClearOutput<F>,
        stage4: &'a Stage4ClearOutput<F>,
        stage5: &'a Stage5ClearOutput<F>,
        witness: &'a W,
    ) -> Self {
        Self {
            config,
            checked,
            stage1,
            stage2,
            stage3,
            stage4,
            stage5,
            witness,
            _field_inline: PhantomData,
        }
    }
}

#[cfg(feature = "field-inline")]
impl<'a, F: Field, W, FI> Stage6ProverInput<'a, F, W, FI> {
    #[expect(
        clippy::too_many_arguments,
        reason = "Stage 6 consumes every prior clear stage output."
    )]
    pub const fn new(
        config: &'a Stage6ProverConfig,
        checked: &'a CheckedInputs,
        stage1: &'a Stage1ClearOutput<F>,
        stage2: &'a Stage2ClearOutput<F>,
        stage3: &'a Stage3ClearOutput<F>,
        stage4: &'a Stage4ClearOutput<F>,
        stage5: &'a Stage5ClearOutput<F>,
        witness: &'a W,
        field_inline_witness: &'a FI,
    ) -> Self {
        Self {
            config,
            checked,
            stage1,
            stage2,
            stage3,
            stage4,
            stage5,
            witness,
            field_inline_witness,
        }
    }
}
