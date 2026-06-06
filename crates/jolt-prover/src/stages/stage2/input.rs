use jolt_claims::protocols::jolt::JoltReadWriteConfig;
use jolt_field::Field;
use jolt_verifier::stages::stage1::Stage1ClearOutput;
use jolt_verifier::CheckedInputs;
#[cfg(not(feature = "field-inline"))]
use std::marker::PhantomData;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2ProverConfig {
    pub log_t: usize,
}

impl Stage2ProverConfig {
    pub const fn new(log_t: usize) -> Self {
        Self { log_t }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage2BatchProverConfig {
    pub log_t: usize,
    pub log_k: usize,
    pub rw_config: JoltReadWriteConfig,
}

impl Stage2BatchProverConfig {
    pub const fn new(log_t: usize, log_k: usize, rw_config: JoltReadWriteConfig) -> Self {
        Self {
            log_t,
            log_k,
            rw_config,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2ProductUniSkipInput<F: Field> {
    pub tau_low: Vec<F>,
    pub product: F,
    pub should_branch: F,
    pub should_jump: F,
    #[cfg(feature = "field-inline")]
    pub field_product: F,
    #[cfg(feature = "field-inline")]
    pub field_inv_product: F,
}

impl<F: Field> Stage2ProductUniSkipInput<F> {
    pub fn from_stage1(stage1: &Stage1ClearOutput<F>) -> Self {
        let mut tau_low = stage1.public.remainder_challenges[1..].to_vec();
        tau_low.reverse();
        Self {
            tau_low,
            product: stage1.outer.product,
            should_branch: stage1.outer.should_branch,
            should_jump: stage1.outer.should_jump,
            #[cfg(feature = "field-inline")]
            field_product: stage1.field_inline.field_product,
            #[cfg(feature = "field-inline")]
            field_inv_product: stage1.field_inline.field_inv_product,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Stage2ProverInput<'a, F: Field, W, FI = ()> {
    pub config: Stage2BatchProverConfig,
    pub checked: &'a CheckedInputs,
    pub stage1: &'a Stage1ClearOutput<F>,
    pub witness: &'a W,
    #[cfg(feature = "field-inline")]
    pub field_inline_witness: &'a FI,
    #[cfg(not(feature = "field-inline"))]
    _field_inline: PhantomData<FI>,
}

#[cfg(not(feature = "field-inline"))]
impl<'a, F: Field, W> Stage2ProverInput<'a, F, W> {
    pub const fn new(
        config: Stage2BatchProverConfig,
        checked: &'a CheckedInputs,
        stage1: &'a Stage1ClearOutput<F>,
        witness: &'a W,
    ) -> Self {
        Self {
            config,
            checked,
            stage1,
            witness,
            _field_inline: PhantomData,
        }
    }
}

#[cfg(feature = "field-inline")]
impl<'a, F: Field, W, FI> Stage2ProverInput<'a, F, W, FI> {
    pub const fn new(
        config: Stage2BatchProverConfig,
        checked: &'a CheckedInputs,
        stage1: &'a Stage1ClearOutput<F>,
        witness: &'a W,
        field_inline_witness: &'a FI,
    ) -> Self {
        Self {
            config,
            checked,
            stage1,
            witness,
            field_inline_witness,
        }
    }
}
