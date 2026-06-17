use jolt_field::Field;

use crate::spec::BatchedSumcheckSpec;

/// Observable protocol step in a modular proof program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProverStep<F: Field> {
    BatchedSumcheck(BatchedSumcheckSpec<F>),
}

/// One labeled stage in a [`ProverProgram`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage<F: Field> {
    pub label: &'static str,
    pub steps: Vec<ProverStep<F>>,
}

/// Top-level proof schedule: commit / batched sumcheck / open steps in transcript order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProverProgram<F: Field> {
    pub stages: Vec<Stage<F>>,
}

impl<F: Field> ProverProgram<F> {
    pub fn new(stages: Vec<Stage<F>>) -> Self {
        Self { stages }
    }
}
