use jolt_crypto::Commitment;
use jolt_sumcheck::CommittedSumcheckConsistency;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Inputs<F, C> {
    pub stages: Vec<StageInput<F, C>>,
}

impl<F, C> Inputs<F, C> {
    pub fn new(stages: Vec<StageInput<F, C>>) -> Self {
        Self { stages }
    }

    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageInput<F, C> {
    pub consistency: CommittedSumcheckConsistency<F, C>,
}

impl<F, C> StageInput<F, C> {
    pub fn new(consistency: CommittedSumcheckConsistency<F, C>) -> Self {
        Self { consistency }
    }
}

pub type VectorStageInput<F, VC> = StageInput<F, <VC as Commitment>::Output>;
pub type VectorInputs<F, VC> = Inputs<F, <VC as Commitment>::Output>;
