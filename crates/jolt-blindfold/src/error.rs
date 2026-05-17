use jolt_r1cs::ClaimLoweringError;
use jolt_sumcheck::SumcheckR1csError;
use thiserror::Error as ThisError;

#[derive(Clone, Debug, ThisError, PartialEq, Eq)]
pub enum Error {
    #[error(transparent)]
    Layout(#[from] LayoutError),
    #[error(transparent)]
    Claim(#[from] ClaimLoweringError),
    #[error("stage {stage_index}: {source}")]
    Sumcheck {
        stage_index: usize,
        source: SumcheckR1csError,
    },
    #[error("layout has {layout_stages} stages but instance has {instance_stages}")]
    LayoutStageCountMismatch {
        instance_stages: usize,
        layout_stages: usize,
    },
}

#[derive(Clone, Debug, ThisError, PartialEq, Eq)]
pub enum LayoutError {
    #[error("instance has {instance_stages} stages but committed inputs have {input_stages}")]
    StageCountMismatch {
        instance_stages: usize,
        input_stages: usize,
    },
    #[error("stage {stage_index}: {source}")]
    Sumcheck {
        stage_index: usize,
        source: SumcheckR1csError,
    },
}
