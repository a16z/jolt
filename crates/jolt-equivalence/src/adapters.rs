//! Thin representation adapters between kernel and generated stage types.

use jolt_field::Fr;
use jolt_kernels::stage4::Stage4ExecutionArtifacts;
use jolt_verifier::{JoltStageExecutionArtifacts, JoltStageProof};

use crate::artifacts::{NamedScalar, StageArtifacts, SumcheckArtifacts};

macro_rules! canonical_stage_artifacts {
    ($stage:literal, $artifacts:expr) => {{
        StageArtifacts {
            stage: $stage.to_owned(),
            sumchecks: $artifacts
                .sumchecks
                .iter()
                .map(|output| SumcheckArtifacts {
                    driver: output.driver.to_owned(),
                    point: output.point.clone(),
                    round_polynomials: output
                        .proof
                        .round_polynomials
                        .iter()
                        .map(|poly| poly.clone().into_coefficients())
                        .collect(),
                    evals: output
                        .evals
                        .iter()
                        .map(|eval| NamedScalar {
                            name: eval.name.to_owned(),
                            value: eval.value,
                        })
                        .collect(),
                })
                .collect(),
            opening_batches: Vec::new(),
        }
    }};
}

macro_rules! define_canonical_stage_adapters {
    ($($function:ident, $stage:literal, $input:ty;)*) => {
        $(
        pub(crate) fn $function(artifacts: &$input) -> StageArtifacts<Fr> {
            canonical_stage_artifacts!($stage, artifacts)
        }
        )*
    };
}

define_canonical_stage_adapters! {
    canonical_stage4_artifacts, "Stage 4", Stage4ExecutionArtifacts<Fr>;
    canonical_generated_stage4_execution_artifacts, "Stage 4", JoltStageExecutionArtifacts;
    canonical_generated_stage5_proof, "Stage 5", JoltStageProof;
    canonical_generated_stage5_execution_artifacts, "Stage 5", JoltStageExecutionArtifacts;
    canonical_generated_stage6_proof, "Stage 6", JoltStageProof;
    canonical_generated_stage6_execution_artifacts, "Stage 6", JoltStageExecutionArtifacts;
    canonical_generated_stage7_proof, "Stage 7", JoltStageProof;
    canonical_generated_stage7_execution_artifacts, "Stage 7", JoltStageExecutionArtifacts;
}
