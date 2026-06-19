pub mod inputs;
pub mod outputs;
pub mod ram_ra_claim_reduction;
mod verify;

pub use inputs::{deps, Deps, Stage5Claims};
pub use outputs::{Stage5ClearOutput, Stage5Output, Stage5ZkOutput};
pub use ram_ra_claim_reduction::RamRaClaimReductionRelation;
pub use verify::{
    append_stage5_opening_claims, stage5_dependency_opening_points, stage5_expected_final_claim,
    stage5_expected_output_claims, stage5_input_claims, stage5_instruction_opening_points,
    stage5_instruction_read_raf_dependencies, stage5_output_claim_values,
    stage5_value_opening_points, verify, Stage5DependencyOpeningPointRequest,
    Stage5DependencyOpeningPoints, Stage5ExpectedOutputClaims, Stage5ExpectedOutputRequest,
    Stage5InputClaimRequest, Stage5InputClaims, Stage5InstructionOpeningPoints,
    Stage5InstructionReadRafDependencyRequest, Stage5ValueOpeningPointRequest,
    Stage5ValueOpeningPoints,
};
