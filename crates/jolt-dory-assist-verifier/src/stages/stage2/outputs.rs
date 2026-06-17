//! Typed outputs produced by stage 2 verification.

use jolt_claims::protocols::dory_assist::{
    formulas::dory_reduce::DoryReducePublicFoldConstraint, DoryAssistCopyConstraint,
};
use jolt_field::Fq;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2Output {
    pub relation_count: u32,
    pub copy_constraints: Vec<Stage2CopyConstraintOutput>,
    pub dory_reduce_public_folds: Vec<Stage2DoryReducePublicFoldOutput>,
    pub challenge: Fq,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2CopyConstraintOutput {
    pub constraint: DoryAssistCopyConstraint,
    pub source_value: Fq,
    pub target_value: Fq,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2DoryReducePublicFoldOutput {
    pub constraint: DoryReducePublicFoldConstraint,
    pub expected_value: Fq,
    pub target_value: Fq,
}
