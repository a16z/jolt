//! Dory-assist verifier stages.

pub mod stage1;
pub mod stage2;
pub mod stage3;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistStageProofs {
    pub stage1: stage1::Stage1Proof,
    pub stage2: stage2::Stage2Proof,
    pub stage3: stage3::Stage3Proof,
}
