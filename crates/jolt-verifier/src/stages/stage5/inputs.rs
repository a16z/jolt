//! Typed inputs consumed by stage 5.

use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::stages::{
    stage2::{Stage2Output, Stage2ZkOutput},
    stage4::{Stage4Output, Stage4ZkOutput},
};

use super::instruction_read_raf::InstructionReadRafOutputClaims;
use super::ram_ra_claim_reduction::RamRaClaimReductionOutputClaims;
use super::registers_val_evaluation::RegistersValEvaluationOutputClaims;

#[derive(Clone, Copy)]
pub enum Deps<'a, F: Field, C> {
    Clear {
        stage2: &'a crate::stages::stage2::Stage2ClearOutput<F>,
        stage4: &'a crate::stages::stage4::Stage4ClearOutput<F>,
    },
    Zk {
        stage2: &'a Stage2ZkOutput<F, C>,
        stage4: &'a Stage4ZkOutput<F, C>,
    },
}

pub fn deps<'a, F: Field, C>(
    stage2: &'a Stage2Output<F, C>,
    stage4: &'a Stage4Output<F, C>,
) -> Result<Deps<'a, F, C>, crate::VerifierError> {
    match (stage2, stage4) {
        (Stage2Output::Clear(stage2), Stage4Output::Clear(stage4)) => {
            Ok(Deps::Clear { stage2, stage4 })
        }
        (Stage2Output::Zk(stage2), Stage4Output::Zk(stage4)) => Ok(Deps::Zk { stage2, stage4 }),
        (Stage2Output::Clear(_), Stage4Output::Zk(_)) => {
            Err(crate::VerifierError::ExpectedClearProof { field: "stage4" })
        }
        (Stage2Output::Zk(_), Stage4Output::Clear(_)) => {
            Err(crate::VerifierError::ExpectedCommittedProof { field: "stage4" })
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Stage5Claims<F: Field> {
    pub instruction_read_raf: InstructionReadRafOutputClaims<F>,
    pub ram_ra_claim_reduction: RamRaClaimReductionOutputClaims<F>,
    pub registers_val_evaluation: RegistersValEvaluationOutputClaims<F>,
}
