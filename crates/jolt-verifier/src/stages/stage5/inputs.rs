//! Typed transparent-mode inputs consumed by stage 5.

use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::stages::{stage2::Stage2Output, stage4::Stage4Output};

#[derive(Clone, Copy)]
pub struct Deps<'a, F: Field> {
    pub stage2: &'a Stage2Output<F>,
    pub stage4: &'a Stage4Output<F>,
}

pub fn deps<'a, F: Field>(stage2: &'a Stage2Output<F>, stage4: &'a Stage4Output<F>) -> Deps<'a, F> {
    Deps { stage2, stage4 }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Stage5Claims<F: Field> {
    pub instruction_read_raf: InstructionReadRafOutputOpeningClaims<F>,
    pub ram_ra_claim_reduction: RamRaClaimReductionOutputOpeningClaims<F>,
    pub registers_val_evaluation: RegistersValEvaluationOutputOpeningClaims<F>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct InstructionReadRafOutputOpeningClaims<F: Field> {
    pub lookup_table_flags: Vec<F>,
    pub instruction_ra: Vec<F>,
    pub instruction_raf_flag: F,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct RamRaClaimReductionOutputOpeningClaims<F: Field> {
    pub ram_ra: F,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct RegistersValEvaluationOutputOpeningClaims<F: Field> {
    pub rd_inc: F,
    pub rd_wa: F,
}
