//! Typed clear-mode inputs consumed by stage 4.

use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::stages::{stage1::Stage1ClearOutput, stage2::Stage2ClearOutput, stage3::Stage3Output};

#[derive(Clone, Copy)]
pub struct Deps<'a, F: Field> {
    pub stage1: &'a Stage1ClearOutput<F>,
    pub stage2: &'a Stage2ClearOutput<F>,
    pub stage3: &'a Stage3Output<F>,
}

pub fn deps<'a, F: Field>(
    stage1: &'a Stage1ClearOutput<F>,
    stage2: &'a Stage2ClearOutput<F>,
    stage3: &'a Stage3Output<F>,
) -> Deps<'a, F> {
    Deps {
        stage1,
        stage2,
        stage3,
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Stage4Claims<F: Field> {
    pub advice: RamValCheckAdviceOpeningClaims<F>,
    pub registers_read_write: RegistersReadWriteOutputOpeningClaims<F>,
    pub ram_val_check: RamValCheckOutputOpeningClaims<F>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct RamValCheckAdviceOpeningClaims<F: Field> {
    pub untrusted: Option<F>,
    pub trusted: Option<F>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct RegistersReadWriteOutputOpeningClaims<F: Field> {
    pub registers_val: F,
    pub rs1_ra: F,
    pub rs2_ra: F,
    pub rd_wa: F,
    pub rd_inc: F,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct RamValCheckOutputOpeningClaims<F: Field> {
    pub ram_ra: F,
    pub ram_inc: F,
}
