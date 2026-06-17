//! Typed inputs consumed by stage 7.

use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::stages::{
    stage4::{Stage4ClearOutput, Stage4Output},
    stage6::{Stage6ClearOutput, Stage6Output, Stage6ZkOutput},
};

#[derive(Clone, Copy)]
pub enum Deps<'a, F: Field, C> {
    Clear {
        stage4: &'a Stage4ClearOutput<F>,
        stage6: &'a Stage6ClearOutput<F>,
    },
    Zk {
        stage6: &'a Stage6ZkOutput<F, C>,
    },
}

pub fn deps<'a, F: Field, C>(
    stage4: &'a Stage4Output<F, C>,
    stage6: &'a Stage6Output<F, C>,
) -> Result<Deps<'a, F, C>, crate::VerifierError> {
    match (stage4, stage6) {
        (Stage4Output::Clear(stage4), Stage6Output::Clear(stage6)) => {
            Ok(Deps::Clear { stage4, stage6 })
        }
        (Stage4Output::Zk(_), Stage6Output::Zk(stage6)) => Ok(Deps::Zk { stage6 }),
        (Stage4Output::Clear(_), Stage6Output::Zk(_)) => {
            Err(crate::VerifierError::ExpectedClearProof { field: "stage6" })
        }
        (Stage4Output::Zk(_), Stage6Output::Clear(_)) => {
            Err(crate::VerifierError::ExpectedCommittedProof { field: "stage6" })
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct Stage7Claims<F: Field> {
    pub hamming_weight_claim_reduction: HammingWeightClaimReductionOutputOpeningClaims<F>,
    pub advice_address_phase: Stage7AdviceAddressPhaseClaims<F>,
    /// Final `BytecodeChunk(i)` claims from the committed-bytecode reduction's
    /// address phase; present only when that phase runs.
    pub bytecode_address_phase: Option<BytecodeAddressPhaseOutputClaims<F>>,
    /// Final `ProgramImageInit` claim from the program-image reduction's
    /// address phase; present only when that phase runs.
    pub program_image_address_phase: Option<ProgramImageAddressPhaseOutputClaim<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct HammingWeightClaimReductionOutputOpeningClaims<F: Field> {
    pub instruction_ra: Vec<F>,
    pub bytecode_ra: Vec<F>,
    pub ram_ra: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct Stage7AdviceAddressPhaseClaims<F: Field> {
    pub trusted: Option<AdviceAddressPhaseOutputClaim<F>>,
    pub untrusted: Option<AdviceAddressPhaseOutputClaim<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct AdviceAddressPhaseOutputClaim<F: Field> {
    pub opening_claim: F,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct BytecodeAddressPhaseOutputClaims<F: Field> {
    pub chunks: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct ProgramImageAddressPhaseOutputClaim<F: Field> {
    pub opening_claim: F,
}
