//! Typed inputs consumed by stage 6.

use jolt_claims::protocols::jolt::formulas::claim_reductions::bytecode::NUM_BYTECODE_VAL_STAGES;
use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::stages::{
    stage1::{Stage1ClearOutput, Stage1Output},
    stage2::{Stage2ClearOutput, Stage2Output},
    stage3::{Stage3ClearOutput, Stage3Output},
    stage4::{Stage4ClearOutput, Stage4Output},
    stage5::{Stage5ClearOutput, Stage5Output, Stage5ZkOutput},
};

// The per-relation produced-claim structs now live in their relation modules
// (cell-generic, `#[derive(OutputClaims)]`); re-export them so `stage6::inputs::*`
// consumers and the `Stage6OutputClaims` aggregate keep resolving them here.
pub use super::inc_claim_reduction::IncClaimReductionOutputClaims;
pub use super::instruction_ra_virtualization::InstructionRaVirtualizationOutputClaims;
pub use super::ram_hamming_booleanity::RamHammingBooleanityOutputClaims;
pub use super::ram_ra_virtualization::RamRaVirtualizationOutputClaims;

#[derive(Clone, Copy)]
pub enum Deps<'a, F: Field, C> {
    Clear {
        stage1: &'a Stage1ClearOutput<F>,
        stage2: &'a Stage2ClearOutput<F>,
        stage3: &'a Stage3ClearOutput<F>,
        stage4: &'a Stage4ClearOutput<F>,
        stage5: &'a Stage5ClearOutput<F>,
    },
    Zk {
        stage5: &'a Stage5ZkOutput<F, C>,
    },
}

pub fn deps<'a, F: Field, C>(
    stage1: &'a Stage1Output<F, C>,
    stage2: &'a Stage2Output<F, C>,
    stage3: &'a Stage3Output<F, C>,
    stage4: &'a Stage4Output<F, C>,
    stage5: &'a Stage5Output<F, C>,
) -> Result<Deps<'a, F, C>, crate::VerifierError> {
    match (stage1, stage2, stage3, stage4, stage5) {
        (
            Stage1Output::Clear(stage1),
            Stage2Output::Clear(stage2),
            Stage3Output::Clear(stage3),
            Stage4Output::Clear(stage4),
            Stage5Output::Clear(stage5),
        ) => Ok(Deps::Clear {
            stage1,
            stage2,
            stage3,
            stage4,
            stage5,
        }),
        (
            Stage1Output::Zk(_),
            Stage2Output::Zk(_),
            Stage3Output::Zk(_),
            Stage4Output::Zk(_),
            Stage5Output::Zk(stage5),
        ) => Ok(Deps::Zk { stage5 }),
        (_, _, _, _, Stage5Output::Clear(_)) => {
            Err(crate::VerifierError::ExpectedClearProof { field: "stage5" })
        }
        (_, _, _, _, Stage5Output::Zk(_)) => {
            Err(crate::VerifierError::ExpectedCommittedProof { field: "stage5" })
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Stage6OutputClaims<F: Field> {
    pub address_phase: Stage6AddressPhaseClaims<F>,
    pub bytecode_read_raf: BytecodeReadRafOutputClaims<F>,
    pub booleanity: BooleanityOutputClaims<F>,
    pub ram_hamming_booleanity: RamHammingBooleanityOutputClaims<F>,
    pub ram_ra_virtualization: RamRaVirtualizationOutputClaims<F>,
    pub instruction_ra_virtualization: InstructionRaVirtualizationOutputClaims<F>,
    pub inc_claim_reduction: IncClaimReductionOutputClaims<F>,
    pub advice_cycle_phase: Stage6AdviceCyclePhaseClaims<F>,
    /// Committed program mode only.
    pub bytecode_claim_reduction: Option<BytecodeCyclePhaseOutputClaims<F>>,
    /// Committed program mode only.
    pub program_image_claim_reduction: Option<ProgramImageCyclePhaseOutputClaim<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Stage6AddressPhaseClaims<F: Field> {
    pub bytecode_read_raf: F,
    pub booleanity: F,
    /// `BytecodeValStage(s)` openings staged at the address-phase point;
    /// present only in committed program mode.
    pub bytecode_val_stages: Option<[F; NUM_BYTECODE_VAL_STAGES]>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct BytecodeReadRafOutputClaims<F: Field> {
    pub bytecode_ra: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct BooleanityOutputClaims<F: Field> {
    pub instruction_ra: Vec<F>,
    pub bytecode_ra: Vec<F>,
    pub ram_ra: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Stage6AdviceCyclePhaseClaims<F: Field> {
    pub trusted: Option<AdviceCyclePhaseOutputClaim<F>>,
    pub untrusted: Option<AdviceCyclePhaseOutputClaim<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct AdviceCyclePhaseOutputClaim<F: Field> {
    pub opening_claim: F,
}

/// Openings cached when the committed-bytecode claim reduction's cycle phase
/// completes: the intermediate claim when address-phase rounds remain, or the
/// per-chunk `BytecodeChunk(i)` claims when the reduction finishes in the
/// cycle phase.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub enum BytecodeCyclePhaseOutputClaims<F: Field> {
    Intermediate(F),
    Chunks(Vec<F>),
}

/// Opening cached when the program-image claim reduction's cycle phase
/// completes (the intermediate or final `ProgramImageInit` claim).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ProgramImageCyclePhaseOutputClaim<F: Field> {
    pub opening_claim: F,
}
