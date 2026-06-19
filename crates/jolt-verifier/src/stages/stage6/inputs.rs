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
pub use super::booleanity::BooleanityOutputClaims;
pub use super::bytecode_read_raf::BytecodeReadRafOutputClaims;
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

/// The stage 6 produced opening claims, generic over the cell (`F` on the wire,
/// `Vec<F>` for derived points, `OpeningClaim<F>` (point + value) on the clear
/// path). The per-relation members are each `#[derive(OutputClaims)]` structs;
/// the address-phase and committed-reduction members are hand-written but follow
/// the same cell convention.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
pub struct Stage6OutputClaims<C> {
    pub address_phase: Stage6AddressPhaseClaims<C>,
    pub bytecode_read_raf: BytecodeReadRafOutputClaims<C>,
    pub booleanity: BooleanityOutputClaims<C>,
    pub ram_hamming_booleanity: RamHammingBooleanityOutputClaims<C>,
    pub ram_ra_virtualization: RamRaVirtualizationOutputClaims<C>,
    pub instruction_ra_virtualization: InstructionRaVirtualizationOutputClaims<C>,
    pub inc_claim_reduction: IncClaimReductionOutputClaims<C>,
    pub advice_cycle_phase: Stage6AdviceCyclePhaseClaims<C>,
    /// Committed program mode only.
    pub bytecode_claim_reduction: Option<BytecodeCyclePhaseOutputClaims<C>>,
    /// Committed program mode only.
    pub program_image_claim_reduction: Option<ProgramImageCyclePhaseOutputClaim<C>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
pub struct Stage6AddressPhaseClaims<C> {
    pub bytecode_read_raf: C,
    pub booleanity: C,
    /// `BytecodeValStage(s)` openings staged at the address-phase point;
    /// present only in committed program mode.
    pub bytecode_val_stages: Option<[C; NUM_BYTECODE_VAL_STAGES]>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
pub struct Stage6AdviceCyclePhaseClaims<C> {
    pub trusted: Option<AdviceCyclePhaseOutputClaim<C>>,
    pub untrusted: Option<AdviceCyclePhaseOutputClaim<C>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
pub struct AdviceCyclePhaseOutputClaim<C> {
    pub opening_claim: C,
}

/// Openings cached when the committed-bytecode claim reduction's cycle phase
/// completes: the intermediate claim when address-phase rounds remain, or the
/// per-chunk `BytecodeChunk(i)` claims when the reduction finishes in the
/// cycle phase.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
pub enum BytecodeCyclePhaseOutputClaims<C> {
    Intermediate(C),
    Chunks(Vec<C>),
}

/// Opening cached when the program-image claim reduction's cycle phase
/// completes (the intermediate or final `ProgramImageInit` claim).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
pub struct ProgramImageCyclePhaseOutputClaim<C> {
    pub opening_claim: C,
}
