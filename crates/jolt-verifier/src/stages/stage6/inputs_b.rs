//! Typed inputs for stage 6b.

use jolt_field::Field;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct BytecodeReadRafOutputOpeningClaims<F: Field> {
    pub bytecode_ra: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct BooleanityOutputOpeningClaims<F: Field> {
    pub instruction_ra: Vec<F>,
    pub bytecode_ra: Vec<F>,
    pub ram_ra: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct RamHammingBooleanityOutputOpeningClaims<F: Field> {
    pub ram_hamming_weight: F,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct RamRaVirtualizationOutputOpeningClaims<F: Field> {
    pub ram_ra: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct InstructionRaVirtualizationOutputOpeningClaims<F: Field> {
    pub committed_instruction_ra: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct IncClaimReductionOutputOpeningClaims<F: Field> {
    pub ram_inc: F,
    pub rd_inc: F,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct FusedIncrementTranslationOutputClaims<F: Field> {
    pub ram_source: F,
    pub magnitude: F,
    pub sign: F,
    pub rd_source: F,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct FusedIncrementSourceLinkOutputClaims<F: Field> {
    pub bytecode_ra: Vec<F>,
    pub store_flag: F,
    pub rd_present: F,
    #[serde(default)]
    pub store_flag_chunks: Vec<F>,
    #[serde(default)]
    pub rd_present_chunks: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct Stage6AdviceCyclePhaseClaims<F: Field> {
    pub trusted: Option<AdviceCyclePhaseOutputClaim<F>>,
    pub untrusted: Option<AdviceCyclePhaseOutputClaim<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct AdviceCyclePhaseOutputClaim<F: Field> {
    pub opening_claim: F,
}

/// Openings cached when the committed-bytecode claim reduction's cycle phase
/// completes: the intermediate claim when address-phase rounds remain, or the
/// per-chunk `BytecodeChunk(i)` claims when the reduction finishes in the
/// cycle phase.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub enum BytecodeCyclePhaseOutputClaims<F: Field> {
    Intermediate(F),
    Chunks(Vec<F>),
}

/// Opening cached when the program-image claim reduction's cycle phase
/// completes (the intermediate or final `ProgramImageInit` claim).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct ProgramImageCyclePhaseOutputClaim<F: Field> {
    pub opening_claim: F,
}
