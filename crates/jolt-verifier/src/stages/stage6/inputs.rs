//! Typed inputs consumed by stage 6.

use jolt_field::Field;
use serde::{Deserialize, Serialize};

pub use super::inputs_a::Stage6AddressPhaseClaims;
pub use super::inputs_b::{
    AdviceCyclePhaseOutputClaim, BooleanityOutputOpeningClaims, BytecodeCyclePhaseOutputClaims,
    BytecodeReadRafOutputOpeningClaims, IncClaimReductionOutputOpeningClaims,
    InstructionRaVirtualizationOutputOpeningClaims, ProgramImageCyclePhaseOutputClaim,
    RamHammingBooleanityOutputOpeningClaims, RamRaVirtualizationOutputOpeningClaims,
    Stage6AdviceCyclePhaseClaims, UnsignedIncClaimReductionOutputOpeningClaims,
};
use crate::stages::{
    stage1::{Stage1ClearOutput, Stage1Output},
    stage2::{Stage2ClearOutput, Stage2Output},
    stage3::{Stage3ClearOutput, Stage3Output},
    stage4::{Stage4ClearOutput, Stage4Output},
    stage5::{Stage5ClearOutput, Stage5Output, Stage5ZkOutput},
    stage5_increment::Stage5IncrementClearOutput,
};

#[derive(Clone, Copy)]
pub enum Deps<'a, F: Field, C> {
    Clear {
        stage1: &'a Stage1ClearOutput<F>,
        stage2: &'a Stage2ClearOutput<F>,
        stage3: &'a Stage3ClearOutput<F>,
        stage4: &'a Stage4ClearOutput<F>,
        stage5: &'a Stage5ClearOutput<F>,
        stage5_increment: Option<&'a Stage5IncrementClearOutput<F>>,
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
    stage5_increment: Option<&'a Stage5IncrementClearOutput<F>>,
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
            stage5_increment,
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
#[serde(deny_unknown_fields)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct Stage6Claims<F: Field> {
    pub address_phase: Stage6AddressPhaseClaims<F>,
    pub bytecode_read_raf: BytecodeReadRafOutputOpeningClaims<F>,
    pub booleanity: BooleanityOutputOpeningClaims<F>,
    pub ram_hamming_booleanity: RamHammingBooleanityOutputOpeningClaims<F>,
    pub ram_ra_virtualization: RamRaVirtualizationOutputOpeningClaims<F>,
    pub instruction_ra_virtualization: InstructionRaVirtualizationOutputOpeningClaims<F>,
    pub inc_claim_reduction: IncClaimReductionOutputOpeningClaims<F>,
    /// Lattice PCS mode only, once Stage 5 increment virtualization is present.
    #[serde(default)]
    pub unsigned_inc_claim_reduction: Option<UnsignedIncClaimReductionOutputOpeningClaims<F>>,
    #[cfg(feature = "field-inline")]
    pub field_inline: FieldInlineStage6Claims<F>,
    pub advice_cycle_phase: Stage6AdviceCyclePhaseClaims<F>,
    /// Committed program mode only.
    pub bytecode_claim_reduction: Option<BytecodeCyclePhaseOutputClaims<F>>,
    /// Committed program mode only.
    pub program_image_claim_reduction: Option<ProgramImageCyclePhaseOutputClaim<F>>,
}

#[cfg(feature = "field-inline")]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct FieldInlineStage6Claims<F: Field> {
    pub field_registers_inc_claim_reduction: FieldRegistersIncClaimReductionOutputOpeningClaims<F>,
}

#[cfg(feature = "field-inline")]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct FieldRegistersIncClaimReductionOutputOpeningClaims<F: Field> {
    pub field_rd_inc: F,
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::expect_used,
        reason = "test setup should fail loudly when serde shape changes"
    )]

    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};

    #[test]
    fn stage6_claims_rejects_unknown_fields() {
        let zero = Fr::from_u64(0);
        let claims = Stage6Claims {
            address_phase: Stage6AddressPhaseClaims {
                bytecode_read_raf: zero,
                booleanity: zero,
                bytecode_val_stages: None,
            },
            bytecode_read_raf: BytecodeReadRafOutputOpeningClaims {
                bytecode_ra: vec![zero],
            },
            booleanity: BooleanityOutputOpeningClaims {
                instruction_ra: vec![zero],
                bytecode_ra: vec![zero],
                ram_ra: vec![zero],
            },
            ram_hamming_booleanity: RamHammingBooleanityOutputOpeningClaims {
                ram_hamming_weight: zero,
            },
            ram_ra_virtualization: RamRaVirtualizationOutputOpeningClaims { ram_ra: vec![zero] },
            instruction_ra_virtualization: InstructionRaVirtualizationOutputOpeningClaims {
                committed_instruction_ra: vec![zero],
            },
            inc_claim_reduction: IncClaimReductionOutputOpeningClaims {
                ram_inc: zero,
                rd_inc: zero,
            },
            unsigned_inc_claim_reduction: None,
            #[cfg(feature = "field-inline")]
            field_inline: FieldInlineStage6Claims {
                field_registers_inc_claim_reduction:
                    FieldRegistersIncClaimReductionOutputOpeningClaims { field_rd_inc: zero },
            },
            advice_cycle_phase: Stage6AdviceCyclePhaseClaims {
                trusted: None,
                untrusted: None,
            },
            bytecode_claim_reduction: None,
            program_image_claim_reduction: None,
        };

        let mut value = serde_json::to_value(claims).expect("claims should serialize");
        let _ = value
            .as_object_mut()
            .expect("claims should serialize to a map")
            .insert("extra_stage6_claim".to_string(), serde_json::json!(null));

        let error = serde_json::from_value::<Stage6Claims<Fr>>(value)
            .expect_err("unknown Stage 6 fields should be rejected");
        assert!(
            error
                .to_string()
                .contains("unknown field `extra_stage6_claim`"),
            "{error}"
        );
    }
}
