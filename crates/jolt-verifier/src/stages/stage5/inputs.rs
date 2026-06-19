//! Typed inputs consumed by stage 5.

use jolt_field::Field;
use jolt_verifier_derive::{InputClaims, OutputClaims};
use serde::{Deserialize, Serialize};

use crate::stages::{
    relations::OpeningClaim,
    stage2::{Stage2ClearOutput, Stage2Output, Stage2ZkOutput},
    stage4::{Stage4ClearOutput, Stage4Output, Stage4ZkOutput},
};

#[derive(Clone, Copy)]
pub enum Deps<'a, F: Field, C> {
    Clear {
        stage2: &'a Stage2ClearOutput<F>,
        stage4: &'a Stage4ClearOutput<F>,
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
    pub instruction_read_raf: InstructionReadRafOutputOpeningClaims<F>,
    pub ram_ra_claim_reduction: RamRaClaimReductionOutputOpeningClaims<F>,
    pub registers_val_evaluation: RegistersValEvaluationOutputOpeningClaims<F>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(InstructionReadRaf)]
pub struct InstructionReadRafOutputOpeningClaims<C> {
    #[opening(LookupTableFlag)]
    pub lookup_table_flags: Vec<C>,
    #[opening(InstructionRa)]
    pub instruction_ra: Vec<C>,
    #[opening(InstructionRafFlag)]
    pub instruction_raf_flag: C,
}

/// Consumed instruction-lookup openings (the reduced lookup output + left/right
/// operands), wired from the upstream instruction claim-reduction.
#[derive(Clone, Debug, InputClaims)]
pub struct InstructionReadRafInputs<C> {
    #[opening(LookupOutput, from = InstructionClaimReduction)]
    pub lookup_output: C,
    #[opening(LeftLookupOperand, from = InstructionClaimReduction)]
    pub left_lookup_operand: C,
    #[opening(RightLookupOperand, from = InstructionClaimReduction)]
    pub right_lookup_operand: C,
}

impl<F: Field> InstructionReadRafInputs<OpeningClaim<F>> {
    /// Wire the consumed openings from the upstream instruction claim-reduction
    /// (stage 2), applying the lookup-output fallback to the product remainder.
    /// All three share the claim-reduction opening point.
    pub fn from_clear(stage2: &Stage2ClearOutput<F>) -> Self {
        let reduction = &stage2.output_claims.instruction_claim_reduction;
        let lookup_output = reduction
            .lookup_output
            .unwrap_or(stage2.output_claims.product_remainder.lookup_output);
        let point = stage2
            .batch
            .instruction_claim_reduction
            .opening_point
            .clone();
        Self {
            lookup_output: OpeningClaim {
                point: point.clone(),
                value: lookup_output,
            },
            left_lookup_operand: OpeningClaim {
                point: point.clone(),
                value: reduction.left_lookup_operand,
            },
            right_lookup_operand: OpeningClaim {
                point,
                value: reduction.right_lookup_operand,
            },
        }
    }
}

/// Produced RAM-RA reduced opening, generic over the cell (`F` on the wire,
/// `Vec<F>` for ZK points, `OpeningClaim<F>` once located on the clear path).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(RamRaClaimReduction)]
pub struct RamRaClaimReductionOutputOpeningClaims<C> {
    #[opening(RamRa)]
    pub ram_ra: C,
}

/// Consumed RAM-RA openings reduced by the `RamRaClaimReduction` sumcheck, wired
/// from the upstream RAF-evaluation, read-write-checking, and val-check
/// relations. Generic over the cell (`OpeningClaim<F>` on the clear path,
/// `Vec<F>` for ZK points).
#[derive(Clone, Debug, InputClaims)]
pub struct RamRaClaimReductionInputs<C> {
    #[opening(RamRa, from = RamRafEvaluation)]
    pub raf: C,
    #[opening(RamRa, from = RamReadWriteChecking)]
    pub read_write: C,
    #[opening(RamRa, from = RamValCheck)]
    pub val_check: C,
}

impl<F: Field> RamRaClaimReductionInputs<OpeningClaim<F>> {
    /// Wire this relation's consumed openings from the upstream clear outputs:
    /// the RAF-evaluation and read-write openings (stage 2) and the val-check
    /// opening (stage 4), each as a located `(point, value)`.
    pub fn from_clear(stage2: &Stage2ClearOutput<F>, stage4: &Stage4ClearOutput<F>) -> Self {
        Self {
            raf: OpeningClaim {
                point: stage2.batch.ram_raf_evaluation.opening_point.clone(),
                value: stage2.output_claims.ram_raf_evaluation,
            },
            read_write: OpeningClaim {
                point: stage2.batch.ram_read_write.opening_point.clone(),
                value: stage2.output_claims.ram_read_write.ra,
            },
            val_check: OpeningClaim {
                point: stage4.batch.ram_val_check.opening_point.clone(),
                value: stage4.output_claims.ram_val_check.ram_ra,
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(RegistersValEvaluation)]
pub struct RegistersValEvaluationOutputOpeningClaims<C> {
    #[opening(committed = RdInc)]
    pub rd_inc: C,
    #[opening(RdWa)]
    pub rd_wa: C,
}

/// Consumed register value-evaluation opening, wired from the upstream register
/// read-write checking.
#[derive(Clone, Debug, InputClaims)]
pub struct RegistersValEvaluationInputs<C> {
    #[opening(RegistersVal, from = RegistersReadWriteChecking)]
    pub registers_val: C,
}

impl<F: Field> RegistersValEvaluationInputs<OpeningClaim<F>> {
    /// Wire the consumed `RegistersVal` opening from the upstream register
    /// read-write checking (stage 4).
    pub fn from_clear(stage4: &Stage4ClearOutput<F>) -> Self {
        Self {
            registers_val: OpeningClaim {
                point: stage4.batch.registers_read_write.opening_point.clone(),
                value: stage4.output_claims.registers_read_write.registers_val,
            },
        }
    }
}
