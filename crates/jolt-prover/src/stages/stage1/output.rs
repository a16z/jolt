use std::collections::{btree_map::Entry, BTreeMap, BTreeSet};

use jolt_backends::{BackendValueSlot, SumcheckEvaluationOutput, SumcheckResult, SumcheckSlot};
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::{FieldInlineOpFlag, FieldInlineVirtualPolynomial};
use jolt_claims::protocols::jolt::JoltVirtualPolynomial;
#[cfg(feature = "zk")]
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_poly::Point;
use jolt_riscv::CircuitFlags;
#[cfg(feature = "zk")]
use jolt_sumcheck::SumcheckProof;
#[cfg(feature = "field-inline")]
use jolt_verifier::stages::stage1::inputs::{FieldInlineStage1Claims, FieldInlineStage1FlagClaims};
use jolt_verifier::stages::stage1::inputs::{
    SpartanOuterClaims, SpartanOuterFlagClaims, Stage1Claims,
};
use jolt_verifier::stages::stage1::outputs::Stage1ClearOutput;
use jolt_verifier::stages::stage1::outputs::{Stage1PublicOutput, VerifiedSpartanOuterSumcheck};

#[cfg(feature = "zk")]
use crate::committed::CommittedSumcheckWitness;
use crate::ProverError;

#[cfg(feature = "field-inline")]
use super::request::Stage1FieldInlineR1csEvaluationRequest;
use super::request::{
    Stage1R1csEvaluationRequest, Stage1Request, STAGE1_REMAINDER_OUTPUT_SLOT,
    STAGE1_REMAINDER_SLOT, STAGE1_UNISKIP_OUTPUT_SLOT, STAGE1_UNISKIP_SLOT,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1SumcheckOutput<F: Field, Proof> {
    pub uniskip_proof: Proof,
    pub remainder_proof: Proof,
    pub uniskip_output_claim: F,
    pub remainder_output_claim: F,
    pub r1cs_input_claims: Vec<Stage1R1csInputClaim<F>>,
    #[cfg(feature = "field-inline")]
    pub field_inline_r1cs_input_claims: Vec<Stage1FieldInlineR1csInputClaim<F>>,
    /// Verifier-mirroring Stage 1 output that downstream stages consume as a
    /// dependency, produced by the clear prover (`prove`). The pure-backend test
    /// path (`from_backend_result`) leaves it `None`; the full-proof
    /// orchestrator requires it.
    pub verifier_output: Option<Stage1ClearOutput<F>>,
}

#[cfg(feature = "zk")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1CommittedBoundaryOutput<F, VC>
where
    F: Field,
    VC: VectorCommitment<Field = F>,
{
    pub uniskip_proof: SumcheckProof<F, VC::Output>,
    pub remainder_proof: SumcheckProof<F, VC::Output>,
    pub public: Stage1PublicOutput<F>,
    pub verifier_output: Stage1ClearOutput<F>,
    pub uniskip_output_claim_values: Vec<F>,
    pub remainder_output_claim_values: Vec<F>,
    pub(crate) uniskip_committed_witness: CommittedSumcheckWitness<F>,
    pub(crate) remainder_committed_witness: CommittedSumcheckWitness<F>,
}

impl<F: Field, Proof> Stage1SumcheckOutput<F, Proof> {
    pub fn from_backend_result(
        request: &Stage1Request,
        result: SumcheckResult<F, Proof>,
    ) -> Result<Self, ProverError> {
        let mut proofs = collect_proofs(result.proofs)?;
        let mut values = collect_values(result.evaluations)?;

        let uniskip_proof = take_proof(&mut proofs, STAGE1_UNISKIP_SLOT)?;
        let remainder_proof = take_proof(&mut proofs, STAGE1_REMAINDER_SLOT)?;
        reject_extra_proofs(&proofs)?;

        let uniskip_output_claim = take_value(
            &mut values,
            STAGE1_UNISKIP_OUTPUT_SLOT,
            "stage1 uniskip output",
        )?;
        let remainder_output_claim = take_value(
            &mut values,
            STAGE1_REMAINDER_OUTPUT_SLOT,
            "stage1 remainder output",
        )?;
        let r1cs_input_claims = request
            .r1cs_inputs
            .iter()
            .map(|input| {
                Ok(Stage1R1csInputClaim {
                    variable: input.variable,
                    slot: input.slot,
                    value: take_value(&mut values, input.slot, "stage1 R1CS input")?,
                })
            })
            .collect::<Result<Vec<_>, ProverError>>()?;
        reject_extra_values(&values, request)?;

        Ok(Self {
            uniskip_proof,
            remainder_proof,
            uniskip_output_claim,
            remainder_output_claim,
            r1cs_input_claims,
            #[cfg(feature = "field-inline")]
            field_inline_r1cs_input_claims: Vec::new(),
            verifier_output: None,
        })
    }
}

pub fn r1cs_input_claims_from_evaluations<F: Field>(
    request: &Stage1R1csEvaluationRequest<F>,
    evaluations: Vec<SumcheckEvaluationOutput<F>>,
) -> Result<Vec<Stage1R1csInputClaim<F>>, ProverError> {
    let mut values = collect_values(evaluations)?;
    let claims = request
        .r1cs_inputs
        .iter()
        .map(|input| {
            Ok(Stage1R1csInputClaim {
                variable: input.variable,
                slot: input.slot,
                value: take_value(&mut values, input.slot, "stage1 R1CS input")?,
            })
        })
        .collect::<Result<Vec<_>, ProverError>>()?;
    if let Some(slot) = values.keys().next() {
        return Err(ProverError::InvalidSumcheckOutput {
            reason: format!("unexpected sumcheck value slot {slot:?}"),
        });
    }
    Ok(claims)
}

#[cfg(feature = "field-inline")]
pub fn field_inline_r1cs_input_claims_from_evaluations<F: Field>(
    request: &Stage1FieldInlineR1csEvaluationRequest<F>,
    evaluations: Vec<SumcheckEvaluationOutput<F>>,
) -> Result<Vec<Stage1FieldInlineR1csInputClaim<F>>, ProverError> {
    let mut values = collect_values(evaluations)?;
    let claims = request
        .r1cs_inputs
        .iter()
        .map(|input| {
            Ok(Stage1FieldInlineR1csInputClaim {
                variable: input.variable,
                slot: input.slot,
                value: take_value(&mut values, input.slot, "stage1 field-inline R1CS input")?,
            })
        })
        .collect::<Result<Vec<_>, ProverError>>()?;
    if let Some(slot) = values.keys().next() {
        return Err(ProverError::InvalidSumcheckOutput {
            reason: format!("unexpected sumcheck value slot {slot:?}"),
        });
    }
    Ok(claims)
}

pub fn spartan_outer_claims_from_r1cs_inputs<F: Field>(
    claims: &[Stage1R1csInputClaim<F>],
) -> Result<SpartanOuterClaims<F>, ProverError> {
    let mut values = collect_r1cs_inputs(claims)?;
    let outer = SpartanOuterClaims {
        left_instruction_input: take_r1cs_input(
            &mut values,
            JoltVirtualPolynomial::LeftInstructionInput,
        )?,
        right_instruction_input: take_r1cs_input(
            &mut values,
            JoltVirtualPolynomial::RightInstructionInput,
        )?,
        product: take_r1cs_input(&mut values, JoltVirtualPolynomial::Product)?,
        should_branch: take_r1cs_input(&mut values, JoltVirtualPolynomial::ShouldBranch)?,
        pc: take_r1cs_input(&mut values, JoltVirtualPolynomial::PC)?,
        unexpanded_pc: take_r1cs_input(&mut values, JoltVirtualPolynomial::UnexpandedPC)?,
        imm: take_r1cs_input(&mut values, JoltVirtualPolynomial::Imm)?,
        ram_address: take_r1cs_input(&mut values, JoltVirtualPolynomial::RamAddress)?,
        rs1_value: take_r1cs_input(&mut values, JoltVirtualPolynomial::Rs1Value)?,
        rs2_value: take_r1cs_input(&mut values, JoltVirtualPolynomial::Rs2Value)?,
        rd_write_value: take_r1cs_input(&mut values, JoltVirtualPolynomial::RdWriteValue)?,
        ram_read_value: take_r1cs_input(&mut values, JoltVirtualPolynomial::RamReadValue)?,
        ram_write_value: take_r1cs_input(&mut values, JoltVirtualPolynomial::RamWriteValue)?,
        left_lookup_operand: take_r1cs_input(
            &mut values,
            JoltVirtualPolynomial::LeftLookupOperand,
        )?,
        right_lookup_operand: take_r1cs_input(
            &mut values,
            JoltVirtualPolynomial::RightLookupOperand,
        )?,
        next_unexpanded_pc: take_r1cs_input(&mut values, JoltVirtualPolynomial::NextUnexpandedPC)?,
        next_pc: take_r1cs_input(&mut values, JoltVirtualPolynomial::NextPC)?,
        next_is_virtual: take_r1cs_input(&mut values, JoltVirtualPolynomial::NextIsVirtual)?,
        next_is_first_in_sequence: take_r1cs_input(
            &mut values,
            JoltVirtualPolynomial::NextIsFirstInSequence,
        )?,
        lookup_output: take_r1cs_input(&mut values, JoltVirtualPolynomial::LookupOutput)?,
        should_jump: take_r1cs_input(&mut values, JoltVirtualPolynomial::ShouldJump)?,
        flags: SpartanOuterFlagClaims {
            add_operands: take_flag(&mut values, CircuitFlags::AddOperands)?,
            subtract_operands: take_flag(&mut values, CircuitFlags::SubtractOperands)?,
            multiply_operands: take_flag(&mut values, CircuitFlags::MultiplyOperands)?,
            load: take_flag(&mut values, CircuitFlags::Load)?,
            store: take_flag(&mut values, CircuitFlags::Store)?,
            jump: take_flag(&mut values, CircuitFlags::Jump)?,
            write_lookup_output_to_rd: take_flag(&mut values, CircuitFlags::WriteLookupOutputToRD)?,
            virtual_instruction: take_flag(&mut values, CircuitFlags::VirtualInstruction)?,
            assert: take_flag(&mut values, CircuitFlags::Assert)?,
            do_not_update_unexpanded_pc: take_flag(
                &mut values,
                CircuitFlags::DoNotUpdateUnexpandedPC,
            )?,
            advice: take_flag(&mut values, CircuitFlags::Advice)?,
            is_compressed: take_flag(&mut values, CircuitFlags::IsCompressed)?,
            is_first_in_sequence: take_flag(&mut values, CircuitFlags::IsFirstInSequence)?,
            is_last_in_sequence: take_flag(&mut values, CircuitFlags::IsLastInSequence)?,
        },
    };
    reject_extra_r1cs_inputs(&values)?;
    Ok(outer)
}

#[cfg(feature = "field-inline")]
pub fn field_inline_stage1_claims_from_r1cs_inputs<F: Field>(
    claims: &[Stage1FieldInlineR1csInputClaim<F>],
) -> Result<FieldInlineStage1Claims<F>, ProverError> {
    let mut values = collect_field_inline_r1cs_inputs(claims)?;
    let field_inline = FieldInlineStage1Claims {
        field_rs1_value: take_field_inline_r1cs_input(
            &mut values,
            FieldInlineVirtualPolynomial::FieldRs1Value,
        )?,
        field_rs2_value: take_field_inline_r1cs_input(
            &mut values,
            FieldInlineVirtualPolynomial::FieldRs2Value,
        )?,
        field_rd_value: take_field_inline_r1cs_input(
            &mut values,
            FieldInlineVirtualPolynomial::FieldRdValue,
        )?,
        field_product: take_field_inline_r1cs_input(
            &mut values,
            FieldInlineVirtualPolynomial::FieldProduct,
        )?,
        field_inv_product: take_field_inline_r1cs_input(
            &mut values,
            FieldInlineVirtualPolynomial::FieldInvProduct,
        )?,
        flags: FieldInlineStage1FlagClaims {
            add: take_field_inline_flag(&mut values, FieldInlineOpFlag::Add)?,
            sub: take_field_inline_flag(&mut values, FieldInlineOpFlag::Sub)?,
            mul: take_field_inline_flag(&mut values, FieldInlineOpFlag::Mul)?,
            inv: take_field_inline_flag(&mut values, FieldInlineOpFlag::Inv)?,
            assert_eq: take_field_inline_flag(&mut values, FieldInlineOpFlag::AssertEq)?,
            load_from_x: take_field_inline_flag(&mut values, FieldInlineOpFlag::LoadFromX)?,
            store_to_x: take_field_inline_flag(&mut values, FieldInlineOpFlag::StoreToX)?,
            load_imm: take_field_inline_flag(&mut values, FieldInlineOpFlag::LoadImm)?,
        },
    };
    reject_extra_field_inline_r1cs_inputs(&values)?;
    Ok(field_inline)
}

#[cfg(not(feature = "field-inline"))]
pub fn stage1_claims_from_r1cs_inputs<F: Field>(
    uniskip_output_claim: F,
    claims: &[Stage1R1csInputClaim<F>],
) -> Result<Stage1Claims<F>, ProverError> {
    Ok(Stage1Claims {
        uniskip_output_claim,
        outer: spartan_outer_claims_from_r1cs_inputs(claims)?,
    })
}

/// Assemble the verifier-mirroring `Stage1ClearOutput` from the transparent
/// prover's Fiat-Shamir state and opening claims.
///
/// Mirrors `jolt-verifier/src/stages/stage1/verify.rs`: the uni-skip statement
/// has input claim `0`, single challenge `uniskip_challenge`, and output claim
/// `uniskip_output_claim`; the remainder statement has input claim
/// `uniskip_output_claim · remainder_batching_coefficient`, sumcheck point over
/// the remainder challenges, and `expected_remainder_output_claim` (already
/// scaled by the batching coefficient). The verifier records
/// `public.remainder_challenges` as `remainder.sumcheck_point.as_slice()`, which
/// `Point::high_to_low` preserves verbatim.
#[expect(
    clippy::too_many_arguments,
    reason = "Mirrors the verifier's Stage1ClearOutput, which decomposes the uni-skip/remainder reductions into distinct Fiat-Shamir values."
)]
pub fn stage1_clear_output<F: Field>(
    tau: Vec<F>,
    uniskip_challenge: F,
    uniskip_output_claim: F,
    remainder_batching_coefficient: F,
    remainder_challenges: Vec<F>,
    remainder_output_claim: F,
    expected_remainder_output_claim: F,
    r1cs_input_claims: &[Stage1R1csInputClaim<F>],
    #[cfg(feature = "field-inline")]
    field_inline_r1cs_input_claims: &[Stage1FieldInlineR1csInputClaim<F>],
) -> Result<Stage1ClearOutput<F>, ProverError> {
    let uniskip = VerifiedSpartanOuterSumcheck {
        input_claim: F::from_u64(0),
        sumcheck_point: Point::high_to_low(vec![uniskip_challenge]),
        sumcheck_final_claim: uniskip_output_claim,
        expected_output_claim: uniskip_output_claim,
    };
    let remainder_point = Point::high_to_low(remainder_challenges);
    let remainder = VerifiedSpartanOuterSumcheck {
        input_claim: uniskip_output_claim * remainder_batching_coefficient,
        sumcheck_point: remainder_point.clone(),
        sumcheck_final_claim: remainder_output_claim,
        expected_output_claim: expected_remainder_output_claim,
    };
    let public = Stage1PublicOutput {
        tau,
        uniskip_challenge,
        remainder_batching_coefficient,
        remainder_challenges: remainder_point.as_slice().to_vec(),
    };
    Ok(Stage1ClearOutput {
        public,
        uniskip,
        remainder,
        outer: spartan_outer_claims_from_r1cs_inputs(r1cs_input_claims)?,
        #[cfg(feature = "field-inline")]
        field_inline: field_inline_stage1_claims_from_r1cs_inputs(field_inline_r1cs_input_claims)?,
    })
}

#[cfg(feature = "field-inline")]
pub fn stage1_claims_from_r1cs_inputs<F: Field>(
    uniskip_output_claim: F,
    claims: &[Stage1R1csInputClaim<F>],
    field_inline_claims: &[Stage1FieldInlineR1csInputClaim<F>],
) -> Result<Stage1Claims<F>, ProverError> {
    Ok(Stage1Claims {
        uniskip_output_claim,
        outer: spartan_outer_claims_from_r1cs_inputs(claims)?,
        field_inline: field_inline_stage1_claims_from_r1cs_inputs(field_inline_claims)?,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage1R1csInputClaim<F: Field> {
    pub variable: JoltVirtualPolynomial,
    pub slot: BackendValueSlot,
    pub value: F,
}

#[cfg(feature = "field-inline")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage1FieldInlineR1csInputClaim<F: Field> {
    pub variable: FieldInlineVirtualPolynomial,
    pub slot: BackendValueSlot,
    pub value: F,
}

fn collect_r1cs_inputs<F: Field>(
    claims: &[Stage1R1csInputClaim<F>],
) -> Result<BTreeMap<JoltVirtualPolynomial, F>, ProverError> {
    let mut values = BTreeMap::new();
    for claim in claims {
        match values.entry(claim.variable) {
            Entry::Vacant(entry) => {
                let _ = entry.insert(claim.value);
            }
            Entry::Occupied(_) => {
                return Err(ProverError::InvalidSumcheckOutput {
                    reason: format!("duplicate Stage 1 R1CS input {:?}", claim.variable),
                });
            }
        }
    }
    Ok(values)
}

#[cfg(feature = "field-inline")]
fn collect_field_inline_r1cs_inputs<F: Field>(
    claims: &[Stage1FieldInlineR1csInputClaim<F>],
) -> Result<BTreeMap<FieldInlineVirtualPolynomial, F>, ProverError> {
    let mut values = BTreeMap::new();
    for claim in claims {
        match values.entry(claim.variable) {
            Entry::Vacant(entry) => {
                let _ = entry.insert(claim.value);
            }
            Entry::Occupied(_) => {
                return Err(ProverError::InvalidSumcheckOutput {
                    reason: format!(
                        "duplicate Stage 1 field-inline R1CS input {:?}",
                        claim.variable
                    ),
                });
            }
        }
    }
    Ok(values)
}

fn take_r1cs_input<F: Field>(
    values: &mut BTreeMap<JoltVirtualPolynomial, F>,
    variable: JoltVirtualPolynomial,
) -> Result<F, ProverError> {
    values
        .remove(&variable)
        .ok_or_else(|| ProverError::InvalidSumcheckOutput {
            reason: format!("missing Stage 1 R1CS input {variable:?}"),
        })
}

#[cfg(feature = "field-inline")]
fn take_field_inline_r1cs_input<F: Field>(
    values: &mut BTreeMap<FieldInlineVirtualPolynomial, F>,
    variable: FieldInlineVirtualPolynomial,
) -> Result<F, ProverError> {
    values
        .remove(&variable)
        .ok_or_else(|| ProverError::InvalidSumcheckOutput {
            reason: format!("missing Stage 1 field-inline R1CS input {variable:?}"),
        })
}

fn take_flag<F: Field>(
    values: &mut BTreeMap<JoltVirtualPolynomial, F>,
    flag: CircuitFlags,
) -> Result<F, ProverError> {
    take_r1cs_input(values, JoltVirtualPolynomial::OpFlags(flag))
}

#[cfg(feature = "field-inline")]
fn take_field_inline_flag<F: Field>(
    values: &mut BTreeMap<FieldInlineVirtualPolynomial, F>,
    flag: FieldInlineOpFlag,
) -> Result<F, ProverError> {
    take_field_inline_r1cs_input(values, FieldInlineVirtualPolynomial::FieldOpFlag(flag))
}

fn reject_extra_r1cs_inputs<F: Field>(
    values: &BTreeMap<JoltVirtualPolynomial, F>,
) -> Result<(), ProverError> {
    if let Some(variable) = values.keys().next() {
        return Err(ProverError::InvalidSumcheckOutput {
            reason: format!("unexpected Stage 1 R1CS input {variable:?}"),
        });
    }
    Ok(())
}

#[cfg(feature = "field-inline")]
fn reject_extra_field_inline_r1cs_inputs<F: Field>(
    values: &BTreeMap<FieldInlineVirtualPolynomial, F>,
) -> Result<(), ProverError> {
    if let Some(variable) = values.keys().next() {
        return Err(ProverError::InvalidSumcheckOutput {
            reason: format!("unexpected Stage 1 field-inline R1CS input {variable:?}"),
        });
    }
    Ok(())
}

fn collect_proofs<Proof>(
    proofs: Vec<jolt_backends::SumcheckProofOutput<Proof>>,
) -> Result<BTreeMap<SumcheckSlot, Proof>, ProverError> {
    let mut by_slot = BTreeMap::new();
    for proof in proofs {
        match by_slot.entry(proof.slot) {
            Entry::Vacant(entry) => {
                let _ = entry.insert(proof.proof);
            }
            Entry::Occupied(_) => {
                return Err(ProverError::InvalidSumcheckOutput {
                    reason: format!("duplicate sumcheck proof slot {:?}", proof.slot),
                });
            }
        }
    }
    Ok(by_slot)
}

fn collect_values<F: Field>(
    values: Vec<jolt_backends::SumcheckEvaluationOutput<F>>,
) -> Result<BTreeMap<BackendValueSlot, F>, ProverError> {
    let mut by_slot = BTreeMap::new();
    for value in values {
        match by_slot.entry(value.slot) {
            Entry::Vacant(entry) => {
                let _ = entry.insert(value.value);
            }
            Entry::Occupied(_) => {
                return Err(ProverError::InvalidSumcheckOutput {
                    reason: format!("duplicate sumcheck value slot {:?}", value.slot),
                });
            }
        }
    }
    Ok(by_slot)
}

fn take_proof<Proof>(
    proofs: &mut BTreeMap<SumcheckSlot, Proof>,
    slot: SumcheckSlot,
) -> Result<Proof, ProverError> {
    proofs
        .remove(&slot)
        .ok_or_else(|| ProverError::InvalidSumcheckOutput {
            reason: format!("missing sumcheck proof slot {slot:?}"),
        })
}

fn take_value<F: Field>(
    values: &mut BTreeMap<BackendValueSlot, F>,
    slot: BackendValueSlot,
    label: &'static str,
) -> Result<F, ProverError> {
    values
        .remove(&slot)
        .ok_or_else(|| ProverError::InvalidSumcheckOutput {
            reason: format!("missing {label} value slot {slot:?}"),
        })
}

fn reject_extra_proofs<Proof>(proofs: &BTreeMap<SumcheckSlot, Proof>) -> Result<(), ProverError> {
    if let Some(slot) = proofs.keys().next() {
        return Err(ProverError::InvalidSumcheckOutput {
            reason: format!("unexpected sumcheck proof slot {slot:?}"),
        });
    }
    Ok(())
}

fn reject_extra_values<F: Field>(
    values: &BTreeMap<BackendValueSlot, F>,
    request: &Stage1Request,
) -> Result<(), ProverError> {
    if values.is_empty() {
        return Ok(());
    }
    let expected = request
        .expected_value_slots()
        .into_iter()
        .collect::<BTreeSet<_>>();
    let unexpected = values
        .keys()
        .find(|slot| !expected.contains(slot))
        .or_else(|| values.keys().next());
    if let Some(slot) = unexpected {
        return Err(ProverError::InvalidSumcheckOutput {
            reason: format!("unexpected sumcheck value slot {slot:?}"),
        });
    }
    Ok(())
}
