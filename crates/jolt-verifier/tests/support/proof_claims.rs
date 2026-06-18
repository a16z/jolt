//! Opening-claim projection for verifier-native prover proofs.

#[cfg(all(any(feature = "prover-fixtures", test), not(feature = "zk")))]
use jolt_claims::protocols::jolt::formulas::spartan::SpartanOuterDimensions;
#[cfg(all(any(feature = "prover-fixtures", test), not(feature = "zk")))]
use jolt_claims::protocols::jolt::{
    self as jolt,
    formulas::{
        booleanity, bytecode,
        claim_reductions::registers as registers_claim_reduction,
        claim_reductions::{advice, increments, instruction as instruction_claim_reduction},
        instruction, ram, registers,
        spartan::{
            outer_opening, outer_uniskip_opening, product_remainder_output_openings,
            product_uniskip_opening, shift_output_openings,
        },
    },
    JoltAdviceKind, JoltCommittedPolynomial, JoltOpeningId, JoltRelationId, JoltVirtualPolynomial,
};
use jolt_field::Field;
use jolt_lookup_tables::{LookupTableKind, XLEN as RISCV_XLEN};
#[cfg(all(any(feature = "prover-fixtures", test), not(feature = "zk")))]
use jolt_riscv::CircuitFlags;
#[cfg(any(feature = "prover-fixtures", test))]
use jolt_verifier::proof::{JoltProof, JoltProofClaims};
#[cfg(all(any(feature = "prover-fixtures", test), not(feature = "zk")))]
use jolt_verifier::stages::stage6::inputs::AdviceCyclePhaseOutputClaim;
use jolt_verifier::{
    proof::ClearProofClaims,
    stages::{
        stage1::inputs::{SpartanOuterClaims, SpartanOuterFlagClaims, Stage1Claims},
        stage2::inputs::{
            InstructionClaimReductionOutputOpeningClaims, ProductRemainderOutputOpeningClaims,
            RamReadWriteOutputOpeningClaims, Stage2BatchOutputOpeningClaims, Stage2Claims,
        },
        stage3::inputs::{
            InstructionInputOutputOpeningClaims, RegistersClaimReductionOutputOpeningClaims,
            SpartanShiftOutputOpeningClaims, Stage3Claims,
        },
        stage4::inputs::{
            RamValCheckAdviceOpeningClaims, RamValCheckOutputOpeningClaims,
            RegistersReadWriteOutputOpeningClaims, Stage4Claims,
        },
        stage5::inputs::{
            InstructionReadRafOutputOpeningClaims, RamRaClaimReductionOutputOpeningClaims,
            RegistersValEvaluationOutputOpeningClaims, Stage5Claims,
        },
        stage6::inputs::{
            BooleanityOutputOpeningClaims, BytecodeReadRafOutputOpeningClaims,
            IncClaimReductionOutputOpeningClaims, InstructionRaVirtualizationOutputOpeningClaims,
            RamHammingBooleanityOutputOpeningClaims, RamRaVirtualizationOutputOpeningClaims,
            Stage6AddressPhaseClaims, Stage6AdviceCyclePhaseClaims, Stage6Claims,
        },
        stage7::inputs::{
            HammingWeightClaimReductionOutputOpeningClaims, Stage7AdviceAddressPhaseClaims,
            Stage7Claims,
        },
    },
};
#[cfg(any(feature = "prover-fixtures", test))]
use {jolt_crypto::VectorCommitment, jolt_openings::CommitmentScheme};

#[doc(hidden)]
#[cfg(any(feature = "prover-fixtures", test))]
pub fn attach_empty_opening_claims<PCS, VC, ZkProof>(proof: &mut JoltProof<PCS, VC, ZkProof>)
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    proof.claims = JoltProofClaims::Clear(empty_clear_claims(proof.trace_length));
}

#[cfg(any(feature = "prover-fixtures", test))]
fn empty_clear_claims<F: Field>(_trace_length: usize) -> ClearProofClaims<F> {
    let zero = F::zero();

    ClearProofClaims {
        stage1: Stage1Claims {
            uniskip_output_claim: zero,
            outer: empty_spartan_outer_claims(),
        },
        stage2: Stage2Claims {
            product_uniskip_output_claim: zero,
            batch_outputs: Stage2BatchOutputOpeningClaims {
                ram_read_write: RamReadWriteOutputOpeningClaims {
                    val: zero,
                    ra: zero,
                    inc: zero,
                },
                product_remainder: ProductRemainderOutputOpeningClaims {
                    left_instruction_input: zero,
                    right_instruction_input: zero,
                    jump_flag: zero,
                    write_lookup_output_to_rd: zero,
                    lookup_output: zero,
                    branch_flag: zero,
                    next_is_noop: zero,
                    virtual_instruction: zero,
                },
                instruction_claim_reduction: InstructionClaimReductionOutputOpeningClaims {
                    lookup_output: None,
                    left_lookup_operand: zero,
                    right_lookup_operand: zero,
                    left_instruction_input: None,
                    right_instruction_input: None,
                },
                ram_raf_evaluation: zero,
                ram_output_check: zero,
            },
        },
        stage3: Stage3Claims {
            shift: SpartanShiftOutputOpeningClaims {
                unexpanded_pc: zero,
                pc: zero,
                is_virtual: zero,
                is_first_in_sequence: zero,
                is_noop: zero,
            },
            instruction_input: InstructionInputOutputOpeningClaims {
                left_operand_is_rs1: zero,
                rs1_value: zero,
                left_operand_is_pc: zero,
                unexpanded_pc: zero,
                right_operand_is_rs2: zero,
                rs2_value: zero,
                right_operand_is_imm: zero,
                imm: zero,
            },
            registers_claim_reduction: RegistersClaimReductionOutputOpeningClaims {
                rd_write_value: zero,
                rs1_value: zero,
                rs2_value: zero,
            },
        },
        stage4: Stage4Claims {
            advice: RamValCheckAdviceOpeningClaims {
                untrusted: None,
                trusted: None,
            },
            program_image_contribution: None,
            registers_read_write: RegistersReadWriteOutputOpeningClaims {
                registers_val: zero,
                rs1_ra: zero,
                rs2_ra: zero,
                rd_wa: zero,
                rd_inc: zero,
            },
            ram_val_check: RamValCheckOutputOpeningClaims {
                ram_ra: zero,
                ram_inc: zero,
            },
        },
        stage5: Stage5Claims {
            instruction_read_raf: InstructionReadRafOutputOpeningClaims {
                lookup_table_flags: vec![zero; LookupTableKind::<RISCV_XLEN>::COUNT],
                instruction_ra: vec![zero],
                instruction_raf_flag: zero,
            },
            ram_ra_claim_reduction: RamRaClaimReductionOutputOpeningClaims { ram_ra: zero },
            registers_val_evaluation: RegistersValEvaluationOutputOpeningClaims {
                rd_inc: zero,
                rd_wa: zero,
            },
        },
        stage6: Stage6Claims {
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
            advice_cycle_phase: Stage6AdviceCyclePhaseClaims {
                trusted: None,
                untrusted: None,
            },
            bytecode_claim_reduction: None,
            program_image_claim_reduction: None,
        },
        stage7: Stage7Claims {
            hamming_weight_claim_reduction: HammingWeightClaimReductionOutputOpeningClaims {
                instruction_ra: vec![zero],
                bytecode_ra: vec![zero],
                ram_ra: vec![zero],
            },
            advice_address_phase: Stage7AdviceAddressPhaseClaims {
                trusted: None,
                untrusted: None,
            },
            bytecode_address_phase: None,
            program_image_address_phase: None,
        },
    }
}

#[cfg(any(feature = "prover-fixtures", test))]
fn empty_spartan_outer_claims<F: Field>() -> SpartanOuterClaims<F> {
    let zero = F::zero();

    SpartanOuterClaims {
        left_instruction_input: zero,
        right_instruction_input: zero,
        product: zero,
        should_branch: zero,
        pc: zero,
        unexpanded_pc: zero,
        imm: zero,
        ram_address: zero,
        rs1_value: zero,
        rs2_value: zero,
        rd_write_value: zero,
        ram_read_value: zero,
        ram_write_value: zero,
        left_lookup_operand: zero,
        right_lookup_operand: zero,
        next_unexpanded_pc: zero,
        next_pc: zero,
        next_is_virtual: zero,
        next_is_first_in_sequence: zero,
        lookup_output: zero,
        should_jump: zero,
        flags: SpartanOuterFlagClaims {
            add_operands: zero,
            subtract_operands: zero,
            multiply_operands: zero,
            load: zero,
            store: zero,
            jump: zero,
            write_lookup_output_to_rd: zero,
            virtual_instruction: zero,
            assert: zero,
            do_not_update_unexpanded_pc: zero,
            advice: zero,
            is_compressed: zero,
            is_first_in_sequence: zero,
            is_last_in_sequence: zero,
        },
    }
}

#[doc(hidden)]
#[cfg(all(any(feature = "prover-fixtures", test), not(feature = "zk")))]
pub fn offset_opening_claim<PCS, VC, ZkProof>(
    proof: &mut JoltProof<PCS, VC, ZkProof>,
    id: jolt::JoltOpeningId,
    delta: PCS::Field,
) -> bool
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let Some(opening_claim) = claim_mut(proof, id) else {
        return false;
    };
    *opening_claim += delta;
    true
}

#[doc(hidden)]
#[cfg(all(any(feature = "prover-fixtures", test), not(feature = "zk")))]
pub fn upsert_opening_claim<PCS, VC, ZkProof>(
    proof: &mut JoltProof<PCS, VC, ZkProof>,
    id: jolt::JoltOpeningId,
    opening_claim: PCS::Field,
) where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let _ = set_claim(proof, id, opening_claim);
}

#[doc(hidden)]
#[cfg(all(any(feature = "prover-fixtures", test), not(feature = "zk")))]
pub fn opening_claim<PCS, VC, ZkProof>(
    proof: &JoltProof<PCS, VC, ZkProof>,
    id: jolt::JoltOpeningId,
) -> Option<PCS::Field>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    claim(proof, id)
}

#[cfg(all(any(feature = "prover-fixtures", test), not(feature = "zk")))]
fn claim<PCS, VC, ZkProof>(
    proof: &JoltProof<PCS, VC, ZkProof>,
    id: jolt::JoltOpeningId,
) -> Option<PCS::Field>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let JoltProofClaims::Clear(claims) = &proof.claims else {
        return None;
    };

    claim_from_clear(claims, proof.trace_length, id)
}

#[cfg(all(any(feature = "prover-fixtures", test), not(feature = "zk")))]
fn claim_mut<PCS, VC, ZkProof>(
    proof: &mut JoltProof<PCS, VC, ZkProof>,
    id: jolt::JoltOpeningId,
) -> Option<&mut PCS::Field>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let JoltProofClaims::Clear(claims) = &mut proof.claims else {
        return None;
    };

    claim_mut_from_clear(claims, proof.trace_length, id)
}

#[cfg(all(any(feature = "prover-fixtures", test), not(feature = "zk")))]
fn set_claim<PCS, VC, ZkProof>(
    proof: &mut JoltProof<PCS, VC, ZkProof>,
    id: jolt::JoltOpeningId,
    opening_claim: PCS::Field,
) -> bool
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let JoltProofClaims::Clear(claims) = &mut proof.claims else {
        return false;
    };

    set_claim_in_clear(claims, proof.trace_length, id, opening_claim)
}

#[cfg(all(any(feature = "prover-fixtures", test), not(feature = "zk")))]
fn claim_from_clear<F: Field>(
    claims: &ClearProofClaims<F>,
    trace_length: usize,
    id: jolt::JoltOpeningId,
) -> Option<F> {
    if id == outer_uniskip_opening() {
        return Some(claims.stage1.uniskip_output_claim);
    }
    if let Some(variable) = stage1_outer_variable(trace_length, id) {
        return claim_from_spartan_outer(&claims.stage1.outer, variable);
    }
    if id == product_uniskip_opening() {
        return Some(claims.stage2.product_uniskip_output_claim);
    }

    claim_from_stage2_batch_outputs(&claims.stage2.batch_outputs, id)
        .or_else(|| claim_from_stage3_outputs(&claims.stage3, id))
        .or_else(|| claim_from_stage4_outputs(&claims.stage4, id))
        .or_else(|| claim_from_stage5_outputs(&claims.stage5, id))
        .or_else(|| claim_from_stage7_outputs(&claims.stage7, id))
        .or_else(|| claim_from_stage6_outputs(&claims.stage6, id))
}

#[cfg(all(any(feature = "prover-fixtures", test), not(feature = "zk")))]
fn claim_mut_from_clear<F: Field>(
    claims: &mut ClearProofClaims<F>,
    trace_length: usize,
    id: jolt::JoltOpeningId,
) -> Option<&mut F> {
    if id == outer_uniskip_opening() {
        return Some(&mut claims.stage1.uniskip_output_claim);
    }
    if let Some(variable) = stage1_outer_variable(trace_length, id) {
        return claim_mut_from_spartan_outer(&mut claims.stage1.outer, variable);
    }
    if id == product_uniskip_opening() {
        return Some(&mut claims.stage2.product_uniskip_output_claim);
    }

    claim_mut_from_stage2_batch_outputs(&mut claims.stage2.batch_outputs, id)
        .or_else(|| claim_mut_from_stage3_outputs(&mut claims.stage3, id))
        .or_else(|| claim_mut_from_stage4_outputs(&mut claims.stage4, id))
        .or_else(|| claim_mut_from_stage5_outputs(&mut claims.stage5, id))
        .or_else(|| claim_mut_from_stage7_outputs(&mut claims.stage7, id))
        .or_else(|| claim_mut_from_stage6_outputs(&mut claims.stage6, id))
}

#[cfg(all(any(feature = "prover-fixtures", test), not(feature = "zk")))]
fn claim_from_spartan_outer<F: Field>(
    claims: &SpartanOuterClaims<F>,
    variable: JoltVirtualPolynomial,
) -> Option<F> {
    match variable {
        JoltVirtualPolynomial::LeftInstructionInput => Some(claims.left_instruction_input),
        JoltVirtualPolynomial::RightInstructionInput => Some(claims.right_instruction_input),
        JoltVirtualPolynomial::Product => Some(claims.product),
        JoltVirtualPolynomial::ShouldBranch => Some(claims.should_branch),
        JoltVirtualPolynomial::PC => Some(claims.pc),
        JoltVirtualPolynomial::UnexpandedPC => Some(claims.unexpanded_pc),
        JoltVirtualPolynomial::Imm => Some(claims.imm),
        JoltVirtualPolynomial::RamAddress => Some(claims.ram_address),
        JoltVirtualPolynomial::Rs1Value => Some(claims.rs1_value),
        JoltVirtualPolynomial::Rs2Value => Some(claims.rs2_value),
        JoltVirtualPolynomial::RdWriteValue => Some(claims.rd_write_value),
        JoltVirtualPolynomial::RamReadValue => Some(claims.ram_read_value),
        JoltVirtualPolynomial::RamWriteValue => Some(claims.ram_write_value),
        JoltVirtualPolynomial::LeftLookupOperand => Some(claims.left_lookup_operand),
        JoltVirtualPolynomial::RightLookupOperand => Some(claims.right_lookup_operand),
        JoltVirtualPolynomial::NextUnexpandedPC => Some(claims.next_unexpanded_pc),
        JoltVirtualPolynomial::NextPC => Some(claims.next_pc),
        JoltVirtualPolynomial::NextIsVirtual => Some(claims.next_is_virtual),
        JoltVirtualPolynomial::NextIsFirstInSequence => Some(claims.next_is_first_in_sequence),
        JoltVirtualPolynomial::LookupOutput => Some(claims.lookup_output),
        JoltVirtualPolynomial::ShouldJump => Some(claims.should_jump),
        JoltVirtualPolynomial::OpFlags(flag) => claim_from_spartan_outer_flag(&claims.flags, flag),
        _ => None,
    }
}

#[cfg(all(any(feature = "prover-fixtures", test), not(feature = "zk")))]
fn claim_from_spartan_outer_flag<F: Field>(
    claims: &SpartanOuterFlagClaims<F>,
    flag: CircuitFlags,
) -> Option<F> {
    match flag {
        CircuitFlags::AddOperands => Some(claims.add_operands),
        CircuitFlags::SubtractOperands => Some(claims.subtract_operands),
        CircuitFlags::MultiplyOperands => Some(claims.multiply_operands),
        CircuitFlags::Load => Some(claims.load),
        CircuitFlags::Store => Some(claims.store),
        CircuitFlags::Jump => Some(claims.jump),
        CircuitFlags::WriteLookupOutputToRD => Some(claims.write_lookup_output_to_rd),
        CircuitFlags::VirtualInstruction => Some(claims.virtual_instruction),
        CircuitFlags::Assert => Some(claims.assert),
        CircuitFlags::DoNotUpdateUnexpandedPC => Some(claims.do_not_update_unexpanded_pc),
        CircuitFlags::Advice => Some(claims.advice),
        CircuitFlags::IsCompressed => Some(claims.is_compressed),
        CircuitFlags::IsFirstInSequence => Some(claims.is_first_in_sequence),
        CircuitFlags::IsLastInSequence => Some(claims.is_last_in_sequence),
    }
}

#[cfg(all(any(feature = "prover-fixtures", test), not(feature = "zk")))]
fn claim_mut_from_spartan_outer<F: Field>(
    claims: &mut SpartanOuterClaims<F>,
    variable: JoltVirtualPolynomial,
) -> Option<&mut F> {
    match variable {
        JoltVirtualPolynomial::LeftInstructionInput => Some(&mut claims.left_instruction_input),
        JoltVirtualPolynomial::RightInstructionInput => Some(&mut claims.right_instruction_input),
        JoltVirtualPolynomial::Product => Some(&mut claims.product),
        JoltVirtualPolynomial::ShouldBranch => Some(&mut claims.should_branch),
        JoltVirtualPolynomial::PC => Some(&mut claims.pc),
        JoltVirtualPolynomial::UnexpandedPC => Some(&mut claims.unexpanded_pc),
        JoltVirtualPolynomial::Imm => Some(&mut claims.imm),
        JoltVirtualPolynomial::RamAddress => Some(&mut claims.ram_address),
        JoltVirtualPolynomial::Rs1Value => Some(&mut claims.rs1_value),
        JoltVirtualPolynomial::Rs2Value => Some(&mut claims.rs2_value),
        JoltVirtualPolynomial::RdWriteValue => Some(&mut claims.rd_write_value),
        JoltVirtualPolynomial::RamReadValue => Some(&mut claims.ram_read_value),
        JoltVirtualPolynomial::RamWriteValue => Some(&mut claims.ram_write_value),
        JoltVirtualPolynomial::LeftLookupOperand => Some(&mut claims.left_lookup_operand),
        JoltVirtualPolynomial::RightLookupOperand => Some(&mut claims.right_lookup_operand),
        JoltVirtualPolynomial::NextUnexpandedPC => Some(&mut claims.next_unexpanded_pc),
        JoltVirtualPolynomial::NextPC => Some(&mut claims.next_pc),
        JoltVirtualPolynomial::NextIsVirtual => Some(&mut claims.next_is_virtual),
        JoltVirtualPolynomial::NextIsFirstInSequence => Some(&mut claims.next_is_first_in_sequence),
        JoltVirtualPolynomial::LookupOutput => Some(&mut claims.lookup_output),
        JoltVirtualPolynomial::ShouldJump => Some(&mut claims.should_jump),
        JoltVirtualPolynomial::OpFlags(flag) => {
            claim_mut_from_spartan_outer_flag(&mut claims.flags, flag)
        }
        _ => None,
    }
}

#[cfg(all(any(feature = "prover-fixtures", test), not(feature = "zk")))]
fn claim_mut_from_spartan_outer_flag<F: Field>(
    claims: &mut SpartanOuterFlagClaims<F>,
    flag: CircuitFlags,
) -> Option<&mut F> {
    match flag {
        CircuitFlags::AddOperands => Some(&mut claims.add_operands),
        CircuitFlags::SubtractOperands => Some(&mut claims.subtract_operands),
        CircuitFlags::MultiplyOperands => Some(&mut claims.multiply_operands),
        CircuitFlags::Load => Some(&mut claims.load),
        CircuitFlags::Store => Some(&mut claims.store),
        CircuitFlags::Jump => Some(&mut claims.jump),
        CircuitFlags::WriteLookupOutputToRD => Some(&mut claims.write_lookup_output_to_rd),
        CircuitFlags::VirtualInstruction => Some(&mut claims.virtual_instruction),
        CircuitFlags::Assert => Some(&mut claims.assert),
        CircuitFlags::DoNotUpdateUnexpandedPC => Some(&mut claims.do_not_update_unexpanded_pc),
        CircuitFlags::Advice => Some(&mut claims.advice),
        CircuitFlags::IsCompressed => Some(&mut claims.is_compressed),
        CircuitFlags::IsFirstInSequence => Some(&mut claims.is_first_in_sequence),
        CircuitFlags::IsLastInSequence => Some(&mut claims.is_last_in_sequence),
    }
}

#[cfg(all(any(feature = "prover-fixtures", test), not(feature = "zk")))]
fn set_claim_in_clear<F: Field>(
    claims: &mut ClearProofClaims<F>,
    trace_length: usize,
    id: jolt::JoltOpeningId,
    opening_claim: F,
) -> bool {
    if let Some(claim) = claim_mut_from_clear(claims, trace_length, id) {
        *claim = opening_claim;
        return true;
    }

    set_optional_stage2_batch_output(&mut claims.stage2.batch_outputs, id, opening_claim)
        || set_optional_stage4_output(&mut claims.stage4, id, opening_claim)
        || set_optional_stage6_output(&mut claims.stage6, id, opening_claim)
}

#[cfg(all(any(feature = "prover-fixtures", test), not(feature = "zk")))]
fn stage1_outer_variable(
    trace_length: usize,
    id: jolt::JoltOpeningId,
) -> Option<JoltVirtualPolynomial> {
    let log_t = trace_length.ilog2() as usize;
    SpartanOuterDimensions::rv64(log_t)
        .variables()
        .iter()
        .copied()
        .find(|variable| id == outer_opening(*variable))
}

#[cfg(all(any(feature = "prover-fixtures", test), not(feature = "zk")))]
fn claim_from_stage2_batch_outputs<F: Field>(
    claims: &Stage2BatchOutputOpeningClaims<F>,
    id: jolt::JoltOpeningId,
) -> Option<F> {
    let [ram_val, ram_ra, ram_inc] = ram::read_write_checking_output_openings();
    let [product_left_instruction_input, product_right_instruction_input, product_jump_flag, product_write_lookup_output_to_rd, product_lookup_output, product_branch_flag, product_next_is_noop, product_virtual_instruction] =
        product_remainder_output_openings();
    let [instruction_lookup_output, instruction_left_lookup_operand, instruction_right_lookup_operand, instruction_left_instruction_input, instruction_right_instruction_input] =
        instruction_claim_reduction::claim_reduction_output_openings();
    let [ram_ra_raf_evaluation] = ram::raf_evaluation_output_openings();
    let [ram_val_final] = ram::output_check_output_openings();

    match id {
        id if id == ram_val => Some(claims.ram_read_write.val),
        id if id == ram_ra => Some(claims.ram_read_write.ra),
        id if id == ram_inc => Some(claims.ram_read_write.inc),
        id if id == product_left_instruction_input => {
            Some(claims.product_remainder.left_instruction_input)
        }
        id if id == product_right_instruction_input => {
            Some(claims.product_remainder.right_instruction_input)
        }
        id if id == product_jump_flag => Some(claims.product_remainder.jump_flag),
        id if id == product_write_lookup_output_to_rd => {
            Some(claims.product_remainder.write_lookup_output_to_rd)
        }
        id if id == product_lookup_output => Some(claims.product_remainder.lookup_output),
        id if id == product_branch_flag => Some(claims.product_remainder.branch_flag),
        id if id == product_next_is_noop => Some(claims.product_remainder.next_is_noop),
        id if id == product_virtual_instruction => {
            Some(claims.product_remainder.virtual_instruction)
        }
        id if id == instruction_lookup_output => claims.instruction_claim_reduction.lookup_output,
        id if id == instruction_left_lookup_operand => {
            Some(claims.instruction_claim_reduction.left_lookup_operand)
        }
        id if id == instruction_right_lookup_operand => {
            Some(claims.instruction_claim_reduction.right_lookup_operand)
        }
        id if id == instruction_left_instruction_input => {
            claims.instruction_claim_reduction.left_instruction_input
        }
        id if id == instruction_right_instruction_input => {
            claims.instruction_claim_reduction.right_instruction_input
        }
        id if id == ram_ra_raf_evaluation => Some(claims.ram_raf_evaluation),
        id if id == ram_val_final => Some(claims.ram_output_check),
        _ => None,
    }
}

#[cfg(all(any(feature = "prover-fixtures", test), not(feature = "zk")))]
fn claim_mut_from_stage2_batch_outputs<F: Field>(
    claims: &mut Stage2BatchOutputOpeningClaims<F>,
    id: jolt::JoltOpeningId,
) -> Option<&mut F> {
    let [ram_val, ram_ra, ram_inc] = ram::read_write_checking_output_openings();
    let [product_left_instruction_input, product_right_instruction_input, product_jump_flag, product_write_lookup_output_to_rd, product_lookup_output, product_branch_flag, product_next_is_noop, product_virtual_instruction] =
        product_remainder_output_openings();
    let [instruction_lookup_output, instruction_left_lookup_operand, instruction_right_lookup_operand, instruction_left_instruction_input, instruction_right_instruction_input] =
        instruction_claim_reduction::claim_reduction_output_openings();
    let [ram_ra_raf_evaluation] = ram::raf_evaluation_output_openings();
    let [ram_val_final] = ram::output_check_output_openings();

    match id {
        id if id == ram_val => Some(&mut claims.ram_read_write.val),
        id if id == ram_ra => Some(&mut claims.ram_read_write.ra),
        id if id == ram_inc => Some(&mut claims.ram_read_write.inc),
        id if id == product_left_instruction_input => {
            Some(&mut claims.product_remainder.left_instruction_input)
        }
        id if id == product_right_instruction_input => {
            Some(&mut claims.product_remainder.right_instruction_input)
        }
        id if id == product_jump_flag => Some(&mut claims.product_remainder.jump_flag),
        id if id == product_write_lookup_output_to_rd => {
            Some(&mut claims.product_remainder.write_lookup_output_to_rd)
        }
        id if id == product_lookup_output => Some(&mut claims.product_remainder.lookup_output),
        id if id == product_branch_flag => Some(&mut claims.product_remainder.branch_flag),
        id if id == product_next_is_noop => Some(&mut claims.product_remainder.next_is_noop),
        id if id == product_virtual_instruction => {
            Some(&mut claims.product_remainder.virtual_instruction)
        }
        id if id == instruction_lookup_output => {
            claims.instruction_claim_reduction.lookup_output.as_mut()
        }
        id if id == instruction_left_lookup_operand => {
            Some(&mut claims.instruction_claim_reduction.left_lookup_operand)
        }
        id if id == instruction_right_lookup_operand => {
            Some(&mut claims.instruction_claim_reduction.right_lookup_operand)
        }
        id if id == instruction_left_instruction_input => claims
            .instruction_claim_reduction
            .left_instruction_input
            .as_mut(),
        id if id == instruction_right_instruction_input => claims
            .instruction_claim_reduction
            .right_instruction_input
            .as_mut(),
        id if id == ram_ra_raf_evaluation => Some(&mut claims.ram_raf_evaluation),
        id if id == ram_val_final => Some(&mut claims.ram_output_check),
        _ => None,
    }
}

#[cfg(all(any(feature = "prover-fixtures", test), not(feature = "zk")))]
fn set_optional_stage2_batch_output<F: Field>(
    claims: &mut Stage2BatchOutputOpeningClaims<F>,
    id: jolt::JoltOpeningId,
    opening_claim: F,
) -> bool {
    let [instruction_lookup_output, _, _, instruction_left_instruction_input, instruction_right_instruction_input] =
        instruction_claim_reduction::claim_reduction_output_openings();

    match id {
        id if id == instruction_lookup_output => {
            claims.instruction_claim_reduction.lookup_output = Some(opening_claim);
            true
        }
        id if id == instruction_left_instruction_input => {
            claims.instruction_claim_reduction.left_instruction_input = Some(opening_claim);
            true
        }
        id if id == instruction_right_instruction_input => {
            claims.instruction_claim_reduction.right_instruction_input = Some(opening_claim);
            true
        }
        _ => false,
    }
}

#[cfg(all(any(feature = "prover-fixtures", test), not(feature = "zk")))]
fn set_optional_stage4_output<F: Field>(
    claims: &mut Stage4Claims<F>,
    id: jolt::JoltOpeningId,
    opening_claim: F,
) -> bool {
    match id {
        id if id == ram::val_check_advice_opening(JoltAdviceKind::Untrusted) => {
            claims.advice.untrusted = Some(opening_claim);
            true
        }
        id if id == ram::val_check_advice_opening(JoltAdviceKind::Trusted) => {
            claims.advice.trusted = Some(opening_claim);
            true
        }
        _ => false,
    }
}

#[cfg(all(any(feature = "prover-fixtures", test), not(feature = "zk")))]
fn set_optional_stage6_output<F: Field>(
    claims: &mut Stage6Claims<F>,
    id: jolt::JoltOpeningId,
    opening_claim: F,
) -> bool {
    match id {
        id if id == bytecode::bytecode_read_raf_address_phase_opening() => {
            claims.address_phase.bytecode_read_raf = opening_claim;
            true
        }
        id if id == booleanity::booleanity_address_phase_opening() => {
            claims.address_phase.booleanity = opening_claim;
            true
        }
        id if id == advice::cycle_phase_advice_opening(JoltAdviceKind::Trusted)
            || id == advice::final_advice_opening(JoltAdviceKind::Trusted) =>
        {
            claims.advice_cycle_phase.trusted = Some(AdviceCyclePhaseOutputClaim { opening_claim });
            true
        }
        id if id == advice::cycle_phase_advice_opening(JoltAdviceKind::Untrusted)
            || id == advice::final_advice_opening(JoltAdviceKind::Untrusted) =>
        {
            claims.advice_cycle_phase.untrusted =
                Some(AdviceCyclePhaseOutputClaim { opening_claim });
            true
        }
        _ => false,
    }
}

#[cfg(all(any(feature = "prover-fixtures", test), not(feature = "zk")))]
fn claim_from_stage3_outputs<F: Field>(
    claims: &Stage3Claims<F>,
    id: jolt::JoltOpeningId,
) -> Option<F> {
    let [unexpanded_pc_shift, pc_shift, is_virtual_shift, is_first_in_sequence_shift, is_noop_shift] =
        shift_output_openings();
    let [right_operand_is_rs2, rs2_value_input, right_operand_is_imm, imm_input, left_operand_is_rs1, rs1_value_input, left_operand_is_pc, unexpanded_pc_input] =
        instruction::input_virtualization_output_openings();
    let [rd_write_value_reduced, rs1_value_reduced, rs2_value_reduced] =
        registers_claim_reduction::claim_reduction_output_openings();

    match id {
        id if id == unexpanded_pc_shift => Some(claims.shift.unexpanded_pc),
        id if id == pc_shift => Some(claims.shift.pc),
        id if id == is_virtual_shift => Some(claims.shift.is_virtual),
        id if id == is_first_in_sequence_shift => Some(claims.shift.is_first_in_sequence),
        id if id == is_noop_shift => Some(claims.shift.is_noop),
        id if id == left_operand_is_rs1 => Some(claims.instruction_input.left_operand_is_rs1),
        id if id == rs1_value_input => Some(claims.instruction_input.rs1_value),
        id if id == left_operand_is_pc => Some(claims.instruction_input.left_operand_is_pc),
        id if id == unexpanded_pc_input => Some(claims.instruction_input.unexpanded_pc),
        id if id == right_operand_is_rs2 => Some(claims.instruction_input.right_operand_is_rs2),
        id if id == rs2_value_input => Some(claims.instruction_input.rs2_value),
        id if id == right_operand_is_imm => Some(claims.instruction_input.right_operand_is_imm),
        id if id == imm_input => Some(claims.instruction_input.imm),
        id if id == rd_write_value_reduced => Some(claims.registers_claim_reduction.rd_write_value),
        id if id == rs1_value_reduced => Some(claims.registers_claim_reduction.rs1_value),
        id if id == rs2_value_reduced => Some(claims.registers_claim_reduction.rs2_value),
        _ => None,
    }
}

#[cfg(all(any(feature = "prover-fixtures", test), not(feature = "zk")))]
fn claim_mut_from_stage3_outputs<F: Field>(
    claims: &mut Stage3Claims<F>,
    id: jolt::JoltOpeningId,
) -> Option<&mut F> {
    let [unexpanded_pc_shift, pc_shift, is_virtual_shift, is_first_in_sequence_shift, is_noop_shift] =
        shift_output_openings();
    let [right_operand_is_rs2, rs2_value_input, right_operand_is_imm, imm_input, left_operand_is_rs1, rs1_value_input, left_operand_is_pc, unexpanded_pc_input] =
        instruction::input_virtualization_output_openings();
    let [rd_write_value_reduced, rs1_value_reduced, rs2_value_reduced] =
        registers_claim_reduction::claim_reduction_output_openings();

    match id {
        id if id == unexpanded_pc_shift => Some(&mut claims.shift.unexpanded_pc),
        id if id == pc_shift => Some(&mut claims.shift.pc),
        id if id == is_virtual_shift => Some(&mut claims.shift.is_virtual),
        id if id == is_first_in_sequence_shift => Some(&mut claims.shift.is_first_in_sequence),
        id if id == is_noop_shift => Some(&mut claims.shift.is_noop),
        id if id == left_operand_is_rs1 => Some(&mut claims.instruction_input.left_operand_is_rs1),
        id if id == rs1_value_input => Some(&mut claims.instruction_input.rs1_value),
        id if id == left_operand_is_pc => Some(&mut claims.instruction_input.left_operand_is_pc),
        id if id == unexpanded_pc_input => Some(&mut claims.instruction_input.unexpanded_pc),
        id if id == right_operand_is_rs2 => {
            Some(&mut claims.instruction_input.right_operand_is_rs2)
        }
        id if id == rs2_value_input => Some(&mut claims.instruction_input.rs2_value),
        id if id == right_operand_is_imm => {
            Some(&mut claims.instruction_input.right_operand_is_imm)
        }
        id if id == imm_input => Some(&mut claims.instruction_input.imm),
        id if id == rd_write_value_reduced => {
            Some(&mut claims.registers_claim_reduction.rd_write_value)
        }
        id if id == rs1_value_reduced => Some(&mut claims.registers_claim_reduction.rs1_value),
        id if id == rs2_value_reduced => Some(&mut claims.registers_claim_reduction.rs2_value),
        _ => None,
    }
}

#[cfg(all(any(feature = "prover-fixtures", test), not(feature = "zk")))]
fn claim_from_stage4_outputs<F: Field>(
    claims: &Stage4Claims<F>,
    id: jolt::JoltOpeningId,
) -> Option<F> {
    let [registers_val, rs1_ra, rs2_ra, rd_wa, rd_inc] =
        registers::read_write_checking_output_openings();
    let [ram_ra, ram_inc] = ram::val_check_output_openings();

    match id {
        id if id == ram::val_check_advice_opening(JoltAdviceKind::Untrusted) => {
            claims.advice.untrusted
        }
        id if id == ram::val_check_advice_opening(JoltAdviceKind::Trusted) => claims.advice.trusted,
        id if id == registers_val => Some(claims.registers_read_write.registers_val),
        id if id == rs1_ra => Some(claims.registers_read_write.rs1_ra),
        id if id == rs2_ra => Some(claims.registers_read_write.rs2_ra),
        id if id == rd_wa => Some(claims.registers_read_write.rd_wa),
        id if id == rd_inc => Some(claims.registers_read_write.rd_inc),
        id if id == ram_ra => Some(claims.ram_val_check.ram_ra),
        id if id == ram_inc => Some(claims.ram_val_check.ram_inc),
        _ => None,
    }
}

#[cfg(all(any(feature = "prover-fixtures", test), not(feature = "zk")))]
fn claim_mut_from_stage4_outputs<F: Field>(
    claims: &mut Stage4Claims<F>,
    id: jolt::JoltOpeningId,
) -> Option<&mut F> {
    let [registers_val, rs1_ra, rs2_ra, rd_wa, rd_inc] =
        registers::read_write_checking_output_openings();
    let [ram_ra, ram_inc] = ram::val_check_output_openings();

    match id {
        id if id == ram::val_check_advice_opening(JoltAdviceKind::Untrusted) => {
            claims.advice.untrusted.as_mut()
        }
        id if id == ram::val_check_advice_opening(JoltAdviceKind::Trusted) => {
            claims.advice.trusted.as_mut()
        }
        id if id == registers_val => Some(&mut claims.registers_read_write.registers_val),
        id if id == rs1_ra => Some(&mut claims.registers_read_write.rs1_ra),
        id if id == rs2_ra => Some(&mut claims.registers_read_write.rs2_ra),
        id if id == rd_wa => Some(&mut claims.registers_read_write.rd_wa),
        id if id == rd_inc => Some(&mut claims.registers_read_write.rd_inc),
        id if id == ram_ra => Some(&mut claims.ram_val_check.ram_ra),
        id if id == ram_inc => Some(&mut claims.ram_val_check.ram_inc),
        _ => None,
    }
}

#[cfg(all(any(feature = "prover-fixtures", test), not(feature = "zk")))]
fn claim_from_stage5_outputs<F: Field>(
    claims: &Stage5Claims<F>,
    id: jolt::JoltOpeningId,
) -> Option<F> {
    for table in LookupTableKind::<RISCV_XLEN>::iter() {
        if id == instruction::read_raf_lookup_table_flag_opening(table) {
            return claims
                .instruction_read_raf
                .lookup_table_flags
                .get(table.index())
                .copied();
        }
    }
    for (index, opening_claim) in claims
        .instruction_read_raf
        .instruction_ra
        .iter()
        .enumerate()
    {
        if id == instruction::read_raf_instruction_ra_opening(index) {
            return Some(*opening_claim);
        }
    }

    let [ram_ra] = ram::ra_claim_reduction_output_openings();
    let [rd_inc, rd_wa] = registers::val_evaluation_output_openings();
    match id {
        id if id == instruction::read_raf_instruction_raf_flag_opening() => {
            Some(claims.instruction_read_raf.instruction_raf_flag)
        }
        id if id == ram_ra => Some(claims.ram_ra_claim_reduction.ram_ra),
        id if id == rd_inc => Some(claims.registers_val_evaluation.rd_inc),
        id if id == rd_wa => Some(claims.registers_val_evaluation.rd_wa),
        _ => None,
    }
}

#[cfg(all(any(feature = "prover-fixtures", test), not(feature = "zk")))]
fn claim_mut_from_stage5_outputs<F: Field>(
    claims: &mut Stage5Claims<F>,
    id: jolt::JoltOpeningId,
) -> Option<&mut F> {
    for table in LookupTableKind::<RISCV_XLEN>::iter() {
        if id == instruction::read_raf_lookup_table_flag_opening(table) {
            return claims
                .instruction_read_raf
                .lookup_table_flags
                .get_mut(table.index());
        }
    }
    for (index, opening_claim) in claims
        .instruction_read_raf
        .instruction_ra
        .iter_mut()
        .enumerate()
    {
        if id == instruction::read_raf_instruction_ra_opening(index) {
            return Some(opening_claim);
        }
    }

    let [ram_ra] = ram::ra_claim_reduction_output_openings();
    let [rd_inc, rd_wa] = registers::val_evaluation_output_openings();
    match id {
        id if id == instruction::read_raf_instruction_raf_flag_opening() => {
            Some(&mut claims.instruction_read_raf.instruction_raf_flag)
        }
        id if id == ram_ra => Some(&mut claims.ram_ra_claim_reduction.ram_ra),
        id if id == rd_inc => Some(&mut claims.registers_val_evaluation.rd_inc),
        id if id == rd_wa => Some(&mut claims.registers_val_evaluation.rd_wa),
        _ => None,
    }
}

#[cfg(all(any(feature = "prover-fixtures", test), not(feature = "zk")))]
fn claim_from_stage6_outputs<F: Field>(
    claims: &Stage6Claims<F>,
    id: jolt::JoltOpeningId,
) -> Option<F> {
    for (index, opening_claim) in claims.bytecode_read_raf.bytecode_ra.iter().enumerate() {
        if id
            == JoltOpeningId::committed(
                JoltCommittedPolynomial::BytecodeRa(index),
                JoltRelationId::BytecodeReadRaf,
            )
        {
            return Some(*opening_claim);
        }
    }
    for (index, opening_claim) in claims.booleanity.instruction_ra.iter().enumerate() {
        if id
            == JoltOpeningId::committed(
                JoltCommittedPolynomial::InstructionRa(index),
                JoltRelationId::Booleanity,
            )
        {
            return Some(*opening_claim);
        }
    }
    for (index, opening_claim) in claims.booleanity.bytecode_ra.iter().enumerate() {
        if id
            == JoltOpeningId::committed(
                JoltCommittedPolynomial::BytecodeRa(index),
                JoltRelationId::Booleanity,
            )
        {
            return Some(*opening_claim);
        }
    }
    for (index, opening_claim) in claims.booleanity.ram_ra.iter().enumerate() {
        if id
            == JoltOpeningId::committed(
                JoltCommittedPolynomial::RamRa(index),
                JoltRelationId::Booleanity,
            )
        {
            return Some(*opening_claim);
        }
    }
    let [ram_hamming_weight] = ram::hamming_booleanity_output_openings();
    if id == ram_hamming_weight {
        return Some(claims.ram_hamming_booleanity.ram_hamming_weight);
    }
    for (index, opening_claim) in claims.ram_ra_virtualization.ram_ra.iter().enumerate() {
        if id == ram::ra_virtualization_committed_ram_ra_opening(index) {
            return Some(*opening_claim);
        }
    }
    for (index, opening_claim) in claims
        .instruction_ra_virtualization
        .committed_instruction_ra
        .iter()
        .enumerate()
    {
        if id == instruction::ra_virtualization_committed_instruction_ra_opening(index) {
            return Some(*opening_claim);
        }
    }
    let [ram_inc, rd_inc] = increments::claim_reduction_output_openings();
    match id {
        id if id == bytecode::bytecode_read_raf_address_phase_opening() => {
            Some(claims.address_phase.bytecode_read_raf)
        }
        id if id == booleanity::booleanity_address_phase_opening() => {
            Some(claims.address_phase.booleanity)
        }
        id if id == ram_inc => Some(claims.inc_claim_reduction.ram_inc),
        id if id == rd_inc => Some(claims.inc_claim_reduction.rd_inc),
        id if id == advice::cycle_phase_advice_opening(JoltAdviceKind::Trusted)
            || id == advice::final_advice_opening(JoltAdviceKind::Trusted) =>
        {
            claims
                .advice_cycle_phase
                .trusted
                .as_ref()
                .map(|claim| claim.opening_claim)
        }
        id if id == advice::cycle_phase_advice_opening(JoltAdviceKind::Untrusted)
            || id == advice::final_advice_opening(JoltAdviceKind::Untrusted) =>
        {
            claims
                .advice_cycle_phase
                .untrusted
                .as_ref()
                .map(|claim| claim.opening_claim)
        }
        _ => None,
    }
}

#[cfg(all(any(feature = "prover-fixtures", test), not(feature = "zk")))]
fn claim_mut_from_stage6_outputs<F: Field>(
    claims: &mut Stage6Claims<F>,
    id: jolt::JoltOpeningId,
) -> Option<&mut F> {
    for (index, opening_claim) in claims.bytecode_read_raf.bytecode_ra.iter_mut().enumerate() {
        if id
            == JoltOpeningId::committed(
                JoltCommittedPolynomial::BytecodeRa(index),
                JoltRelationId::BytecodeReadRaf,
            )
        {
            return Some(opening_claim);
        }
    }
    for (index, opening_claim) in claims.booleanity.instruction_ra.iter_mut().enumerate() {
        if id
            == JoltOpeningId::committed(
                JoltCommittedPolynomial::InstructionRa(index),
                JoltRelationId::Booleanity,
            )
        {
            return Some(opening_claim);
        }
    }
    for (index, opening_claim) in claims.booleanity.bytecode_ra.iter_mut().enumerate() {
        if id
            == JoltOpeningId::committed(
                JoltCommittedPolynomial::BytecodeRa(index),
                JoltRelationId::Booleanity,
            )
        {
            return Some(opening_claim);
        }
    }
    for (index, opening_claim) in claims.booleanity.ram_ra.iter_mut().enumerate() {
        if id
            == JoltOpeningId::committed(
                JoltCommittedPolynomial::RamRa(index),
                JoltRelationId::Booleanity,
            )
        {
            return Some(opening_claim);
        }
    }
    let [ram_hamming_weight] = ram::hamming_booleanity_output_openings();
    if id == ram_hamming_weight {
        return Some(&mut claims.ram_hamming_booleanity.ram_hamming_weight);
    }
    for (index, opening_claim) in claims.ram_ra_virtualization.ram_ra.iter_mut().enumerate() {
        if id == ram::ra_virtualization_committed_ram_ra_opening(index) {
            return Some(opening_claim);
        }
    }
    for (index, opening_claim) in claims
        .instruction_ra_virtualization
        .committed_instruction_ra
        .iter_mut()
        .enumerate()
    {
        if id == instruction::ra_virtualization_committed_instruction_ra_opening(index) {
            return Some(opening_claim);
        }
    }
    let [ram_inc, rd_inc] = increments::claim_reduction_output_openings();
    match id {
        id if id == bytecode::bytecode_read_raf_address_phase_opening() => {
            Some(&mut claims.address_phase.bytecode_read_raf)
        }
        id if id == booleanity::booleanity_address_phase_opening() => {
            Some(&mut claims.address_phase.booleanity)
        }
        id if id == ram_inc => Some(&mut claims.inc_claim_reduction.ram_inc),
        id if id == rd_inc => Some(&mut claims.inc_claim_reduction.rd_inc),
        id if id == advice::cycle_phase_advice_opening(JoltAdviceKind::Trusted)
            || id == advice::final_advice_opening(JoltAdviceKind::Trusted) =>
        {
            claims
                .advice_cycle_phase
                .trusted
                .as_mut()
                .map(|claim| &mut claim.opening_claim)
        }
        id if id == advice::cycle_phase_advice_opening(JoltAdviceKind::Untrusted)
            || id == advice::final_advice_opening(JoltAdviceKind::Untrusted) =>
        {
            claims
                .advice_cycle_phase
                .untrusted
                .as_mut()
                .map(|claim| &mut claim.opening_claim)
        }
        _ => None,
    }
}

#[cfg(all(any(feature = "prover-fixtures", test), not(feature = "zk")))]
fn claim_from_stage7_outputs<F: Field>(
    claims: &Stage7Claims<F>,
    id: jolt::JoltOpeningId,
) -> Option<F> {
    for (index, opening) in claims
        .hamming_weight_claim_reduction
        .instruction_ra
        .iter()
        .enumerate()
    {
        if id
            == JoltOpeningId::committed(
                JoltCommittedPolynomial::InstructionRa(index),
                JoltRelationId::HammingWeightClaimReduction,
            )
        {
            return Some(*opening);
        }
    }
    for (index, opening) in claims
        .hamming_weight_claim_reduction
        .bytecode_ra
        .iter()
        .enumerate()
    {
        if id
            == JoltOpeningId::committed(
                JoltCommittedPolynomial::BytecodeRa(index),
                JoltRelationId::HammingWeightClaimReduction,
            )
        {
            return Some(*opening);
        }
    }
    for (index, opening) in claims
        .hamming_weight_claim_reduction
        .ram_ra
        .iter()
        .enumerate()
    {
        if id
            == JoltOpeningId::committed(
                JoltCommittedPolynomial::RamRa(index),
                JoltRelationId::HammingWeightClaimReduction,
            )
        {
            return Some(*opening);
        }
    }

    match id {
        id if id == advice::final_advice_opening(JoltAdviceKind::Trusted) => claims
            .advice_address_phase
            .trusted
            .as_ref()
            .map(|claim| claim.opening_claim),
        id if id == advice::final_advice_opening(JoltAdviceKind::Untrusted) => claims
            .advice_address_phase
            .untrusted
            .as_ref()
            .map(|claim| claim.opening_claim),
        _ => None,
    }
}

#[cfg(all(any(feature = "prover-fixtures", test), not(feature = "zk")))]
fn claim_mut_from_stage7_outputs<F: Field>(
    claims: &mut Stage7Claims<F>,
    id: jolt::JoltOpeningId,
) -> Option<&mut F> {
    for (index, opening) in claims
        .hamming_weight_claim_reduction
        .instruction_ra
        .iter_mut()
        .enumerate()
    {
        if id
            == JoltOpeningId::committed(
                JoltCommittedPolynomial::InstructionRa(index),
                JoltRelationId::HammingWeightClaimReduction,
            )
        {
            return Some(opening);
        }
    }
    for (index, opening) in claims
        .hamming_weight_claim_reduction
        .bytecode_ra
        .iter_mut()
        .enumerate()
    {
        if id
            == JoltOpeningId::committed(
                JoltCommittedPolynomial::BytecodeRa(index),
                JoltRelationId::HammingWeightClaimReduction,
            )
        {
            return Some(opening);
        }
    }
    for (index, opening) in claims
        .hamming_weight_claim_reduction
        .ram_ra
        .iter_mut()
        .enumerate()
    {
        if id
            == JoltOpeningId::committed(
                JoltCommittedPolynomial::RamRa(index),
                JoltRelationId::HammingWeightClaimReduction,
            )
        {
            return Some(opening);
        }
    }

    match id {
        id if id == advice::final_advice_opening(JoltAdviceKind::Trusted) => claims
            .advice_address_phase
            .trusted
            .as_mut()
            .map(|claim| &mut claim.opening_claim),
        id if id == advice::final_advice_opening(JoltAdviceKind::Untrusted) => claims
            .advice_address_phase
            .untrusted
            .as_mut()
            .map(|claim| &mut claim.opening_claim),
        _ => None,
    }
}
