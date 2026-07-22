//! Opening-claim projection for verifier-native prover proofs.
#[cfg(not(feature = "akita"))]
use jolt_claims::protocols::jolt::geometry::claim_reductions::increments;
use jolt_claims::protocols::jolt::geometry::spartan::SpartanOuterDimensions;
use jolt_claims::protocols::jolt::{
    self as native,
    geometry::{
        booleanity, bytecode,
        claim_reductions::registers as registers_claim_reduction,
        claim_reductions::{advice, instruction as instruction_claim_reduction},
        instruction, ram, registers, spartan,
        spartan::{outer_opening, outer_uniskip_opening, product_uniskip_opening},
    },
    JoltAdviceKind, JoltCommittedPolynomial, JoltOpeningId, JoltRelationId, JoltVirtualPolynomial,
};
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_lookup_tables::{LookupTableKind, XLEN as RISCV_XLEN};
use jolt_openings::CommitmentScheme;
use jolt_riscv::CircuitFlags;
use jolt_verifier::{
    proof::{ClearProofClaims, JoltProof, JoltProofClaims},
    stages::{
        stage1::outputs::Stage1BatchOutputClaims, stage2::outputs::Stage2BatchOutputClaims,
        stage3::outputs::Stage3OutputClaims, stage4::Stage4OutputClaims,
        stage5::Stage5OutputClaims, stage6a::outputs::Stage6aOutputClaims,
        stage6b::outputs::Stage6bOutputClaims, stage7::outputs::Stage7OutputClaims,
    },
};

#[doc(hidden)]
pub fn attach_empty_opening_claims<PCS, VC, ZkProof>(proof: &mut JoltProof<PCS, VC, ZkProof>)
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    proof.claims = JoltProofClaims::Clear(crate::support::tamper_manifest::clear_claims(false));
}

#[doc(hidden)]
pub fn offset_opening_claim<PCS, VC, ZkProof>(
    proof: &mut JoltProof<PCS, VC, ZkProof>,
    id: native::JoltOpeningId,
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
pub fn opening_claim<PCS, VC, ZkProof>(
    proof: &JoltProof<PCS, VC, ZkProof>,
    id: native::JoltOpeningId,
) -> Option<PCS::Field>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    claim(proof, id)
}

fn claim<PCS, VC, ZkProof>(
    proof: &JoltProof<PCS, VC, ZkProof>,
    id: native::JoltOpeningId,
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

fn claim_mut<PCS, VC, ZkProof>(
    proof: &mut JoltProof<PCS, VC, ZkProof>,
    id: native::JoltOpeningId,
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

fn claim_from_clear<F: Field>(
    claims: &ClearProofClaims<F>,
    trace_length: usize,
    id: native::JoltOpeningId,
) -> Option<F> {
    // The &mut chain is the single source of truth for the ID -> field mapping;
    // the read path clones and projects through it to avoid duplicating it.
    let mut copy = claims.clone();
    claim_mut_from_clear(&mut copy, trace_length, id).map(|value| *value)
}

fn claim_mut_from_clear<F: Field>(
    claims: &mut ClearProofClaims<F>,
    trace_length: usize,
    id: native::JoltOpeningId,
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
        .or_else(|| claim_mut_from_stage6_outputs(&mut claims.stage6a, &mut claims.stage6b, id))
}

fn claim_mut_from_spartan_outer<F: Field>(
    claims: &mut Stage1BatchOutputClaims<F>,
    variable: JoltVirtualPolynomial,
) -> Option<&mut F> {
    let claims = &mut claims.outer_remainder;
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
        JoltVirtualPolynomial::OpFlags(flag) => match flag {
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
        },
        _ => None,
    }
}

fn stage1_outer_variable(
    trace_length: usize,
    id: native::JoltOpeningId,
) -> Option<JoltVirtualPolynomial> {
    let log_t = trace_length.ilog2() as usize;
    SpartanOuterDimensions::rv64(log_t)
        .variables()
        .iter()
        .copied()
        .find(|variable| id == outer_opening(*variable))
}

fn claim_mut_from_stage2_batch_outputs<F: Field>(
    claims: &mut Stage2BatchOutputClaims<F>,
    id: native::JoltOpeningId,
) -> Option<&mut F> {
    let [ram_val, ram_ra, ram_inc] = [ram::ram_val(), ram::ram_ra(), ram::ram_inc()];
    let [product_left_instruction_input, product_right_instruction_input, product_jump_flag, product_write_lookup_output_to_rd, product_lookup_output, product_branch_flag, product_next_is_noop, product_virtual_instruction] = [
        spartan::left_instruction_input_product(),
        spartan::right_instruction_input_product(),
        spartan::jump_flag_product(),
        spartan::write_lookup_output_to_rd_product(),
        spartan::lookup_output_product(),
        spartan::branch_flag_product(),
        spartan::next_is_noop_product(),
        spartan::virtual_instruction_product(),
    ];
    let [instruction_lookup_output, instruction_left_lookup_operand, instruction_right_lookup_operand, instruction_left_instruction_input, instruction_right_instruction_input] = [
        instruction_claim_reduction::lookup_output_reduced(),
        instruction_claim_reduction::left_lookup_operand_reduced(),
        instruction_claim_reduction::right_lookup_operand_reduced(),
        instruction_claim_reduction::left_instruction_input_reduced(),
        instruction_claim_reduction::right_instruction_input_reduced(),
    ];
    let ram_ra_raf_evaluation = ram::ram_ra_raf_evaluation();
    let ram_val_final = ram::ram_val_final();

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
            Some(&mut claims.instruction_claim_reduction.lookup_output)
        }
        id if id == instruction_left_lookup_operand => {
            Some(&mut claims.instruction_claim_reduction.left_lookup_operand)
        }
        id if id == instruction_right_lookup_operand => {
            Some(&mut claims.instruction_claim_reduction.right_lookup_operand)
        }
        id if id == instruction_left_instruction_input => {
            Some(&mut claims.instruction_claim_reduction.left_instruction_input)
        }
        id if id == instruction_right_instruction_input => {
            Some(&mut claims.instruction_claim_reduction.right_instruction_input)
        }
        id if id == ram_ra_raf_evaluation => Some(&mut claims.ram_raf_evaluation.ram_ra),
        id if id == ram_val_final => Some(&mut claims.ram_output_check.val_final),
        _ => None,
    }
}

fn claim_mut_from_stage3_outputs<F: Field>(
    claims: &mut Stage3OutputClaims<F>,
    id: native::JoltOpeningId,
) -> Option<&mut F> {
    let [unexpanded_pc_shift, pc_shift, is_virtual_shift, is_first_in_sequence_shift, is_noop_shift] = [
        spartan::unexpanded_pc_shift(),
        spartan::pc_shift(),
        spartan::is_virtual_shift(),
        spartan::is_first_in_sequence_shift(),
        spartan::is_noop_shift(),
    ];
    let [right_operand_is_rs2, rs2_value_input, right_operand_is_imm, imm_input, left_operand_is_rs1, rs1_value_input, left_operand_is_pc, unexpanded_pc_input] = [
        instruction::right_operand_is_rs2(),
        instruction::rs2_value(),
        instruction::right_operand_is_imm(),
        instruction::imm(),
        instruction::left_operand_is_rs1(),
        instruction::rs1_value(),
        instruction::left_operand_is_pc(),
        instruction::unexpanded_pc(),
    ];
    let [rd_write_value_reduced, rs1_value_reduced, rs2_value_reduced] = [
        registers_claim_reduction::rd_write_value_reduced(),
        registers_claim_reduction::rs1_value_reduced(),
        registers_claim_reduction::rs2_value_reduced(),
    ];

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

fn claim_mut_from_stage4_outputs<F: Field>(
    claims: &mut Stage4OutputClaims<F>,
    id: native::JoltOpeningId,
) -> Option<&mut F> {
    let [registers_val, rs1_ra, rs2_ra, rd_wa, rd_inc] = [
        registers::registers_val_read_write(),
        registers::rs1_ra_read_write(),
        registers::rs2_ra_read_write(),
        registers::rd_wa_read_write(),
        registers::rd_inc_read_write(),
    ];
    let [ram_ra, ram_inc] = [ram::ram_ra_val_check(), ram::ram_inc_val_check()];

    match id {
        id if id == ram::val_check_advice_opening(JoltAdviceKind::Untrusted) => {
            claims.ram_val_check.untrusted_advice.as_mut()
        }
        id if id == ram::val_check_advice_opening(JoltAdviceKind::Trusted) => {
            claims.ram_val_check.trusted_advice.as_mut()
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

fn claim_mut_from_stage5_outputs<F: Field>(
    claims: &mut Stage5OutputClaims<F>,
    id: native::JoltOpeningId,
) -> Option<&mut F> {
    for table in LookupTableKind::<RISCV_XLEN>::iter() {
        if id == instruction::lookup_table_flag(table) {
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
        if id == instruction::instruction_ra(index) {
            return Some(opening_claim);
        }
    }

    let ram_ra = ram::ram_ra_claim_reduction();
    let [rd_inc, rd_wa] = [
        registers::rd_inc_val_evaluation(),
        registers::rd_wa_val_evaluation(),
    ];
    match id {
        id if id == instruction::instruction_raf_flag() => {
            Some(&mut claims.instruction_read_raf.instruction_raf_flag)
        }
        id if id == ram_ra => Some(&mut claims.ram_ra_claim_reduction.ram_ra),
        id if id == rd_inc => Some(&mut claims.registers_val_evaluation.rd_inc),
        id if id == rd_wa => Some(&mut claims.registers_val_evaluation.rd_wa),
        _ => None,
    }
}

fn claim_mut_from_stage6_outputs<'a, F: Field>(
    stage6a: &'a mut Stage6aOutputClaims<F>,
    stage6b: &'a mut Stage6bOutputClaims<F>,
    id: native::JoltOpeningId,
) -> Option<&'a mut F> {
    for (index, opening_claim) in stage6b.bytecode_read_raf.bytecode_ra.iter_mut().enumerate() {
        if id
            == JoltOpeningId::committed(
                JoltCommittedPolynomial::BytecodeRa(index),
                JoltRelationId::BytecodeReadRaf,
            )
        {
            return Some(opening_claim);
        }
    }
    for (index, opening_claim) in stage6b.booleanity.instruction_ra.iter_mut().enumerate() {
        if id
            == JoltOpeningId::committed(
                JoltCommittedPolynomial::InstructionRa(index),
                JoltRelationId::Booleanity,
            )
        {
            return Some(opening_claim);
        }
    }
    for (index, opening_claim) in stage6b.booleanity.bytecode_ra.iter_mut().enumerate() {
        if id
            == JoltOpeningId::committed(
                JoltCommittedPolynomial::BytecodeRa(index),
                JoltRelationId::Booleanity,
            )
        {
            return Some(opening_claim);
        }
    }
    for (index, opening_claim) in stage6b.booleanity.ram_ra.iter_mut().enumerate() {
        if id
            == JoltOpeningId::committed(
                JoltCommittedPolynomial::RamRa(index),
                JoltRelationId::Booleanity,
            )
        {
            return Some(opening_claim);
        }
    }
    let ram_hamming_weight = ram::ram_hamming_weight();
    if id == ram_hamming_weight {
        return Some(&mut stage6b.ram_hamming_booleanity.ram_hamming_weight);
    }
    for (index, opening_claim) in stage6b.ram_ra_virtualization.ram_ra.iter_mut().enumerate() {
        if id == ram::committed_ram_ra(index) {
            return Some(opening_claim);
        }
    }
    for (index, opening_claim) in stage6b
        .instruction_ra_virtualization
        .committed_instruction_ra
        .iter_mut()
        .enumerate()
    {
        if id == instruction::committed_instruction_ra(index) {
            return Some(opening_claim);
        }
    }
    #[cfg(not(feature = "akita"))]
    let [ram_inc, rd_inc] = [increments::ram_inc_reduced(), increments::rd_inc_reduced()];
    match id {
        id if id == bytecode::bytecode_read_raf_address_phase_opening() => {
            Some(&mut stage6a.bytecode_read_raf.intermediate)
        }
        id if id == booleanity::booleanity_address_phase_opening() => {
            Some(&mut stage6a.booleanity.intermediate)
        }
        #[cfg(not(feature = "akita"))]
        id if id == ram_inc => Some(&mut stage6b.inc_claim_reduction.ram_inc),
        #[cfg(not(feature = "akita"))]
        id if id == rd_inc => Some(&mut stage6b.inc_claim_reduction.rd_inc),
        id if id == advice::cycle_phase_advice_opening(JoltAdviceKind::Trusted)
            || id == advice::final_advice_opening(JoltAdviceKind::Trusted) =>
        {
            stage6b
                .trusted_advice
                .as_mut()
                .map(|claim| &mut claim.trusted)
        }
        id if id == advice::cycle_phase_advice_opening(JoltAdviceKind::Untrusted)
            || id == advice::final_advice_opening(JoltAdviceKind::Untrusted) =>
        {
            stage6b
                .untrusted_advice
                .as_mut()
                .map(|claim| &mut claim.untrusted)
        }
        _ => None,
    }
}

fn claim_mut_from_stage7_outputs<F: Field>(
    claims: &mut Stage7OutputClaims<F>,
    id: native::JoltOpeningId,
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
            .trusted_advice
            .as_mut()
            .map(|claims| &mut claims.trusted),
        id if id == advice::final_advice_opening(JoltAdviceKind::Untrusted) => claims
            .untrusted_advice
            .as_mut()
            .map(|claims| &mut claims.untrusted),
        _ => None,
    }
}
