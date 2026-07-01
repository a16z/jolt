//! Opening-claim projection for verifier-native prover proofs.

use jolt_claims::protocols::jolt::{
    self as jolt,
    geometry::{
        booleanity, bytecode,
        claim_reductions::registers as registers_claim_reduction,
        claim_reductions::{
            advice, bytecode as bytecode_claim_reduction, increments,
            instruction as instruction_claim_reduction, program_image,
        },
        instruction, ram, registers, spartan,
        spartan::{outer_opening, outer_uniskip_opening, product_uniskip_opening},
    },
    JoltAdviceKind, JoltCommittedPolynomial, JoltOpeningId, JoltRelationId, JoltVirtualPolynomial,
};
use jolt_field::Field;
use jolt_lookup_tables::{LookupTableKind, XLEN as RISCV_XLEN};
use jolt_riscv::CircuitFlags;
use jolt_verifier::{
    proof::ClearProofClaims,
    stages::{
        stage1::outputs::{SpartanOuterClaims, SpartanOuterFlagClaims, Stage1OutputClaims},
        stage2::outputs::{
            InstructionClaimReductionOutputClaims, ProductRemainderOutputClaims,
            RamOutputCheckOutputClaims, RamRafEvaluationOutputClaims, RamReadWriteOutputClaims,
            Stage2BatchOutputClaims, Stage2OutputClaims,
        },
        stage3::outputs::{
            InstructionInputOutputClaims, RegistersClaimReductionOutputClaims,
            SpartanShiftOutputClaims, Stage3OutputClaims,
        },
        stage4::{
            RamValCheckAdviceClaims, RamValCheckOutputClaims, RegistersReadWriteOutputClaims,
            Stage4OutputClaims,
        },
        stage5::{
            InstructionReadRafOutputClaims, RamRaClaimReductionOutputClaims,
            RegistersValEvaluationOutputClaims, Stage5OutputClaims,
        },
        stage6::outputs::{
            AdviceCyclePhaseOutputClaim, BooleanityOutputClaims, BytecodeCyclePhaseOutputClaims,
            BytecodeReadRafOutputClaims, IncClaimReductionOutputClaims,
            InstructionRaVirtualizationOutputClaims, ProgramImageCyclePhaseOutputClaim,
            RamHammingBooleanityOutputClaims, RamRaVirtualizationOutputClaims,
            Stage6AddressPhaseClaims, Stage6AdviceCyclePhaseClaims, Stage6OutputClaims,
        },
        stage7::{
            advice_address_phase::AdviceAddressPhaseOutputClaims,
            committed_reduction_address_phase::{
                BytecodeReductionAddressPhaseOutputClaims,
                ProgramImageReductionAddressPhaseOutputClaims,
            },
            hamming_weight_claim_reduction::HammingWeightClaimReductionOutputClaims,
            outputs::Stage7OutputClaims,
        },
    },
    VerifierError,
};

pub(crate) fn build_clear_claims<F: Field>(
    claims: impl IntoIterator<Item = (jolt::JoltOpeningId, F)>,
    _trace_length: usize,
) -> Result<ClearProofClaims<F>, VerifierError> {
    let claims = OpeningClaimMap {
        claims: claims.into_iter().collect(),
    };
    Ok(ClearProofClaims {
        stage1: Stage1OutputClaims {
            uniskip_output_claim: claims.require(outer_uniskip_opening())?,
            outer: spartan_outer_claims_from_openings(&claims)?,
        },
        stage2: stage2_claims_from_openings(&claims)?,
        stage3: stage3_claims_from_openings(&claims)?,
        stage4: stage4_claims_from_openings(&claims)?,
        stage5: stage5_claims_from_openings(&claims)?,
        stage6: stage6_claims_from_openings(&claims)?,
        stage7: stage7_claims_from_openings(&claims)?,
    })
}

fn spartan_outer_claims_from_openings<F: Field>(
    claims: &OpeningClaimMap<F>,
) -> Result<SpartanOuterClaims<F>, VerifierError> {
    let outer_claim = |variable| claims.require(outer_opening(variable));
    let flag_claim = |flag| outer_claim(JoltVirtualPolynomial::OpFlags(flag));

    Ok(SpartanOuterClaims {
        left_instruction_input: outer_claim(JoltVirtualPolynomial::LeftInstructionInput)?,
        right_instruction_input: outer_claim(JoltVirtualPolynomial::RightInstructionInput)?,
        product: outer_claim(JoltVirtualPolynomial::Product)?,
        should_branch: outer_claim(JoltVirtualPolynomial::ShouldBranch)?,
        pc: outer_claim(JoltVirtualPolynomial::PC)?,
        unexpanded_pc: outer_claim(JoltVirtualPolynomial::UnexpandedPC)?,
        imm: outer_claim(JoltVirtualPolynomial::Imm)?,
        ram_address: outer_claim(JoltVirtualPolynomial::RamAddress)?,
        rs1_value: outer_claim(JoltVirtualPolynomial::Rs1Value)?,
        rs2_value: outer_claim(JoltVirtualPolynomial::Rs2Value)?,
        rd_write_value: outer_claim(JoltVirtualPolynomial::RdWriteValue)?,
        ram_read_value: outer_claim(JoltVirtualPolynomial::RamReadValue)?,
        ram_write_value: outer_claim(JoltVirtualPolynomial::RamWriteValue)?,
        left_lookup_operand: outer_claim(JoltVirtualPolynomial::LeftLookupOperand)?,
        right_lookup_operand: outer_claim(JoltVirtualPolynomial::RightLookupOperand)?,
        next_unexpanded_pc: outer_claim(JoltVirtualPolynomial::NextUnexpandedPC)?,
        next_pc: outer_claim(JoltVirtualPolynomial::NextPC)?,
        next_is_virtual: outer_claim(JoltVirtualPolynomial::NextIsVirtual)?,
        next_is_first_in_sequence: outer_claim(JoltVirtualPolynomial::NextIsFirstInSequence)?,
        lookup_output: outer_claim(JoltVirtualPolynomial::LookupOutput)?,
        should_jump: outer_claim(JoltVirtualPolynomial::ShouldJump)?,
        flags: SpartanOuterFlagClaims {
            add_operands: flag_claim(CircuitFlags::AddOperands)?,
            subtract_operands: flag_claim(CircuitFlags::SubtractOperands)?,
            multiply_operands: flag_claim(CircuitFlags::MultiplyOperands)?,
            load: flag_claim(CircuitFlags::Load)?,
            store: flag_claim(CircuitFlags::Store)?,
            jump: flag_claim(CircuitFlags::Jump)?,
            write_lookup_output_to_rd: flag_claim(CircuitFlags::WriteLookupOutputToRD)?,
            virtual_instruction: flag_claim(CircuitFlags::VirtualInstruction)?,
            assert: flag_claim(CircuitFlags::Assert)?,
            do_not_update_unexpanded_pc: flag_claim(CircuitFlags::DoNotUpdateUnexpandedPC)?,
            advice: flag_claim(CircuitFlags::Advice)?,
            is_compressed: flag_claim(CircuitFlags::IsCompressed)?,
            is_first_in_sequence: flag_claim(CircuitFlags::IsFirstInSequence)?,
            is_last_in_sequence: flag_claim(CircuitFlags::IsLastInSequence)?,
        },
    })
}

fn stage2_claims_from_openings<F: Field>(
    claims: &OpeningClaimMap<F>,
) -> Result<Stage2OutputClaims<F>, VerifierError> {
    Ok(Stage2OutputClaims {
        product_uniskip_output_claim: claims.require(product_uniskip_opening())?,
        batch_outputs: Stage2BatchOutputClaims {
            ram_read_write: RamReadWriteOutputClaims {
                val: claims.get_or_zero(ram::ram_val()),
                ra: claims.get_or_zero(ram::ram_ra()),
                inc: claims.get_or_zero(ram::ram_inc()),
            },
            product_remainder: ProductRemainderOutputClaims {
                left_instruction_input: claims
                    .get_or_zero(spartan::left_instruction_input_product()),
                right_instruction_input: claims
                    .get_or_zero(spartan::right_instruction_input_product()),
                jump_flag: claims.get_or_zero(spartan::jump_flag_product()),
                write_lookup_output_to_rd: claims
                    .get_or_zero(spartan::write_lookup_output_to_rd_product()),
                lookup_output: claims.get_or_zero(spartan::lookup_output_product()),
                branch_flag: claims.get_or_zero(spartan::branch_flag_product()),
                next_is_noop: claims.get_or_zero(spartan::next_is_noop_product()),
                virtual_instruction: claims.get_or_zero(spartan::virtual_instruction_product()),
            },
            instruction_claim_reduction: InstructionClaimReductionOutputClaims {
                lookup_output: claims.get(instruction_claim_reduction::lookup_output_reduced()),
                left_lookup_operand: claims
                    .get_or_zero(instruction_claim_reduction::left_lookup_operand_reduced()),
                right_lookup_operand: claims
                    .get_or_zero(instruction_claim_reduction::right_lookup_operand_reduced()),
                left_instruction_input: claims
                    .get(instruction_claim_reduction::left_instruction_input_reduced()),
                right_instruction_input: claims
                    .get(instruction_claim_reduction::right_instruction_input_reduced()),
            },
            ram_raf_evaluation: RamRafEvaluationOutputClaims {
                ram_ra: claims.get_or_zero(ram::ram_ra_raf_evaluation()),
            },
            ram_output_check: RamOutputCheckOutputClaims {
                val_final: claims.get_or_zero(ram::ram_val_final()),
            },
        },
    })
}

fn stage3_claims_from_openings<F: Field>(
    claims: &OpeningClaimMap<F>,
) -> Result<Stage3OutputClaims<F>, VerifierError> {
    let shift = SpartanShiftOutputClaims {
        unexpanded_pc: claims.require(spartan::unexpanded_pc_shift())?,
        pc: claims.require(spartan::pc_shift())?,
        is_virtual: claims.require(spartan::is_virtual_shift())?,
        is_first_in_sequence: claims.require(spartan::is_first_in_sequence_shift())?,
        is_noop: claims.require(spartan::is_noop_shift())?,
    };
    let instruction_input = InstructionInputOutputClaims {
        left_operand_is_rs1: claims.require(instruction::left_operand_is_rs1())?,
        rs1_value: claims.require(instruction::rs1_value())?,
        left_operand_is_pc: claims.require(instruction::left_operand_is_pc())?,
        unexpanded_pc: claims
            .get(instruction::unexpanded_pc())
            .unwrap_or(shift.unexpanded_pc),
        right_operand_is_rs2: claims.require(instruction::right_operand_is_rs2())?,
        rs2_value: claims.require(instruction::rs2_value())?,
        right_operand_is_imm: claims.require(instruction::right_operand_is_imm())?,
        imm: claims.require(instruction::imm())?,
    };
    let registers_claim_reduction = RegistersClaimReductionOutputClaims {
        rd_write_value: claims.require(registers_claim_reduction::rd_write_value_reduced())?,
        rs1_value: claims
            .get(registers_claim_reduction::rs1_value_reduced())
            .unwrap_or(instruction_input.rs1_value),
        rs2_value: claims
            .get(registers_claim_reduction::rs2_value_reduced())
            .unwrap_or(instruction_input.rs2_value),
    };

    Ok(Stage3OutputClaims {
        shift,
        instruction_input,
        registers_claim_reduction,
    })
}

fn stage4_claims_from_openings<F: Field>(
    claims: &OpeningClaimMap<F>,
) -> Result<Stage4OutputClaims<F>, VerifierError> {
    Ok(Stage4OutputClaims {
        advice: RamValCheckAdviceClaims {
            untrusted: claims.get(ram::val_check_advice_opening(JoltAdviceKind::Untrusted)),
            trusted: claims.get(ram::val_check_advice_opening(JoltAdviceKind::Trusted)),
        },
        program_image_contribution: claims.get(program_image::ram_val_check_contribution_opening()),
        registers_read_write: RegistersReadWriteOutputClaims {
            registers_val: claims.require(registers::registers_val_read_write())?,
            rs1_ra: claims.require(registers::rs1_ra_read_write())?,
            rs2_ra: claims.require(registers::rs2_ra_read_write())?,
            rd_wa: claims.require(registers::rd_wa_read_write())?,
            rd_inc: claims.require(registers::rd_inc_read_write())?,
        },
        ram_val_check: RamValCheckOutputClaims {
            ram_ra: claims.require(ram::ram_ra_val_check())?,
            ram_inc: claims.require(ram::ram_inc_val_check())?,
        },
    })
}

fn stage5_claims_from_openings<F: Field>(
    claims: &OpeningClaimMap<F>,
) -> Result<Stage5OutputClaims<F>, VerifierError> {
    let lookup_table_flags = LookupTableKind::<RISCV_XLEN>::iter()
        .map(|table| claims.require(instruction::read_raf_lookup_table_flag_opening(table)))
        .collect::<Result<Vec<_>, _>>()?;
    let mut instruction_ra = Vec::new();
    for index in 0.. {
        let Some(opening_claim) = claims.get(instruction::read_raf_instruction_ra_opening(index))
        else {
            break;
        };
        instruction_ra.push(opening_claim);
    }
    if instruction_ra.is_empty() {
        return Err(VerifierError::MissingOpeningClaim {
            id: instruction::read_raf_instruction_ra_opening(0),
        });
    }
    Ok(Stage5OutputClaims {
        instruction_read_raf: InstructionReadRafOutputClaims {
            lookup_table_flags,
            instruction_ra,
            instruction_raf_flag: claims
                .require(instruction::read_raf_instruction_raf_flag_opening())?,
        },
        ram_ra_claim_reduction: RamRaClaimReductionOutputClaims {
            ram_ra: claims.require(ram::ram_ra_claim_reduction())?,
        },
        registers_val_evaluation: RegistersValEvaluationOutputClaims {
            rd_inc: claims.require(registers::rd_inc_val_evaluation())?,
            rd_wa: claims.require(registers::rd_wa_val_evaluation())?,
        },
    })
}

fn stage6_claims_from_openings<F: Field>(
    claims: &OpeningClaimMap<F>,
) -> Result<Stage6OutputClaims<F>, VerifierError> {
    let mut bytecode_ra = Vec::new();
    for index in 0.. {
        let id = JoltOpeningId::committed(
            JoltCommittedPolynomial::BytecodeRa(index),
            JoltRelationId::BytecodeReadRaf,
        );
        let Some(opening_claim) = claims.get(id) else {
            break;
        };
        bytecode_ra.push(opening_claim);
    }
    if bytecode_ra.is_empty() {
        return Err(VerifierError::MissingOpeningClaim {
            id: JoltOpeningId::committed(
                JoltCommittedPolynomial::BytecodeRa(0),
                JoltRelationId::BytecodeReadRaf,
            ),
        });
    }

    let mut booleanity_instruction_ra = Vec::new();
    for index in 0.. {
        let id = JoltOpeningId::committed(
            JoltCommittedPolynomial::InstructionRa(index),
            JoltRelationId::Booleanity,
        );
        let Some(opening_claim) = claims.get(id) else {
            break;
        };
        booleanity_instruction_ra.push(opening_claim);
    }
    let mut booleanity_bytecode_ra = Vec::new();
    for index in 0.. {
        let id = JoltOpeningId::committed(
            JoltCommittedPolynomial::BytecodeRa(index),
            JoltRelationId::Booleanity,
        );
        let fallback_id = JoltOpeningId::committed(
            JoltCommittedPolynomial::BytecodeRa(index),
            JoltRelationId::BytecodeReadRaf,
        );
        let Some(opening_claim) = claims.get(id).or_else(|| claims.get(fallback_id)) else {
            break;
        };
        booleanity_bytecode_ra.push(opening_claim);
    }
    let mut booleanity_ram_ra = Vec::new();
    for index in 0.. {
        let id = JoltOpeningId::committed(
            JoltCommittedPolynomial::RamRa(index),
            JoltRelationId::Booleanity,
        );
        let Some(opening_claim) = claims.get(id) else {
            break;
        };
        booleanity_ram_ra.push(opening_claim);
    }
    if booleanity_instruction_ra.is_empty()
        && booleanity_bytecode_ra.is_empty()
        && booleanity_ram_ra.is_empty()
    {
        return Err(VerifierError::MissingOpeningClaim {
            id: JoltOpeningId::committed(
                JoltCommittedPolynomial::InstructionRa(0),
                JoltRelationId::Booleanity,
            ),
        });
    }

    let mut ram_ra = Vec::new();
    for index in 0.. {
        let id = ram::ra_virtualization_committed_ram_ra_opening(index);
        let Some(opening_claim) = claims.get(id) else {
            break;
        };
        ram_ra.push(opening_claim);
    }

    let mut committed_instruction_ra = Vec::new();
    for index in 0.. {
        let id = instruction::ra_virtualization_committed_instruction_ra_opening(index);
        let Some(opening_claim) = claims.get(id) else {
            break;
        };
        committed_instruction_ra.push(opening_claim);
    }
    if committed_instruction_ra.is_empty() {
        return Err(VerifierError::MissingOpeningClaim {
            id: instruction::ra_virtualization_committed_instruction_ra_opening(0),
        });
    }

    let bytecode_read_raf_address = bytecode::bytecode_read_raf_address_phase_opening();
    let booleanity_address = booleanity::booleanity_address_phase_opening();

    Ok(Stage6OutputClaims {
        address_phase: Stage6AddressPhaseClaims {
            bytecode_read_raf: claims.require(bytecode_read_raf_address)?,
            booleanity: claims.require(booleanity_address)?,
            bytecode_val_stages: bytecode_val_stage_claims_from_openings(claims)?,
        },
        bytecode_read_raf: BytecodeReadRafOutputClaims { bytecode_ra },
        booleanity: BooleanityOutputClaims {
            instruction_ra: booleanity_instruction_ra,
            bytecode_ra: booleanity_bytecode_ra,
            ram_ra: booleanity_ram_ra,
        },
        ram_hamming_booleanity: RamHammingBooleanityOutputClaims {
            ram_hamming_weight: claims.require(ram::ram_hamming_weight())?,
        },
        ram_ra_virtualization: RamRaVirtualizationOutputClaims { ram_ra },
        instruction_ra_virtualization: InstructionRaVirtualizationOutputClaims {
            committed_instruction_ra,
        },
        inc_claim_reduction: IncClaimReductionOutputClaims {
            ram_inc: claims.require(increments::ram_inc_reduced())?,
            rd_inc: claims.require(increments::rd_inc_reduced())?,
        },
        advice_cycle_phase: Stage6AdviceCyclePhaseClaims {
            trusted: advice_cycle_phase_claim_from_openings(claims, JoltAdviceKind::Trusted),
            untrusted: advice_cycle_phase_claim_from_openings(claims, JoltAdviceKind::Untrusted),
        },
        bytecode_claim_reduction: bytecode_cycle_phase_claims_from_openings(claims),
        program_image_claim_reduction: claims
            .get(program_image::cycle_phase_program_image_opening())
            .or_else(|| claims.get(program_image::final_program_image_opening()))
            .map(|opening_claim| ProgramImageCyclePhaseOutputClaim { opening_claim }),
    })
}

fn advice_cycle_phase_claim_from_openings<F: Field>(
    claims: &OpeningClaimMap<F>,
    kind: JoltAdviceKind,
) -> Option<AdviceCyclePhaseOutputClaim<F>> {
    claims
        .get(advice::cycle_phase_advice_opening(kind))
        .or_else(|| claims.get(advice::final_advice_opening(kind)))
        .map(|opening_claim| AdviceCyclePhaseOutputClaim { opening_claim })
}

fn bytecode_val_stage_claims_from_openings<F: Field>(
    claims: &OpeningClaimMap<F>,
) -> Result<Option<[F; bytecode_claim_reduction::NUM_BYTECODE_VAL_STAGES]>, VerifierError> {
    if claims
        .get(bytecode_claim_reduction::bytecode_val_stage_opening(0))
        .is_none()
    {
        return Ok(None);
    }
    let mut stage_claims = [F::zero(); bytecode_claim_reduction::NUM_BYTECODE_VAL_STAGES];
    for (stage, stage_claim) in stage_claims.iter_mut().enumerate() {
        *stage_claim =
            claims.require(bytecode_claim_reduction::bytecode_val_stage_opening(stage))?;
    }
    Ok(Some(stage_claims))
}

fn bytecode_cycle_phase_claims_from_openings<F: Field>(
    claims: &OpeningClaimMap<F>,
) -> Option<BytecodeCyclePhaseOutputClaims<F>> {
    if let Some(intermediate) =
        claims.get(bytecode_claim_reduction::cycle_phase_intermediate_opening())
    {
        return Some(BytecodeCyclePhaseOutputClaims::Intermediate(intermediate));
    }
    let chunks = final_bytecode_chunk_claims_from_openings(claims);
    (!chunks.is_empty()).then_some(BytecodeCyclePhaseOutputClaims::Chunks(chunks))
}

fn final_bytecode_chunk_claims_from_openings<F: Field>(claims: &OpeningClaimMap<F>) -> Vec<F> {
    let mut chunks = Vec::new();
    for chunk_idx in 0.. {
        let Some(opening_claim) = claims.get(
            bytecode_claim_reduction::final_bytecode_chunk_opening(chunk_idx),
        ) else {
            break;
        };
        chunks.push(opening_claim);
    }
    chunks
}

fn stage7_claims_from_openings<F: Field>(
    claims: &OpeningClaimMap<F>,
) -> Result<Stage7OutputClaims<F>, VerifierError> {
    let mut instruction_ra = Vec::new();
    for index in 0.. {
        let id = JoltOpeningId::committed(
            JoltCommittedPolynomial::InstructionRa(index),
            JoltRelationId::HammingWeightClaimReduction,
        );
        let Some(opening_claim) = claims.get(id) else {
            break;
        };
        instruction_ra.push(opening_claim);
    }
    let mut bytecode_ra = Vec::new();
    for index in 0.. {
        let id = JoltOpeningId::committed(
            JoltCommittedPolynomial::BytecodeRa(index),
            JoltRelationId::HammingWeightClaimReduction,
        );
        let Some(opening_claim) = claims.get(id) else {
            break;
        };
        bytecode_ra.push(opening_claim);
    }
    let mut ram_ra = Vec::new();
    for index in 0.. {
        let id = JoltOpeningId::committed(
            JoltCommittedPolynomial::RamRa(index),
            JoltRelationId::HammingWeightClaimReduction,
        );
        let Some(opening_claim) = claims.get(id) else {
            break;
        };
        ram_ra.push(opening_claim);
    }
    if instruction_ra.is_empty() && bytecode_ra.is_empty() && ram_ra.is_empty() {
        return Err(VerifierError::MissingOpeningClaim {
            id: JoltOpeningId::committed(
                JoltCommittedPolynomial::InstructionRa(0),
                JoltRelationId::HammingWeightClaimReduction,
            ),
        });
    }

    Ok(Stage7OutputClaims {
        hamming_weight_claim_reduction: HammingWeightClaimReductionOutputClaims {
            instruction_ra,
            bytecode_ra,
            ram_ra,
        },
        advice_address_phase: AdviceAddressPhaseOutputClaims {
            trusted: advice_address_phase_claim_from_openings(claims, JoltAdviceKind::Trusted),
            untrusted: advice_address_phase_claim_from_openings(claims, JoltAdviceKind::Untrusted),
        },
        bytecode_address_phase: bytecode_address_phase_claims_from_openings(claims),
        program_image_address_phase: program_image_address_phase_claim_from_openings(claims),
    })
}

fn advice_address_phase_claim_from_openings<F: Field>(
    claims: &OpeningClaimMap<F>,
    kind: JoltAdviceKind,
) -> Option<F> {
    let _ = claims.get(advice::cycle_phase_advice_opening(kind))?;
    claims.get(advice::final_advice_opening(kind))
}

fn bytecode_address_phase_claims_from_openings<F: Field>(
    claims: &OpeningClaimMap<F>,
) -> Option<BytecodeReductionAddressPhaseOutputClaims<F>> {
    let _ = claims.get(bytecode_claim_reduction::cycle_phase_intermediate_opening())?;
    let chunks = final_bytecode_chunk_claims_from_openings(claims);
    (!chunks.is_empty()).then_some(BytecodeReductionAddressPhaseOutputClaims { chunks })
}

fn program_image_address_phase_claim_from_openings<F: Field>(
    claims: &OpeningClaimMap<F>,
) -> Option<ProgramImageReductionAddressPhaseOutputClaims<F>> {
    let _ = claims.get(program_image::cycle_phase_program_image_opening())?;
    claims
        .get(program_image::final_program_image_opening())
        .map(|program_image| ProgramImageReductionAddressPhaseOutputClaims { program_image })
}

#[derive(Clone, Debug)]
struct OpeningClaimMap<F: Field> {
    claims: Vec<(jolt::JoltOpeningId, F)>,
}

impl<F: Field> OpeningClaimMap<F> {
    fn get(&self, id: jolt::JoltOpeningId) -> Option<F> {
        self.claims
            .iter()
            .find_map(|&(claim_id, opening_claim)| (claim_id == id).then_some(opening_claim))
    }

    fn require(&self, id: jolt::JoltOpeningId) -> Result<F, VerifierError> {
        self.get(id)
            .ok_or(VerifierError::MissingOpeningClaim { id })
    }

    fn get_or_zero(&self, id: jolt::JoltOpeningId) -> F {
        self.get(id).unwrap_or_else(F::zero)
    }
}
