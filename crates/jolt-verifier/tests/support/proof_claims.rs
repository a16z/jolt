//! Opening-claim projection for verifier-native prover proofs.
#[cfg(any(feature = "prover-fixtures", test))]
use jolt_claims::protocols::jolt::geometry::spartan::SpartanOuterDimensions;
use jolt_claims::protocols::jolt::{
    self as native,
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
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_lookup_tables::{LookupTableKind, XLEN as RISCV_XLEN};
use jolt_openings::CommitmentScheme;
use jolt_riscv::CircuitFlags;
use jolt_verifier::{
    proof::{ClearProofClaims, JoltProof, JoltProofClaims},
    stages::{
        stage1::{
            outputs::{Stage1BatchOutputClaims, Stage1OutputClaims},
            OuterRemainderOutputClaims,
        },
        stage2::outputs::{
            InstructionClaimReductionOutputClaims, ProductRemainderOutputClaims,
            RamOutputCheckOutputClaims, RamRafEvaluationOutputClaims, RamReadWriteOutputClaims,
            Stage2BatchOutputClaims, Stage2OutputClaims,
        },
        stage3::outputs::{
            InstructionInputOutputClaims, RegistersClaimReductionOutputClaims,
            SpartanShiftOutputClaims, Stage3OutputClaims,
        },
        stage4::{RamValCheckOutputClaims, RegistersReadWriteOutputClaims, Stage4OutputClaims},
        stage5::{
            InstructionReadRafOutputClaims, RamRaClaimReductionOutputClaims,
            RegistersValEvaluationOutputClaims, Stage5OutputClaims,
        },
        stage6::outputs::{
            AdviceCyclePhaseOutputClaims, BooleanityAddressPhaseOutputClaims,
            BooleanityOutputClaims, BytecodeReadRafAddressPhaseOutputClaims,
            BytecodeReadRafOutputClaims, BytecodeReductionCyclePhaseOutputClaims,
            IncClaimReductionOutputClaims, InstructionRaVirtualizationOutputClaims,
            ProgramImageReductionCyclePhaseOutputClaims, RamHammingBooleanityOutputClaims,
            RamRaVirtualizationOutputClaims, Stage6AddressPhaseOutputClaims,
            Stage6CyclePhaseOutputClaims, Stage6OutputClaims,
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

pub(crate) fn clear_claims_from_native<F: Field>(
    claims: impl IntoIterator<Item = (native::JoltOpeningId, F)>,
    _trace_length: usize,
) -> Result<ClearProofClaims<F>, VerifierError> {
    let claims = NativeOpeningClaims {
        claims: claims.into_iter().collect(),
    };
    Ok(ClearProofClaims {
        stage1: Stage1OutputClaims {
            uniskip_output_claim: claims.require(outer_uniskip_opening())?,
            outer: spartan_outer_claims_from_native(&claims)?,
        },
        stage2: stage2_claims_from_native(&claims)?,
        stage3: stage3_claims_from_native(&claims)?,
        stage4: stage4_claims_from_native(&claims)?,
        stage5: stage5_claims_from_native(&claims)?,
        stage6: stage6_claims_from_native(&claims)?,
        stage7: stage7_claims_from_native(&claims)?,
    })
}

fn spartan_outer_claims_from_native<F: Field>(
    claims: &NativeOpeningClaims<F>,
) -> Result<Stage1BatchOutputClaims<F, F>, VerifierError> {
    let outer_claim = |variable| claims.require(outer_opening(variable));
    let flag_claim = |flag| outer_claim(JoltVirtualPolynomial::OpFlags(flag));

    Ok(Stage1BatchOutputClaims {
        outer_remainder: OuterRemainderOutputClaims {
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

fn stage2_claims_from_native<F: Field>(
    claims: &NativeOpeningClaims<F>,
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

fn stage3_claims_from_native<F: Field>(
    claims: &NativeOpeningClaims<F>,
) -> Result<Stage3OutputClaims<F, F>, VerifierError> {
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

fn stage4_claims_from_native<F: Field>(
    claims: &NativeOpeningClaims<F>,
) -> Result<Stage4OutputClaims<F, F>, VerifierError> {
    Ok(Stage4OutputClaims {
        registers_read_write: RegistersReadWriteOutputClaims {
            registers_val: claims.require(registers::registers_val_read_write())?,
            rs1_ra: claims.require(registers::rs1_ra_read_write())?,
            rs2_ra: claims.require(registers::rs2_ra_read_write())?,
            rd_wa: claims.require(registers::rd_wa_read_write())?,
            rd_inc: claims.require(registers::rd_inc_read_write())?,
        },
        ram_val_check: RamValCheckOutputClaims {
            untrusted_advice: claims.get(ram::val_check_advice_opening(JoltAdviceKind::Untrusted)),
            trusted_advice: claims.get(ram::val_check_advice_opening(JoltAdviceKind::Trusted)),
            program_image: claims.get(program_image::ram_val_check_contribution_opening()),
            ram_ra: claims.require(ram::ram_ra_val_check())?,
            ram_inc: claims.require(ram::ram_inc_val_check())?,
        },
    })
}

fn stage5_claims_from_native<F: Field>(
    claims: &NativeOpeningClaims<F>,
) -> Result<Stage5OutputClaims<F, F>, VerifierError> {
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

fn stage6_claims_from_native<F: Field>(
    claims: &NativeOpeningClaims<F>,
) -> Result<Stage6OutputClaims<F, F>, VerifierError> {
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

    let ram_hamming_weight = ram::ram_hamming_weight();
    let [ram_inc, rd_inc] = [increments::ram_inc_reduced(), increments::rd_inc_reduced()];
    let bytecode_read_raf_address = bytecode::bytecode_read_raf_address_phase_opening();
    let booleanity_address = booleanity::booleanity_address_phase_opening();

    Ok(Stage6OutputClaims {
        address_phase: Stage6AddressPhaseOutputClaims {
            bytecode_read_raf: BytecodeReadRafAddressPhaseOutputClaims {
                intermediate: claims.require(bytecode_read_raf_address)?,
                val_stages: bytecode_val_stage_claims_from_native(claims)?,
            },
            booleanity: BooleanityAddressPhaseOutputClaims {
                intermediate: claims.require(booleanity_address)?,
            },
        },
        cycle_phase: Stage6CyclePhaseOutputClaims {
            bytecode_read_raf: BytecodeReadRafOutputClaims { bytecode_ra },
            booleanity: BooleanityOutputClaims {
                instruction_ra: booleanity_instruction_ra,
                bytecode_ra: booleanity_bytecode_ra,
                ram_ra: booleanity_ram_ra,
            },
            ram_hamming_booleanity: RamHammingBooleanityOutputClaims {
                ram_hamming_weight: claims.require(ram_hamming_weight)?,
            },
            ram_ra_virtualization: RamRaVirtualizationOutputClaims { ram_ra },
            instruction_ra_virtualization: InstructionRaVirtualizationOutputClaims {
                committed_instruction_ra,
            },
            inc_claim_reduction: IncClaimReductionOutputClaims {
                ram_inc: claims.require(ram_inc)?,
                rd_inc: claims.require(rd_inc)?,
            },
            trusted_advice: advice_cycle_phase_claim_from_native(claims, JoltAdviceKind::Trusted),
            untrusted_advice: advice_cycle_phase_claim_from_native(
                claims,
                JoltAdviceKind::Untrusted,
            ),
            bytecode_reduction: bytecode_cycle_phase_claims_from_native(claims),
            program_image_reduction: claims
                .get(program_image::cycle_phase_program_image_opening())
                .or_else(|| claims.get(program_image::final_program_image_opening()))
                .map(|program_image| ProgramImageReductionCyclePhaseOutputClaims { program_image }),
        },
    })
}

fn advice_cycle_phase_claim_from_native<F: Field>(
    claims: &NativeOpeningClaims<F>,
    kind: JoltAdviceKind,
) -> Option<AdviceCyclePhaseOutputClaims<F>> {
    let opening_claim = claims
        .get(advice::cycle_phase_advice_opening(kind))
        .or_else(|| claims.get(advice::final_advice_opening(kind)))?;
    Some(match kind {
        JoltAdviceKind::Trusted => AdviceCyclePhaseOutputClaims {
            trusted: Some(opening_claim),
            untrusted: None,
        },
        JoltAdviceKind::Untrusted => AdviceCyclePhaseOutputClaims {
            trusted: None,
            untrusted: Some(opening_claim),
        },
    })
}

fn bytecode_val_stage_claims_from_native<F: Field>(
    claims: &NativeOpeningClaims<F>,
) -> Result<Vec<F>, VerifierError> {
    if claims
        .get(bytecode_claim_reduction::bytecode_val_stage_opening(0))
        .is_none()
    {
        return Ok(Vec::new());
    }
    let mut stage_claims = Vec::with_capacity(bytecode_claim_reduction::NUM_BYTECODE_VAL_STAGES);
    for stage in 0..bytecode_claim_reduction::NUM_BYTECODE_VAL_STAGES {
        stage_claims
            .push(claims.require(bytecode_claim_reduction::bytecode_val_stage_opening(stage))?);
    }
    Ok(stage_claims)
}

fn bytecode_cycle_phase_claims_from_native<F: Field>(
    claims: &NativeOpeningClaims<F>,
) -> Option<BytecodeReductionCyclePhaseOutputClaims<F>> {
    if let Some(intermediate) =
        claims.get(bytecode_claim_reduction::cycle_phase_intermediate_opening())
    {
        return Some(BytecodeReductionCyclePhaseOutputClaims {
            intermediate: Some(intermediate),
            chunks: Vec::new(),
        });
    }
    let chunks = final_bytecode_chunk_claims_from_native(claims);
    (!chunks.is_empty()).then_some(BytecodeReductionCyclePhaseOutputClaims {
        intermediate: None,
        chunks,
    })
}

fn final_bytecode_chunk_claims_from_native<F: Field>(claims: &NativeOpeningClaims<F>) -> Vec<F> {
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

fn stage7_claims_from_native<F: Field>(
    claims: &NativeOpeningClaims<F>,
) -> Result<Stage7OutputClaims<F, F>, VerifierError> {
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
            trusted: advice_address_phase_claim_from_native(claims, JoltAdviceKind::Trusted),
            untrusted: advice_address_phase_claim_from_native(claims, JoltAdviceKind::Untrusted),
        },
        bytecode_address_phase: bytecode_address_phase_claims_from_native(claims),
        program_image_address_phase: program_image_address_phase_claim_from_native(claims),
    })
}

fn advice_address_phase_claim_from_native<F: Field>(
    claims: &NativeOpeningClaims<F>,
    kind: JoltAdviceKind,
) -> Option<F> {
    let _ = claims.get(advice::cycle_phase_advice_opening(kind))?;
    claims.get(advice::final_advice_opening(kind))
}

fn bytecode_address_phase_claims_from_native<F: Field>(
    claims: &NativeOpeningClaims<F>,
) -> Option<BytecodeReductionAddressPhaseOutputClaims<F>> {
    let _ = claims.get(bytecode_claim_reduction::cycle_phase_intermediate_opening())?;
    let chunks = final_bytecode_chunk_claims_from_native(claims);
    (!chunks.is_empty()).then_some(BytecodeReductionAddressPhaseOutputClaims { chunks })
}

fn program_image_address_phase_claim_from_native<F: Field>(
    claims: &NativeOpeningClaims<F>,
) -> Option<ProgramImageReductionAddressPhaseOutputClaims<F>> {
    let _ = claims.get(program_image::cycle_phase_program_image_opening())?;
    claims
        .get(program_image::final_program_image_opening())
        .map(|program_image| ProgramImageReductionAddressPhaseOutputClaims { program_image })
}

#[derive(Clone, Debug)]
struct NativeOpeningClaims<F: Field> {
    claims: Vec<(native::JoltOpeningId, F)>,
}

impl<F: Field> NativeOpeningClaims<F> {
    fn get(&self, id: native::JoltOpeningId) -> Option<F> {
        self.claims
            .iter()
            .find_map(|&(claim_id, opening_claim)| (claim_id == id).then_some(opening_claim))
    }

    fn require(&self, id: native::JoltOpeningId) -> Result<F, VerifierError> {
        self.get(id)
            .ok_or(VerifierError::MissingOpeningClaim { id })
    }

    fn get_or_zero(&self, id: native::JoltOpeningId) -> F {
        self.get(id).unwrap_or_else(F::zero)
    }
}

#[doc(hidden)]
pub fn attach_opening_claims<PCS, VC, ZkProof>(
    proof: &mut JoltProof<PCS, VC, ZkProof>,
    claims: impl IntoIterator<Item = (native::JoltOpeningId, PCS::Field)>,
) -> Result<(), VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    proof.claims = JoltProofClaims::Clear(clear_claims_from_native(claims, proof.trace_length)?);
    Ok(())
}

#[doc(hidden)]
pub fn attach_empty_opening_claims<PCS, VC, ZkProof>(proof: &mut JoltProof<PCS, VC, ZkProof>)
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    proof.claims = JoltProofClaims::Clear(empty_clear_claims(proof.trace_length));
}

fn empty_clear_claims<F: Field>(_trace_length: usize) -> ClearProofClaims<F> {
    let zero = F::zero();

    ClearProofClaims {
        stage1: Stage1OutputClaims {
            uniskip_output_claim: zero,
            outer: empty_spartan_outer_claims(),
        },
        stage2: Stage2OutputClaims {
            product_uniskip_output_claim: zero,
            batch_outputs: Stage2BatchOutputClaims {
                ram_read_write: RamReadWriteOutputClaims {
                    val: zero,
                    ra: zero,
                    inc: zero,
                },
                product_remainder: ProductRemainderOutputClaims {
                    left_instruction_input: zero,
                    right_instruction_input: zero,
                    jump_flag: zero,
                    write_lookup_output_to_rd: zero,
                    lookup_output: zero,
                    branch_flag: zero,
                    next_is_noop: zero,
                    virtual_instruction: zero,
                },
                instruction_claim_reduction: InstructionClaimReductionOutputClaims {
                    lookup_output: None,
                    left_lookup_operand: zero,
                    right_lookup_operand: zero,
                    left_instruction_input: None,
                    right_instruction_input: None,
                },
                ram_raf_evaluation: RamRafEvaluationOutputClaims { ram_ra: zero },
                ram_output_check: RamOutputCheckOutputClaims { val_final: zero },
            },
        },
        stage3: Stage3OutputClaims {
            shift: SpartanShiftOutputClaims {
                unexpanded_pc: zero,
                pc: zero,
                is_virtual: zero,
                is_first_in_sequence: zero,
                is_noop: zero,
            },
            instruction_input: InstructionInputOutputClaims {
                left_operand_is_rs1: zero,
                rs1_value: zero,
                left_operand_is_pc: zero,
                unexpanded_pc: zero,
                right_operand_is_rs2: zero,
                rs2_value: zero,
                right_operand_is_imm: zero,
                imm: zero,
            },
            registers_claim_reduction: RegistersClaimReductionOutputClaims {
                rd_write_value: zero,
                rs1_value: zero,
                rs2_value: zero,
            },
        },
        stage4: Stage4OutputClaims {
            registers_read_write: RegistersReadWriteOutputClaims {
                registers_val: zero,
                rs1_ra: zero,
                rs2_ra: zero,
                rd_wa: zero,
                rd_inc: zero,
            },
            ram_val_check: RamValCheckOutputClaims {
                untrusted_advice: None,
                trusted_advice: None,
                program_image: None,
                ram_ra: zero,
                ram_inc: zero,
            },
        },
        stage5: Stage5OutputClaims {
            instruction_read_raf: InstructionReadRafOutputClaims {
                lookup_table_flags: vec![zero; LookupTableKind::<RISCV_XLEN>::COUNT],
                instruction_ra: vec![zero],
                instruction_raf_flag: zero,
            },
            ram_ra_claim_reduction: RamRaClaimReductionOutputClaims { ram_ra: zero },
            registers_val_evaluation: RegistersValEvaluationOutputClaims {
                rd_inc: zero,
                rd_wa: zero,
            },
        },
        stage6: Stage6OutputClaims {
            address_phase: Stage6AddressPhaseOutputClaims {
                bytecode_read_raf: BytecodeReadRafAddressPhaseOutputClaims {
                    intermediate: zero,
                    val_stages: Vec::new(),
                },
                booleanity: BooleanityAddressPhaseOutputClaims { intermediate: zero },
            },
            cycle_phase: Stage6CyclePhaseOutputClaims {
                bytecode_read_raf: BytecodeReadRafOutputClaims {
                    bytecode_ra: vec![zero],
                },
                booleanity: BooleanityOutputClaims {
                    instruction_ra: vec![zero],
                    bytecode_ra: vec![zero],
                    ram_ra: vec![zero],
                },
                ram_hamming_booleanity: RamHammingBooleanityOutputClaims {
                    ram_hamming_weight: zero,
                },
                ram_ra_virtualization: RamRaVirtualizationOutputClaims { ram_ra: vec![zero] },
                instruction_ra_virtualization: InstructionRaVirtualizationOutputClaims {
                    committed_instruction_ra: vec![zero],
                },
                inc_claim_reduction: IncClaimReductionOutputClaims {
                    ram_inc: zero,
                    rd_inc: zero,
                },
                trusted_advice: None,
                untrusted_advice: None,
                bytecode_reduction: None,
                program_image_reduction: None,
            },
        },
        stage7: Stage7OutputClaims {
            hamming_weight_claim_reduction: HammingWeightClaimReductionOutputClaims {
                instruction_ra: vec![zero],
                bytecode_ra: vec![zero],
                ram_ra: vec![zero],
            },
            advice_address_phase: AdviceAddressPhaseOutputClaims {
                trusted: None,
                untrusted: None,
            },
            bytecode_address_phase: None,
            program_image_address_phase: None,
        },
    }
}

fn empty_spartan_outer_claims<F: Field>() -> Stage1BatchOutputClaims<F, F> {
    let zero = F::zero();

    Stage1BatchOutputClaims {
        outer_remainder: OuterRemainderOutputClaims {
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
#[cfg(any(feature = "prover-fixtures", test))]
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
#[cfg(any(feature = "prover-fixtures", test))]
pub fn upsert_opening_claim<PCS, VC, ZkProof>(
    proof: &mut JoltProof<PCS, VC, ZkProof>,
    id: native::JoltOpeningId,
    opening_claim: PCS::Field,
) where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let _ = set_claim(proof, id, opening_claim);
}

#[doc(hidden)]
#[cfg(any(feature = "prover-fixtures", test))]
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

#[cfg(any(feature = "prover-fixtures", test))]
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

#[cfg(any(feature = "prover-fixtures", test))]
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

#[cfg(any(feature = "prover-fixtures", test))]
fn set_claim<PCS, VC, ZkProof>(
    proof: &mut JoltProof<PCS, VC, ZkProof>,
    id: native::JoltOpeningId,
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

#[cfg(any(feature = "prover-fixtures", test))]
fn claim_from_clear<F: Field>(
    claims: &ClearProofClaims<F>,
    trace_length: usize,
    id: native::JoltOpeningId,
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

#[cfg(any(feature = "prover-fixtures", test))]
fn claim_from_spartan_outer<F: Field>(
    claims: &Stage1BatchOutputClaims<F, F>,
    variable: JoltVirtualPolynomial,
) -> Option<F> {
    let claims = &claims.outer_remainder;
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
        JoltVirtualPolynomial::OpFlags(flag) => match flag {
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
        },
        _ => None,
    }
}

#[cfg(any(feature = "prover-fixtures", test))]
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
        .or_else(|| claim_mut_from_stage6_outputs(&mut claims.stage6, id))
}

#[cfg(any(feature = "prover-fixtures", test))]
fn claim_mut_from_spartan_outer<F: Field>(
    claims: &mut Stage1BatchOutputClaims<F, F>,
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

#[cfg(any(feature = "prover-fixtures", test))]
fn set_claim_in_clear<F: Field>(
    claims: &mut ClearProofClaims<F>,
    trace_length: usize,
    id: native::JoltOpeningId,
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

#[cfg(any(feature = "prover-fixtures", test))]
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

#[cfg(any(feature = "prover-fixtures", test))]
fn claim_from_stage2_batch_outputs<F: Field>(
    claims: &Stage2BatchOutputClaims<F, F>,
    id: native::JoltOpeningId,
) -> Option<F> {
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
        id if id == ram_ra_raf_evaluation => Some(claims.ram_raf_evaluation.ram_ra),
        id if id == ram_val_final => Some(claims.ram_output_check.val_final),
        _ => None,
    }
}

#[cfg(any(feature = "prover-fixtures", test))]
fn claim_mut_from_stage2_batch_outputs<F: Field>(
    claims: &mut Stage2BatchOutputClaims<F, F>,
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
        id if id == ram_ra_raf_evaluation => Some(&mut claims.ram_raf_evaluation.ram_ra),
        id if id == ram_val_final => Some(&mut claims.ram_output_check.val_final),
        _ => None,
    }
}

#[cfg(any(feature = "prover-fixtures", test))]
fn set_optional_stage2_batch_output<F: Field>(
    claims: &mut Stage2BatchOutputClaims<F, F>,
    id: native::JoltOpeningId,
    opening_claim: F,
) -> bool {
    let instruction_lookup_output = instruction_claim_reduction::lookup_output_reduced();
    let instruction_left_instruction_input =
        instruction_claim_reduction::left_instruction_input_reduced();
    let instruction_right_instruction_input =
        instruction_claim_reduction::right_instruction_input_reduced();

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

#[cfg(any(feature = "prover-fixtures", test))]
fn set_optional_stage4_output<F: Field>(
    claims: &mut Stage4OutputClaims<F, F>,
    id: native::JoltOpeningId,
    opening_claim: F,
) -> bool {
    match id {
        id if id == ram::val_check_advice_opening(JoltAdviceKind::Untrusted) => {
            claims.ram_val_check.untrusted_advice = Some(opening_claim);
            true
        }
        id if id == ram::val_check_advice_opening(JoltAdviceKind::Trusted) => {
            claims.ram_val_check.trusted_advice = Some(opening_claim);
            true
        }
        _ => false,
    }
}

#[cfg(any(feature = "prover-fixtures", test))]
fn set_optional_stage6_output<F: Field>(
    claims: &mut Stage6OutputClaims<F, F>,
    id: native::JoltOpeningId,
    opening_claim: F,
) -> bool {
    match id {
        id if id == bytecode::bytecode_read_raf_address_phase_opening() => {
            claims.address_phase.bytecode_read_raf.intermediate = opening_claim;
            true
        }
        id if id == booleanity::booleanity_address_phase_opening() => {
            claims.address_phase.booleanity.intermediate = opening_claim;
            true
        }
        id if id == advice::cycle_phase_advice_opening(JoltAdviceKind::Trusted)
            || id == advice::final_advice_opening(JoltAdviceKind::Trusted) =>
        {
            claims.cycle_phase.trusted_advice = Some(AdviceCyclePhaseOutputClaims {
                trusted: Some(opening_claim),
                untrusted: None,
            });
            true
        }
        id if id == advice::cycle_phase_advice_opening(JoltAdviceKind::Untrusted)
            || id == advice::final_advice_opening(JoltAdviceKind::Untrusted) =>
        {
            claims.cycle_phase.untrusted_advice = Some(AdviceCyclePhaseOutputClaims {
                trusted: None,
                untrusted: Some(opening_claim),
            });
            true
        }
        _ => false,
    }
}

#[cfg(any(feature = "prover-fixtures", test))]
fn claim_from_stage3_outputs<F: Field>(
    claims: &Stage3OutputClaims<F, F>,
    id: native::JoltOpeningId,
) -> Option<F> {
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

#[cfg(any(feature = "prover-fixtures", test))]
fn claim_mut_from_stage3_outputs<F: Field>(
    claims: &mut Stage3OutputClaims<F, F>,
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

#[cfg(any(feature = "prover-fixtures", test))]
fn claim_from_stage4_outputs<F: Field>(
    claims: &Stage4OutputClaims<F, F>,
    id: native::JoltOpeningId,
) -> Option<F> {
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
            claims.ram_val_check.untrusted_advice
        }
        id if id == ram::val_check_advice_opening(JoltAdviceKind::Trusted) => {
            claims.ram_val_check.trusted_advice
        }
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

#[cfg(any(feature = "prover-fixtures", test))]
fn claim_mut_from_stage4_outputs<F: Field>(
    claims: &mut Stage4OutputClaims<F, F>,
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

#[cfg(any(feature = "prover-fixtures", test))]
fn claim_from_stage5_outputs<F: Field>(
    claims: &Stage5OutputClaims<F, F>,
    id: native::JoltOpeningId,
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

    let ram_ra = ram::ram_ra_claim_reduction();
    let [rd_inc, rd_wa] = [
        registers::rd_inc_val_evaluation(),
        registers::rd_wa_val_evaluation(),
    ];
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

#[cfg(any(feature = "prover-fixtures", test))]
fn claim_mut_from_stage5_outputs<F: Field>(
    claims: &mut Stage5OutputClaims<F, F>,
    id: native::JoltOpeningId,
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

    let ram_ra = ram::ram_ra_claim_reduction();
    let [rd_inc, rd_wa] = [
        registers::rd_inc_val_evaluation(),
        registers::rd_wa_val_evaluation(),
    ];
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

#[cfg(any(feature = "prover-fixtures", test))]
fn claim_from_stage6_outputs<F: Field>(
    claims: &Stage6OutputClaims<F, F>,
    id: native::JoltOpeningId,
) -> Option<F> {
    for (index, opening_claim) in claims
        .cycle_phase
        .bytecode_read_raf
        .bytecode_ra
        .iter()
        .enumerate()
    {
        if id
            == JoltOpeningId::committed(
                JoltCommittedPolynomial::BytecodeRa(index),
                JoltRelationId::BytecodeReadRaf,
            )
        {
            return Some(*opening_claim);
        }
    }
    for (index, opening_claim) in claims
        .cycle_phase
        .booleanity
        .instruction_ra
        .iter()
        .enumerate()
    {
        if id
            == JoltOpeningId::committed(
                JoltCommittedPolynomial::InstructionRa(index),
                JoltRelationId::Booleanity,
            )
        {
            return Some(*opening_claim);
        }
    }
    for (index, opening_claim) in claims.cycle_phase.booleanity.bytecode_ra.iter().enumerate() {
        if id
            == JoltOpeningId::committed(
                JoltCommittedPolynomial::BytecodeRa(index),
                JoltRelationId::Booleanity,
            )
        {
            return Some(*opening_claim);
        }
    }
    for (index, opening_claim) in claims.cycle_phase.booleanity.ram_ra.iter().enumerate() {
        if id
            == JoltOpeningId::committed(
                JoltCommittedPolynomial::RamRa(index),
                JoltRelationId::Booleanity,
            )
        {
            return Some(*opening_claim);
        }
    }
    let ram_hamming_weight = ram::ram_hamming_weight();
    if id == ram_hamming_weight {
        return Some(claims.cycle_phase.ram_hamming_booleanity.ram_hamming_weight);
    }
    for (index, opening_claim) in claims
        .cycle_phase
        .ram_ra_virtualization
        .ram_ra
        .iter()
        .enumerate()
    {
        if id == ram::ra_virtualization_committed_ram_ra_opening(index) {
            return Some(*opening_claim);
        }
    }
    for (index, opening_claim) in claims
        .cycle_phase
        .instruction_ra_virtualization
        .committed_instruction_ra
        .iter()
        .enumerate()
    {
        if id == instruction::ra_virtualization_committed_instruction_ra_opening(index) {
            return Some(*opening_claim);
        }
    }
    let [ram_inc, rd_inc] = [increments::ram_inc_reduced(), increments::rd_inc_reduced()];
    match id {
        id if id == bytecode::bytecode_read_raf_address_phase_opening() => {
            Some(claims.address_phase.bytecode_read_raf.intermediate)
        }
        id if id == booleanity::booleanity_address_phase_opening() => {
            Some(claims.address_phase.booleanity.intermediate)
        }
        id if id == ram_inc => Some(claims.cycle_phase.inc_claim_reduction.ram_inc),
        id if id == rd_inc => Some(claims.cycle_phase.inc_claim_reduction.rd_inc),
        id if id == advice::cycle_phase_advice_opening(JoltAdviceKind::Trusted)
            || id == advice::final_advice_opening(JoltAdviceKind::Trusted) =>
        {
            claims
                .cycle_phase
                .trusted_advice
                .as_ref()
                .and_then(|claim| claim.trusted)
        }
        id if id == advice::cycle_phase_advice_opening(JoltAdviceKind::Untrusted)
            || id == advice::final_advice_opening(JoltAdviceKind::Untrusted) =>
        {
            claims
                .cycle_phase
                .untrusted_advice
                .as_ref()
                .and_then(|claim| claim.untrusted)
        }
        _ => None,
    }
}

#[cfg(any(feature = "prover-fixtures", test))]
fn claim_mut_from_stage6_outputs<F: Field>(
    claims: &mut Stage6OutputClaims<F, F>,
    id: native::JoltOpeningId,
) -> Option<&mut F> {
    for (index, opening_claim) in claims
        .cycle_phase
        .bytecode_read_raf
        .bytecode_ra
        .iter_mut()
        .enumerate()
    {
        if id
            == JoltOpeningId::committed(
                JoltCommittedPolynomial::BytecodeRa(index),
                JoltRelationId::BytecodeReadRaf,
            )
        {
            return Some(opening_claim);
        }
    }
    for (index, opening_claim) in claims
        .cycle_phase
        .booleanity
        .instruction_ra
        .iter_mut()
        .enumerate()
    {
        if id
            == JoltOpeningId::committed(
                JoltCommittedPolynomial::InstructionRa(index),
                JoltRelationId::Booleanity,
            )
        {
            return Some(opening_claim);
        }
    }
    for (index, opening_claim) in claims
        .cycle_phase
        .booleanity
        .bytecode_ra
        .iter_mut()
        .enumerate()
    {
        if id
            == JoltOpeningId::committed(
                JoltCommittedPolynomial::BytecodeRa(index),
                JoltRelationId::Booleanity,
            )
        {
            return Some(opening_claim);
        }
    }
    for (index, opening_claim) in claims.cycle_phase.booleanity.ram_ra.iter_mut().enumerate() {
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
        return Some(&mut claims.cycle_phase.ram_hamming_booleanity.ram_hamming_weight);
    }
    for (index, opening_claim) in claims
        .cycle_phase
        .ram_ra_virtualization
        .ram_ra
        .iter_mut()
        .enumerate()
    {
        if id == ram::ra_virtualization_committed_ram_ra_opening(index) {
            return Some(opening_claim);
        }
    }
    for (index, opening_claim) in claims
        .cycle_phase
        .instruction_ra_virtualization
        .committed_instruction_ra
        .iter_mut()
        .enumerate()
    {
        if id == instruction::ra_virtualization_committed_instruction_ra_opening(index) {
            return Some(opening_claim);
        }
    }
    let [ram_inc, rd_inc] = [increments::ram_inc_reduced(), increments::rd_inc_reduced()];
    match id {
        id if id == bytecode::bytecode_read_raf_address_phase_opening() => {
            Some(&mut claims.address_phase.bytecode_read_raf.intermediate)
        }
        id if id == booleanity::booleanity_address_phase_opening() => {
            Some(&mut claims.address_phase.booleanity.intermediate)
        }
        id if id == ram_inc => Some(&mut claims.cycle_phase.inc_claim_reduction.ram_inc),
        id if id == rd_inc => Some(&mut claims.cycle_phase.inc_claim_reduction.rd_inc),
        id if id == advice::cycle_phase_advice_opening(JoltAdviceKind::Trusted)
            || id == advice::final_advice_opening(JoltAdviceKind::Trusted) =>
        {
            claims
                .cycle_phase
                .trusted_advice
                .as_mut()
                .and_then(|claim| claim.trusted.as_mut())
        }
        id if id == advice::cycle_phase_advice_opening(JoltAdviceKind::Untrusted)
            || id == advice::final_advice_opening(JoltAdviceKind::Untrusted) =>
        {
            claims
                .cycle_phase
                .untrusted_advice
                .as_mut()
                .and_then(|claim| claim.untrusted.as_mut())
        }
        _ => None,
    }
}

#[cfg(any(feature = "prover-fixtures", test))]
fn claim_from_stage7_outputs<F: Field>(
    claims: &Stage7OutputClaims<F, F>,
    id: native::JoltOpeningId,
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
        id if id == advice::final_advice_opening(JoltAdviceKind::Trusted) => {
            claims.advice_address_phase.trusted
        }
        id if id == advice::final_advice_opening(JoltAdviceKind::Untrusted) => {
            claims.advice_address_phase.untrusted
        }
        _ => None,
    }
}

#[cfg(any(feature = "prover-fixtures", test))]
fn claim_mut_from_stage7_outputs<F: Field>(
    claims: &mut Stage7OutputClaims<F, F>,
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
        id if id == advice::final_advice_opening(JoltAdviceKind::Trusted) => {
            claims.advice_address_phase.trusted.as_mut()
        }
        id if id == advice::final_advice_opening(JoltAdviceKind::Untrusted) => {
            claims.advice_address_phase.untrusted.as_mut()
        }
        _ => None,
    }
}
