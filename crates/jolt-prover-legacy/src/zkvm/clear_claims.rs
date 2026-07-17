//! Opening-claim projection for verifier-native prover proofs.

use jolt_claims::protocols::jolt::{
    self as jolt,
    geometry::{
        booleanity, bytecode,
        claim_reductions::registers as registers_claim_reduction,
        claim_reductions::{
            advice, bytecode as bytecode_claim_reduction,
            instruction as instruction_claim_reduction, program_image,
        },
        instruction, ram, registers, spartan,
        spartan::{outer_opening, product_uniskip_opening},
    },
    JoltAdviceKind, JoltVirtualPolynomial,
};
#[cfg(not(feature = "akita"))]
use jolt_claims::protocols::jolt::{
    geometry::{claim_reductions::increments, spartan::outer_uniskip_opening},
    JoltCommittedPolynomial, JoltOpeningId, JoltRelationId,
};
use jolt_field::Field;
use jolt_lookup_tables::{LookupTableKind, XLEN as RISCV_XLEN};
use jolt_riscv::CircuitFlags;
#[cfg(not(feature = "akita"))]
use jolt_verifier::{
    proof::ClearProofClaims,
    stages::{
        stage1::outputs::Stage1OutputClaims,
        stage6b::outputs::{
            BooleanityOutputClaims, IncClaimReductionOutputClaims, Stage6bOutputClaims,
        },
        stage7::{
            advice_address_phase::{
                TrustedAdviceAddressPhaseOutputClaims, UntrustedAdviceAddressPhaseOutputClaims,
            },
            hamming_weight_claim_reduction::HammingWeightClaimReductionOutputClaims,
            outputs::Stage7OutputClaims,
        },
    },
};
use jolt_verifier::{
    stages::{
        stage1::{outputs::Stage1BatchOutputClaims, OuterRemainderOutputClaims},
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
        stage6a::outputs::{
            BooleanityAddressPhaseOutputClaims, BytecodeReadRafAddressPhaseOutputClaims,
            Stage6aOutputClaims,
        },
        stage6b::outputs::{
            BytecodeReadRafOutputClaims, BytecodeReductionCyclePhaseOutputClaims,
            InstructionRaVirtualizationOutputClaims, ProgramImageReductionCyclePhaseOutputClaims,
            RamHammingBooleanityOutputClaims, RamRaVirtualizationOutputClaims,
            TrustedAdviceCyclePhaseOutputClaims, UntrustedAdviceCyclePhaseOutputClaims,
        },
        stage7::committed_reduction_address_phase::{
            BytecodeReductionAddressPhaseOutputClaims,
            ProgramImageReductionAddressPhaseOutputClaims,
        },
    },
    VerifierError,
};

// The akita (packed) prove path assembles the akita-shaped
// `ClearProofClaims` itself; the base builder and its stage-6b/7 pieces
// target the base wire shape and are compiled out with it.
#[cfg(not(feature = "akita"))]
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
        stage6a: stage6a_claims_from_openings(&claims)?,
        stage6b: stage6b_claims_from_openings(&claims)?,
        stage7: stage7_claims_from_openings(&claims)?,
    })
}

fn spartan_outer_claims_from_openings<F: Field>(
    claims: &OpeningClaimMap<F>,
) -> Result<Stage1BatchOutputClaims<F>, VerifierError> {
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

fn stage2_claims_from_openings<F: Field>(
    claims: &OpeningClaimMap<F>,
) -> Result<Stage2OutputClaims<F>, VerifierError> {
    let product_remainder = ProductRemainderOutputClaims {
        left_instruction_input: claims.get_or_zero(spartan::left_instruction_input_product()),
        right_instruction_input: claims.get_or_zero(spartan::right_instruction_input_product()),
        jump_flag: claims.get_or_zero(spartan::jump_flag_product()),
        write_lookup_output_to_rd: claims.get_or_zero(spartan::write_lookup_output_to_rd_product()),
        lookup_output: claims.get_or_zero(spartan::lookup_output_product()),
        branch_flag: claims.get_or_zero(spartan::branch_flag_product()),
        next_is_noop: claims.get_or_zero(spartan::next_is_noop_product()),
        virtual_instruction: claims.get_or_zero(spartan::virtual_instruction_product()),
    };
    // The three aliased reduced openings are deduplicated into their
    // product-remainder sources by the accumulator (never absorbed separately),
    // so the wire cells are back-filled from the product values — the same idiom
    // the stage-3 projection uses for its aliases.
    let instruction_claim_reduction = InstructionClaimReductionOutputClaims {
        lookup_output: claims
            .get(instruction_claim_reduction::lookup_output_reduced())
            .unwrap_or(product_remainder.lookup_output),
        left_lookup_operand: claims
            .get_or_zero(instruction_claim_reduction::left_lookup_operand_reduced()),
        right_lookup_operand: claims
            .get_or_zero(instruction_claim_reduction::right_lookup_operand_reduced()),
        left_instruction_input: claims
            .get(instruction_claim_reduction::left_instruction_input_reduced())
            .unwrap_or(product_remainder.left_instruction_input),
        right_instruction_input: claims
            .get(instruction_claim_reduction::right_instruction_input_reduced())
            .unwrap_or(product_remainder.right_instruction_input),
    };
    Ok(Stage2OutputClaims {
        product_uniskip_output_claim: claims.require(product_uniskip_opening())?,
        batch_outputs: Stage2BatchOutputClaims {
            ram_read_write: RamReadWriteOutputClaims {
                val: claims.get_or_zero(ram::ram_val()),
                ra: claims.get_or_zero(ram::ram_ra()),
                inc: claims.get_or_zero(ram::ram_inc()),
            },
            product_remainder,
            instruction_claim_reduction,
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
        registers_read_write: RegistersReadWriteOutputClaims {
            registers_val: claims.require(registers::registers_val_read_write())?,
            rs1_ra: claims.require(registers::rs1_ra_read_write())?,
            rs2_ra: claims.require(registers::rs2_ra_read_write())?,
            rd_wa: claims.require(registers::rd_wa_read_write())?,
            rd_inc: claims.require(registers::rd_inc_read_write())?,
        },
        // The advice / program-image openings are produced by the RAM value-check
        // instance, so they are folded into `RamValCheckOutputClaims`. Their values
        // are sourced identically to before; only the struct shape changed.
        ram_val_check: RamValCheckOutputClaims {
            untrusted_advice: claims.get(ram::val_check_advice_opening(JoltAdviceKind::Untrusted)),
            trusted_advice: claims.get(ram::val_check_advice_opening(JoltAdviceKind::Trusted)),
            program_image: claims.get(program_image::ram_val_check_contribution_opening()),
            ram_ra: claims.require(ram::ram_ra_val_check())?,
            ram_inc: claims.require(ram::ram_inc_val_check())?,
        },
    })
}

fn stage5_claims_from_openings<F: Field>(
    claims: &OpeningClaimMap<F>,
) -> Result<Stage5OutputClaims<F>, VerifierError> {
    let lookup_table_flags = LookupTableKind::<RISCV_XLEN>::iter()
        .map(|table| claims.require(instruction::lookup_table_flag(table)))
        .collect::<Result<Vec<_>, _>>()?;
    let mut instruction_ra = Vec::new();
    for index in 0.. {
        let Some(opening_claim) = claims.get(instruction::instruction_ra(index)) else {
            break;
        };
        instruction_ra.push(opening_claim);
    }
    if instruction_ra.is_empty() {
        return Err(VerifierError::MissingOpeningClaim {
            id: instruction::instruction_ra(0),
        });
    }
    Ok(Stage5OutputClaims {
        instruction_read_raf: InstructionReadRafOutputClaims {
            lookup_table_flags,
            instruction_ra,
            instruction_raf_flag: claims.require(instruction::instruction_raf_flag())?,
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

fn stage6a_claims_from_openings<F: Field>(
    claims: &OpeningClaimMap<F>,
) -> Result<Stage6aOutputClaims<F>, VerifierError> {
    let bytecode_read_raf_address = bytecode::bytecode_read_raf_address_phase_opening();
    let booleanity_address = booleanity::booleanity_address_phase_opening();

    Ok(Stage6aOutputClaims {
        bytecode_read_raf: BytecodeReadRafAddressPhaseOutputClaims {
            intermediate: claims.require(bytecode_read_raf_address)?,
            val_stages: bytecode_val_stage_claims_from_openings(claims)?,
        },
        booleanity: BooleanityAddressPhaseOutputClaims {
            intermediate: claims.require(booleanity_address)?,
        },
    })
}

#[cfg(not(feature = "akita"))]
fn stage6b_claims_from_openings<F: Field>(
    claims: &OpeningClaimMap<F>,
) -> Result<Stage6bOutputClaims<F>, VerifierError> {
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
        let id = ram::committed_ram_ra(index);
        let Some(opening_claim) = claims.get(id) else {
            break;
        };
        ram_ra.push(opening_claim);
    }

    let mut committed_instruction_ra = Vec::new();
    for index in 0.. {
        let id = instruction::committed_instruction_ra(index);
        let Some(opening_claim) = claims.get(id) else {
            break;
        };
        committed_instruction_ra.push(opening_claim);
    }
    if committed_instruction_ra.is_empty() {
        return Err(VerifierError::MissingOpeningClaim {
            id: instruction::committed_instruction_ra(0),
        });
    }

    Ok(Stage6bOutputClaims {
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
        trusted_advice: trusted_advice_cycle_phase_claim_from_openings(claims),
        untrusted_advice: untrusted_advice_cycle_phase_claim_from_openings(claims),
        bytecode_reduction: bytecode_cycle_phase_claims_from_openings(claims),
        program_image_reduction: claims
            .get(program_image::cycle_phase_program_image_opening())
            .or_else(|| claims.get(program_image::final_program_image_opening()))
            .map(|program_image| ProgramImageReductionCyclePhaseOutputClaims { program_image }),
    })
}

fn trusted_advice_cycle_phase_claim_from_openings<F: Field>(
    claims: &OpeningClaimMap<F>,
) -> Option<TrustedAdviceCyclePhaseOutputClaims<F>> {
    let opening_claim = claims
        .get(advice::cycle_phase_advice_opening(JoltAdviceKind::Trusted))
        .or_else(|| claims.get(advice::final_advice_opening(JoltAdviceKind::Trusted)))?;
    Some(TrustedAdviceCyclePhaseOutputClaims {
        trusted: opening_claim,
    })
}

fn untrusted_advice_cycle_phase_claim_from_openings<F: Field>(
    claims: &OpeningClaimMap<F>,
) -> Option<UntrustedAdviceCyclePhaseOutputClaims<F>> {
    let opening_claim = claims
        .get(advice::cycle_phase_advice_opening(
            JoltAdviceKind::Untrusted,
        ))
        .or_else(|| claims.get(advice::final_advice_opening(JoltAdviceKind::Untrusted)))?;
    Some(UntrustedAdviceCyclePhaseOutputClaims {
        untrusted: opening_claim,
    })
}

fn bytecode_val_stage_claims_from_openings<F: Field>(
    claims: &OpeningClaimMap<F>,
) -> Result<Vec<F>, VerifierError> {
    if claims
        .get(bytecode_claim_reduction::bytecode_val_stage_opening(0))
        .is_none()
    {
        return Ok(Vec::new());
    }
    // Five staged vals in base mode, six on the packed path (the store
    // stage); the verifier's count validation enforces the mode.
    let mut stage_claims =
        Vec::with_capacity(bytecode_claim_reduction::NUM_BYTECODE_VAL_STAGES + 1);
    for stage in 0.. {
        let Some(opening_claim) =
            claims.get(bytecode_claim_reduction::bytecode_val_stage_opening(stage))
        else {
            break;
        };
        stage_claims.push(opening_claim);
    }
    Ok(stage_claims)
}

fn bytecode_cycle_phase_claims_from_openings<F: Field>(
    claims: &OpeningClaimMap<F>,
) -> Option<BytecodeReductionCyclePhaseOutputClaims<F>> {
    if let Some(intermediate) =
        claims.get(bytecode_claim_reduction::cycle_phase_intermediate_opening())
    {
        return Some(BytecodeReductionCyclePhaseOutputClaims {
            intermediate: Some(intermediate),
            chunks: Vec::new(),
        });
    }
    let chunks = final_bytecode_chunk_claims_from_openings(claims);
    (!chunks.is_empty()).then_some(BytecodeReductionCyclePhaseOutputClaims {
        intermediate: None,
        chunks,
    })
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

#[cfg(not(feature = "akita"))]
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
        trusted_advice: advice_address_phase_claim_from_openings(claims, JoltAdviceKind::Trusted)
            .map(|opening| TrustedAdviceAddressPhaseOutputClaims { trusted: opening }),
        untrusted_advice: advice_address_phase_claim_from_openings(
            claims,
            JoltAdviceKind::Untrusted,
        )
        .map(|opening| UntrustedAdviceAddressPhaseOutputClaims { untrusted: opening }),
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

#[cfg(feature = "akita")]
mod packed {
    use super::*;
    use jolt_claims::protocols::jolt::lattice::relations::advice_reconstruction::{
        self, TrustedAdviceReconstructionOutputClaims, UntrustedAdviceReconstructionOutputClaims,
    };
    use jolt_claims::protocols::jolt::lattice::relations::booleanity::LatticeBooleanityOutputClaims;
    use jolt_claims::protocols::jolt::lattice::relations::bytecode_reconstruction::{
        self, BytecodeChunkReconstructionOutputClaims,
    };
    use jolt_claims::protocols::jolt::lattice::relations::fused_inc_claim_reduction::{
        self, FusedIncClaimReductionOutputClaims,
    };
    use jolt_claims::protocols::jolt::lattice::relations::hamming_weight as lattice_hamming;
    use jolt_claims::protocols::jolt::lattice::relations::inc_virtualization::{
        self, IncVirtualizationOutputClaims,
    };
    use jolt_claims::protocols::jolt::lattice::relations::program_image_reconstruction::{
        self, ProgramImageReconstructionOutputClaims,
    };
    use jolt_claims::protocols::jolt::{
        BytecodeRegisterLane, JoltCommittedPolynomial, JoltOpeningId, JoltRelationId,
    };
    use jolt_riscv::{NUM_CIRCUIT_FLAGS, NUM_INSTRUCTION_FLAGS};
    use jolt_verifier::proof::ClearProofClaims;
    use jolt_verifier::stages::inc_virtualization::IncVirtualizationPhaseOutputClaims;
    use jolt_verifier::stages::stage1::outputs::Stage1OutputClaims;
    use jolt_verifier::stages::stage6b::outputs::Stage6bOutputClaims;
    use jolt_verifier::stages::stage7::advice_address_phase::{
        TrustedAdviceAddressPhaseOutputClaims, UntrustedAdviceAddressPhaseOutputClaims,
    };
    use jolt_verifier::stages::stage7::hamming_weight_claim_reduction::HammingWeightClaimReductionOutputClaims;
    use jolt_verifier::stages::stage7::outputs::Stage7OutputClaims;
    use jolt_verifier::stages::stage8::reconstruction::ReconstructionOutputClaims;
    use spartan::outer_uniskip_opening;

    /// The packed (akita) analog of the base clear-claims projection: the
    /// base stage payloads plus the inc-virtualization and reconstruction
    /// phase cells, with the lattice stage-6b/7 shapes (booleanity carries
    /// the unsigned-inc columns; stage 7 carries the chunk reconstruction;
    /// there is no stage-6b inc slot).
    pub(crate) fn build_packed_clear_claims<F: Field>(
        claims: impl IntoIterator<Item = (jolt::JoltOpeningId, F)>,
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
            inc_virtualization: IncVirtualizationPhaseOutputClaims {
                inc_virtualization: IncVirtualizationOutputClaims {
                    fused_inc: claims.require(inc_virtualization::fused_inc_opening())?,
                    store: claims.require(inc_virtualization::fused_inc_store_opening())?,
                },
            },
            stage6a: stage6a_claims_from_openings(&claims)?,
            stage6b: packed_stage6b_claims_from_openings(&claims)?,
            stage7: packed_stage7_claims_from_openings(&claims)?,
            reconstruction: reconstruction_claims_from_openings(&claims),
        })
    }

    fn indexed_family<F: Field>(
        claims: &OpeningClaimMap<F>,
        id: impl Fn(usize) -> JoltOpeningId,
    ) -> Vec<F> {
        let mut family = Vec::new();
        for index in 0.. {
            let Some(opening_claim) = claims.get(id(index)) else {
                break;
            };
            family.push(opening_claim);
        }
        family
    }

    fn packed_stage6b_claims_from_openings<F: Field>(
        claims: &OpeningClaimMap<F>,
    ) -> Result<Stage6bOutputClaims<F>, VerifierError> {
        let bytecode_ra = indexed_family(claims, |index| {
            JoltOpeningId::committed(
                JoltCommittedPolynomial::BytecodeRa(index),
                JoltRelationId::BytecodeReadRaf,
            )
        });
        if bytecode_ra.is_empty() {
            return Err(VerifierError::MissingOpeningClaim {
                id: JoltOpeningId::committed(
                    JoltCommittedPolynomial::BytecodeRa(0),
                    JoltRelationId::BytecodeReadRaf,
                ),
            });
        }

        let booleanity_instruction_ra = indexed_family(claims, |index| {
            JoltOpeningId::committed(
                JoltCommittedPolynomial::InstructionRa(index),
                JoltRelationId::Booleanity,
            )
        });
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
        let booleanity_ram_ra = indexed_family(claims, |index| {
            JoltOpeningId::committed(
                JoltCommittedPolynomial::RamRa(index),
                JoltRelationId::Booleanity,
            )
        });
        let unsigned_inc_chunks = indexed_family(claims, |index| {
            JoltOpeningId::committed(
                JoltCommittedPolynomial::UnsignedIncChunk(index),
                JoltRelationId::Booleanity,
            )
        });
        let unsigned_inc_msb = claims.require(JoltOpeningId::committed(
            JoltCommittedPolynomial::UnsignedIncMsb,
            JoltRelationId::Booleanity,
        ))?;

        let ram_ra = indexed_family(claims, ram::committed_ram_ra);
        let committed_instruction_ra =
            indexed_family(claims, instruction::committed_instruction_ra);
        if committed_instruction_ra.is_empty() {
            return Err(VerifierError::MissingOpeningClaim {
                id: instruction::committed_instruction_ra(0),
            });
        }

        Ok(Stage6bOutputClaims {
            bytecode_read_raf: BytecodeReadRafOutputClaims { bytecode_ra },
            booleanity: LatticeBooleanityOutputClaims {
                instruction_ra: booleanity_instruction_ra,
                bytecode_ra: booleanity_bytecode_ra,
                ram_ra: booleanity_ram_ra,
                unsigned_inc_chunks,
                unsigned_inc_msb,
            },
            ram_hamming_booleanity: RamHammingBooleanityOutputClaims {
                ram_hamming_weight: claims.require(ram::ram_hamming_weight())?,
            },
            ram_ra_virtualization: RamRaVirtualizationOutputClaims { ram_ra },
            instruction_ra_virtualization: InstructionRaVirtualizationOutputClaims {
                committed_instruction_ra,
            },
            fused_inc_claim_reduction: FusedIncClaimReductionOutputClaims {
                fused_inc: claims.require(fused_inc_claim_reduction::fused_inc_reduced_opening())?,
            },
            trusted_advice: trusted_advice_cycle_phase_claim_from_openings(claims),
            untrusted_advice: untrusted_advice_cycle_phase_claim_from_openings(claims),
            bytecode_reduction: bytecode_cycle_phase_claims_from_openings(claims),
            program_image_reduction: claims
                .get(program_image::cycle_phase_program_image_opening())
                .or_else(|| claims.get(program_image::final_program_image_opening()))
                .map(|program_image| ProgramImageReductionCyclePhaseOutputClaims { program_image }),
        })
    }

    fn packed_stage7_claims_from_openings<F: Field>(
        claims: &OpeningClaimMap<F>,
    ) -> Result<Stage7OutputClaims<F>, VerifierError> {
        let instruction_ra = indexed_family(claims, |index| {
            JoltOpeningId::committed(
                JoltCommittedPolynomial::InstructionRa(index),
                JoltRelationId::HammingWeightClaimReduction,
            )
        });
        let bytecode_ra = indexed_family(claims, |index| {
            JoltOpeningId::committed(
                JoltCommittedPolynomial::BytecodeRa(index),
                JoltRelationId::HammingWeightClaimReduction,
            )
        });
        let ram_ra = indexed_family(claims, |index| {
            JoltOpeningId::committed(
                JoltCommittedPolynomial::RamRa(index),
                JoltRelationId::HammingWeightClaimReduction,
            )
        });
        if instruction_ra.is_empty() && bytecode_ra.is_empty() && ram_ra.is_empty() {
            return Err(VerifierError::MissingOpeningClaim {
                id: JoltOpeningId::committed(
                    JoltCommittedPolynomial::InstructionRa(0),
                    JoltRelationId::HammingWeightClaimReduction,
                ),
            });
        }

        let chunks = indexed_family(claims, lattice_hamming::reduced_unsigned_inc_chunk_opening);
        if chunks.is_empty() {
            return Err(VerifierError::MissingOpeningClaim {
                id: lattice_hamming::reduced_unsigned_inc_chunk_opening(0),
            });
        }

        Ok(Stage7OutputClaims {
            hamming_weight_claim_reduction: HammingWeightClaimReductionOutputClaims {
                instruction_ra,
                bytecode_ra,
                ram_ra,
                unsigned_inc_chunks: chunks,
                unsigned_inc_msb: claims
                    .require(lattice_hamming::reduced_unsigned_inc_msb_opening())?,
            },
            trusted_advice: advice_address_phase_claim_from_openings(
                claims,
                JoltAdviceKind::Trusted,
            )
            .map(|opening| TrustedAdviceAddressPhaseOutputClaims { trusted: opening }),
            untrusted_advice: advice_address_phase_claim_from_openings(
                claims,
                JoltAdviceKind::Untrusted,
            )
            .map(|opening| UntrustedAdviceAddressPhaseOutputClaims { untrusted: opening }),
            bytecode_address_phase: bytecode_address_phase_claims_from_openings(claims),
            program_image_address_phase: program_image_address_phase_claim_from_openings(claims),
        })
    }

    fn reconstruction_claims_from_openings<F: Field>(
        claims: &OpeningClaimMap<F>,
    ) -> ReconstructionOutputClaims<F> {
        ReconstructionOutputClaims {
            untrusted_advice: claims
                .get(advice_reconstruction::untrusted_advice_bytes_opening())
                .map(|bytes| UntrustedAdviceReconstructionOutputClaims { bytes }),
            trusted_advice: claims
                .get(advice_reconstruction::trusted_advice_bytes_opening())
                .map(|bytes| TrustedAdviceReconstructionOutputClaims { bytes }),
            bytecode: bytecode_reconstruction_claims_from_openings(claims),
            program_image: claims
                .get(program_image_reconstruction::program_image_bytes_opening())
                .map(|bytes| ProgramImageReconstructionOutputClaims { bytes }),
        }
    }

    /// Every per-chunk lane family, in the relation's family-major layout;
    /// `None` when no bytecode reconstruction ran (full-program mode).
    fn bytecode_reconstruction_claims_from_openings<F: Field>(
        claims: &OpeningClaimMap<F>,
    ) -> Option<BytecodeChunkReconstructionOutputClaims<F>> {
        let mut chunk_count = 0;
        while claims
            .get(bytecode_reconstruction::bytecode_lookup_selector_opening(
                chunk_count,
            ))
            .is_some()
        {
            chunk_count += 1;
        }
        if chunk_count == 0 {
            return None;
        }
        let mut register_selectors = Vec::new();
        let mut circuit_flags = Vec::new();
        let mut instruction_flags = Vec::new();
        let mut lookup_selectors = Vec::new();
        let mut raf_flags = Vec::new();
        let mut pc_bytes = Vec::new();
        let mut imm_bytes = Vec::new();
        for chunk in 0..chunk_count {
            for lane in BytecodeRegisterLane::ALL {
                register_selectors.push(claims.get(
                    bytecode_reconstruction::bytecode_register_selector_opening(chunk, lane),
                )?);
            }
            for flag in 0..NUM_CIRCUIT_FLAGS {
                circuit_flags.push(claims.get(
                    bytecode_reconstruction::bytecode_circuit_flag_opening(chunk, flag),
                )?);
            }
            for flag in 0..NUM_INSTRUCTION_FLAGS {
                instruction_flags.push(claims.get(
                    bytecode_reconstruction::bytecode_instruction_flag_opening(chunk, flag),
                )?);
            }
            lookup_selectors.push(claims.get(
                bytecode_reconstruction::bytecode_lookup_selector_opening(chunk),
            )?);
            raf_flags.push(claims.get(bytecode_reconstruction::bytecode_raf_flag_opening(chunk))?);
            pc_bytes.push(
                claims.get(bytecode_reconstruction::bytecode_unexpanded_pc_bytes_opening(chunk))?,
            );
            imm_bytes.push(claims.get(bytecode_reconstruction::bytecode_imm_bytes_opening(chunk))?);
        }
        Some(BytecodeChunkReconstructionOutputClaims {
            register_selectors,
            circuit_flags,
            instruction_flags,
            lookup_selectors,
            raf_flags,
            pc_bytes,
            imm_bytes,
        })
    }
}

#[cfg(feature = "akita")]
pub(crate) use packed::build_packed_clear_claims;
