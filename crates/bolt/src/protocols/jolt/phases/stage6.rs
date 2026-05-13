use std::collections::BTreeSet;

use melior::ir::operation::OperationRef;
use melior::ir::Value;

use crate::ir::{BoltModule, Compute, Party, Protocol};
use crate::mlir::{verify_module, MeliorContext, MlirError};
use crate::schema::{verify_protocol_schema, SchemaError};

use super::super::oracles;
use super::super::params::JoltProtocolParams;
use super::lowering::{lower_party_to_compute, transcript_squeeze_protocol_result_type};

const BOOLEANITY_DEGREE: usize = 3;
const HAMMING_BOOLEANITY_DEGREE: usize = 3;
const INC_CLAIM_REDUCTION_DEGREE: usize = 2;

#[derive(Clone, Copy)]
enum BytecodeStageGamma {
    Stage1,
    Stage2,
    Stage3,
    Stage4,
    Stage5,
}

pub fn build_stage6_protocol<'c>(
    context: &'c MeliorContext,
    params: &JoltProtocolParams,
) -> Result<BoltModule<'c, Protocol>, MlirError> {
    let module = context.new_module::<Protocol>("jolt.stage6", None);
    oracles::append_foundation_ops(context, &module, params)?;
    context.append_op_with_owned_attrs(
        &module,
        "protocol.params",
        Some("jolt.params"),
        &params.attrs(),
    )?;
    context.append_op(
        &module,
        "protocol.boundary",
        Some("jolt.stage6"),
        &[("roles", r#"["prover", "verifier"]"#)],
    )?;
    append_stage6_domains(context, &module, params)?;
    append_stage6_oracles(context, &module, params)?;
    append_stage6_relations(context, &module, params)?;
    let inputs = append_stage6_opening_inputs(context, &module, params)?;

    let fs = context.append_typed_op(
        &module,
        "transcript.state",
        Some("fs_after_stage5"),
        &[("scheme", "@blake2b_transcript")],
        &[],
        &["!transcript.state_type"],
    )?;
    let state = first_result(fs, "transcript.state")?;
    let stage = context.append_typed_op(
        &module,
        "piop.stage",
        Some("stage6"),
        &[
            (
                "name",
                r#""bytecode_booleanity_and_virtual_address_reductions""#,
            ),
            ("order", "6 : i64"),
            ("roles", r#"["prover", "verifier"]"#),
        ],
        &[],
        &["!piop.stage_type"],
    )?;
    let stage = first_result(stage, "piop.stage")?;

    let (state, bc_gamma) = append_transcript_squeeze(
        context,
        &module,
        state,
        "stage6.bytecode_read_raf.gamma",
        "bc_raf_gamma",
        "challenge_scalar",
        1,
    )?;
    let (state, bc_stage1_gamma) = append_transcript_squeeze(
        context,
        &module,
        state,
        "stage6.bytecode_read_raf.stage1_gamma",
        "bc_raf_stage1_gamma",
        "challenge_scalar",
        1,
    )?;
    let (state, bc_stage2_gamma) = append_transcript_squeeze(
        context,
        &module,
        state,
        "stage6.bytecode_read_raf.stage2_gamma",
        "bc_raf_stage2_gamma",
        "challenge_scalar",
        1,
    )?;
    let (state, bc_stage3_gamma) = append_transcript_squeeze(
        context,
        &module,
        state,
        "stage6.bytecode_read_raf.stage3_gamma",
        "bc_raf_stage3_gamma",
        "challenge_scalar",
        1,
    )?;
    let (state, bc_stage4_gamma) = append_transcript_squeeze(
        context,
        &module,
        state,
        "stage6.bytecode_read_raf.stage4_gamma",
        "bc_raf_stage4_gamma",
        "challenge_scalar",
        1,
    )?;
    let (state, bc_stage5_gamma) = append_transcript_squeeze(
        context,
        &module,
        state,
        "stage6.bytecode_read_raf.stage5_gamma",
        "bc_raf_stage5_gamma",
        "challenge_scalar",
        1,
    )?;
    let (state, booleanity_gamma) = append_transcript_squeeze(
        context,
        &module,
        state,
        "stage6.booleanity.gamma",
        "booleanity_gamma",
        "challenge_scalar",
        1,
    )?;
    append_booleanity_power_placeholders(context, &module, params, booleanity_gamma)?;
    let (state, inst_ra_gamma) = append_transcript_squeeze(
        context,
        &module,
        state,
        "stage6.instruction_ra_virtual.gamma",
        "inst_ra_virtual_gamma",
        "challenge_scalar",
        1,
    )?;
    let (state, inc_gamma) = append_transcript_squeeze(
        context,
        &module,
        state,
        "stage6.inc_claim_reduction.gamma",
        "inc_reduction_gamma",
        "challenge_scalar",
        1,
    )?;

    let _state = append_stage6_batched_sumcheck(
        context,
        &module,
        params,
        Stage6BatchedSumcheckInputs {
            state,
            stage,
            openings: &inputs,
            bc_gamma,
            bc_stage1_gamma,
            bc_stage2_gamma,
            bc_stage3_gamma,
            bc_stage4_gamma,
            bc_stage5_gamma,
            inst_ra_gamma,
            inc_gamma,
        },
    )?;

    verify_module(&module)?;
    verify_protocol_schema(&module)?;
    Ok(module)
}

pub fn lower_stage6_to_compute<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Party>,
) -> Result<BoltModule<'c, Compute>, MlirError> {
    lower_party_to_compute(context, module, "jolt.stage6", "jolt.stage6", "stage6")
}

fn append_stage6_domains<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Protocol>,
    params: &JoltProtocolParams,
) -> Result<(), MlirError> {
    append_domain(
        context,
        module,
        "jolt.stage2_ram_rw_domain",
        params.log_k_ram + params.log_t,
    )?;
    append_domain(
        context,
        module,
        "jolt.stage4_registers_rw_domain",
        params.register_log_k + params.log_t,
    )?;
    append_domain(
        context,
        module,
        "jolt.stage5_instruction_ra_chunk_domain",
        params.lookups_ra_virtual_log_k_chunk + params.log_t,
    )?;
    append_domain(
        context,
        module,
        "jolt.stage6_bytecode_read_raf_domain",
        stage6_max_rounds(params),
    )?;
    append_domain(
        context,
        module,
        "jolt.stage6_booleanity_domain",
        booleanity_rounds(params),
    )
}

fn append_domain<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Protocol>,
    symbol: &str,
    log_size: usize,
) -> Result<(), MlirError> {
    context.append_op(
        module,
        "poly.domain",
        Some(symbol),
        &[("field", "@bn254_fr"), ("log_size", &int_attr(log_size))],
    )
}

fn append_stage6_oracles<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Protocol>,
    params: &JoltProtocolParams,
) -> Result<(), MlirError> {
    let mut trace_oracles = BTreeSet::new();
    trace_oracles.extend(
        [
            "HammingWeight",
            "Imm",
            "InstructionFlagBranch",
            "InstructionFlagIsNoop",
            "InstructionFlagLeftOperandIsPC",
            "InstructionFlagLeftOperandIsRs1Value",
            "InstructionFlagRightOperandIsImm",
            "InstructionFlagRightOperandIsRs2Value",
            "InstructionRafFlag",
            "LookupOutput",
            "OpFlagAddOperands",
            "OpFlagAdvice",
            "OpFlagAssert",
            "OpFlagDoNotUpdateUnexpandedPC",
            "OpFlagIsCompressed",
            "OpFlagIsFirstInSequence",
            "OpFlagIsLastInSequence",
            "OpFlagJump",
            "OpFlagLoad",
            "OpFlagMultiplyOperands",
            "OpFlagStore",
            "OpFlagSubtractOperands",
            "OpFlagVirtualInstruction",
            "OpFlagWriteLookupOutputToRD",
            "PC",
            "UnexpandedPC",
        ]
        .into_iter()
        .map(str::to_owned),
    );
    for index in 0..params.lookup_table_count {
        let _inserted = trace_oracles.insert(format!("LookupTableFlag_{index}"));
    }
    for oracle in trace_oracles {
        append_virtual_oracle(context, module, &oracle, "jolt.trace_domain")?;
    }

    append_virtual_oracle(context, module, "RamRa", "jolt.stage2_ram_rw_domain")?;
    for oracle in ["RdWa", "Rs1Ra", "Rs2Ra"] {
        append_virtual_oracle(context, module, oracle, "jolt.stage4_registers_rw_domain")?;
    }

    append_committed_trace_oracle(context, module, "RamInc")?;
    append_committed_trace_oracle(context, module, "RdInc")?;
    for index in 0..params.instruction_d {
        append_committed_main_witness_oracle(context, module, &format!("InstructionRa_{index}"))?;
    }
    for index in 0..params.bytecode_d {
        append_committed_main_witness_oracle(context, module, &format!("BytecodeRa_{index}"))?;
    }
    for index in 0..params.ram_d {
        append_committed_main_witness_oracle(context, module, &format!("RamRa_{index}"))?;
    }
    Ok(())
}

fn append_virtual_oracle<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Protocol>,
    symbol: &str,
    domain: &str,
) -> Result<(), MlirError> {
    context.append_op(
        module,
        "piop.oracle",
        Some(symbol),
        &[
            ("field", "@bn254_fr"),
            ("domain", &format!("@{domain}")),
            ("commit_domain", &format!("@{domain}")),
            ("visibility", r#""virtual""#),
            ("layout", r#""virtual""#),
        ],
    )
}

fn append_committed_trace_oracle<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Protocol>,
    symbol: &str,
) -> Result<(), MlirError> {
    context.append_op(
        module,
        "piop.oracle",
        Some(symbol),
        &[
            ("field", "@bn254_fr"),
            ("domain", "@jolt.trace_domain"),
            ("commit_domain", "@jolt.main_witness_commit_domain"),
            ("visibility", r#""committed""#),
            ("layout", r#""dense_trace""#),
        ],
    )
}

fn append_committed_main_witness_oracle<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Protocol>,
    symbol: &str,
) -> Result<(), MlirError> {
    context.append_op(
        module,
        "piop.oracle",
        Some(symbol),
        &[
            ("field", "@bn254_fr"),
            ("domain", "@jolt.main_witness_commit_domain"),
            ("commit_domain", "@jolt.main_witness_commit_domain"),
            ("visibility", r#""committed""#),
            ("layout", r#""onehot_expanded""#),
        ],
    )
}

fn append_stage6_relations<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Protocol>,
    params: &JoltProtocolParams,
) -> Result<(), MlirError> {
    append_relation(
        context,
        module,
        RelationSpec {
            symbol: "jolt.stage6.bytecode_read_raf",
            kind: "sumcheck",
            domain: "jolt.stage6_bytecode_read_raf_domain",
            num_rounds: stage6_max_rounds(params),
            degree: params.bytecode_d + 1,
            output_count: params.bytecode_d,
        },
    )?;
    append_relation(
        context,
        module,
        RelationSpec {
            symbol: "jolt.stage6.booleanity",
            kind: "sumcheck",
            domain: "jolt.stage6_booleanity_domain",
            num_rounds: booleanity_rounds(params),
            degree: BOOLEANITY_DEGREE,
            output_count: total_ra_oracles(params),
        },
    )?;
    append_relation(
        context,
        module,
        RelationSpec {
            symbol: "jolt.stage6.hamming_booleanity",
            kind: "sumcheck",
            domain: "jolt.trace_domain",
            num_rounds: params.log_t,
            degree: HAMMING_BOOLEANITY_DEGREE,
            output_count: 1,
        },
    )?;
    append_relation(
        context,
        module,
        RelationSpec {
            symbol: "jolt.stage6.ram_ra_virtual",
            kind: "sumcheck",
            domain: "jolt.trace_domain",
            num_rounds: params.log_t,
            degree: params.ram_d + 1,
            output_count: params.ram_d,
        },
    )?;
    append_relation(
        context,
        module,
        RelationSpec {
            symbol: "jolt.stage6.instruction_ra_virtual",
            kind: "sumcheck",
            domain: "jolt.trace_domain",
            num_rounds: params.log_t,
            degree: n_committed_per_virtual(params) + 1,
            output_count: params.instruction_d,
        },
    )?;
    append_relation(
        context,
        module,
        RelationSpec {
            symbol: "jolt.stage6.inc_claim_reduction",
            kind: "sumcheck",
            domain: "jolt.trace_domain",
            num_rounds: params.log_t,
            degree: INC_CLAIM_REDUCTION_DEGREE,
            output_count: 2,
        },
    )?;
    append_relation(
        context,
        module,
        RelationSpec {
            symbol: "jolt.stage6.batched",
            kind: "batched_sumcheck",
            domain: "jolt.stage6_bytecode_read_raf_domain",
            num_rounds: stage6_max_rounds(params),
            degree: stage6_batched_degree(params),
            output_count: stage6_output_count(params),
        },
    )
}

fn append_relation<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Protocol>,
    spec: RelationSpec<'_>,
) -> Result<(), MlirError> {
    context.append_op(
        module,
        "piop.relation",
        Some(spec.symbol),
        &[
            ("kind", &format!("\"{}\"", spec.kind)),
            ("domain", &format!("@{}", spec.domain)),
            ("num_rounds", &int_attr(spec.num_rounds)),
            ("degree", &int_attr(spec.degree)),
            ("output_count", &int_attr(spec.output_count)),
        ],
    )
}

fn append_stage6_opening_inputs<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    params: &JoltProtocolParams,
) -> Result<Stage6OpeningInputs<'c, 'a>, MlirError> {
    let mut bytecode_terms = Vec::new();
    append_bytecode_term(
        context,
        module,
        params,
        &mut bytecode_terms,
        BytecodeTermSpec::trace(
            "stage6.input.stage1.UnexpandedPC",
            "stage1",
            "stage1.outer_remaining.opening.UnexpandedPC",
            "UnexpandedPC",
            0,
            Some(BytecodeStageGamma::Stage1),
            0,
        ),
    )?;
    append_bytecode_term(
        context,
        module,
        params,
        &mut bytecode_terms,
        BytecodeTermSpec::trace(
            "stage6.input.stage1.Imm",
            "stage1",
            "stage1.outer_remaining.opening.Imm",
            "Imm",
            0,
            Some(BytecodeStageGamma::Stage1),
            1,
        ),
    )?;
    for (index, oracle) in STAGE1_OP_FLAGS.iter().enumerate() {
        append_bytecode_term(
            context,
            module,
            params,
            &mut bytecode_terms,
            BytecodeTermSpec::trace(
                &format!("stage6.input.stage1.{oracle}"),
                "stage1",
                &format!("stage1.outer_remaining.opening.{oracle}"),
                oracle,
                0,
                Some(BytecodeStageGamma::Stage1),
                2 + index,
            ),
        )?;
    }
    for (oracle, stage_gamma_power) in [
        ("OpFlagJump", 0),
        ("InstructionFlagBranch", 1),
        ("OpFlagWriteLookupOutputToRD", 2),
        ("OpFlagVirtualInstruction", 3),
    ] {
        append_bytecode_term(
            context,
            module,
            params,
            &mut bytecode_terms,
            BytecodeTermSpec::trace(
                &format!("stage6.input.stage2.{oracle}"),
                "stage2",
                &format!("stage2.product_virtual.remainder.opening.{oracle}"),
                oracle,
                1,
                Some(BytecodeStageGamma::Stage2),
                stage_gamma_power,
            ),
        )?;
    }
    for (symbol, source_claim, oracle, stage_gamma_power) in [
        (
            "stage6.input.stage3.instruction_input.Imm",
            "stage3.instruction_input.opening.Imm",
            "Imm",
            0,
        ),
        (
            "stage6.input.stage3.spartan_shift.UnexpandedPC",
            "stage3.spartan_shift.opening.UnexpandedPC",
            "UnexpandedPC",
            1,
        ),
        (
            "stage6.input.stage3.instruction_input.InstructionFlagLeftOperandIsRs1Value",
            "stage3.instruction_input.opening.InstructionFlagLeftOperandIsRs1Value",
            "InstructionFlagLeftOperandIsRs1Value",
            2,
        ),
        (
            "stage6.input.stage3.instruction_input.InstructionFlagLeftOperandIsPC",
            "stage3.instruction_input.opening.InstructionFlagLeftOperandIsPC",
            "InstructionFlagLeftOperandIsPC",
            3,
        ),
        (
            "stage6.input.stage3.instruction_input.InstructionFlagRightOperandIsRs2Value",
            "stage3.instruction_input.opening.InstructionFlagRightOperandIsRs2Value",
            "InstructionFlagRightOperandIsRs2Value",
            4,
        ),
        (
            "stage6.input.stage3.instruction_input.InstructionFlagRightOperandIsImm",
            "stage3.instruction_input.opening.InstructionFlagRightOperandIsImm",
            "InstructionFlagRightOperandIsImm",
            5,
        ),
        (
            "stage6.input.stage3.spartan_shift.InstructionFlagIsNoop",
            "stage3.spartan_shift.opening.InstructionFlagIsNoop",
            "InstructionFlagIsNoop",
            6,
        ),
        (
            "stage6.input.stage3.spartan_shift.OpFlagVirtualInstruction",
            "stage3.spartan_shift.opening.OpFlagVirtualInstruction",
            "OpFlagVirtualInstruction",
            7,
        ),
        (
            "stage6.input.stage3.spartan_shift.OpFlagIsFirstInSequence",
            "stage3.spartan_shift.opening.OpFlagIsFirstInSequence",
            "OpFlagIsFirstInSequence",
            8,
        ),
    ] {
        append_bytecode_term(
            context,
            module,
            params,
            &mut bytecode_terms,
            BytecodeTermSpec::trace(
                symbol,
                "stage3",
                source_claim,
                oracle,
                2,
                Some(BytecodeStageGamma::Stage3),
                stage_gamma_power,
            ),
        )?;
    }
    for (oracle, stage_gamma_power) in [("RdWa", 0), ("Rs1Ra", 1), ("Rs2Ra", 2)] {
        append_bytecode_term(
            context,
            module,
            params,
            &mut bytecode_terms,
            BytecodeTermSpec {
                input: StageOpeningInputSpec {
                    symbol: &format!("stage6.input.stage4.{oracle}"),
                    source_stage: "stage4",
                    source_claim: &format!("stage4.registers_read_write.opening.{oracle}"),
                    oracle,
                    domain: "jolt.stage4_registers_rw_domain",
                    point_arity: params.register_log_k + params.log_t,
                    claim_kind: "virtual",
                },
                gamma_power: 3,
                stage_gamma: Some(BytecodeStageGamma::Stage4),
                stage_gamma_power,
            },
        )?;
    }
    append_bytecode_term(
        context,
        module,
        params,
        &mut bytecode_terms,
        BytecodeTermSpec {
            input: StageOpeningInputSpec {
                symbol: "stage6.input.stage5.registers_val_evaluation.RdWa",
                source_stage: "stage5",
                source_claim: "stage5.registers_val_evaluation.opening.RdWa",
                oracle: "RdWa",
                domain: "jolt.stage4_registers_rw_domain",
                point_arity: params.register_log_k + params.log_t,
                claim_kind: "virtual",
            },
            gamma_power: 4,
            stage_gamma: Some(BytecodeStageGamma::Stage5),
            stage_gamma_power: 0,
        },
    )?;
    append_bytecode_term(
        context,
        module,
        params,
        &mut bytecode_terms,
        BytecodeTermSpec::trace(
            "stage6.input.stage5.InstructionRafFlag",
            "stage5",
            "stage5.instruction_read_raf.opening.InstructionRafFlag",
            "InstructionRafFlag",
            4,
            Some(BytecodeStageGamma::Stage5),
            1,
        ),
    )?;
    for index in 0..params.lookup_table_count {
        let oracle = format!("LookupTableFlag_{index}");
        append_bytecode_term(
            context,
            module,
            params,
            &mut bytecode_terms,
            BytecodeTermSpec::trace(
                &format!("stage6.input.stage5.{oracle}"),
                "stage5",
                &format!("stage5.instruction_read_raf.opening.{oracle}"),
                &oracle,
                4,
                Some(BytecodeStageGamma::Stage5),
                2 + index,
            ),
        )?;
    }
    append_bytecode_term(
        context,
        module,
        params,
        &mut bytecode_terms,
        BytecodeTermSpec::trace(
            "stage6.input.stage1.PC",
            "stage1",
            "stage1.outer_remaining.opening.PC",
            "PC",
            5,
            None,
            0,
        ),
    )?;
    append_bytecode_term(
        context,
        module,
        params,
        &mut bytecode_terms,
        BytecodeTermSpec::trace(
            "stage6.input.stage3.spartan_shift.PC",
            "stage3",
            "stage3.spartan_shift.opening.PC",
            "PC",
            6,
            None,
            0,
        ),
    )?;

    let ram_ra_virtual = append_stage_input(
        context,
        module,
        StageOpeningInputSpec {
            symbol: "stage6.input.stage5.ram_ra_claim_reduction.RamRa",
            source_stage: "stage5",
            source_claim: "stage5.ram_ra_claim_reduction.opening.RamRa",
            oracle: "RamRa",
            domain: "jolt.stage2_ram_rw_domain",
            point_arity: params.log_k_ram + params.log_t,
            claim_kind: "virtual",
        },
    )?;

    let mut instruction_ra_virtual = Vec::with_capacity(params.instruction_ra_virtual_d);
    for index in 0..params.instruction_ra_virtual_d {
        instruction_ra_virtual.push(append_stage_input(
            context,
            module,
            StageOpeningInputSpec {
                symbol: &format!("stage6.input.stage5.instruction_read_raf.InstructionRa_{index}"),
                source_stage: "stage5",
                source_claim: &format!("stage5.instruction_read_raf.opening.InstructionRa_{index}"),
                oracle: &format!("InstructionRa_{index}"),
                domain: "jolt.stage5_instruction_ra_chunk_domain",
                point_arity: params.lookups_ra_virtual_log_k_chunk + params.log_t,
                claim_kind: "virtual",
            },
        )?);
    }

    Ok(Stage6OpeningInputs {
        bytecode_terms,
        hamming_lookup_output: append_stage_input(
            context,
            module,
            StageOpeningInputSpec {
                symbol: "stage6.input.stage1.LookupOutput",
                source_stage: "stage1",
                source_claim: "stage1.outer_remaining.opening.LookupOutput",
                oracle: "LookupOutput",
                domain: "jolt.trace_domain",
                point_arity: params.log_t,
                claim_kind: "virtual",
            },
        )?,
        ram_ra_virtual,
        instruction_ra_virtual,
        ram_inc_stage2: append_stage_input(
            context,
            module,
            StageOpeningInputSpec {
                symbol: "stage6.input.stage2.ram_read_write.RamInc",
                source_stage: "stage2",
                source_claim: "stage2.ram_read_write.opening.RamInc",
                oracle: "RamInc",
                domain: "jolt.trace_domain",
                point_arity: params.log_t,
                claim_kind: "committed",
            },
        )?,
        ram_inc_stage4: append_stage_input(
            context,
            module,
            StageOpeningInputSpec {
                symbol: "stage6.input.stage4.ram_val_check.RamInc",
                source_stage: "stage4",
                source_claim: "stage4.ram_val_check.opening.RamInc",
                oracle: "RamInc",
                domain: "jolt.trace_domain",
                point_arity: params.log_t,
                claim_kind: "committed",
            },
        )?,
        rd_inc_stage4: append_stage_input(
            context,
            module,
            StageOpeningInputSpec {
                symbol: "stage6.input.stage4.registers_read_write.RdInc",
                source_stage: "stage4",
                source_claim: "stage4.registers_read_write.opening.RdInc",
                oracle: "RdInc",
                domain: "jolt.trace_domain",
                point_arity: params.log_t,
                claim_kind: "committed",
            },
        )?,
        rd_inc_stage5: append_stage_input(
            context,
            module,
            StageOpeningInputSpec {
                symbol: "stage6.input.stage5.registers_val_evaluation.RdInc",
                source_stage: "stage5",
                source_claim: "stage5.registers_val_evaluation.opening.RdInc",
                oracle: "RdInc",
                domain: "jolt.trace_domain",
                point_arity: params.log_t,
                claim_kind: "committed",
            },
        )?,
    })
}

fn append_bytecode_term<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    params: &JoltProtocolParams,
    terms: &mut Vec<Stage6BytecodeTerm<'c, 'a>>,
    mut spec: BytecodeTermSpec<'_>,
) -> Result<(), MlirError> {
    if spec.input.point_arity == 0 {
        spec.input.point_arity = params.log_t;
    }
    let input = append_stage_input(context, module, spec.input)?;
    terms.push(Stage6BytecodeTerm {
        eval: input.eval,
        claim: input.claim,
        gamma_power: spec.gamma_power,
        stage_gamma: spec.stage_gamma,
        stage_gamma_power: spec.stage_gamma_power,
    });
    Ok(())
}

fn append_stage_input<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    spec: StageOpeningInputSpec<'_>,
) -> Result<Stage6OpeningInput<'c, 'a>, MlirError> {
    let op = context.append_typed_op(
        module,
        "piop.opening_input",
        Some(spec.symbol),
        &[
            ("source_stage", &format!("@{}", spec.source_stage)),
            ("source_claim", &format!("@{}", spec.source_claim)),
            ("oracle", &format!("@{}", spec.oracle)),
            ("domain", &format!("@{}", spec.domain)),
            ("point_arity", &int_attr(spec.point_arity)),
            ("claim_kind", &format!("\"{}\"", spec.claim_kind)),
        ],
        &[],
        &["!poly.point", "!field.scalar", "!piop.opening_claim_type"],
    )?;
    Ok(Stage6OpeningInput {
        point: result(op, 0, "piop.opening_input")?,
        eval: result(op, 1, "piop.opening_input")?,
        claim: result(op, 2, "piop.opening_input")?,
    })
}

fn append_transcript_squeeze<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    state: Value<'c, 'a>,
    symbol: &str,
    label: &str,
    kind: &str,
    count: usize,
) -> Result<(Value<'c, 'a>, Value<'c, 'a>), MlirError> {
    let op = context.append_typed_op(
        module,
        "transcript.squeeze",
        Some(symbol),
        &[
            ("label", &format!("\"{label}\"")),
            ("kind", &format!("\"{kind}\"")),
            ("count", &int_attr(count)),
        ],
        &[state],
        &[
            "!transcript.state_type",
            transcript_squeeze_protocol_result_type(kind)?,
        ],
    )?;
    Ok((
        result(op, 0, "transcript.squeeze")?,
        result(op, 1, "transcript.squeeze")?,
    ))
}

fn append_booleanity_power_placeholders<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    params: &JoltProtocolParams,
    booleanity_gamma: Value<'c, 'a>,
) -> Result<(), MlirError> {
    let total = total_ra_oracles(params);
    for index in 0..total {
        let _ = append_field_pow(
            context,
            module,
            &format!("stage6.booleanity.gamma_sq_{index}"),
            booleanity_gamma,
            2 * index,
        )?;
    }
    for index in 0..total {
        let _ = append_field_pow(
            context,
            module,
            &format!("stage6.booleanity.gamma_pow_{index}"),
            booleanity_gamma,
            index,
        )?;
    }
    Ok(())
}

fn append_stage6_batched_sumcheck<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    params: &JoltProtocolParams,
    spec: Stage6BatchedSumcheckInputs<'c, 'a, '_>,
) -> Result<Value<'c, 'a>, MlirError> {
    let inputs = spec.openings;
    let bytecode_claim = append_bytecode_read_raf_claim(context, module, inputs, &spec)?;
    let zero = append_field_zero(context, module, "stage6.zero")?;
    let ram_ra_virtual_claim = inputs.ram_ra_virtual.eval;
    let inst_ra_virtual_claim =
        append_instruction_ra_virtual_claim(context, module, inputs, spec.inst_ra_gamma)?;
    let inc_claim = append_inc_claim_reduction_claim(context, module, inputs, spec.inc_gamma)?;

    let bytecode_inputs = inputs
        .bytecode_terms
        .iter()
        .map(|term| term.claim)
        .collect::<Vec<_>>();
    let instruction_inputs = inputs
        .instruction_ra_virtual
        .iter()
        .map(|input| input.claim)
        .collect::<Vec<_>>();
    let inc_inputs = [
        inputs.ram_inc_stage2.claim,
        inputs.ram_inc_stage4.claim,
        inputs.rd_inc_stage4.claim,
        inputs.rd_inc_stage5.claim,
    ];
    let claims = [
        append_sumcheck_claim(
            context,
            module,
            SumcheckClaimSpec {
                symbol: "stage6.bytecode_read_raf.input",
                stage: "stage6",
                domain: "jolt.stage6_bytecode_read_raf_domain",
                num_rounds: stage6_max_rounds(params),
                degree: params.bytecode_d + 1,
                claim: "stage6.bytecode_read_raf.weighted_prior_stage_values",
                relation: "jolt.stage6.bytecode_read_raf",
            },
            bytecode_claim,
            &bytecode_inputs,
        )?,
        append_sumcheck_claim(
            context,
            module,
            SumcheckClaimSpec {
                symbol: "stage6.booleanity.input",
                stage: "stage6",
                domain: "jolt.stage6_booleanity_domain",
                num_rounds: booleanity_rounds(params),
                degree: BOOLEANITY_DEGREE,
                claim: "stage6.booleanity.zero",
                relation: "jolt.stage6.booleanity",
            },
            zero,
            &[],
        )?,
        append_sumcheck_claim(
            context,
            module,
            SumcheckClaimSpec {
                symbol: "stage6.hamming_booleanity.input",
                stage: "stage6",
                domain: "jolt.trace_domain",
                num_rounds: params.log_t,
                degree: HAMMING_BOOLEANITY_DEGREE,
                claim: "stage6.hamming_booleanity.zero",
                relation: "jolt.stage6.hamming_booleanity",
            },
            zero,
            &[inputs.hamming_lookup_output.claim],
        )?,
        append_sumcheck_claim(
            context,
            module,
            SumcheckClaimSpec {
                symbol: "stage6.ram_ra_virtual.input",
                stage: "stage6",
                domain: "jolt.trace_domain",
                num_rounds: params.log_t,
                degree: params.ram_d + 1,
                claim: "stage6.ram_ra_virtual.weighted_ram_ra",
                relation: "jolt.stage6.ram_ra_virtual",
            },
            ram_ra_virtual_claim,
            &[inputs.ram_ra_virtual.claim],
        )?,
        append_sumcheck_claim(
            context,
            module,
            SumcheckClaimSpec {
                symbol: "stage6.instruction_ra_virtual.input",
                stage: "stage6",
                domain: "jolt.trace_domain",
                num_rounds: params.log_t,
                degree: n_committed_per_virtual(params) + 1,
                claim: "stage6.instruction_ra_virtual.weighted_instruction_ra",
                relation: "jolt.stage6.instruction_ra_virtual",
            },
            inst_ra_virtual_claim,
            &instruction_inputs,
        )?,
        append_sumcheck_claim(
            context,
            module,
            SumcheckClaimSpec {
                symbol: "stage6.inc_claim_reduction.input",
                stage: "stage6",
                domain: "jolt.trace_domain",
                num_rounds: params.log_t,
                degree: INC_CLAIM_REDUCTION_DEGREE,
                claim: "stage6.inc_claim_reduction.weighted_increments",
                relation: "jolt.stage6.inc_claim_reduction",
            },
            inc_claim,
            &inc_inputs,
        )?,
    ];
    let round_schedule = format!("[{}, {}]", params.log_k_bytecode, params.log_t);
    let batch = append_sumcheck_batch(
        context,
        module,
        spec.stage,
        &claims,
        SumcheckBatchSpec {
            symbol: "stage6.batch",
            stage: "stage6",
            proof_slot: "stage6.sumcheck",
            policy: "jolt_core_stage6_aligned",
            ordered_claims: &[
                "stage6.bytecode_read_raf.input",
                "stage6.booleanity.input",
                "stage6.hamming_booleanity.input",
                "stage6.ram_ra_virtual.input",
                "stage6.instruction_ra_virtual.input",
                "stage6.inc_claim_reduction.input",
            ],
            claim_label: "sumcheck_claim",
            round_label: "sumcheck_poly",
            round_schedule: &round_schedule,
        },
    )?;
    let (state, point, result_value) = append_sumcheck(
        context,
        module,
        spec.state,
        batch,
        SumcheckDriverSpec {
            symbol: "stage6.sumcheck",
            stage: "stage6",
            proof_slot: "stage6.sumcheck",
            relation: "jolt.stage6.batched",
            policy: "jolt_core_stage6_aligned",
            round_schedule: &round_schedule,
            claim_label: "sumcheck_claim",
            round_label: "sumcheck_poly",
            num_rounds: stage6_max_rounds(params),
            degree: stage6_batched_degree(params),
        },
    )?;
    let bytecode = append_sumcheck_instance_result(
        context,
        module,
        SumcheckInstanceResultSpec {
            symbol: "stage6.bytecode_read_raf.instance",
            source: "stage6.sumcheck",
            claim: "stage6.bytecode_read_raf.input",
            relation: "jolt.stage6.bytecode_read_raf",
            index: 0,
            point_arity: stage6_max_rounds(params),
            num_rounds: stage6_max_rounds(params),
            round_offset: 0,
            point_order: "bytecode_read_raf",
            degree: params.bytecode_d + 1,
        },
        point,
        result_value,
    )?;
    let booleanity = append_sumcheck_instance_result(
        context,
        module,
        SumcheckInstanceResultSpec {
            symbol: "stage6.booleanity.instance",
            source: "stage6.sumcheck",
            claim: "stage6.booleanity.input",
            relation: "jolt.stage6.booleanity",
            index: 1,
            point_arity: booleanity_rounds(params),
            num_rounds: booleanity_rounds(params),
            round_offset: params.log_k_bytecode.saturating_sub(params.log_k_chunk),
            point_order: "stage6_booleanity",
            degree: BOOLEANITY_DEGREE,
        },
        point,
        result_value,
    )?;
    let hamming = append_stage6_trace_instance_result(
        context,
        module,
        params,
        point,
        result_value,
        Stage6TraceInstanceSpec {
            symbol: "stage6.hamming_booleanity.instance",
            claim: "stage6.hamming_booleanity.input",
            relation: "jolt.stage6.hamming_booleanity",
            index: 2,
            degree: HAMMING_BOOLEANITY_DEGREE,
        },
    )?;
    let ram = append_stage6_trace_instance_result(
        context,
        module,
        params,
        point,
        result_value,
        Stage6TraceInstanceSpec {
            symbol: "stage6.ram_ra_virtual.instance",
            claim: "stage6.ram_ra_virtual.input",
            relation: "jolt.stage6.ram_ra_virtual",
            index: 3,
            degree: params.ram_d + 1,
        },
    )?;
    let instruction = append_stage6_trace_instance_result(
        context,
        module,
        params,
        point,
        result_value,
        Stage6TraceInstanceSpec {
            symbol: "stage6.instruction_ra_virtual.instance",
            claim: "stage6.instruction_ra_virtual.input",
            relation: "jolt.stage6.instruction_ra_virtual",
            index: 4,
            degree: n_committed_per_virtual(params) + 1,
        },
    )?;
    let inc = append_stage6_trace_instance_result(
        context,
        module,
        params,
        point,
        result_value,
        Stage6TraceInstanceSpec {
            symbol: "stage6.inc_claim_reduction.instance",
            claim: "stage6.inc_claim_reduction.input",
            relation: "jolt.stage6.inc_claim_reduction",
            index: 5,
            degree: INC_CLAIM_REDUCTION_DEGREE,
        },
    )?;
    append_stage6_output_openings(
        context,
        module,
        params,
        inputs,
        bytecode,
        booleanity,
        hamming,
        ram,
        instruction,
        inc,
    )?;
    Ok(state)
}

fn append_bytecode_read_raf_claim<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    inputs: &Stage6OpeningInputs<'c, 'a>,
    spec: &Stage6BatchedSumcheckInputs<'c, 'a, '_>,
) -> Result<Value<'c, 'a>, MlirError> {
    let mut terms = Vec::with_capacity(inputs.bytecode_terms.len() + 1);
    for (index, term) in inputs.bytecode_terms.iter().enumerate() {
        terms.push(append_weighted_eval(
            context,
            module,
            &format!("stage6.bytecode_read_raf.claim.term{index}"),
            term.eval,
            WeightedEvalSpec {
                gamma: spec.bc_gamma,
                gamma_power: term.gamma_power,
                stage_gamma: term.stage_gamma.map(|gamma| stage_gamma_value(gamma, spec)),
                stage_gamma_power: term.stage_gamma_power,
            },
        )?);
    }
    terms.push(append_field_pow(
        context,
        module,
        "stage6.bytecode_read_raf.claim.entry_constant",
        spec.bc_gamma,
        7,
    )?);
    append_field_sum(
        context,
        module,
        "stage6.bytecode_read_raf.claim_expr",
        &terms,
    )
}

fn append_instruction_ra_virtual_claim<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    inputs: &Stage6OpeningInputs<'c, 'a>,
    gamma: Value<'c, 'a>,
) -> Result<Value<'c, 'a>, MlirError> {
    let mut terms = Vec::with_capacity(inputs.instruction_ra_virtual.len());
    for (index, input) in inputs.instruction_ra_virtual.iter().enumerate() {
        terms.push(append_weighted_eval(
            context,
            module,
            &format!("stage6.instruction_ra_virtual.claim.term{index}"),
            input.eval,
            WeightedEvalSpec {
                gamma,
                gamma_power: index,
                stage_gamma: None,
                stage_gamma_power: 0,
            },
        )?);
    }
    append_field_sum(
        context,
        module,
        "stage6.instruction_ra_virtual.claim_expr",
        &terms,
    )
}

fn append_inc_claim_reduction_claim<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    inputs: &Stage6OpeningInputs<'c, 'a>,
    gamma: Value<'c, 'a>,
) -> Result<Value<'c, 'a>, MlirError> {
    let terms = [
        inputs.ram_inc_stage2.eval,
        append_weighted_eval(
            context,
            module,
            "stage6.inc_claim_reduction.claim.ram_inc_stage4",
            inputs.ram_inc_stage4.eval,
            WeightedEvalSpec {
                gamma,
                gamma_power: 1,
                stage_gamma: None,
                stage_gamma_power: 0,
            },
        )?,
        append_weighted_eval(
            context,
            module,
            "stage6.inc_claim_reduction.claim.rd_inc_stage4",
            inputs.rd_inc_stage4.eval,
            WeightedEvalSpec {
                gamma,
                gamma_power: 2,
                stage_gamma: None,
                stage_gamma_power: 0,
            },
        )?,
        append_weighted_eval(
            context,
            module,
            "stage6.inc_claim_reduction.claim.rd_inc_stage5",
            inputs.rd_inc_stage5.eval,
            WeightedEvalSpec {
                gamma,
                gamma_power: 3,
                stage_gamma: None,
                stage_gamma_power: 0,
            },
        )?,
    ];
    append_field_sum(
        context,
        module,
        "stage6.inc_claim_reduction.claim_expr",
        &terms,
    )
}

fn append_weighted_eval<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    symbol_prefix: &str,
    eval: Value<'c, 'a>,
    spec: WeightedEvalSpec<'c, 'a>,
) -> Result<Value<'c, 'a>, MlirError> {
    let mut value = eval;
    if spec.stage_gamma_power > 0 {
        let power = append_field_pow(
            context,
            module,
            &format!("{symbol_prefix}.stage_gamma_pow"),
            spec.stage_gamma.ok_or_else(|| MlirError::Schema {
                message: format!(
                    "{symbol_prefix} requires stage gamma when stage_gamma_power is non-zero"
                ),
            })?,
            spec.stage_gamma_power,
        )?;
        value = append_field_mul(
            context,
            module,
            &format!("{symbol_prefix}.stage_gamma_term"),
            power,
            value,
        )?;
    }
    if spec.gamma_power > 0 {
        let power = append_field_pow(
            context,
            module,
            &format!("{symbol_prefix}.gamma_pow"),
            spec.gamma,
            spec.gamma_power,
        )?;
        value = append_field_mul(
            context,
            module,
            &format!("{symbol_prefix}.gamma_term"),
            power,
            value,
        )?;
    }
    Ok(value)
}

fn stage_gamma_value<'c, 'a>(
    gamma: BytecodeStageGamma,
    spec: &Stage6BatchedSumcheckInputs<'c, 'a, '_>,
) -> Value<'c, 'a> {
    match gamma {
        BytecodeStageGamma::Stage1 => spec.bc_stage1_gamma,
        BytecodeStageGamma::Stage2 => spec.bc_stage2_gamma,
        BytecodeStageGamma::Stage3 => spec.bc_stage3_gamma,
        BytecodeStageGamma::Stage4 => spec.bc_stage4_gamma,
        BytecodeStageGamma::Stage5 => spec.bc_stage5_gamma,
    }
}

fn append_stage6_trace_instance_result<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    params: &JoltProtocolParams,
    point: Value<'c, 'a>,
    result_value: Value<'c, 'a>,
    spec: Stage6TraceInstanceSpec<'_>,
) -> Result<(Value<'c, 'a>, Value<'c, 'a>), MlirError> {
    append_sumcheck_instance_result(
        context,
        module,
        SumcheckInstanceResultSpec {
            symbol: spec.symbol,
            source: "stage6.sumcheck",
            claim: spec.claim,
            relation: spec.relation,
            index: spec.index,
            point_arity: params.log_t,
            num_rounds: params.log_t,
            round_offset: params.log_k_bytecode,
            point_order: "reverse",
            degree: spec.degree,
        },
        point,
        result_value,
    )
}

#[expect(clippy::too_many_arguments)]
fn append_stage6_output_openings<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    params: &JoltProtocolParams,
    inputs: &Stage6OpeningInputs<'c, 'a>,
    bytecode: (Value<'c, 'a>, Value<'c, 'a>),
    booleanity: (Value<'c, 'a>, Value<'c, 'a>),
    hamming: (Value<'c, 'a>, Value<'c, 'a>),
    ram: (Value<'c, 'a>, Value<'c, 'a>),
    instruction: (Value<'c, 'a>, Value<'c, 'a>),
    inc: (Value<'c, 'a>, Value<'c, 'a>),
) -> Result<(), MlirError> {
    let mut claims = Vec::new();
    let mut claim_symbols = Vec::new();

    let bytecode_cycle = append_point_slice(
        context,
        module,
        "stage6.bytecode_read_raf.point.Cycle",
        "stage6.bytecode_read_raf.instance",
        params.log_k_bytecode,
        params.log_t,
        bytecode.0,
    )?;
    for index in 0..params.bytecode_d {
        let oracle = format!("BytecodeRa_{index}");
        let eval_symbol = format!("stage6.bytecode_read_raf.eval.{oracle}");
        let eval = append_sumcheck_eval(
            context,
            module,
            &eval_symbol,
            "stage6.sumcheck",
            &oracle,
            index,
            bytecode.1,
        )?;
        let address = append_padded_address_chunk(
            context,
            module,
            &format!("stage6.bytecode_read_raf.point.{oracle}.address"),
            "stage6.bytecode_read_raf.instance",
            params.log_k_bytecode,
            index,
            params.log_k_chunk,
            bytecode.0,
        )?;
        let point = append_point_concat(
            context,
            module,
            &format!("stage6.bytecode_read_raf.point.{oracle}"),
            "address_chunk_then_cycle",
            params.log_k_chunk + params.log_t,
            &[address, bytecode_cycle],
        )?;
        let symbol = format!("stage6.bytecode_read_raf.opening.{oracle}");
        claim_symbols.push(symbol.clone());
        claims.push(append_opening_claim(
            context,
            module,
            point,
            eval,
            OpeningClaimSpec {
                symbol: &symbol,
                oracle: &oracle,
                domain: "jolt.main_witness_commit_domain",
                point_arity: params.log_k_chunk + params.log_t,
                claim_kind: "committed",
            },
        )?);
    }

    let mut eval_index = 0;
    for index in 0..params.instruction_d {
        append_booleanity_output_opening(
            context,
            module,
            params,
            &mut claims,
            &mut claim_symbols,
            booleanity,
            &format!("InstructionRa_{index}"),
            eval_index,
        )?;
        eval_index += 1;
    }
    for index in 0..params.bytecode_d {
        append_booleanity_output_opening(
            context,
            module,
            params,
            &mut claims,
            &mut claim_symbols,
            booleanity,
            &format!("BytecodeRa_{index}"),
            eval_index,
        )?;
        eval_index += 1;
    }
    for index in 0..params.ram_d {
        append_booleanity_output_opening(
            context,
            module,
            params,
            &mut claims,
            &mut claim_symbols,
            booleanity,
            &format!("RamRa_{index}"),
            eval_index,
        )?;
        eval_index += 1;
    }

    let hamming_eval = append_sumcheck_eval(
        context,
        module,
        "stage6.hamming_booleanity.eval.HammingWeight",
        "stage6.sumcheck",
        "HammingWeight",
        0,
        hamming.1,
    )?;
    claim_symbols.push("stage6.hamming_booleanity.opening.HammingWeight".to_owned());
    claims.push(append_opening_claim(
        context,
        module,
        hamming.0,
        hamming_eval,
        OpeningClaimSpec {
            symbol: "stage6.hamming_booleanity.opening.HammingWeight",
            oracle: "HammingWeight",
            domain: "jolt.trace_domain",
            point_arity: params.log_t,
            claim_kind: "virtual",
        },
    )?);

    for index in 0..params.ram_d {
        let oracle = format!("RamRa_{index}");
        let symbol = format!("stage6.ram_ra_virtual.opening.{oracle}");
        let address = append_padded_address_chunk(
            context,
            module,
            &format!("stage6.ram_ra_virtual.point.{oracle}.address"),
            "stage6.input.stage5.ram_ra_claim_reduction.RamRa",
            params.log_k_ram,
            index,
            params.log_k_chunk,
            inputs.ram_ra_virtual.point,
        )?;
        let point = append_point_concat(
            context,
            module,
            &format!("stage6.ram_ra_virtual.point.{oracle}"),
            "address_chunk_then_cycle",
            params.log_k_chunk + params.log_t,
            &[address, ram.0],
        )?;
        let eval = append_sumcheck_eval(
            context,
            module,
            &format!("stage6.ram_ra_virtual.eval.{oracle}"),
            "stage6.sumcheck",
            &oracle,
            index,
            ram.1,
        )?;
        claim_symbols.push(symbol.clone());
        claims.push(append_opening_claim(
            context,
            module,
            point,
            eval,
            OpeningClaimSpec {
                symbol: &symbol,
                oracle: &oracle,
                domain: "jolt.main_witness_commit_domain",
                point_arity: params.log_k_chunk + params.log_t,
                claim_kind: "committed",
            },
        )?);
    }
    for index in 0..params.instruction_d {
        let oracle = format!("InstructionRa_{index}");
        let symbol = format!("stage6.instruction_ra_virtual.opening.{oracle}");
        let virtual_index = index / n_committed_per_virtual(params);
        let chunk_index = index % n_committed_per_virtual(params);
        let virtual_input = inputs.instruction_ra_virtual[virtual_index].point;
        let address = append_padded_address_chunk(
            context,
            module,
            &format!("stage6.instruction_ra_virtual.point.{oracle}.address"),
            &format!("stage6.input.stage5.instruction_read_raf.InstructionRa_{virtual_index}"),
            params.lookups_ra_virtual_log_k_chunk,
            chunk_index,
            params.log_k_chunk,
            virtual_input,
        )?;
        let point = append_point_concat(
            context,
            module,
            &format!("stage6.instruction_ra_virtual.point.{oracle}"),
            "address_chunk_then_cycle",
            params.log_k_chunk + params.log_t,
            &[address, instruction.0],
        )?;
        let eval = append_sumcheck_eval(
            context,
            module,
            &format!("stage6.instruction_ra_virtual.eval.{oracle}"),
            "stage6.sumcheck",
            &oracle,
            index,
            instruction.1,
        )?;
        claim_symbols.push(symbol.clone());
        claims.push(append_opening_claim(
            context,
            module,
            point,
            eval,
            OpeningClaimSpec {
                symbol: &symbol,
                oracle: &oracle,
                domain: "jolt.main_witness_commit_domain",
                point_arity: params.log_k_chunk + params.log_t,
                claim_kind: "committed",
            },
        )?);
    }

    for (index, oracle) in ["RamInc", "RdInc"].iter().enumerate() {
        let symbol = format!("stage6.inc_claim_reduction.opening.{oracle}");
        let eval = append_sumcheck_eval(
            context,
            module,
            &format!("stage6.inc_claim_reduction.eval.{oracle}"),
            "stage6.sumcheck",
            oracle,
            index,
            inc.1,
        )?;
        claim_symbols.push(symbol.clone());
        claims.push(append_opening_claim(
            context,
            module,
            inc.0,
            eval,
            OpeningClaimSpec {
                symbol: &symbol,
                oracle,
                domain: "jolt.trace_domain",
                point_arity: params.log_t,
                claim_kind: "committed",
            },
        )?);
    }

    let claim_names = claim_symbols.iter().map(String::as_str).collect::<Vec<_>>();
    let _batch = context.append_typed_op(
        module,
        "piop.opening_batch",
        Some("stage6.openings"),
        &[
            ("stage", "@stage6"),
            ("proof_slot", "@stage6.openings"),
            ("policy", r#""jolt_stage6_output_order""#),
            ("count", &int_attr(claims.len())),
            ("ordered_claims", &symbol_array_attr(&claim_names)),
        ],
        &claims,
        &["!piop.opening_batch_type"],
    )?;
    Ok(())
}

#[expect(clippy::too_many_arguments)]
fn append_booleanity_output_opening<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    params: &JoltProtocolParams,
    claims: &mut Vec<Value<'c, 'a>>,
    claim_symbols: &mut Vec<String>,
    booleanity: (Value<'c, 'a>, Value<'c, 'a>),
    oracle: &str,
    eval_index: usize,
) -> Result<(), MlirError> {
    let symbol = format!("stage6.booleanity.opening.{oracle}");
    let eval = append_sumcheck_eval(
        context,
        module,
        &format!("stage6.booleanity.eval.{oracle}"),
        "stage6.sumcheck",
        oracle,
        eval_index,
        booleanity.1,
    )?;
    claim_symbols.push(symbol.clone());
    claims.push(append_opening_claim(
        context,
        module,
        booleanity.0,
        eval,
        OpeningClaimSpec {
            symbol: &symbol,
            oracle,
            domain: "jolt.main_witness_commit_domain",
            point_arity: booleanity_rounds(params),
            claim_kind: "committed",
        },
    )?);
    Ok(())
}

fn append_field_zero<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    symbol: &str,
) -> Result<Value<'c, 'a>, MlirError> {
    let op = context.append_typed_op(
        module,
        "field.zero",
        Some(symbol),
        &[("field", "@bn254_fr")],
        &[],
        &["!field.scalar"],
    )?;
    first_result(op, "field.zero")
}

fn append_field_binary<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    op_name: &str,
    symbol: &str,
    lhs: Value<'c, 'a>,
    rhs: Value<'c, 'a>,
) -> Result<Value<'c, 'a>, MlirError> {
    let op = context.append_typed_op(
        module,
        op_name,
        Some(symbol),
        &[],
        &[lhs, rhs],
        &["!field.scalar"],
    )?;
    first_result(op, op_name)
}

fn append_field_add<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    symbol: &str,
    lhs: Value<'c, 'a>,
    rhs: Value<'c, 'a>,
) -> Result<Value<'c, 'a>, MlirError> {
    append_field_binary(context, module, "field.add", symbol, lhs, rhs)
}

fn append_field_mul<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    symbol: &str,
    lhs: Value<'c, 'a>,
    rhs: Value<'c, 'a>,
) -> Result<Value<'c, 'a>, MlirError> {
    append_field_binary(context, module, "field.mul", symbol, lhs, rhs)
}

fn append_field_pow<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    symbol: &str,
    base: Value<'c, 'a>,
    exponent: usize,
) -> Result<Value<'c, 'a>, MlirError> {
    let op = context.append_typed_op(
        module,
        "field.pow",
        Some(symbol),
        &[("exponent", &int_attr(exponent))],
        &[base],
        &["!field.scalar"],
    )?;
    first_result(op, "field.pow")
}

fn append_field_sum<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    symbol_prefix: &str,
    terms: &[Value<'c, 'a>],
) -> Result<Value<'c, 'a>, MlirError> {
    let Some((&first, rest)) = terms.split_first() else {
        return append_field_zero(context, module, symbol_prefix);
    };
    let mut value = first;
    for (index, &term) in rest.iter().enumerate() {
        value = append_field_add(
            context,
            module,
            &format!("{symbol_prefix}.partial{index}"),
            value,
            term,
        )?;
    }
    Ok(value)
}

fn append_sumcheck_claim<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    spec: SumcheckClaimSpec<'_>,
    input_claim: Value<'c, 'a>,
    inputs: &[Value<'c, 'a>],
) -> Result<Value<'c, 'a>, MlirError> {
    let mut operands = Vec::with_capacity(inputs.len() + 1);
    operands.push(input_claim);
    operands.extend_from_slice(inputs);
    let op = context.append_typed_op(
        module,
        "piop.sumcheck_claim",
        Some(spec.symbol),
        &[
            ("stage", &format!("@{}", spec.stage)),
            ("domain", &format!("@{}", spec.domain)),
            ("num_rounds", &int_attr(spec.num_rounds)),
            ("degree", &int_attr(spec.degree)),
            ("claim", &format!("@{}", spec.claim)),
            ("relation", &format!("@{}", spec.relation)),
        ],
        &operands,
        &["!piop.sumcheck_claim_type"],
    )?;
    first_result(op, "piop.sumcheck_claim")
}

fn append_sumcheck_batch<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    stage: Value<'c, 'a>,
    claims: &[Value<'c, 'a>],
    spec: SumcheckBatchSpec<'_>,
) -> Result<Value<'c, 'a>, MlirError> {
    let mut operands = Vec::with_capacity(claims.len() + 1);
    operands.push(stage);
    operands.extend_from_slice(claims);
    let op = context.append_typed_op(
        module,
        "piop.sumcheck_batch",
        Some(spec.symbol),
        &[
            ("stage", &format!("@{}", spec.stage)),
            ("proof_slot", &format!("@{}", spec.proof_slot)),
            ("policy", &format!("\"{}\"", spec.policy)),
            ("count", &int_attr(spec.ordered_claims.len())),
            ("ordered_claims", &symbol_array_attr(spec.ordered_claims)),
            ("claim_label", &format!("\"{}\"", spec.claim_label)),
            ("round_label", &format!("\"{}\"", spec.round_label)),
            ("round_schedule", spec.round_schedule),
        ],
        &operands,
        &["!piop.sumcheck_batch_type"],
    )?;
    first_result(op, "piop.sumcheck_batch")
}

fn append_sumcheck<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    state: Value<'c, 'a>,
    batch: Value<'c, 'a>,
    spec: SumcheckDriverSpec<'_>,
) -> Result<(Value<'c, 'a>, Value<'c, 'a>, Value<'c, 'a>), MlirError> {
    let op = context.append_typed_op(
        module,
        "piop.sumcheck",
        Some(spec.symbol),
        &[
            ("stage", &format!("@{}", spec.stage)),
            ("proof_slot", &format!("@{}", spec.proof_slot)),
            ("relation", &format!("@{}", spec.relation)),
            ("policy", &format!("\"{}\"", spec.policy)),
            ("round_schedule", spec.round_schedule),
            ("claim_label", &format!("\"{}\"", spec.claim_label)),
            ("round_label", &format!("\"{}\"", spec.round_label)),
            ("num_rounds", &int_attr(spec.num_rounds)),
            ("degree", &int_attr(spec.degree)),
        ],
        &[state, batch],
        &[
            "!transcript.state_type",
            "!poly.point",
            "!piop.sumcheck_result_type",
            "!piop.sumcheck_proof_type",
        ],
    )?;
    Ok((
        result(op, 0, "piop.sumcheck")?,
        result(op, 1, "piop.sumcheck")?,
        result(op, 2, "piop.sumcheck")?,
    ))
}

fn append_sumcheck_instance_result<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    spec: SumcheckInstanceResultSpec<'_>,
    point: Value<'c, 'a>,
    result_value: Value<'c, 'a>,
) -> Result<(Value<'c, 'a>, Value<'c, 'a>), MlirError> {
    let op = context.append_typed_op(
        module,
        "piop.sumcheck_instance_result",
        Some(spec.symbol),
        &[
            ("source", &format!("@{}", spec.source)),
            ("claim", &format!("@{}", spec.claim)),
            ("relation", &format!("@{}", spec.relation)),
            ("index", &int_attr(spec.index)),
            ("point_arity", &int_attr(spec.point_arity)),
            ("num_rounds", &int_attr(spec.num_rounds)),
            ("round_offset", &int_attr(spec.round_offset)),
            ("point_order", &format!("\"{}\"", spec.point_order)),
            ("degree", &int_attr(spec.degree)),
        ],
        &[point, result_value],
        &["!poly.point", "!piop.sumcheck_result_type"],
    )?;
    Ok((
        result(op, 0, "piop.sumcheck_instance_result")?,
        result(op, 1, "piop.sumcheck_instance_result")?,
    ))
}

fn append_sumcheck_eval<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    symbol: &str,
    source: &str,
    oracle: &str,
    index: usize,
    result_value: Value<'c, 'a>,
) -> Result<Value<'c, 'a>, MlirError> {
    let op = context.append_typed_op(
        module,
        "piop.sumcheck_eval",
        Some(symbol),
        &[
            ("source", &format!("@{}", source)),
            ("name", &format!("@{}", symbol)),
            ("index", &int_attr(index)),
            ("oracle", &format!("@{}", oracle)),
        ],
        &[result_value],
        &["!field.scalar"],
    )?;
    first_result(op, "piop.sumcheck_eval")
}

fn append_point_slice<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    symbol: &str,
    source: &str,
    offset: usize,
    length: usize,
    input: Value<'c, 'a>,
) -> Result<Value<'c, 'a>, MlirError> {
    let op = context.append_typed_op(
        module,
        "poly.point_slice",
        Some(symbol),
        &[
            ("source", &format!("@{}", source)),
            ("offset", &int_attr(offset)),
            ("length", &int_attr(length)),
        ],
        &[input],
        &["!poly.point"],
    )?;
    first_result(op, "poly.point_slice")
}

#[expect(clippy::too_many_arguments)]
fn append_padded_address_chunk<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    symbol: &str,
    source: &str,
    address_len: usize,
    chunk_index: usize,
    chunk_len: usize,
    input: Value<'c, 'a>,
) -> Result<Value<'c, 'a>, MlirError> {
    let pad_len = (chunk_len - (address_len % chunk_len)) % chunk_len;
    let padded_offset = chunk_index * chunk_len;
    let zero_len = pad_len.saturating_sub(padded_offset).min(chunk_len);
    let source_offset = padded_offset.saturating_sub(pad_len);
    let source_len = chunk_len - zero_len;
    if source_offset + source_len > address_len {
        return Err(schema_error(format!(
            "address chunk {chunk_index} exceeds source point @{source}"
        )));
    }

    let source_chunk = if source_len == 0 {
        None
    } else {
        let source_symbol = if zero_len == 0 {
            symbol.to_owned()
        } else {
            format!("{symbol}.source")
        };
        Some(append_point_slice(
            context,
            module,
            &source_symbol,
            source,
            source_offset,
            source_len,
            input,
        )?)
    };

    if zero_len == 0 {
        return source_chunk.ok_or_else(|| {
            schema_error(format!("address chunk {chunk_index} has no source point"))
        });
    }

    let zero = append_point_zero(context, module, &format!("{symbol}.zero_pad"), zero_len)?;
    let inputs = match source_chunk {
        Some(source_chunk) => vec![zero, source_chunk],
        None => vec![zero],
    };
    append_point_concat(
        context,
        module,
        symbol,
        "left_zero_padded_address_chunk",
        chunk_len,
        &inputs,
    )
}

fn append_point_zero<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    symbol: &str,
    arity: usize,
) -> Result<Value<'c, 'a>, MlirError> {
    let op = context.append_typed_op(
        module,
        "poly.point_zero",
        Some(symbol),
        &[("field", "@bn254_fr"), ("arity", &int_attr(arity))],
        &[],
        &["!poly.point"],
    )?;
    first_result(op, "poly.point_zero")
}

fn append_point_concat<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    symbol: &str,
    layout: &str,
    arity: usize,
    inputs: &[Value<'c, 'a>],
) -> Result<Value<'c, 'a>, MlirError> {
    let op = context.append_typed_op(
        module,
        "poly.point_concat",
        Some(symbol),
        &[
            ("layout", &format!("\"{}\"", layout)),
            ("arity", &int_attr(arity)),
        ],
        inputs,
        &["!poly.point"],
    )?;
    first_result(op, "poly.point_concat")
}

fn append_opening_claim<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    point: Value<'c, 'a>,
    eval: Value<'c, 'a>,
    spec: OpeningClaimSpec<'_>,
) -> Result<Value<'c, 'a>, MlirError> {
    let op = context.append_typed_op(
        module,
        "piop.opening_claim",
        Some(spec.symbol),
        &[
            ("oracle", &format!("@{}", spec.oracle)),
            ("domain", &format!("@{}", spec.domain)),
            ("point_arity", &int_attr(spec.point_arity)),
            ("claim_kind", &format!("\"{}\"", spec.claim_kind)),
        ],
        &[point, eval],
        &["!piop.opening_claim_type"],
    )?;
    first_result(op, "piop.opening_claim")
}

fn stage6_max_rounds(params: &JoltProtocolParams) -> usize {
    params.log_k_bytecode + params.log_t
}

fn booleanity_rounds(params: &JoltProtocolParams) -> usize {
    params.log_k_chunk + params.log_t
}

fn n_committed_per_virtual(params: &JoltProtocolParams) -> usize {
    params.lookups_ra_virtual_log_k_chunk / params.log_k_chunk
}

fn total_ra_oracles(params: &JoltProtocolParams) -> usize {
    params.instruction_d + params.bytecode_d + params.ram_d
}

fn stage6_batched_degree(params: &JoltProtocolParams) -> usize {
    [
        params.bytecode_d + 1,
        BOOLEANITY_DEGREE,
        HAMMING_BOOLEANITY_DEGREE,
        params.ram_d + 1,
        n_committed_per_virtual(params) + 1,
        INC_CLAIM_REDUCTION_DEGREE,
    ]
    .into_iter()
    .max()
    .unwrap_or(INC_CLAIM_REDUCTION_DEGREE)
}

fn stage6_output_count(params: &JoltProtocolParams) -> usize {
    params.bytecode_d + total_ra_oracles(params) + 1 + params.ram_d + params.instruction_d + 2
}

fn int_attr(value: usize) -> String {
    format!("{value} : i64")
}

fn symbol_array_attr(values: &[&str]) -> String {
    let values = values
        .iter()
        .map(|value| format!("@{value}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{values}]")
}

fn first_result<'c, 'a>(
    op: OperationRef<'c, 'a>,
    context: &str,
) -> Result<Value<'c, 'a>, MlirError> {
    result(op, 0, context)
}

fn result<'c, 'a>(
    op: OperationRef<'c, 'a>,
    index: usize,
    context: &str,
) -> Result<Value<'c, 'a>, MlirError> {
    op.result(index)
        .map(Into::into)
        .map_err(|_| schema_error(format!("{context} missing result {index}")))
}

fn schema_error(message: impl Into<String>) -> MlirError {
    SchemaError::new(message).into()
}

const STAGE1_OP_FLAGS: [&str; 14] = [
    "OpFlagAddOperands",
    "OpFlagSubtractOperands",
    "OpFlagMultiplyOperands",
    "OpFlagLoad",
    "OpFlagStore",
    "OpFlagJump",
    "OpFlagWriteLookupOutputToRD",
    "OpFlagVirtualInstruction",
    "OpFlagAssert",
    "OpFlagDoNotUpdateUnexpandedPC",
    "OpFlagAdvice",
    "OpFlagIsCompressed",
    "OpFlagIsFirstInSequence",
    "OpFlagIsLastInSequence",
];

struct Stage6BatchedSumcheckInputs<'c, 'a, 'b> {
    state: Value<'c, 'a>,
    stage: Value<'c, 'a>,
    openings: &'b Stage6OpeningInputs<'c, 'a>,
    bc_gamma: Value<'c, 'a>,
    bc_stage1_gamma: Value<'c, 'a>,
    bc_stage2_gamma: Value<'c, 'a>,
    bc_stage3_gamma: Value<'c, 'a>,
    bc_stage4_gamma: Value<'c, 'a>,
    bc_stage5_gamma: Value<'c, 'a>,
    inst_ra_gamma: Value<'c, 'a>,
    inc_gamma: Value<'c, 'a>,
}

struct Stage6OpeningInputs<'c, 'a> {
    bytecode_terms: Vec<Stage6BytecodeTerm<'c, 'a>>,
    hamming_lookup_output: Stage6OpeningInput<'c, 'a>,
    ram_ra_virtual: Stage6OpeningInput<'c, 'a>,
    instruction_ra_virtual: Vec<Stage6OpeningInput<'c, 'a>>,
    ram_inc_stage2: Stage6OpeningInput<'c, 'a>,
    ram_inc_stage4: Stage6OpeningInput<'c, 'a>,
    rd_inc_stage4: Stage6OpeningInput<'c, 'a>,
    rd_inc_stage5: Stage6OpeningInput<'c, 'a>,
}

struct Stage6OpeningInput<'c, 'a> {
    point: Value<'c, 'a>,
    eval: Value<'c, 'a>,
    claim: Value<'c, 'a>,
}

struct Stage6BytecodeTerm<'c, 'a> {
    eval: Value<'c, 'a>,
    claim: Value<'c, 'a>,
    gamma_power: usize,
    stage_gamma: Option<BytecodeStageGamma>,
    stage_gamma_power: usize,
}

struct BytecodeTermSpec<'a> {
    input: StageOpeningInputSpec<'a>,
    gamma_power: usize,
    stage_gamma: Option<BytecodeStageGamma>,
    stage_gamma_power: usize,
}

impl<'a> BytecodeTermSpec<'a> {
    fn trace(
        symbol: &'a str,
        source_stage: &'a str,
        source_claim: &'a str,
        oracle: &'a str,
        gamma_power: usize,
        stage_gamma: Option<BytecodeStageGamma>,
        stage_gamma_power: usize,
    ) -> Self {
        Self {
            input: StageOpeningInputSpec {
                symbol,
                source_stage,
                source_claim,
                oracle,
                domain: "jolt.trace_domain",
                point_arity: 0,
                claim_kind: "virtual",
            },
            gamma_power,
            stage_gamma,
            stage_gamma_power,
        }
    }
}

struct WeightedEvalSpec<'c, 'a> {
    gamma: Value<'c, 'a>,
    gamma_power: usize,
    stage_gamma: Option<Value<'c, 'a>>,
    stage_gamma_power: usize,
}

struct StageOpeningInputSpec<'a> {
    symbol: &'a str,
    source_stage: &'a str,
    source_claim: &'a str,
    oracle: &'a str,
    domain: &'a str,
    point_arity: usize,
    claim_kind: &'a str,
}

struct RelationSpec<'a> {
    symbol: &'a str,
    kind: &'a str,
    domain: &'a str,
    num_rounds: usize,
    degree: usize,
    output_count: usize,
}

struct SumcheckClaimSpec<'a> {
    symbol: &'a str,
    stage: &'a str,
    domain: &'a str,
    num_rounds: usize,
    degree: usize,
    claim: &'a str,
    relation: &'a str,
}

struct SumcheckBatchSpec<'a> {
    symbol: &'a str,
    stage: &'a str,
    proof_slot: &'a str,
    policy: &'a str,
    ordered_claims: &'a [&'a str],
    claim_label: &'a str,
    round_label: &'a str,
    round_schedule: &'a str,
}

struct SumcheckDriverSpec<'a> {
    symbol: &'a str,
    stage: &'a str,
    proof_slot: &'a str,
    relation: &'a str,
    policy: &'a str,
    round_schedule: &'a str,
    claim_label: &'a str,
    round_label: &'a str,
    num_rounds: usize,
    degree: usize,
}

struct SumcheckInstanceResultSpec<'a> {
    symbol: &'a str,
    source: &'a str,
    claim: &'a str,
    relation: &'a str,
    index: usize,
    point_arity: usize,
    num_rounds: usize,
    round_offset: usize,
    point_order: &'a str,
    degree: usize,
}

struct Stage6TraceInstanceSpec<'a> {
    symbol: &'a str,
    claim: &'a str,
    relation: &'a str,
    index: usize,
    degree: usize,
}

struct OpeningClaimSpec<'a> {
    symbol: &'a str,
    oracle: &'a str,
    domain: &'a str,
    point_arity: usize,
    claim_kind: &'a str,
}
