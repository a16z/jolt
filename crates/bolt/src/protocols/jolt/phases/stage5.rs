use melior::ir::operation::OperationRef;
use melior::ir::Value;

use crate::ir::{BoltModule, Compute, Party, Protocol};
use crate::mlir::{verify_module, MeliorContext, MlirError};
use crate::schema::{verify_protocol_schema, SchemaError};

use super::super::oracles;
use super::super::params::JoltProtocolParams;
use super::lowering::{lower_party_to_compute, transcript_squeeze_protocol_result_type};

const RAM_RA_CLAIM_REDUCTION_DEGREE: usize = 2;
const REGISTERS_VAL_EVALUATION_DEGREE: usize = 3;
const FIELD_REGISTERS_VAL_EVALUATION_DEGREE: usize = 3;

pub fn build_stage5_protocol<'c>(
    context: &'c MeliorContext,
    params: &JoltProtocolParams,
) -> Result<BoltModule<'c, Protocol>, MlirError> {
    let module = context.new_module::<Protocol>("jolt.stage5", None);
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
        Some("jolt.stage5"),
        &[("roles", r#"["prover", "verifier"]"#)],
    )?;
    append_stage5_domains(context, &module, params)?;
    append_stage5_oracles(context, &module, params)?;
    append_stage5_relations(context, &module, params)?;
    let inputs = append_stage5_opening_inputs(context, &module, params)?;

    let fs = context.append_typed_op(
        &module,
        "transcript.state",
        Some("fs_after_stage4"),
        &[("scheme", "@blake2b_transcript")],
        &[],
        &["!transcript.state_type"],
    )?;
    let state = first_result(fs, "transcript.state")?;
    let stage = context.append_typed_op(
        &module,
        "piop.stage",
        Some("stage5"),
        &[
            ("name", r#""instruction_ram_and_register_value_reductions""#),
            ("order", "5 : i64"),
            ("roles", r#"["prover", "verifier"]"#),
        ],
        &[],
        &["!piop.stage_type"],
    )?;
    let stage = first_result(stage, "piop.stage")?;

    let (state, instruction_gamma) = append_transcript_squeeze(
        context,
        &module,
        state,
        "stage5.instruction_read_raf.gamma",
        "instruction_read_raf_gamma",
        "challenge_scalar",
        1,
    )?;
    let (state, ram_gamma) = append_transcript_squeeze(
        context,
        &module,
        state,
        "stage5.ram_ra_claim_reduction.gamma",
        "ram_ra_claim_reduction_gamma",
        "challenge_scalar",
        1,
    )?;
    let _state = append_stage5_batched_sumcheck(
        context,
        &module,
        params,
        Stage5BatchedSumcheckInputs {
            state,
            stage,
            openings: &inputs,
            instruction_gamma,
            ram_gamma,
        },
    )?;

    verify_module(&module)?;
    verify_protocol_schema(&module)?;
    Ok(module)
}

pub fn lower_stage5_to_compute<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Party>,
) -> Result<BoltModule<'c, Compute>, MlirError> {
    lower_party_to_compute(context, module, "jolt.stage5", "jolt.stage5", "stage5")
}

fn append_stage5_domains<'c>(
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
        "jolt.stage4_field_registers_rw_domain",
        params.field_register_log_k + params.log_t,
    )?;
    append_domain(
        context,
        module,
        "jolt.stage5_instruction_read_raf_domain",
        params.instruction_log_k + params.log_t,
    )?;
    append_domain(
        context,
        module,
        "jolt.stage5_instruction_ra_chunk_domain",
        params.lookups_ra_virtual_log_k_chunk + params.log_t,
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

fn append_stage5_oracles<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Protocol>,
    params: &JoltProtocolParams,
) -> Result<(), MlirError> {
    append_virtual_oracle(context, module, "LookupOutput", "jolt.trace_domain")?;
    append_virtual_oracle(context, module, "LeftLookupOperand", "jolt.trace_domain")?;
    append_virtual_oracle(context, module, "RightLookupOperand", "jolt.trace_domain")?;
    append_virtual_oracle(context, module, "RamRa", "jolt.stage2_ram_rw_domain")?;
    append_virtual_oracle(
        context,
        module,
        "RegistersVal",
        "jolt.stage4_registers_rw_domain",
    )?;
    append_virtual_oracle(context, module, "RdWa", "jolt.stage4_registers_rw_domain")?;
    append_committed_trace_oracle(context, module, "RdInc")?;
    // BN254 Fr coprocessor oracles for FR ValEvaluation: reduces virtual
    // FieldRegistersVal to the committed FieldRdInc via the same shape as
    // the integer RegistersValEvaluation.
    append_virtual_oracle(
        context,
        module,
        "FieldRegistersVal",
        "jolt.stage4_field_registers_rw_domain",
    )?;
    append_virtual_oracle(
        context,
        module,
        "FieldRdWa",
        "jolt.stage4_field_registers_rw_domain",
    )?;
    append_committed_trace_oracle(context, module, "FieldRdInc")?;
    append_virtual_oracle(context, module, "InstructionRafFlag", "jolt.trace_domain")?;
    for index in 0..params.lookup_table_count {
        append_virtual_oracle(
            context,
            module,
            &format!("LookupTableFlag_{index}"),
            "jolt.trace_domain",
        )?;
    }
    for index in 0..params.instruction_ra_virtual_d {
        append_virtual_oracle(
            context,
            module,
            &format!("InstructionRa_{index}"),
            "jolt.stage5_instruction_ra_chunk_domain",
        )?;
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

fn append_stage5_relations<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Protocol>,
    params: &JoltProtocolParams,
) -> Result<(), MlirError> {
    append_relation(
        context,
        module,
        RelationSpec {
            symbol: "jolt.stage5.instruction_read_raf",
            kind: "sumcheck",
            domain: "jolt.stage5_instruction_read_raf_domain",
            num_rounds: stage5_instruction_rounds(params),
            degree: instruction_read_raf_degree(params),
            output_count: instruction_read_raf_output_count(params),
        },
    )?;
    append_relation(
        context,
        module,
        RelationSpec {
            symbol: "jolt.stage5.ram_ra_claim_reduction",
            kind: "sumcheck",
            domain: "jolt.trace_domain",
            num_rounds: params.log_t,
            degree: RAM_RA_CLAIM_REDUCTION_DEGREE,
            output_count: 1,
        },
    )?;
    append_relation(
        context,
        module,
        RelationSpec {
            symbol: "jolt.stage5.registers_val_evaluation",
            kind: "sumcheck",
            domain: "jolt.trace_domain",
            num_rounds: params.log_t,
            degree: REGISTERS_VAL_EVALUATION_DEGREE,
            output_count: 2,
        },
    )?;
    append_relation(
        context,
        module,
        RelationSpec {
            symbol: "jolt.stage5.field_registers_val_evaluation",
            kind: "sumcheck",
            domain: "jolt.trace_domain",
            num_rounds: params.log_t,
            degree: FIELD_REGISTERS_VAL_EVALUATION_DEGREE,
            output_count: 2,
        },
    )?;
    append_relation(
        context,
        module,
        RelationSpec {
            symbol: "jolt.stage5.batched",
            kind: "batched_sumcheck",
            domain: "jolt.stage5_instruction_read_raf_domain",
            num_rounds: stage5_instruction_rounds(params),
            degree: instruction_read_raf_degree(params),
            output_count: stage5_output_count(params),
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

fn append_stage5_opening_inputs<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    params: &JoltProtocolParams,
) -> Result<Stage5OpeningInputs<'c, 'a>, MlirError> {
    Ok(Stage5OpeningInputs {
        lookup_output_instruction: append_stage_input(
            context,
            module,
            StageOpeningInputSpec {
                symbol: "stage5.input.stage2.instruction.LookupOutput",
                source_stage: "stage2",
                source_claim: "stage2.instruction_lookup.claim_reduction.opening.LookupOutput",
                oracle: "LookupOutput",
                domain: "jolt.trace_domain",
                point_arity: params.log_t,
            },
        )?,
        lookup_output_product: append_stage_input(
            context,
            module,
            StageOpeningInputSpec {
                symbol: "stage5.input.stage2.product_virtual.LookupOutput",
                source_stage: "stage2",
                source_claim: "stage2.product_virtual.remainder.opening.LookupOutput",
                oracle: "LookupOutput",
                domain: "jolt.trace_domain",
                point_arity: params.log_t,
            },
        )?,
        left_lookup_operand: append_stage_input(
            context,
            module,
            StageOpeningInputSpec {
                symbol: "stage5.input.stage2.instruction.LeftLookupOperand",
                source_stage: "stage2",
                source_claim: "stage2.instruction_lookup.claim_reduction.opening.LeftLookupOperand",
                oracle: "LeftLookupOperand",
                domain: "jolt.trace_domain",
                point_arity: params.log_t,
            },
        )?,
        right_lookup_operand: append_stage_input(
            context,
            module,
            StageOpeningInputSpec {
                symbol: "stage5.input.stage2.instruction.RightLookupOperand",
                source_stage: "stage2",
                source_claim:
                    "stage2.instruction_lookup.claim_reduction.opening.RightLookupOperand",
                oracle: "RightLookupOperand",
                domain: "jolt.trace_domain",
                point_arity: params.log_t,
            },
        )?,
        ram_ra_raf: append_stage_input(
            context,
            module,
            StageOpeningInputSpec {
                symbol: "stage5.input.stage2.ram_raf.RamRa",
                source_stage: "stage2",
                source_claim: "stage2.ram_raf.opening.RamRa",
                oracle: "RamRa",
                domain: "jolt.stage2_ram_rw_domain",
                point_arity: params.log_k_ram + params.log_t,
            },
        )?,
        ram_ra_rw: append_stage_input(
            context,
            module,
            StageOpeningInputSpec {
                symbol: "stage5.input.stage2.ram_read_write.RamRa",
                source_stage: "stage2",
                source_claim: "stage2.ram_read_write.opening.RamRa",
                oracle: "RamRa",
                domain: "jolt.stage2_ram_rw_domain",
                point_arity: params.log_k_ram + params.log_t,
            },
        )?,
        ram_ra_val: append_stage_input(
            context,
            module,
            StageOpeningInputSpec {
                symbol: "stage5.input.stage4.ram_val_check.RamRa",
                source_stage: "stage4",
                source_claim: "stage4.ram_val_check.opening.RamRa",
                oracle: "RamRa",
                domain: "jolt.stage2_ram_rw_domain",
                point_arity: params.log_k_ram + params.log_t,
            },
        )?,
        registers_val: append_stage_input(
            context,
            module,
            StageOpeningInputSpec {
                symbol: "stage5.input.stage4.registers.RegistersVal",
                source_stage: "stage4",
                source_claim: "stage4.registers_read_write.opening.RegistersVal",
                oracle: "RegistersVal",
                domain: "jolt.stage4_registers_rw_domain",
                point_arity: params.register_log_k + params.log_t,
            },
        )?,
        field_registers_val: append_stage_input(
            context,
            module,
            StageOpeningInputSpec {
                symbol: "stage5.input.stage4.field_registers.FieldRegistersVal",
                source_stage: "stage4",
                source_claim: "stage4.field_registers_read_write.opening.FieldRegistersVal",
                oracle: "FieldRegistersVal",
                domain: "jolt.stage4_field_registers_rw_domain",
                point_arity: params.field_register_log_k + params.log_t,
            },
        )?,
    })
}

fn append_stage_input<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    spec: StageOpeningInputSpec<'_>,
) -> Result<Stage5OpeningInput<'c, 'a>, MlirError> {
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
            ("claim_kind", r#""virtual""#),
        ],
        &[],
        &["!poly.point", "!field.scalar", "!piop.opening_claim_type"],
    )?;
    Ok(Stage5OpeningInput {
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

fn append_stage5_batched_sumcheck<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    params: &JoltProtocolParams,
    spec: Stage5BatchedSumcheckInputs<'c, 'a, '_>,
) -> Result<Value<'c, 'a>, MlirError> {
    let inputs = spec.openings;
    append_opening_claim_equal(
        context,
        module,
        "stage5.instruction.lookup_output_claim_consistency",
        inputs.lookup_output_instruction.claim,
        inputs.lookup_output_product.claim,
    )?;

    let instruction_gamma2 = append_field_pow(
        context,
        module,
        "stage5.instruction_read_raf.gamma2",
        spec.instruction_gamma,
        2,
    )?;
    let left_term = append_field_mul(
        context,
        module,
        "stage5.instruction_read_raf.term.LeftLookupOperand",
        spec.instruction_gamma,
        inputs.left_lookup_operand.eval,
    )?;
    let right_term = append_field_mul(
        context,
        module,
        "stage5.instruction_read_raf.term.RightLookupOperand",
        instruction_gamma2,
        inputs.right_lookup_operand.eval,
    )?;
    let lookup_left_sum = append_field_add(
        context,
        module,
        "stage5.instruction_read_raf.partial.LookupOutputLeftOperand",
        inputs.lookup_output_instruction.eval,
        left_term,
    )?;
    let instruction_claim = append_field_add(
        context,
        module,
        "stage5.instruction_read_raf.claim_expr",
        lookup_left_sum,
        right_term,
    )?;

    let ram_gamma2 = append_field_pow(
        context,
        module,
        "stage5.ram_ra_claim_reduction.gamma2",
        spec.ram_gamma,
        2,
    )?;
    let ram_rw_term = append_field_mul(
        context,
        module,
        "stage5.ram_ra_claim_reduction.term.RamRaReadWrite",
        spec.ram_gamma,
        inputs.ram_ra_rw.eval,
    )?;
    let ram_val_term = append_field_mul(
        context,
        module,
        "stage5.ram_ra_claim_reduction.term.RamRaValCheck",
        ram_gamma2,
        inputs.ram_ra_val.eval,
    )?;
    let ram_raf_rw_sum = append_field_add(
        context,
        module,
        "stage5.ram_ra_claim_reduction.partial.RafReadWrite",
        inputs.ram_ra_raf.eval,
        ram_rw_term,
    )?;
    let ram_claim = append_field_add(
        context,
        module,
        "stage5.ram_ra_claim_reduction.claim_expr",
        ram_raf_rw_sum,
        ram_val_term,
    )?;

    let claims = [
        append_sumcheck_claim(
            context,
            module,
            SumcheckClaimSpec {
                symbol: "stage5.instruction_read_raf.input",
                stage: "stage5",
                domain: "jolt.stage5_instruction_read_raf_domain",
                num_rounds: stage5_instruction_rounds(params),
                degree: instruction_read_raf_degree(params),
                claim: "stage5.instruction_read_raf.weighted_lookup_values",
                relation: "jolt.stage5.instruction_read_raf",
            },
            instruction_claim,
            &[
                inputs.lookup_output_instruction.claim,
                inputs.left_lookup_operand.claim,
                inputs.right_lookup_operand.claim,
            ],
        )?,
        append_sumcheck_claim(
            context,
            module,
            SumcheckClaimSpec {
                symbol: "stage5.ram_ra_claim_reduction.input",
                stage: "stage5",
                domain: "jolt.trace_domain",
                num_rounds: params.log_t,
                degree: RAM_RA_CLAIM_REDUCTION_DEGREE,
                claim: "stage5.ram_ra_claim_reduction.weighted_ram_ra",
                relation: "jolt.stage5.ram_ra_claim_reduction",
            },
            ram_claim,
            &[
                inputs.ram_ra_raf.claim,
                inputs.ram_ra_rw.claim,
                inputs.ram_ra_val.claim,
            ],
        )?,
        append_sumcheck_claim(
            context,
            module,
            SumcheckClaimSpec {
                symbol: "stage5.registers_val_evaluation.input",
                stage: "stage5",
                domain: "jolt.trace_domain",
                num_rounds: params.log_t,
                degree: REGISTERS_VAL_EVALUATION_DEGREE,
                claim: "stage5.registers_val_evaluation.registers_val",
                relation: "jolt.stage5.registers_val_evaluation",
            },
            inputs.registers_val.eval,
            &[inputs.registers_val.claim],
        )?,
        append_sumcheck_claim(
            context,
            module,
            SumcheckClaimSpec {
                symbol: "stage5.field_registers_val_evaluation.input",
                stage: "stage5",
                domain: "jolt.trace_domain",
                num_rounds: params.log_t,
                degree: FIELD_REGISTERS_VAL_EVALUATION_DEGREE,
                claim: "stage5.field_registers_val_evaluation.field_registers_val",
                relation: "jolt.stage5.field_registers_val_evaluation",
            },
            inputs.field_registers_val.eval,
            &[inputs.field_registers_val.claim],
        )?,
    ];
    let round_schedule = format!("[{}, {}]", params.instruction_log_k, params.log_t);
    let batch = append_sumcheck_batch(
        context,
        module,
        spec.stage,
        &claims,
        SumcheckBatchSpec {
            symbol: "stage5.batch",
            stage: "stage5",
            proof_slot: "stage5.sumcheck",
            policy: "jolt_core_stage5_aligned",
            ordered_claims: &[
                "stage5.instruction_read_raf.input",
                "stage5.ram_ra_claim_reduction.input",
                "stage5.registers_val_evaluation.input",
                "stage5.field_registers_val_evaluation.input",
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
            symbol: "stage5.sumcheck",
            stage: "stage5",
            proof_slot: "stage5.sumcheck",
            relation: "jolt.stage5.batched",
            policy: "jolt_core_stage5_aligned",
            round_schedule: &round_schedule,
            claim_label: "sumcheck_claim",
            round_label: "sumcheck_poly",
            num_rounds: stage5_instruction_rounds(params),
            degree: instruction_read_raf_degree(params),
        },
    )?;
    let instruction = append_sumcheck_instance_result(
        context,
        module,
        SumcheckInstanceResultSpec {
            symbol: "stage5.instruction_read_raf.instance",
            source: "stage5.sumcheck",
            claim: "stage5.instruction_read_raf.input",
            relation: "jolt.stage5.instruction_read_raf",
            index: 0,
            point_arity: stage5_instruction_rounds(params),
            num_rounds: stage5_instruction_rounds(params),
            round_offset: 0,
            point_order: "instruction_read_raf",
            degree: instruction_read_raf_degree(params),
        },
        point,
        result_value,
    )?;
    let ram = append_sumcheck_instance_result(
        context,
        module,
        SumcheckInstanceResultSpec {
            symbol: "stage5.ram_ra_claim_reduction.instance",
            source: "stage5.sumcheck",
            claim: "stage5.ram_ra_claim_reduction.input",
            relation: "jolt.stage5.ram_ra_claim_reduction",
            index: 1,
            point_arity: params.log_t,
            num_rounds: params.log_t,
            round_offset: params.instruction_log_k,
            point_order: "reverse",
            degree: RAM_RA_CLAIM_REDUCTION_DEGREE,
        },
        point,
        result_value,
    )?;
    let registers = append_sumcheck_instance_result(
        context,
        module,
        SumcheckInstanceResultSpec {
            symbol: "stage5.registers_val_evaluation.instance",
            source: "stage5.sumcheck",
            claim: "stage5.registers_val_evaluation.input",
            relation: "jolt.stage5.registers_val_evaluation",
            index: 2,
            point_arity: params.log_t,
            num_rounds: params.log_t,
            round_offset: params.instruction_log_k,
            point_order: "reverse",
            degree: REGISTERS_VAL_EVALUATION_DEGREE,
        },
        point,
        result_value,
    )?;
    let field_registers = append_sumcheck_instance_result(
        context,
        module,
        SumcheckInstanceResultSpec {
            symbol: "stage5.field_registers_val_evaluation.instance",
            source: "stage5.sumcheck",
            claim: "stage5.field_registers_val_evaluation.input",
            relation: "jolt.stage5.field_registers_val_evaluation",
            index: 3,
            point_arity: params.log_t,
            num_rounds: params.log_t,
            round_offset: params.instruction_log_k,
            point_order: "reverse",
            degree: FIELD_REGISTERS_VAL_EVALUATION_DEGREE,
        },
        point,
        result_value,
    )?;
    append_stage5_output_openings(
        context,
        module,
        params,
        inputs,
        instruction,
        ram,
        registers,
        field_registers,
    )?;
    Ok(state)
}

#[expect(clippy::too_many_arguments)]
fn append_stage5_output_openings<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    params: &JoltProtocolParams,
    inputs: &Stage5OpeningInputs<'c, 'a>,
    instruction: (Value<'c, 'a>, Value<'c, 'a>),
    ram: (Value<'c, 'a>, Value<'c, 'a>),
    registers: (Value<'c, 'a>, Value<'c, 'a>),
    field_registers: (Value<'c, 'a>, Value<'c, 'a>),
) -> Result<(), MlirError> {
    let mut claims = Vec::new();
    let mut claim_symbols = Vec::new();

    let instruction_cycle = append_point_slice(
        context,
        module,
        "stage5.instruction_read_raf.point.Cycle",
        "stage5.instruction_read_raf.instance",
        params.instruction_log_k,
        params.log_t,
        instruction.0,
    )?;
    for index in 0..params.lookup_table_count {
        let oracle = format!("LookupTableFlag_{index}");
        let symbol = format!("stage5.instruction_read_raf.opening.{oracle}");
        let eval_symbol = format!("stage5.instruction_read_raf.eval.{oracle}");
        let eval = append_sumcheck_eval(
            context,
            module,
            &eval_symbol,
            "stage5.sumcheck",
            &oracle,
            index,
            instruction.1,
        )?;
        claim_symbols.push(symbol.clone());
        claims.push(append_opening_claim(
            context,
            module,
            instruction_cycle,
            eval,
            OpeningClaimSpec {
                symbol: &symbol,
                oracle: &oracle,
                domain: "jolt.trace_domain",
                point_arity: params.log_t,
                claim_kind: "virtual",
            },
        )?);
    }

    for index in 0..params.instruction_ra_virtual_d {
        let oracle = format!("InstructionRa_{index}");
        let symbol = format!("stage5.instruction_read_raf.opening.{oracle}");
        let address_chunk = append_point_slice(
            context,
            module,
            &format!("stage5.instruction_read_raf.point.{oracle}.address"),
            "stage5.instruction_read_raf.instance",
            index * params.lookups_ra_virtual_log_k_chunk,
            params.lookups_ra_virtual_log_k_chunk,
            instruction.0,
        )?;
        let ra_point = append_point_concat(
            context,
            module,
            &format!("stage5.instruction_read_raf.point.{oracle}"),
            "address_chunk_then_cycle",
            params.lookups_ra_virtual_log_k_chunk + params.log_t,
            &[address_chunk, instruction_cycle],
        )?;
        let eval_symbol = format!("stage5.instruction_read_raf.eval.{oracle}");
        let eval = append_sumcheck_eval(
            context,
            module,
            &eval_symbol,
            "stage5.sumcheck",
            &oracle,
            params.lookup_table_count + index,
            instruction.1,
        )?;
        claim_symbols.push(symbol.clone());
        claims.push(append_opening_claim(
            context,
            module,
            ra_point,
            eval,
            OpeningClaimSpec {
                symbol: &symbol,
                oracle: &oracle,
                domain: "jolt.stage5_instruction_ra_chunk_domain",
                point_arity: params.lookups_ra_virtual_log_k_chunk + params.log_t,
                claim_kind: "virtual",
            },
        )?);
    }

    let raf_flag_eval_index = params.lookup_table_count + params.instruction_ra_virtual_d;
    let raf_flag_eval = append_sumcheck_eval(
        context,
        module,
        "stage5.instruction_read_raf.eval.InstructionRafFlag",
        "stage5.sumcheck",
        "InstructionRafFlag",
        raf_flag_eval_index,
        instruction.1,
    )?;
    claim_symbols.push("stage5.instruction_read_raf.opening.InstructionRafFlag".to_owned());
    claims.push(append_opening_claim(
        context,
        module,
        instruction_cycle,
        raf_flag_eval,
        OpeningClaimSpec {
            symbol: "stage5.instruction_read_raf.opening.InstructionRafFlag",
            oracle: "InstructionRafFlag",
            domain: "jolt.trace_domain",
            point_arity: params.log_t,
            claim_kind: "virtual",
        },
    )?);

    let ram_address = append_point_slice(
        context,
        module,
        "stage5.ram_ra_claim_reduction.point.RamAddress",
        "stage5.input.stage2.ram_raf.RamRa",
        0,
        params.log_k_ram,
        inputs.ram_ra_raf.point,
    )?;
    let ram_ra_point = append_point_concat(
        context,
        module,
        "stage5.ram_ra_claim_reduction.point.RamRa",
        "address_then_cycle",
        params.log_k_ram + params.log_t,
        &[ram_address, ram.0],
    )?;
    let ram_ra_eval = append_sumcheck_eval(
        context,
        module,
        "stage5.ram_ra_claim_reduction.eval.RamRa",
        "stage5.sumcheck",
        "RamRa",
        0,
        ram.1,
    )?;
    claim_symbols.push("stage5.ram_ra_claim_reduction.opening.RamRa".to_owned());
    claims.push(append_opening_claim(
        context,
        module,
        ram_ra_point,
        ram_ra_eval,
        OpeningClaimSpec {
            symbol: "stage5.ram_ra_claim_reduction.opening.RamRa",
            oracle: "RamRa",
            domain: "jolt.stage2_ram_rw_domain",
            point_arity: params.log_k_ram + params.log_t,
            claim_kind: "virtual",
        },
    )?);

    let rd_inc_eval = append_sumcheck_eval(
        context,
        module,
        "stage5.registers_val_evaluation.eval.RdInc",
        "stage5.sumcheck",
        "RdInc",
        0,
        registers.1,
    )?;
    claim_symbols.push("stage5.registers_val_evaluation.opening.RdInc".to_owned());
    claims.push(append_opening_claim(
        context,
        module,
        registers.0,
        rd_inc_eval,
        OpeningClaimSpec {
            symbol: "stage5.registers_val_evaluation.opening.RdInc",
            oracle: "RdInc",
            domain: "jolt.trace_domain",
            point_arity: params.log_t,
            claim_kind: "committed",
        },
    )?);

    let register_address = append_point_slice(
        context,
        module,
        "stage5.registers_val_evaluation.point.RegisterAddress",
        "stage5.input.stage4.registers.RegistersVal",
        0,
        params.register_log_k,
        inputs.registers_val.point,
    )?;
    let rd_wa_point = append_point_concat(
        context,
        module,
        "stage5.registers_val_evaluation.point.RdWa",
        "register_address_then_cycle",
        params.register_log_k + params.log_t,
        &[register_address, registers.0],
    )?;
    let rd_wa_eval = append_sumcheck_eval(
        context,
        module,
        "stage5.registers_val_evaluation.eval.RdWa",
        "stage5.sumcheck",
        "RdWa",
        1,
        registers.1,
    )?;
    claim_symbols.push("stage5.registers_val_evaluation.opening.RdWa".to_owned());
    claims.push(append_opening_claim(
        context,
        module,
        rd_wa_point,
        rd_wa_eval,
        OpeningClaimSpec {
            symbol: "stage5.registers_val_evaluation.opening.RdWa",
            oracle: "RdWa",
            domain: "jolt.stage4_registers_rw_domain",
            point_arity: params.register_log_k + params.log_t,
            claim_kind: "virtual",
        },
    )?);

    // BN254 Fr coprocessor ValEvaluation outputs: FieldRdInc (committed,
    // trace_domain) + FieldRdWa (virtual, FR rw domain).
    let field_rd_inc_eval = append_sumcheck_eval(
        context,
        module,
        "stage5.field_registers_val_evaluation.eval.FieldRdInc",
        "stage5.sumcheck",
        "FieldRdInc",
        0,
        field_registers.1,
    )?;
    claim_symbols.push("stage5.field_registers_val_evaluation.opening.FieldRdInc".to_owned());
    claims.push(append_opening_claim(
        context,
        module,
        field_registers.0,
        field_rd_inc_eval,
        OpeningClaimSpec {
            symbol: "stage5.field_registers_val_evaluation.opening.FieldRdInc",
            oracle: "FieldRdInc",
            domain: "jolt.trace_domain",
            point_arity: params.log_t,
            claim_kind: "committed",
        },
    )?);

    let field_register_address = append_point_slice(
        context,
        module,
        "stage5.field_registers_val_evaluation.point.FieldRegisterAddress",
        "stage5.input.stage4.field_registers.FieldRegistersVal",
        0,
        params.field_register_log_k,
        inputs.field_registers_val.point,
    )?;
    let field_rd_wa_point = append_point_concat(
        context,
        module,
        "stage5.field_registers_val_evaluation.point.FieldRdWa",
        "register_address_then_cycle",
        params.field_register_log_k + params.log_t,
        &[field_register_address, field_registers.0],
    )?;
    let field_rd_wa_eval = append_sumcheck_eval(
        context,
        module,
        "stage5.field_registers_val_evaluation.eval.FieldRdWa",
        "stage5.sumcheck",
        "FieldRdWa",
        1,
        field_registers.1,
    )?;
    claim_symbols.push("stage5.field_registers_val_evaluation.opening.FieldRdWa".to_owned());
    claims.push(append_opening_claim(
        context,
        module,
        field_rd_wa_point,
        field_rd_wa_eval,
        OpeningClaimSpec {
            symbol: "stage5.field_registers_val_evaluation.opening.FieldRdWa",
            oracle: "FieldRdWa",
            domain: "jolt.stage4_field_registers_rw_domain",
            point_arity: params.field_register_log_k + params.log_t,
            claim_kind: "virtual",
        },
    )?);

    let claim_names = claim_symbols.iter().map(String::as_str).collect::<Vec<_>>();
    let _batch = context.append_typed_op(
        module,
        "piop.opening_batch",
        Some("stage5.openings"),
        &[
            ("stage", "@stage5"),
            ("proof_slot", "@stage5.openings"),
            ("policy", r#""jolt_stage5_output_order""#),
            ("count", &int_attr(claims.len())),
            ("ordered_claims", &symbol_array_attr(&claim_names)),
        ],
        &claims,
        &["!piop.opening_batch_type"],
    )?;
    Ok(())
}

fn append_opening_claim_equal<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Protocol>,
    symbol: &str,
    left: Value<'c, '_>,
    right: Value<'c, '_>,
) -> Result<(), MlirError> {
    let _operation = context.append_typed_op(
        module,
        "piop.opening_claim_equal",
        Some(symbol),
        &[("mode", r#""point_and_eval""#)],
        &[left, right],
        &[],
    )?;
    Ok(())
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

fn stage5_instruction_rounds(params: &JoltProtocolParams) -> usize {
    params.instruction_log_k + params.log_t
}

fn instruction_read_raf_degree(params: &JoltProtocolParams) -> usize {
    params.instruction_ra_virtual_d + 2
}

fn instruction_read_raf_output_count(params: &JoltProtocolParams) -> usize {
    params.lookup_table_count + params.instruction_ra_virtual_d + 1
}

fn stage5_output_count(params: &JoltProtocolParams) -> usize {
    // instruction_read_raf outputs + ram_ra_claim_reduction (1) +
    // registers_val_evaluation (2: RdInc, RdWa) + field_registers_val_evaluation
    // (2: FieldRdInc, FieldRdWa).
    instruction_read_raf_output_count(params) + 1 + 2 + 2
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
        .map_err(|_| schema_error(format!("{context} expected result {index}")))
}

fn schema_error(message: impl Into<String>) -> MlirError {
    SchemaError::new(message).into()
}

struct Stage5OpeningInput<'c, 'a> {
    point: Value<'c, 'a>,
    eval: Value<'c, 'a>,
    claim: Value<'c, 'a>,
}

struct Stage5OpeningInputs<'c, 'a> {
    lookup_output_instruction: Stage5OpeningInput<'c, 'a>,
    lookup_output_product: Stage5OpeningInput<'c, 'a>,
    left_lookup_operand: Stage5OpeningInput<'c, 'a>,
    right_lookup_operand: Stage5OpeningInput<'c, 'a>,
    ram_ra_raf: Stage5OpeningInput<'c, 'a>,
    ram_ra_rw: Stage5OpeningInput<'c, 'a>,
    ram_ra_val: Stage5OpeningInput<'c, 'a>,
    registers_val: Stage5OpeningInput<'c, 'a>,
    field_registers_val: Stage5OpeningInput<'c, 'a>,
}

struct StageOpeningInputSpec<'a> {
    symbol: &'a str,
    source_stage: &'a str,
    source_claim: &'a str,
    oracle: &'a str,
    domain: &'a str,
    point_arity: usize,
}

struct Stage5BatchedSumcheckInputs<'c, 'a, 'b> {
    state: Value<'c, 'a>,
    stage: Value<'c, 'a>,
    openings: &'b Stage5OpeningInputs<'c, 'a>,
    instruction_gamma: Value<'c, 'a>,
    ram_gamma: Value<'c, 'a>,
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

struct OpeningClaimSpec<'a> {
    symbol: &'a str,
    oracle: &'a str,
    domain: &'a str,
    point_arity: usize,
    claim_kind: &'a str,
}
