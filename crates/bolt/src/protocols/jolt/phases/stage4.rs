use melior::ir::operation::OperationRef;
use melior::ir::Value;

use crate::ir::{BoltModule, Compute, Party, Protocol};
use crate::mlir::{verify_module, MeliorContext, MlirError};
use crate::schema::{verify_protocol_schema, SchemaError};

use super::super::oracles;
use super::super::params::JoltProtocolParams;
use super::field_formula::{FieldFormulaBuilder, FieldFormulaStep};
use super::lowering::{lower_party_to_compute, transcript_squeeze_protocol_result_type};
use super::sumcheck_output::{
    append_structured_polynomial_eval, append_sumcheck_output_claim, OutputClaimSpec,
    StructuredPolynomialPointSpec, StructuredPolynomialSpec,
};

const REGISTERS_RW_DEGREE: usize = 3;
const RAM_VAL_CHECK_DEGREE: usize = 3;
const STAGE4_BATCHED_DEGREE: usize = 3;

const STAGE4_REGISTER_INPUTS: [&str; 3] = ["RdWriteValue", "Rs1Value", "Rs2Value"];
const STAGE4_REGISTER_OUTPUTS: [&str; 5] = ["RegistersVal", "Rs1Ra", "Rs2Ra", "RdWa", "RdInc"];
const STAGE4_RAM_VAL_OUTPUTS: [&str; 2] = ["RamRa", "RamInc"];

const STAGE4_REGISTERS_OUTPUT_FORMULAS: [FieldFormulaStep; 9] = [
    FieldFormulaStep::add(
        "stage4.registers_read_write.output.sum.RegistersValRdInc",
        "stage4.registers_read_write.eval.RegistersVal",
        "stage4.registers_read_write.eval.RdInc",
    ),
    FieldFormulaStep::mul(
        "stage4.registers_read_write.output.term.RdWa",
        "stage4.registers_read_write.eval.RdWa",
        "stage4.registers_read_write.output.sum.RegistersValRdInc",
    ),
    FieldFormulaStep::mul(
        "stage4.registers_read_write.output.term.Rs1Ra",
        "stage4.registers_read_write.eval.Rs1Ra",
        "stage4.registers_read_write.eval.RegistersVal",
    ),
    FieldFormulaStep::mul(
        "stage4.registers_read_write.output.term.Rs2Ra",
        "stage4.registers_read_write.eval.Rs2Ra",
        "stage4.registers_read_write.eval.RegistersVal",
    ),
    FieldFormulaStep::mul(
        "stage4.registers_read_write.output.weighted.Rs2Ra",
        "stage4.registers_read_write.gamma",
        "stage4.registers_read_write.output.term.Rs2Ra",
    ),
    FieldFormulaStep::add(
        "stage4.registers_read_write.output.read_terms",
        "stage4.registers_read_write.output.term.Rs1Ra",
        "stage4.registers_read_write.output.weighted.Rs2Ra",
    ),
    FieldFormulaStep::mul(
        "stage4.registers_read_write.output.weighted.read_terms",
        "stage4.registers_read_write.gamma",
        "stage4.registers_read_write.output.read_terms",
    ),
    FieldFormulaStep::add(
        "stage4.registers_read_write.output.weighted_values",
        "stage4.registers_read_write.output.term.RdWa",
        "stage4.registers_read_write.output.weighted.read_terms",
    ),
    FieldFormulaStep::mul(
        "stage4.registers_read_write.output.claim_expr",
        "stage4.registers_read_write.output.eq.RdWriteValue",
        "stage4.registers_read_write.output.weighted_values",
    ),
];

const STAGE4_RAM_VAL_OUTPUT_FORMULAS: [FieldFormulaStep; 3] = [
    FieldFormulaStep::add(
        "stage4.ram_val_check.output.lt_plus_gamma",
        "stage4.ram_val_check.output.lt.RamValCycle",
        "stage4.ram_val_check.gamma",
    ),
    FieldFormulaStep::mul(
        "stage4.ram_val_check.output.term.RamIncRamRa",
        "stage4.ram_val_check.eval.RamInc",
        "stage4.ram_val_check.eval.RamRa",
    ),
    FieldFormulaStep::mul(
        "stage4.ram_val_check.output.claim_expr",
        "stage4.ram_val_check.output.term.RamIncRamRa",
        "stage4.ram_val_check.output.lt_plus_gamma",
    ),
];

pub fn build_stage4_protocol<'c>(
    context: &'c MeliorContext,
    params: &JoltProtocolParams,
) -> Result<BoltModule<'c, Protocol>, MlirError> {
    let module = context.new_module::<Protocol>("jolt.stage4", None);
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
        Some("jolt.stage4"),
        &[("roles", r#"["prover", "verifier"]"#)],
    )?;
    append_stage4_domains(context, &module, params)?;
    append_stage4_oracles(context, &module)?;
    append_stage4_relations(context, &module, params)?;
    let inputs = append_stage4_opening_inputs(context, &module, params)?;

    let fs = context.append_typed_op(
        &module,
        "transcript.state",
        Some("fs_after_stage3"),
        &[("scheme", "@blake2b_transcript")],
        &[],
        &["!transcript.state_type"],
    )?;
    let state = first_result(fs, "transcript.state")?;
    let stage = context.append_typed_op(
        &module,
        "piop.stage",
        Some("stage4"),
        &[
            ("name", r#""registers_rw_and_ram_val_check""#),
            ("order", "4 : i64"),
            ("roles", r#"["prover", "verifier"]"#),
        ],
        &[],
        &["!piop.stage_type"],
    )?;
    let stage = first_result(stage, "piop.stage")?;

    let (state, registers_gamma) = append_transcript_squeeze(
        context,
        &module,
        state,
        "stage4.registers_read_write.gamma",
        "registers_read_write_gamma",
        "challenge_scalar",
        1,
    )?;
    let state = append_transcript_absorb_bytes(
        context,
        &module,
        state,
        "stage4.ram_val_check.domain_separator",
        "ram_val_check_gamma",
        "",
    )?;
    let (state, ram_val_check_gamma) = append_transcript_squeeze(
        context,
        &module,
        state,
        "stage4.ram_val_check.gamma",
        "ram_val_check_gamma",
        "challenge_scalar",
        1,
    )?;
    let _state = append_stage4_batched_sumcheck(
        context,
        &module,
        params,
        Stage4BatchedSumcheckInputs {
            state,
            stage,
            openings: &inputs,
            registers_gamma,
            ram_val_check_gamma,
        },
    )?;

    verify_module(&module)?;
    verify_protocol_schema(&module)?;
    Ok(module)
}

pub fn lower_stage4_to_compute<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Party>,
) -> Result<BoltModule<'c, Compute>, MlirError> {
    lower_party_to_compute(context, module, "jolt.stage4", "jolt.stage4", "stage4")
}

fn append_stage4_domains<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Protocol>,
    params: &JoltProtocolParams,
) -> Result<(), MlirError> {
    context.append_op(
        module,
        "poly.domain",
        Some("jolt.stage4_registers_rw_domain"),
        &[
            ("field", "@bn254_fr"),
            ("log_size", &int_attr(stage4_registers_rw_rounds(params))),
        ],
    )?;
    context.append_op(
        module,
        "poly.domain",
        Some("jolt.stage2_ram_rw_domain"),
        &[
            ("field", "@bn254_fr"),
            ("log_size", &int_attr(params.log_k_ram + params.log_t)),
        ],
    )?;
    context.append_op(
        module,
        "poly.domain",
        Some("jolt.ram_address_domain"),
        &[
            ("field", "@bn254_fr"),
            ("log_size", &int_attr(params.log_k_ram)),
        ],
    )
}

fn append_stage4_oracles<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Protocol>,
) -> Result<(), MlirError> {
    for oracle in STAGE4_REGISTER_INPUTS {
        append_virtual_oracle(context, module, oracle, "jolt.trace_domain")?;
    }
    append_virtual_oracle(
        context,
        module,
        "RegistersVal",
        "jolt.stage4_registers_rw_domain",
    )?;
    append_virtual_oracle(context, module, "Rs1Ra", "jolt.stage4_registers_rw_domain")?;
    append_virtual_oracle(context, module, "Rs2Ra", "jolt.stage4_registers_rw_domain")?;
    append_virtual_oracle(context, module, "RdWa", "jolt.stage4_registers_rw_domain")?;
    append_virtual_oracle(context, module, "RamVal", "jolt.stage2_ram_rw_domain")?;
    append_virtual_oracle(context, module, "RamRa", "jolt.stage2_ram_rw_domain")?;
    append_virtual_oracle(context, module, "RamValFinal", "jolt.ram_address_domain")?;
    append_virtual_oracle(context, module, "RamValInit", "jolt.ram_address_domain")?;
    append_committed_trace_oracle(context, module, "RdInc")?;
    append_committed_trace_oracle(context, module, "RamInc")
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

fn append_stage4_relations<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Protocol>,
    params: &JoltProtocolParams,
) -> Result<(), MlirError> {
    append_relation(
        context,
        module,
        RelationSpec {
            symbol: "jolt.stage4.registers_read_write",
            kind: "sumcheck",
            domain: "jolt.stage4_registers_rw_domain",
            num_rounds: stage4_registers_rw_rounds(params),
            degree: REGISTERS_RW_DEGREE,
            output_count: STAGE4_REGISTER_OUTPUTS.len(),
        },
    )?;
    append_relation(
        context,
        module,
        RelationSpec {
            symbol: "jolt.stage4.ram_val_check",
            kind: "sumcheck",
            domain: "jolt.trace_domain",
            num_rounds: params.log_t,
            degree: RAM_VAL_CHECK_DEGREE,
            output_count: STAGE4_RAM_VAL_OUTPUTS.len(),
        },
    )?;
    append_relation(
        context,
        module,
        RelationSpec {
            symbol: "jolt.stage4.batched",
            kind: "batched_sumcheck",
            domain: "jolt.stage4_registers_rw_domain",
            num_rounds: stage4_registers_rw_rounds(params),
            degree: STAGE4_BATCHED_DEGREE,
            output_count: stage4_output_count(),
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

fn append_stage4_opening_inputs<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    params: &JoltProtocolParams,
) -> Result<Stage4OpeningInputs<'c, 'a>, MlirError> {
    Ok(Stage4OpeningInputs {
        rd_write_value: append_stage_input(
            context,
            module,
            StageOpeningInputSpec {
                symbol: "stage4.input.stage3.registers.RdWriteValue",
                source_stage: "stage3",
                source_claim: "stage3.registers_claim_reduction.opening.RdWriteValue",
                oracle: "RdWriteValue",
                domain: "jolt.trace_domain",
                point_arity: params.log_t,
            },
        )?,
        rs1_registers: append_stage_input(
            context,
            module,
            StageOpeningInputSpec {
                symbol: "stage4.input.stage3.registers.Rs1Value",
                source_stage: "stage3",
                source_claim: "stage3.registers_claim_reduction.opening.Rs1Value",
                oracle: "Rs1Value",
                domain: "jolt.trace_domain",
                point_arity: params.log_t,
            },
        )?,
        rs2_registers: append_stage_input(
            context,
            module,
            StageOpeningInputSpec {
                symbol: "stage4.input.stage3.registers.Rs2Value",
                source_stage: "stage3",
                source_claim: "stage3.registers_claim_reduction.opening.Rs2Value",
                oracle: "Rs2Value",
                domain: "jolt.trace_domain",
                point_arity: params.log_t,
            },
        )?,
        rs1_instruction: append_stage_input(
            context,
            module,
            StageOpeningInputSpec {
                symbol: "stage4.input.stage3.instruction.Rs1Value",
                source_stage: "stage3",
                source_claim: "stage3.instruction_input.opening.Rs1Value",
                oracle: "Rs1Value",
                domain: "jolt.trace_domain",
                point_arity: params.log_t,
            },
        )?,
        rs2_instruction: append_stage_input(
            context,
            module,
            StageOpeningInputSpec {
                symbol: "stage4.input.stage3.instruction.Rs2Value",
                source_stage: "stage3",
                source_claim: "stage3.instruction_input.opening.Rs2Value",
                oracle: "Rs2Value",
                domain: "jolt.trace_domain",
                point_arity: params.log_t,
            },
        )?,
        ram_val: append_stage_input(
            context,
            module,
            StageOpeningInputSpec {
                symbol: "stage4.input.stage2.RamVal",
                source_stage: "stage2",
                source_claim: "stage2.ram_read_write.opening.RamVal",
                oracle: "RamVal",
                domain: "jolt.stage2_ram_rw_domain",
                point_arity: params.log_k_ram + params.log_t,
            },
        )?,
        ram_val_final: append_stage_input(
            context,
            module,
            StageOpeningInputSpec {
                symbol: "stage4.input.stage2.RamValFinal",
                source_stage: "stage2",
                source_claim: "stage2.ram_output.opening.RamValFinal",
                oracle: "RamValFinal",
                domain: "jolt.ram_address_domain",
                point_arity: params.log_k_ram,
            },
        )?,
        ram_val_init: append_stage_input(
            context,
            module,
            StageOpeningInputSpec {
                symbol: "stage4.input.initial_ram.RamValInit",
                source_stage: "stage4_precomputed",
                source_claim: "stage4.ram_val_check.initial_ram_eval",
                oracle: "RamValInit",
                domain: "jolt.ram_address_domain",
                point_arity: params.log_k_ram,
            },
        )?,
    })
}

fn append_stage_input<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    spec: StageOpeningInputSpec<'_>,
) -> Result<Stage4OpeningInput<'c, 'a>, MlirError> {
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
    Ok(Stage4OpeningInput {
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

fn append_transcript_absorb_bytes<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    state: Value<'c, 'a>,
    symbol: &str,
    label: &str,
    payload: &str,
) -> Result<Value<'c, 'a>, MlirError> {
    let op = context.append_typed_op(
        module,
        "transcript.absorb_bytes",
        Some(symbol),
        &[
            ("label", &format!("\"{label}\"")),
            ("payload", &format!("\"{payload}\"")),
        ],
        &[state],
        &["!transcript.state_type"],
    )?;
    first_result(op, "transcript.absorb_bytes")
}

fn append_stage4_batched_sumcheck<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    params: &JoltProtocolParams,
    spec: Stage4BatchedSumcheckInputs<'c, 'a, '_>,
) -> Result<Value<'c, 'a>, MlirError> {
    let inputs = spec.openings;
    append_opening_claim_equal(
        context,
        module,
        "stage4.registers.rs1_claim_consistency",
        inputs.rs1_registers.claim,
        inputs.rs1_instruction.claim,
    )?;
    append_opening_claim_equal(
        context,
        module,
        "stage4.registers.rs2_claim_consistency",
        inputs.rs2_registers.claim,
        inputs.rs2_instruction.claim,
    )?;
    let registers_gamma2 = append_field_pow(
        context,
        module,
        "stage4.registers_read_write.gamma2",
        spec.registers_gamma,
        2,
    )?;
    let rs1_term = append_field_mul(
        context,
        module,
        "stage4.registers_read_write.term.Rs1Value",
        spec.registers_gamma,
        inputs.rs1_registers.eval,
    )?;
    let rs2_term = append_field_mul(
        context,
        module,
        "stage4.registers_read_write.term.Rs2Value",
        registers_gamma2,
        inputs.rs2_registers.eval,
    )?;
    let registers_sum = append_field_add(
        context,
        module,
        "stage4.registers_read_write.partial.RdWriteValueRs1Value",
        inputs.rd_write_value.eval,
        rs1_term,
    )?;
    let registers_claim = append_field_add(
        context,
        module,
        "stage4.registers_read_write.claim_expr",
        registers_sum,
        rs2_term,
    )?;

    let ram_val_delta = append_field_sub(
        context,
        module,
        "stage4.ram_val_check.delta.RamVal",
        inputs.ram_val.eval,
        inputs.ram_val_init.eval,
    )?;
    let ram_final_delta = append_field_sub(
        context,
        module,
        "stage4.ram_val_check.delta.RamValFinal",
        inputs.ram_val_final.eval,
        inputs.ram_val_init.eval,
    )?;
    let ram_final_term = append_field_mul(
        context,
        module,
        "stage4.ram_val_check.term.RamValFinal",
        spec.ram_val_check_gamma,
        ram_final_delta,
    )?;
    let ram_val_claim = append_field_add(
        context,
        module,
        "stage4.ram_val_check.claim_expr",
        ram_val_delta,
        ram_final_term,
    )?;

    let claims = [
        append_sumcheck_claim(
            context,
            module,
            SumcheckClaimSpec {
                symbol: "stage4.registers_read_write.input",
                stage: "stage4",
                domain: "jolt.stage4_registers_rw_domain",
                num_rounds: stage4_registers_rw_rounds(params),
                degree: REGISTERS_RW_DEGREE,
                claim: "stage4.registers_read_write.weighted_values",
                relation: "jolt.stage4.registers_read_write",
            },
            registers_claim,
            &[
                inputs.rd_write_value.claim,
                inputs.rs1_registers.claim,
                inputs.rs2_registers.claim,
            ],
        )?,
        append_sumcheck_claim(
            context,
            module,
            SumcheckClaimSpec {
                symbol: "stage4.ram_val_check.input",
                stage: "stage4",
                domain: "jolt.trace_domain",
                num_rounds: params.log_t,
                degree: RAM_VAL_CHECK_DEGREE,
                claim: "stage4.ram_val_check.weighted_values",
                relation: "jolt.stage4.ram_val_check",
            },
            ram_val_claim,
            &[
                inputs.ram_val.claim,
                inputs.ram_val_final.claim,
                inputs.ram_val_init.claim,
            ],
        )?,
    ];
    let batch = append_sumcheck_batch(
        context,
        module,
        spec.stage,
        &claims,
        SumcheckBatchSpec {
            symbol: "stage4.batch",
            stage: "stage4",
            proof_slot: "stage4.sumcheck",
            policy: "jolt_core_stage4_aligned",
            ordered_claims: &[
                "stage4.registers_read_write.input",
                "stage4.ram_val_check.input",
            ],
            claim_label: "sumcheck_claim",
            round_label: "sumcheck_poly",
            round_schedule: format!("[{}, {}]", params.log_t, params.register_log_k),
        },
    )?;
    let (state, point, result_value) = append_sumcheck(
        context,
        module,
        spec.state,
        batch,
        SumcheckDriverSpec {
            symbol: "stage4.sumcheck",
            stage: "stage4",
            proof_slot: "stage4.sumcheck",
            relation: "jolt.stage4.batched",
            policy: "jolt_core_stage4_aligned",
            round_schedule: format!("[{}, {}]", params.log_t, params.register_log_k),
            claim_label: "sumcheck_claim",
            round_label: "sumcheck_poly",
            num_rounds: stage4_registers_rw_rounds(params),
            degree: STAGE4_BATCHED_DEGREE,
        },
    )?;
    let registers = append_sumcheck_instance_result(
        context,
        module,
        SumcheckInstanceResultSpec {
            symbol: "stage4.registers_read_write.instance",
            source: "stage4.sumcheck",
            claim: "stage4.registers_read_write.input",
            relation: "jolt.stage4.registers_read_write",
            index: 0,
            point_arity: stage4_registers_rw_rounds(params),
            num_rounds: stage4_registers_rw_rounds(params),
            round_offset: 0,
            point_order: "stage4_registers_rw",
            degree: REGISTERS_RW_DEGREE,
        },
        point,
        result_value,
    )?;
    let ram_val_check = append_sumcheck_instance_result(
        context,
        module,
        SumcheckInstanceResultSpec {
            symbol: "stage4.ram_val_check.instance",
            source: "stage4.sumcheck",
            claim: "stage4.ram_val_check.input",
            relation: "jolt.stage4.ram_val_check",
            index: 1,
            point_arity: params.log_t,
            num_rounds: params.log_t,
            round_offset: params.register_log_k,
            point_order: "reverse",
            degree: RAM_VAL_CHECK_DEGREE,
        },
        point,
        result_value,
    )?;
    append_stage4_output_openings(
        context,
        module,
        params,
        Stage4OutputOpeningsInputs {
            openings: inputs,
            registers,
            ram_val_check,
            registers_gamma: spec.registers_gamma,
            ram_val_check_gamma: spec.ram_val_check_gamma,
        },
    )?;
    Ok(state)
}

fn append_stage4_output_openings<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    params: &JoltProtocolParams,
    spec: Stage4OutputOpeningsInputs<'c, 'a, '_>,
) -> Result<(), MlirError> {
    let inputs = spec.openings;
    let registers = spec.registers;
    let ram_val_check = spec.ram_val_check;
    let mut claims = Vec::new();
    let mut claim_symbols = Vec::new();
    let mut register_evals = Vec::new();

    for (index, &oracle) in ["RegistersVal", "Rs1Ra", "Rs2Ra", "RdWa"]
        .iter()
        .enumerate()
    {
        let symbol = format!("stage4.registers_read_write.opening.{oracle}");
        let eval = append_sumcheck_eval(
            context,
            module,
            &format!("stage4.registers_read_write.eval.{oracle}"),
            "stage4.sumcheck",
            oracle,
            index,
            registers.1,
        )?;
        register_evals.push(eval);
        claim_symbols.push(symbol.clone());
        claims.push(append_opening_claim(
            context,
            module,
            registers.0,
            eval,
            OpeningClaimSpec {
                symbol: &symbol,
                oracle,
                domain: "jolt.stage4_registers_rw_domain",
                point_arity: stage4_registers_rw_rounds(params),
                claim_kind: "virtual",
            },
        )?);
    }

    let rd_inc_point = append_point_slice(
        context,
        module,
        "stage4.registers_read_write.point.RdInc",
        "stage4.registers_read_write.instance",
        params.register_log_k,
        params.log_t,
        registers.0,
    )?;
    let rd_inc_eval = append_sumcheck_eval(
        context,
        module,
        "stage4.registers_read_write.eval.RdInc",
        "stage4.sumcheck",
        "RdInc",
        4,
        registers.1,
    )?;
    claim_symbols.push("stage4.registers_read_write.opening.RdInc".to_owned());
    claims.push(append_opening_claim(
        context,
        module,
        rd_inc_point,
        rd_inc_eval,
        OpeningClaimSpec {
            symbol: "stage4.registers_read_write.opening.RdInc",
            oracle: "RdInc",
            domain: "jolt.trace_domain",
            point_arity: params.log_t,
            claim_kind: "committed",
        },
    )?);

    let ram_address_point = append_point_slice(
        context,
        module,
        "stage4.ram_val_check.point.RamAddress",
        "stage4.input.stage2.RamVal",
        0,
        params.log_k_ram,
        inputs.ram_val.point,
    )?;
    let ram_ra_point = append_point_concat(
        context,
        module,
        "stage4.ram_val_check.point.RamRa",
        "address_then_cycle",
        params.log_k_ram + params.log_t,
        &[ram_address_point, ram_val_check.0],
    )?;
    let ram_ra_eval = append_sumcheck_eval(
        context,
        module,
        "stage4.ram_val_check.eval.RamRa",
        "stage4.sumcheck",
        "RamRa",
        0,
        ram_val_check.1,
    )?;
    claim_symbols.push("stage4.ram_val_check.opening.RamRa".to_owned());
    claims.push(append_opening_claim(
        context,
        module,
        ram_ra_point,
        ram_ra_eval,
        OpeningClaimSpec {
            symbol: "stage4.ram_val_check.opening.RamRa",
            oracle: "RamRa",
            domain: "jolt.stage2_ram_rw_domain",
            point_arity: params.log_k_ram + params.log_t,
            claim_kind: "virtual",
        },
    )?);

    let ram_inc_eval = append_sumcheck_eval(
        context,
        module,
        "stage4.ram_val_check.eval.RamInc",
        "stage4.sumcheck",
        "RamInc",
        1,
        ram_val_check.1,
    )?;
    claim_symbols.push("stage4.ram_val_check.opening.RamInc".to_owned());
    claims.push(append_opening_claim(
        context,
        module,
        ram_val_check.0,
        ram_inc_eval,
        OpeningClaimSpec {
            symbol: "stage4.ram_val_check.opening.RamInc",
            oracle: "RamInc",
            domain: "jolt.trace_domain",
            point_arity: params.log_t,
            claim_kind: "committed",
        },
    )?);

    let [registers_val_eval, rs1_ra_eval, rs2_ra_eval, rd_wa_eval] = register_evals.as_slice()
    else {
        return Err(schema_error(format!(
            "stage4 registers output eval count mismatch: expected 4, got {}",
            register_evals.len()
        )));
    };
    append_stage4_output_claims(
        context,
        module,
        Stage4OutputClaimInputs {
            openings: inputs,
            registers,
            ram_val_check,
            registers_gamma: spec.registers_gamma,
            ram_val_check_gamma: spec.ram_val_check_gamma,
            registers_val_eval: *registers_val_eval,
            rs1_ra_eval: *rs1_ra_eval,
            rs2_ra_eval: *rs2_ra_eval,
            rd_wa_eval: *rd_wa_eval,
            rd_inc_eval,
            ram_ra_eval,
            ram_inc_eval,
        },
    )?;

    let claim_names = claim_symbols.iter().map(String::as_str).collect::<Vec<_>>();
    let _batch = context.append_typed_op(
        module,
        "piop.opening_batch",
        Some("stage4.openings"),
        &[
            ("stage", "@stage4"),
            ("proof_slot", "@stage4.openings"),
            ("policy", r#""jolt_stage4_output_order""#),
            ("count", &int_attr(claims.len())),
            ("ordered_claims", &symbol_array_attr(&claim_names)),
        ],
        &claims,
        &["!piop.opening_batch_type"],
    )?;
    Ok(())
}

fn append_stage4_output_claims<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    spec: Stage4OutputClaimInputs<'c, 'a, '_>,
) -> Result<(), MlirError> {
    let registers_eq_trace = append_structured_polynomial_eval(
        context,
        module,
        StructuredPolynomialSpec {
            symbol: "stage4.registers_read_write.output.eq.RdWriteValue",
            polynomial: "eq",
            x_point: StructuredPolynomialPointSpec::prefix("y_point", "reverse"),
            y_point: StructuredPolynomialPointSpec::full("as_is"),
        },
        spec.registers.0,
        spec.openings.rd_write_value.point,
    )?;
    let mut formula = FieldFormulaBuilder::new(context, module);
    formula.bind_all(&[
        ("stage4.registers_read_write.gamma", spec.registers_gamma),
        (
            "stage4.registers_read_write.eval.RegistersVal",
            spec.registers_val_eval,
        ),
        ("stage4.registers_read_write.eval.Rs1Ra", spec.rs1_ra_eval),
        ("stage4.registers_read_write.eval.Rs2Ra", spec.rs2_ra_eval),
        ("stage4.registers_read_write.eval.RdWa", spec.rd_wa_eval),
        ("stage4.registers_read_write.eval.RdInc", spec.rd_inc_eval),
        (
            "stage4.registers_read_write.output.eq.RdWriteValue",
            registers_eq_trace,
        ),
    ]);
    formula.append_all(&STAGE4_REGISTERS_OUTPUT_FORMULAS)?;
    let registers_claim = formula.value("stage4.registers_read_write.output.claim_expr")?;
    append_sumcheck_output_claim(
        context,
        module,
        OutputClaimSpec {
            symbol: "stage4.registers_read_write.output.claim",
            stage: "stage4",
            relation: "jolt.stage4.registers_read_write",
        },
        registers_claim,
        &[(
            "stage4.registers_read_write.output.eq.RdWriteValue",
            registers_eq_trace,
        )],
    )?;

    let ram_lt = append_structured_polynomial_eval(
        context,
        module,
        StructuredPolynomialSpec {
            symbol: "stage4.ram_val_check.output.lt.RamValCycle",
            polynomial: "lt",
            x_point: StructuredPolynomialPointSpec::full("reverse"),
            y_point: StructuredPolynomialPointSpec::suffix("x_point", "as_is"),
        },
        spec.ram_val_check.0,
        spec.openings.ram_val.point,
    )?;
    let mut formula = FieldFormulaBuilder::new(context, module);
    formula.bind_all(&[
        ("stage4.ram_val_check.gamma", spec.ram_val_check_gamma),
        ("stage4.ram_val_check.eval.RamRa", spec.ram_ra_eval),
        ("stage4.ram_val_check.eval.RamInc", spec.ram_inc_eval),
        ("stage4.ram_val_check.output.lt.RamValCycle", ram_lt),
    ]);
    formula.append_all(&STAGE4_RAM_VAL_OUTPUT_FORMULAS)?;
    let ram_val_claim = formula.value("stage4.ram_val_check.output.claim_expr")?;
    append_sumcheck_output_claim(
        context,
        module,
        OutputClaimSpec {
            symbol: "stage4.ram_val_check.output.claim",
            stage: "stage4",
            relation: "jolt.stage4.ram_val_check",
        },
        ram_val_claim,
        &[("stage4.ram_val_check.output.lt.RamValCycle", ram_lt)],
    )
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

fn append_field_sub<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    symbol: &str,
    lhs: Value<'c, 'a>,
    rhs: Value<'c, 'a>,
) -> Result<Value<'c, 'a>, MlirError> {
    append_field_binary(context, module, "field.sub", symbol, lhs, rhs)
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
            ("round_schedule", &spec.round_schedule),
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
            ("round_schedule", &spec.round_schedule),
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
            ("source", &format!("@{source}")),
            ("name", &format!("@{symbol}")),
            ("index", &int_attr(index)),
            ("oracle", &format!("@{oracle}")),
        ],
        &[result_value],
        &["!field.scalar"],
    )?;
    first_result(op, "piop.sumcheck_eval")
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

fn append_point_slice<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    symbol: &str,
    source: &str,
    offset: usize,
    length: usize,
    point: Value<'c, 'a>,
) -> Result<Value<'c, 'a>, MlirError> {
    let op = context.append_typed_op(
        module,
        "poly.point_slice",
        Some(symbol),
        &[
            ("source", &format!("@{source}")),
            ("offset", &int_attr(offset)),
            ("length", &int_attr(length)),
        ],
        &[point],
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
    points: &[Value<'c, 'a>],
) -> Result<Value<'c, 'a>, MlirError> {
    let op = context.append_typed_op(
        module,
        "poly.point_concat",
        Some(symbol),
        &[
            ("layout", &format!("\"{layout}\"")),
            ("arity", &int_attr(arity)),
        ],
        points,
        &["!poly.point"],
    )?;
    first_result(op, "poly.point_concat")
}

fn first_result<'c, 'a>(
    operation: OperationRef<'c, 'a>,
    operation_name: &str,
) -> Result<Value<'c, 'a>, MlirError> {
    result(operation, 0, operation_name)
}

fn result<'c, 'a>(
    operation: OperationRef<'c, 'a>,
    index: usize,
    operation_name: &str,
) -> Result<Value<'c, 'a>, MlirError> {
    operation
        .result(index)
        .map(Into::into)
        .map_err(|_| schema_error(format!("{operation_name} requires result {index}")))
}

struct RelationSpec<'a> {
    symbol: &'a str,
    kind: &'a str,
    domain: &'a str,
    num_rounds: usize,
    degree: usize,
    output_count: usize,
}

struct Stage4OpeningInputs<'c, 'a> {
    rd_write_value: Stage4OpeningInput<'c, 'a>,
    rs1_registers: Stage4OpeningInput<'c, 'a>,
    rs2_registers: Stage4OpeningInput<'c, 'a>,
    rs1_instruction: Stage4OpeningInput<'c, 'a>,
    rs2_instruction: Stage4OpeningInput<'c, 'a>,
    ram_val: Stage4OpeningInput<'c, 'a>,
    ram_val_final: Stage4OpeningInput<'c, 'a>,
    ram_val_init: Stage4OpeningInput<'c, 'a>,
}

struct Stage4OpeningInput<'c, 'a> {
    point: Value<'c, 'a>,
    eval: Value<'c, 'a>,
    claim: Value<'c, 'a>,
}

struct StageOpeningInputSpec<'a> {
    symbol: &'a str,
    source_stage: &'a str,
    source_claim: &'a str,
    oracle: &'a str,
    domain: &'a str,
    point_arity: usize,
}

struct Stage4BatchedSumcheckInputs<'c, 'a, 'b> {
    state: Value<'c, 'a>,
    stage: Value<'c, 'a>,
    openings: &'b Stage4OpeningInputs<'c, 'a>,
    registers_gamma: Value<'c, 'a>,
    ram_val_check_gamma: Value<'c, 'a>,
}

struct Stage4OutputOpeningsInputs<'c, 'a, 'b> {
    openings: &'b Stage4OpeningInputs<'c, 'a>,
    registers: (Value<'c, 'a>, Value<'c, 'a>),
    ram_val_check: (Value<'c, 'a>, Value<'c, 'a>),
    registers_gamma: Value<'c, 'a>,
    ram_val_check_gamma: Value<'c, 'a>,
}

struct Stage4OutputClaimInputs<'c, 'a, 'b> {
    openings: &'b Stage4OpeningInputs<'c, 'a>,
    registers: (Value<'c, 'a>, Value<'c, 'a>),
    ram_val_check: (Value<'c, 'a>, Value<'c, 'a>),
    registers_gamma: Value<'c, 'a>,
    ram_val_check_gamma: Value<'c, 'a>,
    registers_val_eval: Value<'c, 'a>,
    rs1_ra_eval: Value<'c, 'a>,
    rs2_ra_eval: Value<'c, 'a>,
    rd_wa_eval: Value<'c, 'a>,
    rd_inc_eval: Value<'c, 'a>,
    ram_ra_eval: Value<'c, 'a>,
    ram_inc_eval: Value<'c, 'a>,
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
    round_schedule: String,
}

struct SumcheckDriverSpec<'a> {
    symbol: &'a str,
    stage: &'a str,
    proof_slot: &'a str,
    relation: &'a str,
    policy: &'a str,
    round_schedule: String,
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

fn stage4_registers_rw_rounds(params: &JoltProtocolParams) -> usize {
    params.log_t + params.register_log_k
}

fn stage4_output_count() -> usize {
    STAGE4_REGISTER_OUTPUTS.len() + STAGE4_RAM_VAL_OUTPUTS.len()
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

fn schema_error(message: impl Into<String>) -> MlirError {
    SchemaError::new(message).into()
}
