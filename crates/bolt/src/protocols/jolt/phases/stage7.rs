use melior::ir::operation::OperationRef;
use melior::ir::Value;

use crate::ir::{BoltModule, Compute, Party, Protocol};
use crate::mlir::{verify_module, MeliorContext, MlirError};
use crate::schema::{verify_protocol_schema, SchemaError};

use super::super::oracles;
use super::super::params::JoltProtocolParams;
use super::lowering::{lower_party_to_compute, transcript_squeeze_protocol_result_type};
use super::sumcheck_output::{
    append_structured_polynomial_eval, append_sumcheck_output_claim,
    append_sumcheck_output_eval_family, OutputClaimSpec, OutputEvalFamilySpec,
    StructuredPolynomialPointSpec, StructuredPolynomialSpec,
};

const HAMMING_WEIGHT_CLAIM_REDUCTION_DEGREE: usize = 2;

pub fn build_stage7_protocol<'c>(
    context: &'c MeliorContext,
    params: &JoltProtocolParams,
) -> Result<BoltModule<'c, Protocol>, MlirError> {
    let module = context.new_module::<Protocol>("jolt.stage7", None);
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
        Some("jolt.stage7"),
        &[("roles", r#"["prover", "verifier"]"#)],
    )?;
    append_stage7_domains(context, &module, params)?;
    append_stage7_oracles(context, &module, params)?;
    append_stage7_relations(context, &module, params)?;
    let inputs = append_stage7_opening_inputs(context, &module, params)?;

    let fs = context.append_typed_op(
        &module,
        "transcript.state",
        Some("fs_after_stage6"),
        &[("scheme", "@blake2b_transcript")],
        &[],
        &["!transcript.state_type"],
    )?;
    let state = first_result(fs, "transcript.state")?;
    let stage = context.append_typed_op(
        &module,
        "piop.stage",
        Some("stage7"),
        &[
            ("name", r#""hamming_weight_claim_reduction""#),
            ("order", "7 : i64"),
            ("roles", r#"["prover", "verifier"]"#),
        ],
        &[],
        &["!piop.stage_type"],
    )?;
    let stage = first_result(stage, "piop.stage")?;

    let (state, gamma) = append_transcript_squeeze(
        context,
        &module,
        state,
        "stage7.hamming_weight_claim_reduction.gamma",
        "hamming_weight_claim_reduction_gamma",
        "challenge_scalar",
        1,
    )?;
    let _state = append_stage7_sumcheck(
        context,
        &module,
        params,
        Stage7SumcheckInputs {
            state,
            stage,
            openings: &inputs,
            gamma,
        },
    )?;

    verify_module(&module)?;
    verify_protocol_schema(&module)?;
    Ok(module)
}

pub fn lower_stage7_to_compute<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Party>,
) -> Result<BoltModule<'c, Compute>, MlirError> {
    lower_party_to_compute(context, module, "jolt.stage7", "jolt.stage7", "stage7")
}

fn append_stage7_domains<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Protocol>,
    params: &JoltProtocolParams,
) -> Result<(), MlirError> {
    append_domain(
        context,
        module,
        "jolt.stage7_hamming_weight_claim_reduction_domain",
        params.log_k_chunk,
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

fn append_stage7_oracles<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Protocol>,
    params: &JoltProtocolParams,
) -> Result<(), MlirError> {
    append_virtual_oracle(context, module, "HammingWeight", "jolt.trace_domain")?;
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

fn append_stage7_relations<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Protocol>,
    params: &JoltProtocolParams,
) -> Result<(), MlirError> {
    append_relation(
        context,
        module,
        RelationSpec {
            symbol: "jolt.stage7.hamming_weight_claim_reduction",
            kind: "sumcheck",
            domain: "jolt.stage7_hamming_weight_claim_reduction_domain",
            num_rounds: params.log_k_chunk,
            degree: HAMMING_WEIGHT_CLAIM_REDUCTION_DEGREE,
            output_count: total_ra_oracles(params),
        },
    )?;
    append_relation(
        context,
        module,
        RelationSpec {
            symbol: "jolt.stage7.batched",
            kind: "batched_sumcheck",
            domain: "jolt.stage7_hamming_weight_claim_reduction_domain",
            num_rounds: params.log_k_chunk,
            degree: HAMMING_WEIGHT_CLAIM_REDUCTION_DEGREE,
            output_count: total_ra_oracles(params),
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

fn append_stage7_opening_inputs<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    params: &JoltProtocolParams,
) -> Result<Stage7OpeningInputs<'c, 'a>, MlirError> {
    let ram_hamming = append_stage_input(
        context,
        module,
        StageOpeningInputSpec {
            symbol: "stage7.input.stage6.hamming_booleanity.HammingWeight",
            source_stage: "stage6",
            source_claim: "stage6.hamming_booleanity.opening.HammingWeight",
            oracle: "HammingWeight",
            domain: "jolt.trace_domain",
            point_arity: params.log_t,
            claim_kind: "virtual",
        },
    )?;

    let mut ra_inputs = Vec::with_capacity(total_ra_oracles(params));
    for index in 0..params.instruction_d {
        let oracle = format!("InstructionRa_{index}");
        ra_inputs.push(append_ra_inputs(
            context,
            module,
            params,
            &oracle,
            Stage7RaKind::Instruction,
            &format!("stage6.instruction_ra_virtual.opening.{oracle}"),
            &format!("stage7.input.stage6.instruction_ra_virtual.{oracle}"),
        )?);
    }
    for index in 0..params.bytecode_d {
        let oracle = format!("BytecodeRa_{index}");
        ra_inputs.push(append_ra_inputs(
            context,
            module,
            params,
            &oracle,
            Stage7RaKind::Bytecode,
            &format!("stage6.bytecode_read_raf.opening.{oracle}"),
            &format!("stage7.input.stage6.bytecode_read_raf.{oracle}"),
        )?);
    }
    for index in 0..params.ram_d {
        let oracle = format!("RamRa_{index}");
        ra_inputs.push(append_ra_inputs(
            context,
            module,
            params,
            &oracle,
            Stage7RaKind::Ram,
            &format!("stage6.ram_ra_virtual.opening.{oracle}"),
            &format!("stage7.input.stage6.ram_ra_virtual.{oracle}"),
        )?);
    }

    let booleanity_point = ra_inputs
        .first()
        .ok_or_else(|| schema_error("Stage 7 requires at least one RA oracle"))?
        .booleanity
        .point;

    Ok(Stage7OpeningInputs {
        ra_inputs,
        ram_hamming,
        booleanity_point,
    })
}

fn append_ra_inputs<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    params: &JoltProtocolParams,
    oracle: &str,
    kind: Stage7RaKind,
    source_virtual_claim: &str,
    virtual_input_symbol: &str,
) -> Result<Stage7RaInput<'c, 'a>, MlirError> {
    let booleanity = append_stage_input(
        context,
        module,
        StageOpeningInputSpec {
            symbol: &format!("stage7.input.stage6.booleanity.{oracle}"),
            source_stage: "stage6",
            source_claim: &format!("stage6.booleanity.opening.{oracle}"),
            oracle,
            domain: "jolt.main_witness_commit_domain",
            point_arity: params.log_k_chunk + params.log_t,
            claim_kind: "committed",
        },
    )?;
    let virtualization = append_stage_input(
        context,
        module,
        StageOpeningInputSpec {
            symbol: virtual_input_symbol,
            source_stage: "stage6",
            source_claim: source_virtual_claim,
            oracle,
            domain: "jolt.main_witness_commit_domain",
            point_arity: params.log_k_chunk + params.log_t,
            claim_kind: "committed",
        },
    )?;
    Ok(Stage7RaInput {
        oracle: oracle.to_owned(),
        kind,
        booleanity,
        virtualization,
    })
}

fn append_stage_input<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    spec: StageOpeningInputSpec<'_>,
) -> Result<Stage7OpeningInput<'c, 'a>, MlirError> {
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
    Ok(Stage7OpeningInput {
        point: result(op, 0, "piop.opening_input")?,
        eval: result(op, 1, "piop.opening_input")?,
        claim: result(op, 2, "piop.opening_input")?,
    })
}

fn append_stage7_sumcheck<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    params: &JoltProtocolParams,
    spec: Stage7SumcheckInputs<'c, 'a, '_>,
) -> Result<Value<'c, 'a>, MlirError> {
    let input_claim = append_hamming_weight_claim_reduction_input_claim(
        context,
        module,
        spec.openings,
        spec.gamma,
    )?;
    let mut input_openings = Vec::with_capacity(2 * spec.openings.ra_inputs.len() + 1);
    input_openings.push(spec.openings.ram_hamming.claim);
    for input in &spec.openings.ra_inputs {
        input_openings.push(input.booleanity.claim);
        input_openings.push(input.virtualization.claim);
    }
    let claim = append_sumcheck_claim(
        context,
        module,
        SumcheckClaimSpec {
            symbol: "stage7.hamming_weight_claim_reduction.input",
            stage: "stage7",
            domain: "jolt.stage7_hamming_weight_claim_reduction_domain",
            num_rounds: params.log_k_chunk,
            degree: HAMMING_WEIGHT_CLAIM_REDUCTION_DEGREE,
            claim: "stage7.hamming_weight_claim_reduction.weighted_stage6_claims",
            relation: "jolt.stage7.hamming_weight_claim_reduction",
        },
        input_claim,
        &input_openings,
    )?;
    let round_schedule = format!("[{}]", params.log_k_chunk);
    let batch = append_sumcheck_batch(
        context,
        module,
        spec.stage,
        &[claim],
        SumcheckBatchSpec {
            symbol: "stage7.batch",
            stage: "stage7",
            proof_slot: "stage7.sumcheck",
            policy: "jolt_core_stage7_aligned",
            ordered_claims: &["stage7.hamming_weight_claim_reduction.input"],
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
            symbol: "stage7.sumcheck",
            stage: "stage7",
            proof_slot: "stage7.sumcheck",
            relation: "jolt.stage7.batched",
            policy: "jolt_core_stage7_aligned",
            round_schedule: &round_schedule,
            claim_label: "sumcheck_claim",
            round_label: "sumcheck_poly",
            num_rounds: params.log_k_chunk,
            degree: HAMMING_WEIGHT_CLAIM_REDUCTION_DEGREE,
        },
    )?;
    let instance = append_sumcheck_instance_result(
        context,
        module,
        SumcheckInstanceResultSpec {
            symbol: "stage7.hamming_weight_claim_reduction.instance",
            source: "stage7.sumcheck",
            claim: "stage7.hamming_weight_claim_reduction.input",
            relation: "jolt.stage7.hamming_weight_claim_reduction",
            index: 0,
            point_arity: params.log_k_chunk,
            num_rounds: params.log_k_chunk,
            round_offset: 0,
            point_order: "reverse",
            degree: HAMMING_WEIGHT_CLAIM_REDUCTION_DEGREE,
        },
        point,
        result_value,
    )?;
    append_stage7_output_openings(context, module, params, spec.openings, spec.gamma, instance)?;
    Ok(state)
}

fn append_hamming_weight_claim_reduction_input_claim<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    inputs: &Stage7OpeningInputs<'c, 'a>,
    gamma: Value<'c, 'a>,
) -> Result<Value<'c, 'a>, MlirError> {
    let one = append_field_one(context, module, "stage7.field.one")?;
    let mut terms = Vec::with_capacity(3 * inputs.ra_inputs.len());
    for (index, input) in inputs.ra_inputs.iter().enumerate() {
        let hamming_eval = match input.kind {
            Stage7RaKind::Instruction | Stage7RaKind::Bytecode => one,
            Stage7RaKind::Ram => inputs.ram_hamming.eval,
        };
        terms.push(append_weighted_eval(
            context,
            module,
            &format!("stage7.hamming_weight_claim_reduction.claim.{index}.hw"),
            hamming_eval,
            gamma,
            3 * index,
        )?);
        terms.push(append_weighted_eval(
            context,
            module,
            &format!("stage7.hamming_weight_claim_reduction.claim.{index}.booleanity"),
            input.booleanity.eval,
            gamma,
            3 * index + 1,
        )?);
        terms.push(append_weighted_eval(
            context,
            module,
            &format!("stage7.hamming_weight_claim_reduction.claim.{index}.virtualization"),
            input.virtualization.eval,
            gamma,
            3 * index + 2,
        )?);
    }
    append_field_sum(
        context,
        module,
        "stage7.hamming_weight_claim_reduction.claim_expr",
        &terms,
    )
}

fn append_weighted_eval<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    symbol_prefix: &str,
    eval: Value<'c, 'a>,
    gamma: Value<'c, 'a>,
    gamma_power: usize,
) -> Result<Value<'c, 'a>, MlirError> {
    if gamma_power == 0 {
        return Ok(eval);
    }
    let power = append_field_pow(
        context,
        module,
        &format!("{symbol_prefix}.gamma_pow"),
        gamma,
        gamma_power,
    )?;
    append_field_mul(
        context,
        module,
        &format!("{symbol_prefix}.gamma_term"),
        power,
        eval,
    )
}

fn append_stage7_output_openings<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    params: &JoltProtocolParams,
    inputs: &Stage7OpeningInputs<'c, 'a>,
    gamma: Value<'c, 'a>,
    instance: (Value<'c, 'a>, Value<'c, 'a>),
) -> Result<(), MlirError> {
    let cycle = append_point_slice(
        context,
        module,
        "stage7.hamming_weight_claim_reduction.point.cycle",
        "stage7.input.stage6.booleanity.InstructionRa_0",
        params.log_k_chunk,
        params.log_t,
        inputs.booleanity_point,
    )?;
    let full_point = append_point_concat(
        context,
        module,
        "stage7.hamming_weight_claim_reduction.point",
        "address_chunk_then_cycle",
        params.log_k_chunk + params.log_t,
        &[instance.0, cycle],
    )?;

    let mut claims = Vec::with_capacity(inputs.ra_inputs.len());
    let mut claim_symbols = Vec::with_capacity(inputs.ra_inputs.len());
    let mut output_evals = Vec::with_capacity(inputs.ra_inputs.len());
    for (index, input) in inputs.ra_inputs.iter().enumerate() {
        let eval = append_sumcheck_eval(
            context,
            module,
            &format!(
                "stage7.hamming_weight_claim_reduction.eval.{}",
                input.oracle
            ),
            "stage7.sumcheck",
            &input.oracle,
            index,
            instance.1,
        )?;
        output_evals.push(eval);
        let symbol = format!(
            "stage7.hamming_weight_claim_reduction.opening.{}",
            input.oracle
        );
        claim_symbols.push(symbol.clone());
        claims.push(append_opening_claim(
            context,
            module,
            full_point,
            eval,
            OpeningClaimSpec {
                symbol: &symbol,
                oracle: &input.oracle,
                domain: "jolt.main_witness_commit_domain",
                point_arity: params.log_k_chunk + params.log_t,
                claim_kind: "committed",
            },
        )?);
    }
    append_stage7_output_claim(context, module, inputs, instance.0, gamma, &output_evals)?;
    let claim_names = claim_symbols.iter().map(String::as_str).collect::<Vec<_>>();
    let _batch = context.append_typed_op(
        module,
        "piop.opening_batch",
        Some("stage7.openings"),
        &[
            ("stage", "@stage7"),
            ("proof_slot", "@stage7.openings"),
            ("policy", r#""jolt_stage7_output_order""#),
            ("count", &int_attr(claims.len())),
            ("ordered_claims", &symbol_array_attr(&claim_names)),
        ],
        &claims,
        &["!piop.opening_batch_type"],
    )?;
    Ok(())
}

fn append_stage7_output_claim<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    inputs: &Stage7OpeningInputs<'c, 'a>,
    instance_point: Value<'c, 'a>,
    gamma: Value<'c, 'a>,
    output_evals: &[Value<'c, 'a>],
) -> Result<(), MlirError> {
    let booleanity_eq_symbol = "stage7.hamming_weight_claim_reduction.output.eq.Booleanity";
    let booleanity_eq = append_structured_polynomial_eval(
        context,
        module,
        StructuredPolynomialSpec {
            symbol: booleanity_eq_symbol,
            polynomial: "eq",
            x_point: StructuredPolynomialPointSpec::full("reverse"),
            y_point: StructuredPolynomialPointSpec::prefix("x_point", "as_is"),
        },
        instance_point,
        inputs.booleanity_point,
    )?;

    let mut polynomial_evals = Vec::with_capacity(output_evals.len() + 1);
    polynomial_evals.push((booleanity_eq_symbol.to_owned(), booleanity_eq));
    let mut eval_terms = Vec::with_capacity(output_evals.len());
    let mut item_terms = Vec::with_capacity(output_evals.len());
    for (input, &output_eval) in inputs.ra_inputs.iter().zip(output_evals) {
        let virtualization_eq_symbol = format!(
            "stage7.hamming_weight_claim_reduction.output.eq.{}.virtualization",
            input.oracle
        );
        let virtualization_eq = append_structured_polynomial_eval(
            context,
            module,
            StructuredPolynomialSpec {
                symbol: &virtualization_eq_symbol,
                polynomial: "eq",
                x_point: StructuredPolynomialPointSpec::full("reverse"),
                y_point: StructuredPolynomialPointSpec::prefix("x_point", "as_is"),
            },
            instance_point,
            input.virtualization.point,
        )?;
        polynomial_evals.push((virtualization_eq_symbol.clone(), virtualization_eq));
        eval_terms.push((
            format!(
                "stage7.hamming_weight_claim_reduction.eval.{}",
                input.oracle
            ),
            output_eval,
        ));
        item_terms.push((virtualization_eq_symbol, virtualization_eq));
    }
    let eval_term_refs = eval_terms
        .iter()
        .map(|(symbol, value)| (symbol.as_str(), *value))
        .collect::<Vec<_>>();
    let item_term_refs = item_terms
        .iter()
        .map(|(symbol, value)| (symbol.as_str(), *value))
        .collect::<Vec<_>>();
    let output_claim = append_sumcheck_output_eval_family(
        context,
        module,
        OutputEvalFamilySpec {
            symbol: "stage7.hamming_weight_claim_reduction.output.family",
            power_stride: 3,
            value_term_offsets: &[0],
            shared_term_offsets: &[1],
            item_term_offsets: &[2],
        },
        gamma,
        &eval_term_refs,
        &[(booleanity_eq_symbol, booleanity_eq)],
        &item_term_refs,
    )?;
    let polynomial_eval_refs = polynomial_evals
        .iter()
        .map(|(symbol, value)| (symbol.as_str(), *value))
        .collect::<Vec<_>>();
    append_sumcheck_output_claim(
        context,
        module,
        OutputClaimSpec {
            symbol: "stage7.hamming_weight_claim_reduction.output",
            stage: "stage7",
            relation: "jolt.stage7.hamming_weight_claim_reduction",
        },
        output_claim,
        &polynomial_eval_refs,
    )
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

fn append_field_one<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    symbol: &str,
) -> Result<Value<'c, 'a>, MlirError> {
    let op = context.append_typed_op(
        module,
        "field.one",
        Some(symbol),
        &[("field", "@bn254_fr")],
        &[],
        &["!field.scalar"],
    )?;
    first_result(op, "field.one")
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
        return append_field_one(context, module, symbol_prefix);
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

fn total_ra_oracles(params: &JoltProtocolParams) -> usize {
    params.instruction_d + params.bytecode_d + params.ram_d
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

#[derive(Clone, Copy)]
enum Stage7RaKind {
    Instruction,
    Bytecode,
    Ram,
}

struct Stage7SumcheckInputs<'c, 'a, 'b> {
    state: Value<'c, 'a>,
    stage: Value<'c, 'a>,
    openings: &'b Stage7OpeningInputs<'c, 'a>,
    gamma: Value<'c, 'a>,
}

struct Stage7OpeningInputs<'c, 'a> {
    ra_inputs: Vec<Stage7RaInput<'c, 'a>>,
    ram_hamming: Stage7OpeningInput<'c, 'a>,
    booleanity_point: Value<'c, 'a>,
}

struct Stage7RaInput<'c, 'a> {
    oracle: String,
    kind: Stage7RaKind,
    booleanity: Stage7OpeningInput<'c, 'a>,
    virtualization: Stage7OpeningInput<'c, 'a>,
}

struct Stage7OpeningInput<'c, 'a> {
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

struct OpeningClaimSpec<'a> {
    symbol: &'a str,
    oracle: &'a str,
    domain: &'a str,
    point_arity: usize,
    claim_kind: &'a str,
}
