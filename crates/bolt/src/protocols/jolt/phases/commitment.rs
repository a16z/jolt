use std::collections::BTreeMap;

use melior::ir::block::BlockLike;
use melior::ir::operation::{OperationLike, OperationResult};
use melior::ir::{OperationRef, Value};

use crate::ir::{BoltModule, Compute, Cpu, Party, Protocol, Role};
use crate::mlir::{verify_module, MeliorContext, MlirError};
use crate::schema::{
    int_attr, operation_name, symbol_array_attr, symbol_attr, verify_compute_schema,
    verify_cpu_schema, SchemaError,
};

use super::super::oracles::{self, ADVICE_FAMILY_SYMBOL, MAIN_WITNESS_FAMILY_SYMBOL, PCS_SYMBOL};
use super::super::params::JoltProtocolParams;
use super::super::validate::{verify_jolt_party_schema, verify_jolt_protocol_schema};
use super::lowering::{
    copy_attrs, field_lowering_attrs as compute_field_attrs, string_attr,
    transcript_squeeze_cpu_result_types,
};

pub fn build_commitment_protocol<'c>(
    context: &'c MeliorContext,
    params: &JoltProtocolParams,
) -> Result<BoltModule<'c, Protocol>, MlirError> {
    let module = context.new_module::<Protocol>("jolt.commitment_phase", None);
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
        Some("jolt.commitment_phase"),
        &[("roles", r#"["prover", "verifier"]"#)],
    )?;
    oracles::append_committed_oracles(context, &module, params)?;
    context.append_op(
        &module,
        "piop.oracle_family",
        Some(MAIN_WITNESS_FAMILY_SYMBOL),
        &[
            (
                "ordered_oracles",
                &oracles::main_witness_oracle_attr(params),
            ),
            ("count", &format!("{} : i64", params.num_committed)),
            ("domain", "@jolt.main_witness_commit_domain"),
            ("visibility", r#""committed""#),
        ],
    )?;
    context.append_op(
        &module,
        "piop.oracle_family",
        Some(ADVICE_FAMILY_SYMBOL),
        &[
            ("ordered_oracles", "[@UntrustedAdvice, @TrustedAdvice]"),
            ("count", "2 : i64"),
            ("domain", "@jolt.trace_domain"),
            ("visibility", r#""optional_committed""#),
        ],
    )?;
    let state = context.append_typed_op(
        &module,
        "transcript.state",
        Some("fs0"),
        &[("scheme", "@blake2b_transcript")],
        &[],
        &["!transcript.state_type"],
    )?;
    let mut state = state
        .result(0)
        .map_err(|_| schema_error("transcript.state requires one result"))?
        .into();
    let main_commitments = context.append_typed_op(
        &module,
        "commit.publish_batch",
        Some("jolt.main_witness_commitments"),
        &[
            ("oracle_family", "@jolt.main_witness_polys"),
            ("label", r#""commitment""#),
        ],
        &[],
        &["!commit.artifact"],
    )?;
    let main_commitments = main_commitments
        .result(0)
        .map_err(|_| schema_error("commit.publish_batch requires one result"))?
        .into();
    let _pcs_commit = context.append_typed_op(
        &module,
        "pcs.commit_batch",
        Some("jolt.dory_main_witness_commit"),
        &[("scheme", &format!("@{PCS_SYMBOL}"))],
        &[main_commitments],
        &[],
    )?;
    let untrusted_advice = context.append_typed_op(
        &module,
        "commit.publish_optional",
        Some("jolt.untrusted_advice_commitment"),
        &[
            ("oracle", "@UntrustedAdvice"),
            ("label", r#""untrusted_advice""#),
            ("skip_policy", r#""missing_or_zero""#),
        ],
        &[],
        &["!commit.artifact"],
    )?;
    let untrusted_advice = untrusted_advice
        .result(0)
        .map_err(|_| schema_error("commit.publish_optional requires one result"))?
        .into();
    let trusted_advice = context.append_typed_op(
        &module,
        "commit.publish_optional",
        Some("jolt.trusted_advice_commitment"),
        &[
            ("oracle", "@TrustedAdvice"),
            ("label", r#""trusted_advice""#),
            ("skip_policy", r#""missing_or_zero""#),
        ],
        &[],
        &["!commit.artifact"],
    )?;
    let trusted_advice = trusted_advice
        .result(0)
        .map_err(|_| schema_error("commit.publish_optional requires one result"))?
        .into();
    let absorb = context.append_typed_op(
        &module,
        "transcript.absorb",
        Some("jolt.absorb_main_witness_commitments"),
        &[("label", r#""commitment""#)],
        &[state, main_commitments],
        &["!transcript.state_type"],
    )?;
    state = absorb
        .result(0)
        .map_err(|_| schema_error("transcript.absorb requires one result"))?
        .into();
    let absorb = context.append_typed_op(
        &module,
        "transcript.absorb_optional",
        Some("jolt.absorb_untrusted_advice"),
        &[("label", r#""untrusted_advice""#)],
        &[state, untrusted_advice],
        &["!transcript.state_type"],
    )?;
    state = absorb
        .result(0)
        .map_err(|_| schema_error("transcript.absorb_optional requires one result"))?
        .into();
    let _absorb = context.append_typed_op(
        &module,
        "transcript.absorb_optional",
        Some("jolt.absorb_trusted_advice"),
        &[("label", r#""trusted_advice""#)],
        &[state, trusted_advice],
        &["!transcript.state_type"],
    )?;
    verify_module(&module)?;
    verify_jolt_protocol_schema(&module)?;
    Ok(module)
}

pub fn lower_commitment_to_compute<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Party>,
) -> Result<BoltModule<'c, Compute>, MlirError> {
    verify_jolt_party_schema(module)?;
    let role = module
        .role()
        .ok_or_else(|| schema_error("commitment lowering requires party role"))?;
    let concrete = analyze_concrete(module)?;
    let (batch_op, optional_op) = match role {
        Role::Prover => ("compute.pcs_commit_batch", "compute.pcs_commit_optional"),
        Role::Verifier => ("compute.pcs_receive_batch", "compute.pcs_receive_optional"),
    };
    let module_name = module.name();
    let compute = context.new_module::<Compute>(&module_name, Some(role.clone()));
    context.append_op_with_owned_attrs(
        &compute,
        "compute.params",
        Some("jolt.compute_params"),
        &[
            ("field".to_owned(), symbol_ref(&concrete.params.field)),
            ("pcs".to_owned(), symbol_ref(&concrete.params.pcs)),
            (
                "transcript".to_owned(),
                symbol_ref(&concrete.params.transcript),
            ),
        ],
    )?;
    context.append_op(
        &compute,
        "compute.function",
        Some("jolt.commitment_phase"),
        &[("source", "@jolt.commitment_phase")],
    )?;

    let mut artifact_values = BTreeMap::new();
    let transcript_scheme = symbol_ref(&concrete.params.transcript);
    let transcript_init = context.append_typed_op(
        &compute,
        "compute.transcript_init",
        Some("fs0"),
        &[("scheme", transcript_scheme.as_str())],
        &[],
        &["!compute.transcript_state"],
    )?;
    let mut transcript_value = first_result(transcript_init, "compute.transcript_init")?;

    for plan in &concrete.batch_plans {
        let family_symbol = format!("{}.oracle_family.compute", plan.oracle_family);
        let family_init = context.append_typed_op_with_owned_attrs(
            &compute,
            "compute.oracle_family_init",
            Some(&family_symbol),
            &[
                ("family".to_owned(), symbol_ref(&plan.oracle_family)),
                ("count".to_owned(), int_attr_source(plan.count)),
            ],
            &[],
            &["!compute.oracle_family"],
        )?;
        let mut family_value = first_result(family_init, "compute.oracle_family_init")?;
        for (index, oracle) in plan.oracles.iter().enumerate() {
            let oracle_buffer = concrete.oracle_buffers.get(oracle).ok_or_else(|| {
                schema_error(format!(
                    "batch commitment references missing oracle buffer @{oracle}"
                ))
            })?;
            let oracle_value = append_oracle_buffer(
                context,
                &compute,
                &role,
                &concrete.params,
                oracle,
                &oracle_buffer.domain,
                oracle_buffer.num_vars,
            )?;
            let append_symbol = format!("{}.append_{index}.compute", plan.oracle_family);
            let append = context.append_typed_op_with_owned_attrs(
                &compute,
                "compute.oracle_family_append",
                Some(&append_symbol),
                &[
                    ("family".to_owned(), symbol_ref(&plan.oracle_family)),
                    ("oracle".to_owned(), symbol_ref(oracle)),
                    ("index".to_owned(), int_attr_source(index)),
                ],
                &[family_value, oracle_value],
                &["!compute.oracle_family"],
            )?;
            family_value = first_result(append, "compute.oracle_family_append")?;
        }
        let symbol = format!("{}.compute", plan.artifact);
        let attrs = vec![
            ("artifact".to_owned(), symbol_ref(&plan.artifact)),
            ("count".to_owned(), int_attr_source(plan.count)),
            ("domain".to_owned(), symbol_ref(&plan.domain)),
            ("label".to_owned(), string_attr_source(&plan.label)),
            ("num_vars".to_owned(), int_attr_source(plan.num_vars)),
            ("oracle_family".to_owned(), symbol_ref(&plan.oracle_family)),
            (
                "ordered_oracles".to_owned(),
                symbol_array_attr_source(&plan.oracles),
            ),
            ("pcs".to_owned(), symbol_ref(&plan.pcs)),
        ];
        let operation = context.append_typed_op_with_owned_attrs(
            &compute,
            batch_op,
            Some(&symbol),
            &attrs,
            &[family_value],
            &["!compute.commitment_artifact"],
        )?;
        let value = first_result(operation, batch_op)?;
        let inserted = artifact_values.insert(plan.artifact.clone(), value);
        debug_assert!(inserted.is_none());
    }
    for plan in &concrete.optional_plans {
        let oracle_value = append_optional_oracle_buffer(
            context,
            &compute,
            &role,
            &plan.oracle,
            &plan.domain,
            plan.num_vars,
            &plan.skip_policy,
        )?;
        let symbol = format!("{}.compute", plan.artifact);
        let attrs = vec![
            ("artifact".to_owned(), symbol_ref(&plan.artifact)),
            ("domain".to_owned(), symbol_ref(&plan.domain)),
            ("label".to_owned(), string_attr_source(&plan.label)),
            ("num_vars".to_owned(), int_attr_source(plan.num_vars)),
            ("oracle".to_owned(), symbol_ref(&plan.oracle)),
            ("pcs".to_owned(), symbol_ref(&plan.pcs)),
            (
                "skip_policy".to_owned(),
                string_attr_source(&plan.skip_policy),
            ),
        ];
        let operation = context.append_typed_op_with_owned_attrs(
            &compute,
            optional_op,
            Some(&symbol),
            &attrs,
            &[oracle_value],
            &["!compute.commitment_artifact"],
        )?;
        let value = first_result(operation, optional_op)?;
        let inserted = artifact_values.insert(plan.artifact.clone(), value);
        debug_assert!(inserted.is_none());
    }
    for step in &concrete.transcript_steps {
        let artifact = artifact_values.get(&step.source).copied().ok_or_else(|| {
            schema_error(format!(
                "transcript absorb @{} references missing commitment artifact @{}",
                step.symbol, step.source
            ))
        })?;
        let symbol = format!("{}.compute", step.symbol);
        let attrs = vec![
            ("label".to_owned(), string_attr_source(&step.label)),
            (
                "optional".to_owned(),
                bool_attr_source(step.optional).to_owned(),
            ),
        ];
        let operation = context.append_typed_op_with_owned_attrs(
            &compute,
            "compute.transcript_absorb",
            Some(&symbol),
            &attrs,
            &[transcript_value, artifact],
            &["!compute.transcript_state"],
        )?;
        transcript_value = first_result(operation, "compute.transcript_absorb")?;
    }
    verify_module(&compute)?;
    verify_compute_schema(&compute)?;
    Ok(compute)
}

pub fn lower_compute_to_cpu<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Compute>,
) -> Result<BoltModule<'c, Cpu>, MlirError> {
    verify_compute_schema(module)?;
    let role = module
        .role()
        .ok_or_else(|| schema_error("CPU lowering requires compute party role"))?;
    let module_name = module.name();
    let cpu = context.new_module::<Cpu>(&module_name, Some(role));
    let mut value_map = BTreeMap::new();
    let mut operation = module.as_mlir_module().body().first_operation();
    while let Some(op) = operation {
        operation = op.next_in_block();
        match operation_name(op).as_str() {
            "compute.function" => {
                let source = symbol_ref(&symbol_attr(op, "source")?);
                let symbol = string_attr(op, "sym_name")?;
                context.append_op(&cpu, "cpu.function", Some(&symbol), &[("source", &source)])?;
            }
            "compute.params" => {
                let symbol = string_attr(op, "sym_name")?;
                context.append_op_with_owned_attrs(
                    &cpu,
                    "cpu.params",
                    Some(&symbol),
                    &[
                        ("field".to_owned(), symbol_ref(&symbol_attr(op, "field")?)),
                        ("pcs".to_owned(), symbol_ref(&symbol_attr(op, "pcs")?)),
                        (
                            "transcript".to_owned(),
                            symbol_ref(&symbol_attr(op, "transcript")?),
                        ),
                    ],
                )?;
            }
            "compute.kernel" => {
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(op, &["relation", "kind", "backend", "abi"])?;
                context.append_op_with_owned_attrs(&cpu, "cpu.kernel", Some(&symbol), &attrs)?;
            }
            "compute.transcript_init" => {
                let attrs = vec![("scheme".to_owned(), symbol_ref(&symbol_attr(op, "scheme")?))];
                let symbol = string_attr(op, "sym_name")?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &cpu,
                    "cpu.transcript_init",
                    Some(&symbol),
                    &attrs,
                    &[],
                    &["!cpu.transcript_state"],
                )?;
                let value = first_result(operation, "cpu.transcript_init")?;
                let inserted = value_map.insert(operation_result_key(op)?, value);
                debug_assert!(inserted.is_none());
            }
            "compute.oracle_dense_trace"
            | "compute.oracle_one_hot_chunk"
            | "compute.oracle_optional_advice"
            | "compute.oracle_ref" => {
                let target_op = operation_name(op).replacen("compute.", "cpu.", 1);
                let attrs = copy_attrs(
                    op,
                    &[
                        "oracle",
                        "source",
                        "domain",
                        "num_vars",
                        "trace_num_vars",
                        "chunk",
                        "num_chunks",
                        "chunk_bits",
                        "padding",
                        "layout",
                        "skip_policy",
                    ],
                )?;
                let symbol = string_attr(op, "sym_name")?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &cpu,
                    &target_op,
                    Some(&symbol),
                    &attrs,
                    &[],
                    &["!cpu.oracle_buffer"],
                )?;
                let value = first_result(operation, &target_op)?;
                let inserted = value_map.insert(operation_result_key(op)?, value);
                debug_assert!(inserted.is_none());
            }
            "compute.oracle_family_init" => {
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(op, &["family", "count"])?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &cpu,
                    "cpu.oracle_family_init",
                    Some(&symbol),
                    &attrs,
                    &[],
                    &["!cpu.oracle_family"],
                )?;
                let value = first_result(operation, "cpu.oracle_family_init")?;
                let inserted = value_map.insert(operation_result_key(op)?, value);
                debug_assert!(inserted.is_none());
            }
            "compute.oracle_family_append" => {
                let input = operand_key(op, 0)?;
                let oracle = operand_key(op, 1)?;
                let input = value_map.get(&input).copied().ok_or_else(|| {
                    schema_error("compute.oracle_family_append input operand was not lowered")
                })?;
                let oracle = value_map.get(&oracle).copied().ok_or_else(|| {
                    schema_error("compute.oracle_family_append oracle operand was not lowered")
                })?;
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(op, &["family", "oracle", "index"])?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &cpu,
                    "cpu.oracle_family_append",
                    Some(&symbol),
                    &attrs,
                    &[input, oracle],
                    &["!cpu.oracle_family"],
                )?;
                let value = first_result(operation, "cpu.oracle_family_append")?;
                let inserted = value_map.insert(operation_result_key(op)?, value);
                debug_assert!(inserted.is_none());
            }
            "compute.pcs_commit_batch" | "compute.pcs_receive_batch" => {
                let target_op = match operation_name(op).as_str() {
                    "compute.pcs_commit_batch" => "cpu.pcs_commit_batch",
                    "compute.pcs_receive_batch" => "cpu.pcs_receive_batch",
                    _ => unreachable!(),
                };
                let attrs = vec![
                    (
                        "artifact".to_owned(),
                        symbol_ref(&symbol_attr(op, "artifact")?),
                    ),
                    ("count".to_owned(), int_attr_source(int_attr(op, "count")?)),
                    ("domain".to_owned(), symbol_ref(&symbol_attr(op, "domain")?)),
                    (
                        "label".to_owned(),
                        string_attr_source(&string_attr(op, "label")?),
                    ),
                    (
                        "num_vars".to_owned(),
                        int_attr_source(int_attr(op, "num_vars")?),
                    ),
                    (
                        "oracle_family".to_owned(),
                        symbol_ref(&symbol_attr(op, "oracle_family")?),
                    ),
                    (
                        "ordered_oracles".to_owned(),
                        symbol_array_attr_source(&symbol_array_attr(op, "ordered_oracles")?),
                    ),
                    ("pcs".to_owned(), symbol_ref(&symbol_attr(op, "pcs")?)),
                ];
                let oracles = operand_key(op, 0)?;
                let oracles = value_map.get(&oracles).copied().ok_or_else(|| {
                    schema_error("compute.pcs batch oracle family was not lowered")
                })?;
                let symbol = string_attr(op, "sym_name")?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &cpu,
                    target_op,
                    Some(&symbol),
                    &attrs,
                    &[oracles],
                    &["!cpu.commitment_artifact"],
                )?;
                let value = first_result(operation, target_op)?;
                let inserted = value_map.insert(operation_result_key(op)?, value);
                debug_assert!(inserted.is_none());
            }
            "compute.pcs_commit_optional" | "compute.pcs_receive_optional" => {
                let target_op = match operation_name(op).as_str() {
                    "compute.pcs_commit_optional" => "cpu.pcs_commit_optional",
                    "compute.pcs_receive_optional" => "cpu.pcs_receive_optional",
                    _ => unreachable!(),
                };
                let attrs = vec![
                    (
                        "artifact".to_owned(),
                        symbol_ref(&symbol_attr(op, "artifact")?),
                    ),
                    ("domain".to_owned(), symbol_ref(&symbol_attr(op, "domain")?)),
                    (
                        "label".to_owned(),
                        string_attr_source(&string_attr(op, "label")?),
                    ),
                    (
                        "num_vars".to_owned(),
                        int_attr_source(int_attr(op, "num_vars")?),
                    ),
                    ("oracle".to_owned(), symbol_ref(&symbol_attr(op, "oracle")?)),
                    ("pcs".to_owned(), symbol_ref(&symbol_attr(op, "pcs")?)),
                    (
                        "skip_policy".to_owned(),
                        string_attr_source(&string_attr(op, "skip_policy")?),
                    ),
                ];
                let oracle = operand_key(op, 0)?;
                let oracle = value_map
                    .get(&oracle)
                    .copied()
                    .ok_or_else(|| schema_error("compute.pcs optional oracle was not lowered"))?;
                let symbol = string_attr(op, "sym_name")?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &cpu,
                    target_op,
                    Some(&symbol),
                    &attrs,
                    &[oracle],
                    &["!cpu.commitment_artifact"],
                )?;
                let value = first_result(operation, target_op)?;
                let inserted = value_map.insert(operation_result_key(op)?, value);
                debug_assert!(inserted.is_none());
            }
            "compute.transcript_absorb" => {
                let input = operand_key(op, 0)?;
                let artifact = operand_key(op, 1)?;
                let input = value_map.get(&input).copied().ok_or_else(|| {
                    schema_error("compute.transcript_absorb input operand was not lowered")
                })?;
                let artifact = value_map.get(&artifact).copied().ok_or_else(|| {
                    schema_error("compute.transcript_absorb artifact operand was not lowered")
                })?;
                let attrs = vec![
                    (
                        "label".to_owned(),
                        string_attr_source(&string_attr(op, "label")?),
                    ),
                    (
                        "optional".to_owned(),
                        bool_attr_source(bool_attr(op, "optional")?).to_owned(),
                    ),
                ];
                let symbol = string_attr(op, "sym_name")?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &cpu,
                    "cpu.transcript_absorb",
                    Some(&symbol),
                    &attrs,
                    &[input, artifact],
                    &["!cpu.transcript_state"],
                )?;
                let output = first_result(operation, "cpu.transcript_absorb")?;
                let inserted = value_map.insert(operation_result_key(op)?, output);
                debug_assert!(inserted.is_none());
            }
            "compute.transcript_absorb_bytes" => {
                let operands = lowered_operands(op, &value_map)?;
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(op, &["label", "payload"])?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &cpu,
                    "cpu.transcript_absorb_bytes",
                    Some(&symbol),
                    &attrs,
                    &operands,
                    &["!cpu.transcript_state"],
                )?;
                let output = first_result(operation, "cpu.transcript_absorb_bytes")?;
                let inserted = value_map.insert(operation_result_key(op)?, output);
                debug_assert!(inserted.is_none());
            }
            "compute.transcript_squeeze" => {
                let operands = lowered_operands(op, &value_map)?;
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(op, &["label", "kind", "count"])?;
                let result_types = transcript_squeeze_cpu_result_types(op)?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &cpu,
                    "cpu.transcript_squeeze",
                    Some(&symbol),
                    &attrs,
                    &operands,
                    &result_types,
                )?;
                insert_result_mapping(&mut value_map, op, operation, 0, 0)?;
                insert_result_mapping(&mut value_map, op, operation, 1, 1)?;
            }
            "compute.opening_input" => {
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(
                    op,
                    &[
                        "source_stage",
                        "source_claim",
                        "oracle",
                        "domain",
                        "point_arity",
                        "claim_kind",
                    ],
                )?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &cpu,
                    "cpu.opening_input",
                    Some(&symbol),
                    &attrs,
                    &[],
                    &["!cpu.point", "!cpu.field_value", "!cpu.opening_claim_type"],
                )?;
                for index in 0..3 {
                    insert_result_mapping(&mut value_map, op, operation, index, index)?;
                }
            }
            "compute.point_slice" => {
                let operands = lowered_operands(op, &value_map)?;
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(op, &["source", "offset", "length"])?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &cpu,
                    "cpu.point_slice",
                    Some(&symbol),
                    &attrs,
                    &operands,
                    &["!cpu.point"],
                )?;
                insert_result_mapping(&mut value_map, op, operation, 0, 0)?;
            }
            "compute.point_zero" => {
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(op, &["field", "arity"])?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &cpu,
                    "cpu.point_zero",
                    Some(&symbol),
                    &attrs,
                    &[],
                    &["!cpu.point"],
                )?;
                insert_result_mapping(&mut value_map, op, operation, 0, 0)?;
            }
            "compute.point_concat" => {
                let operands = lowered_operands(op, &value_map)?;
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(op, &["layout", "arity"])?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &cpu,
                    "cpu.point_concat",
                    Some(&symbol),
                    &attrs,
                    &operands,
                    &["!cpu.point"],
                )?;
                insert_result_mapping(&mut value_map, op, operation, 0, 0)?;
            }
            "compute.field_const" => {
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(op, &["field", "value"])?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &cpu,
                    "cpu.field_const",
                    Some(&symbol),
                    &attrs,
                    &[],
                    &["!cpu.field_value"],
                )?;
                insert_result_mapping(&mut value_map, op, operation, 0, 0)?;
            }
            "compute.field_zero" | "compute.field_one" => {
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(op, &["field"])?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &cpu,
                    &operation_name(op).replace("compute.", "cpu."),
                    Some(&symbol),
                    &attrs,
                    &[],
                    &["!cpu.field_value"],
                )?;
                insert_result_mapping(&mut value_map, op, operation, 0, 0)?;
            }
            "compute.field_add"
            | "compute.field_sub"
            | "compute.field_mul"
            | "compute.field_neg"
            | "compute.field_pow"
            | "compute.poly_lagrange_basis_eval" => {
                let operands = lowered_operands(op, &value_map)?;
                let symbol = string_attr(op, "sym_name")?;
                let attrs = compute_field_attrs(op)?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &cpu,
                    &operation_name(op).replace("compute.", "cpu."),
                    Some(&symbol),
                    &attrs,
                    &operands,
                    &["!cpu.field_value"],
                )?;
                insert_result_mapping(&mut value_map, op, operation, 0, 0)?;
            }
            "compute.sumcheck_kernel_claim" => {
                let operands = lowered_operands(op, &value_map)?;
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(
                    op,
                    &["stage", "domain", "num_rounds", "degree", "claim", "kernel"],
                )?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &cpu,
                    "cpu.sumcheck_claim",
                    Some(&symbol),
                    &attrs,
                    &operands,
                    &["!cpu.sumcheck_claim_type"],
                )?;
                insert_result_mapping(&mut value_map, op, operation, 0, 0)?;
            }
            "compute.sumcheck_verify_claim" => {
                let operands = lowered_operands(op, &value_map)?;
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(
                    op,
                    &[
                        "stage",
                        "domain",
                        "num_rounds",
                        "degree",
                        "claim",
                        "relation",
                    ],
                )?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &cpu,
                    "cpu.sumcheck_verify_claim",
                    Some(&symbol),
                    &attrs,
                    &operands,
                    &["!cpu.sumcheck_claim_type"],
                )?;
                insert_result_mapping(&mut value_map, op, operation, 0, 0)?;
            }
            "compute.sumcheck_batch" => {
                let operands = lowered_operands(op, &value_map)?;
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(
                    op,
                    &[
                        "stage",
                        "proof_slot",
                        "policy",
                        "count",
                        "ordered_claims",
                        "claim_label",
                        "round_label",
                        "round_schedule",
                    ],
                )?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &cpu,
                    "cpu.sumcheck_batch",
                    Some(&symbol),
                    &attrs,
                    &operands,
                    &["!cpu.sumcheck_batch_type"],
                )?;
                insert_result_mapping(&mut value_map, op, operation, 0, 0)?;
            }
            "compute.sumcheck_kernel_driver" => {
                let operands = lowered_operands(op, &value_map)?;
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(
                    op,
                    &[
                        "stage",
                        "proof_slot",
                        "kernel",
                        "policy",
                        "round_schedule",
                        "claim_label",
                        "round_label",
                        "num_rounds",
                        "degree",
                    ],
                )?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &cpu,
                    "cpu.sumcheck_driver",
                    Some(&symbol),
                    &attrs,
                    &operands,
                    &[
                        "!cpu.transcript_state",
                        "!cpu.point",
                        "!cpu.sumcheck_result_type",
                        "!cpu.sumcheck_proof_type",
                    ],
                )?;
                for index in 0..4 {
                    insert_result_mapping(&mut value_map, op, operation, index, index)?;
                }
            }
            "compute.sumcheck_verify" => {
                let operands = lowered_operands(op, &value_map)?;
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(
                    op,
                    &[
                        "stage",
                        "proof_slot",
                        "relation",
                        "policy",
                        "round_schedule",
                        "claim_label",
                        "round_label",
                        "num_rounds",
                        "degree",
                    ],
                )?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &cpu,
                    "cpu.sumcheck_verify",
                    Some(&symbol),
                    &attrs,
                    &operands,
                    &[
                        "!cpu.transcript_state",
                        "!cpu.point",
                        "!cpu.sumcheck_result_type",
                        "!cpu.sumcheck_proof_type",
                    ],
                )?;
                for index in 0..4 {
                    insert_result_mapping(&mut value_map, op, operation, index, index)?;
                }
            }
            "compute.sumcheck_eval" => {
                let operands = lowered_operands(op, &value_map)?;
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(op, &["source", "name", "index", "oracle"])?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &cpu,
                    "cpu.sumcheck_eval",
                    Some(&symbol),
                    &attrs,
                    &operands,
                    &["!cpu.field_value"],
                )?;
                insert_result_mapping(&mut value_map, op, operation, 0, 0)?;
            }
            "compute.sumcheck_instance_result" => {
                let operands = lowered_operands(op, &value_map)?;
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(
                    op,
                    &[
                        "source",
                        "claim",
                        "relation",
                        "index",
                        "point_arity",
                        "num_rounds",
                        "round_offset",
                        "point_order",
                        "degree",
                    ],
                )?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &cpu,
                    "cpu.sumcheck_instance_result",
                    Some(&symbol),
                    &attrs,
                    &operands,
                    &["!cpu.point", "!cpu.sumcheck_result_type"],
                )?;
                insert_result_mapping(&mut value_map, op, operation, 0, 0)?;
                insert_result_mapping(&mut value_map, op, operation, 1, 1)?;
            }
            "compute.structured_polynomial_eval" => {
                let operands = lowered_operands(op, &value_map)?;
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(
                    op,
                    &[
                        "polynomial",
                        "x_point_segment",
                        "x_point_length",
                        "x_point_order",
                        "y_point_segment",
                        "y_point_length",
                        "y_point_order",
                    ],
                )?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &cpu,
                    "cpu.structured_polynomial_eval",
                    Some(&symbol),
                    &attrs,
                    &operands,
                    &["!cpu.field_value"],
                )?;
                insert_result_mapping(&mut value_map, op, operation, 0, 0)?;
            }
            "compute.sumcheck_output_eval_family" => {
                let operands = lowered_operands(op, &value_map)?;
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(
                    op,
                    &[
                        "power_stride",
                        "value_term_offsets",
                        "shared_term_offsets",
                        "item_term_offsets",
                        "evals",
                        "shared_terms",
                        "item_terms",
                    ],
                )?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &cpu,
                    "cpu.sumcheck_output_eval_family",
                    Some(&symbol),
                    &attrs,
                    &operands,
                    &["!cpu.field_value"],
                )?;
                insert_result_mapping(&mut value_map, op, operation, 0, 0)?;
            }
            "compute.sumcheck_output_claim" => {
                let operands = lowered_operands(op, &value_map)?;
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(op, &["stage", "relation", "count", "polynomial_evals"])?;
                let _operation = context.append_typed_op_with_owned_attrs(
                    &cpu,
                    "cpu.sumcheck_output_claim",
                    Some(&symbol),
                    &attrs,
                    &operands,
                    &[],
                )?;
            }
            "compute.opening_claim" => {
                let operands = lowered_operands(op, &value_map)?;
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(op, &["oracle", "domain", "point_arity", "claim_kind"])?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &cpu,
                    "cpu.opening_claim",
                    Some(&symbol),
                    &attrs,
                    &operands,
                    &["!cpu.opening_claim_type"],
                )?;
                insert_result_mapping(&mut value_map, op, operation, 0, 0)?;
            }
            "compute.opening_claim_equal" => {
                let operands = lowered_operands(op, &value_map)?;
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(op, &["mode"])?;
                let _operation = context.append_typed_op_with_owned_attrs(
                    &cpu,
                    "cpu.opening_claim_equal",
                    Some(&symbol),
                    &attrs,
                    &operands,
                    &[],
                )?;
            }
            "compute.opening_batch" => {
                let operands = lowered_operands(op, &value_map)?;
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(
                    op,
                    &["stage", "proof_slot", "policy", "count", "ordered_claims"],
                )?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &cpu,
                    "cpu.opening_batch",
                    Some(&symbol),
                    &attrs,
                    &operands,
                    &["!cpu.opening_batch_type"],
                )?;
                insert_result_mapping(&mut value_map, op, operation, 0, 0)?;
            }
            "compute.pcs_opening_claim" => {
                let operands = lowered_operands(op, &value_map)?;
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(op, &["oracle", "family", "domain", "point_arity"])?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &cpu,
                    "cpu.pcs_opening_claim",
                    Some(&symbol),
                    &attrs,
                    &operands,
                    &["!cpu.opening_claim_type"],
                )?;
                insert_result_mapping(&mut value_map, op, operation, 0, 0)?;
            }
            "compute.pcs_opening_batch" => {
                let operands = lowered_operands(op, &value_map)?;
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(op, &["proof_slot", "policy", "count", "ordered_claims"])?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &cpu,
                    "cpu.pcs_opening_batch",
                    Some(&symbol),
                    &attrs,
                    &operands,
                    &["!cpu.opening_batch_type"],
                )?;
                insert_result_mapping(&mut value_map, op, operation, 0, 0)?;
            }
            "compute.pcs_batch_open" | "compute.pcs_batch_verify" => {
                let target_op = match operation_name(op).as_str() {
                    "compute.pcs_batch_open" => "cpu.pcs_batch_open",
                    "compute.pcs_batch_verify" => "cpu.pcs_batch_verify",
                    _ => unreachable!(),
                };
                let operands = lowered_operands(op, &value_map)?;
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(op, &["pcs", "proof_slot", "transcript_label"])?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &cpu,
                    target_op,
                    Some(&symbol),
                    &attrs,
                    &operands,
                    &["!cpu.transcript_state", "!cpu.opening_proof_type"],
                )?;
                insert_result_mapping(&mut value_map, op, operation, 0, 0)?;
                insert_result_mapping(&mut value_map, op, operation, 1, 1)?;
            }
            _ => {}
        }
    }
    verify_module(&cpu)?;
    verify_cpu_schema(&cpu)?;
    Ok(cpu)
}

#[derive(Clone, Debug)]
struct ConcreteCommitmentAst {
    params: ParamsAst,
    oracle_buffers: BTreeMap<String, OracleBufferAst>,
    batch_plans: Vec<BatchPlanAst>,
    optional_plans: Vec<OptionalPlanAst>,
    transcript_steps: Vec<TranscriptStepAst>,
}

#[derive(Clone, Debug)]
struct ParamsAst {
    field: String,
    pcs: String,
    transcript: String,
    log_t: usize,
    log_k_chunk: usize,
    instruction_d: usize,
    bytecode_d: usize,
    ram_d: usize,
}

#[derive(Clone, Debug)]
struct DomainAst {
    num_vars: usize,
}

#[derive(Clone, Debug)]
struct OracleAst {
    domain: String,
    commit_domain: String,
    layout: String,
}

#[derive(Clone, Debug)]
struct OracleBufferAst {
    domain: String,
    num_vars: usize,
}

#[derive(Clone, Debug)]
struct OracleFamilyAst {
    oracles: Vec<String>,
    count: usize,
    domain: String,
}

#[derive(Clone, Debug)]
struct PublishedBatchAst {
    artifact: String,
    oracle_family: String,
    label: String,
}

#[derive(Clone, Debug)]
struct PublishedOptionalAst {
    artifact: String,
    oracle: String,
    label: String,
    skip_policy: String,
}

#[derive(Clone, Debug)]
struct BatchPlanAst {
    artifact: String,
    pcs: String,
    oracle_family: String,
    oracles: Vec<String>,
    label: String,
    domain: String,
    num_vars: usize,
    count: usize,
}

#[derive(Clone, Debug)]
struct OptionalPlanAst {
    artifact: String,
    pcs: String,
    oracle: String,
    label: String,
    domain: String,
    num_vars: usize,
    skip_policy: String,
}

#[derive(Clone, Debug)]
struct TranscriptStepAst {
    symbol: String,
    label: String,
    source: String,
    optional: bool,
}

fn analyze_concrete<P>(module: &BoltModule<'_, P>) -> Result<ConcreteCommitmentAst, MlirError>
where
    P: crate::ir::Phase,
{
    let mut params = None;
    let mut domains = BTreeMap::new();
    let mut oracles = BTreeMap::new();
    let mut families = BTreeMap::new();
    let mut published_batches = Vec::new();
    let mut published_optional = Vec::new();
    let mut pcs_by_artifact = BTreeMap::new();
    let mut transcript_steps = Vec::new();

    let mut operation = module.as_mlir_module().body().first_operation();
    while let Some(op) = operation {
        operation = op.next_in_block();
        match operation_name(op).as_str() {
            "protocol.params" => {
                params = Some(ParamsAst {
                    field: symbol_attr(op, "field")?,
                    pcs: symbol_attr(op, "pcs")?,
                    transcript: symbol_attr(op, "transcript")?,
                    log_t: int_attr(op, "log_t")?,
                    log_k_chunk: int_attr(op, "log_k_chunk")?,
                    instruction_d: int_attr(op, "instruction_d")?,
                    bytecode_d: int_attr(op, "bytecode_d")?,
                    ram_d: int_attr(op, "ram_d")?,
                });
            }
            "poly.domain" => {
                let _ = domains.insert(
                    string_attr(op, "sym_name")?,
                    DomainAst {
                        num_vars: int_attr(op, "log_size")?,
                    },
                );
            }
            "piop.oracle" => {
                let _ = oracles.insert(
                    string_attr(op, "sym_name")?,
                    OracleAst {
                        domain: symbol_attr(op, "domain")?,
                        commit_domain: symbol_attr(op, "commit_domain")?,
                        layout: string_attr(op, "layout")?,
                    },
                );
            }
            "piop.oracle_family" => {
                let _ = families.insert(
                    string_attr(op, "sym_name")?,
                    OracleFamilyAst {
                        oracles: symbol_array_attr(op, "ordered_oracles")?,
                        count: int_attr(op, "count")?,
                        domain: symbol_attr(op, "domain")?,
                    },
                );
            }
            "commit.publish_batch" => published_batches.push(PublishedBatchAst {
                artifact: string_attr(op, "sym_name")?,
                oracle_family: symbol_attr(op, "oracle_family")?,
                label: string_attr(op, "label")?,
            }),
            "commit.publish_optional" => published_optional.push(PublishedOptionalAst {
                artifact: string_attr(op, "sym_name")?,
                oracle: symbol_attr(op, "oracle")?,
                label: string_attr(op, "label")?,
                skip_policy: string_attr(op, "skip_policy")?,
            }),
            "pcs.commit_batch" => {
                let _ = pcs_by_artifact
                    .insert(pcs_commitment_artifact(op)?, symbol_attr(op, "scheme")?);
            }
            "transcript.absorb" | "transcript.absorb_optional" => {
                transcript_steps.push(TranscriptStepAst {
                    symbol: string_attr(op, "sym_name")?,
                    label: string_attr(op, "label")?,
                    source: transcript_artifact_source(op)?,
                    optional: operation_name(op) == "transcript.absorb_optional",
                });
            }
            _ => {}
        }
    }

    let params = params.ok_or_else(|| schema_error("missing protocol.params"))?;
    let mut oracle_buffers = BTreeMap::new();
    for (symbol, oracle) in &oracles {
        let buffer_domain = oracle_buffer_domain(oracle);
        let domain = domains.get(buffer_domain).ok_or_else(|| {
            schema_error(format!(
                "oracle @{symbol} references missing buffer domain @{buffer_domain}"
            ))
        })?;
        let _ = oracle_buffers.insert(
            symbol.clone(),
            OracleBufferAst {
                domain: buffer_domain.to_owned(),
                num_vars: domain.num_vars,
            },
        );
    }

    let mut batch_plans = Vec::new();
    for batch in published_batches {
        let family = families.get(&batch.oracle_family).ok_or_else(|| {
            schema_error(format!(
                "commitment artifact @{} references missing oracle family @{}",
                batch.artifact, batch.oracle_family
            ))
        })?;
        let domain = domains.get(&family.domain).ok_or_else(|| {
            schema_error(format!(
                "oracle family @{} references missing domain @{}",
                batch.oracle_family, family.domain
            ))
        })?;
        batch_plans.push(BatchPlanAst {
            pcs: pcs_by_artifact
                .get(&batch.artifact)
                .cloned()
                .unwrap_or_else(|| params.pcs.clone()),
            artifact: batch.artifact,
            oracle_family: batch.oracle_family,
            oracles: family.oracles.clone(),
            label: batch.label,
            domain: family.domain.clone(),
            num_vars: domain.num_vars,
            count: family.count,
        });
    }

    let mut optional_plans = Vec::new();
    for optional in published_optional {
        let oracle = oracles.get(&optional.oracle).ok_or_else(|| {
            schema_error(format!(
                "commitment artifact @{} references missing oracle @{}",
                optional.artifact, optional.oracle
            ))
        })?;
        let domain = domains.get(&oracle.commit_domain).ok_or_else(|| {
            schema_error(format!(
                "oracle @{} references missing commit domain @{}",
                optional.oracle, oracle.commit_domain
            ))
        })?;
        optional_plans.push(OptionalPlanAst {
            pcs: params.pcs.clone(),
            artifact: optional.artifact,
            oracle: optional.oracle,
            label: optional.label,
            domain: oracle.commit_domain.clone(),
            num_vars: domain.num_vars,
            skip_policy: optional.skip_policy,
        });
    }

    Ok(ConcreteCommitmentAst {
        params,
        oracle_buffers,
        batch_plans,
        optional_plans,
        transcript_steps,
    })
}

fn oracle_buffer_domain(oracle: &OracleAst) -> &str {
    if oracle.layout == "onehot_expanded" {
        &oracle.commit_domain
    } else {
        &oracle.domain
    }
}

fn append_oracle_buffer<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Compute>,
    role: &Role,
    params: &ParamsAst,
    oracle: &str,
    domain: &str,
    num_vars: usize,
) -> Result<Value<'c, 'a>, MlirError> {
    let symbol = format!("jolt.oracle.{oracle}.compute");
    match role {
        Role::Verifier => append_oracle_ref(context, module, &symbol, oracle, domain, num_vars),
        Role::Prover => {
            let recipe = oracle_recipe(oracle, params)?;
            let attrs = recipe.attrs(oracle, domain, num_vars, params);
            let operation = context.append_typed_op_with_owned_attrs(
                module,
                recipe.op_name(),
                Some(&symbol),
                &attrs,
                &[],
                &["!compute.oracle_buffer"],
            )?;
            first_result(operation, recipe.op_name())
        }
    }
}

fn append_optional_oracle_buffer<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Compute>,
    role: &Role,
    oracle: &str,
    domain: &str,
    num_vars: usize,
    skip_policy: &str,
) -> Result<Value<'c, 'a>, MlirError> {
    let symbol = format!("jolt.oracle.{oracle}.compute");
    match role {
        Role::Verifier => append_oracle_ref(context, module, &symbol, oracle, domain, num_vars),
        Role::Prover => {
            let operation = context.append_typed_op_with_owned_attrs(
                module,
                "compute.oracle_optional_advice",
                Some(&symbol),
                &[
                    ("oracle".to_owned(), symbol_ref(oracle)),
                    (
                        "source".to_owned(),
                        symbol_ref(&optional_advice_source(oracle)?),
                    ),
                    ("domain".to_owned(), symbol_ref(domain)),
                    ("num_vars".to_owned(), int_attr_source(num_vars)),
                    ("skip_policy".to_owned(), string_attr_source(skip_policy)),
                ],
                &[],
                &["!compute.oracle_buffer"],
            )?;
            first_result(operation, "compute.oracle_optional_advice")
        }
    }
}

fn append_oracle_ref<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Compute>,
    symbol: &str,
    oracle: &str,
    domain: &str,
    num_vars: usize,
) -> Result<Value<'c, 'a>, MlirError> {
    let operation = context.append_typed_op_with_owned_attrs(
        module,
        "compute.oracle_ref",
        Some(symbol),
        &[
            ("oracle".to_owned(), symbol_ref(oracle)),
            ("domain".to_owned(), symbol_ref(domain)),
            ("num_vars".to_owned(), int_attr_source(num_vars)),
        ],
        &[],
        &["!compute.oracle_buffer"],
    )?;
    first_result(operation, "compute.oracle_ref")
}

#[derive(Clone, Debug)]
enum OracleRecipe {
    DenseTrace {
        source: &'static str,
    },
    OneHotChunk {
        source: &'static str,
        chunk: usize,
        num_chunks: usize,
        padding: &'static str,
    },
}

impl OracleRecipe {
    fn op_name(&self) -> &'static str {
        match self {
            Self::DenseTrace { .. } => "compute.oracle_dense_trace",
            Self::OneHotChunk { .. } => "compute.oracle_one_hot_chunk",
        }
    }

    fn attrs(
        &self,
        oracle: &str,
        domain: &str,
        num_vars: usize,
        params: &ParamsAst,
    ) -> Vec<(String, String)> {
        let mut attrs = vec![
            ("oracle".to_owned(), symbol_ref(oracle)),
            ("domain".to_owned(), symbol_ref(domain)),
            ("num_vars".to_owned(), int_attr_source(num_vars)),
        ];
        match self {
            Self::DenseTrace { source } => {
                attrs.push(("source".to_owned(), symbol_ref(source)));
                attrs.push(("padding".to_owned(), string_attr_source("zero")));
            }
            Self::OneHotChunk {
                source,
                chunk,
                num_chunks,
                padding,
            } => {
                attrs.push(("source".to_owned(), symbol_ref(source)));
                attrs.push(("trace_num_vars".to_owned(), int_attr_source(params.log_t)));
                attrs.push(("chunk".to_owned(), int_attr_source(*chunk)));
                attrs.push(("num_chunks".to_owned(), int_attr_source(*num_chunks)));
                attrs.push(("chunk_bits".to_owned(), int_attr_source(params.log_k_chunk)));
                attrs.push(("padding".to_owned(), string_attr_source(padding)));
                attrs.push(("layout".to_owned(), string_attr_source("address_major")));
            }
        }
        attrs
    }
}

fn oracle_recipe(oracle: &str, params: &ParamsAst) -> Result<OracleRecipe, MlirError> {
    if oracle == "RdInc" {
        return Ok(OracleRecipe::DenseTrace {
            source: "trace.rd_inc",
        });
    }
    if oracle == "RamInc" {
        return Ok(OracleRecipe::DenseTrace {
            source: "trace.ram_inc",
        });
    }
    if let Some(index) = parse_indexed_oracle(oracle, "InstructionRa") {
        return Ok(OracleRecipe::OneHotChunk {
            source: "trace.instruction_keys",
            chunk: index,
            num_chunks: params.instruction_d,
            padding: "zero",
        });
    }
    if let Some(index) = parse_indexed_oracle(oracle, "RamRa") {
        return Ok(OracleRecipe::OneHotChunk {
            source: "trace.ram_addresses",
            chunk: index,
            num_chunks: params.ram_d,
            padding: "none",
        });
    }
    if let Some(index) = parse_indexed_oracle(oracle, "BytecodeRa") {
        return Ok(OracleRecipe::OneHotChunk {
            source: "trace.bytecode_indices",
            chunk: index,
            num_chunks: params.bytecode_d,
            padding: "zero",
        });
    }
    Err(schema_error(format!(
        "unsupported commitment oracle @{oracle}"
    )))
}

fn parse_indexed_oracle(oracle: &str, prefix: &str) -> Option<usize> {
    oracle.strip_prefix(prefix)?.strip_prefix('_')?.parse().ok()
}

fn optional_advice_source(oracle: &str) -> Result<String, MlirError> {
    match oracle {
        "UntrustedAdvice" => Ok("advice.untrusted".to_owned()),
        "TrustedAdvice" => Ok("advice.trusted".to_owned()),
        _ => Err(schema_error(format!(
            "unsupported optional advice oracle @{oracle}"
        ))),
    }
}

fn bool_attr(operation: OperationRef<'_, '_>, attr: &str) -> Result<bool, MlirError> {
    operation
        .attribute(attr)
        .map(|attribute| match attribute.to_string().as_str() {
            "true" => Some(true),
            "false" => Some(false),
            _ => None,
        })
        .ok()
        .flatten()
        .ok_or_else(|| {
            schema_error(format!(
                "{} attr `{attr}` is not a bool",
                operation_name(operation)
            ))
        })
}

fn operation_result_key(operation: OperationRef<'_, '_>) -> Result<String, MlirError> {
    operation_result_key_at(operation, 0)
}

fn operation_result_key_at(
    operation: OperationRef<'_, '_>,
    index: usize,
) -> Result<String, MlirError> {
    let result = operation.result(index).map_err(|_| {
        schema_error(format!(
            "{} requires result {index}",
            operation_name(operation)
        ))
    })?;
    result_key(result.owner(), result.result_number())
}

fn result_key(operation: OperationRef<'_, '_>, result_number: usize) -> Result<String, MlirError> {
    Ok(format!(
        "{}#{result_number}",
        string_attr(operation, "sym_name")?
    ))
}

fn operand_key(operation: OperationRef<'_, '_>, index: usize) -> Result<String, MlirError> {
    let operand = operation.operand(index).map_err(|_| {
        schema_error(format!(
            "{} requires operand {index}",
            operation_name(operation)
        ))
    })?;
    let owner = OperationResult::try_from(operand).map_err(|_| {
        schema_error(format!(
            "{} operand {index} must be an op result",
            operation_name(operation)
        ))
    })?;
    result_key(owner.owner(), owner.result_number()).map_err(|_| {
        schema_error(format!(
            "{} operand {index} owner missing sym_name",
            operation_name(operation)
        ))
    })
}

fn lowered_operands<'c, 'a>(
    operation: OperationRef<'_, '_>,
    value_map: &BTreeMap<String, Value<'c, 'a>>,
) -> Result<Vec<Value<'c, 'a>>, MlirError> {
    (0..operation.operand_count())
        .map(|index| {
            let key = operand_key(operation, index)?;
            value_map.get(&key).copied().ok_or_else(|| {
                schema_error(format!(
                    "{} operand {index} was not lowered",
                    operation_name(operation)
                ))
            })
        })
        .collect()
}

fn insert_result_mapping<'c, 'a>(
    value_map: &mut BTreeMap<String, Value<'c, 'a>>,
    source: OperationRef<'_, '_>,
    target: OperationRef<'c, 'a>,
    source_index: usize,
    target_index: usize,
) -> Result<(), MlirError> {
    let key = operation_result_key_at(source, source_index)?;
    let value = target.result(target_index).map(Into::into).map_err(|_| {
        schema_error(format!(
            "{} requires result {target_index}",
            operation_name(target)
        ))
    })?;
    let inserted = value_map.insert(key, value);
    debug_assert!(inserted.is_none());
    Ok(())
}

fn first_result<'c, 'a>(
    operation: OperationRef<'c, 'a>,
    operation_name: &str,
) -> Result<Value<'c, 'a>, MlirError> {
    operation
        .result(0)
        .map(Into::into)
        .map_err(|_| schema_error(format!("{operation_name} requires one result")))
}

fn pcs_commitment_artifact(operation: OperationRef<'_, '_>) -> Result<String, MlirError> {
    let artifact = operation.operand(0).map_err(|_| {
        schema_error(format!(
            "{} requires commitment artifact operand 0",
            operation_name(operation)
        ))
    })?;
    let owner = OperationResult::try_from(artifact).map_err(|_| {
        schema_error(format!(
            "{} commitment operand must be an op result",
            operation_name(operation)
        ))
    })?;
    string_attr(owner.owner(), "sym_name")
}

fn transcript_artifact_source(operation: OperationRef<'_, '_>) -> Result<String, MlirError> {
    if let Ok(source) = symbol_attr(operation, "source") {
        return Ok(source);
    }
    let artifact = operation.operand(1).map_err(|_| {
        schema_error(format!(
            "{} requires commitment artifact operand 1",
            operation_name(operation)
        ))
    })?;
    let owner = OperationResult::try_from(artifact).map_err(|_| {
        schema_error(format!(
            "{} artifact operand must be an op result",
            operation_name(operation)
        ))
    })?;
    string_attr(owner.owner(), "sym_name")
}

fn schema_error(message: impl Into<String>) -> MlirError {
    let error = SchemaError::new(message);
    error.into()
}

fn symbol_ref(value: &str) -> String {
    format!("@{value}")
}

fn string_attr_source(value: &str) -> String {
    format!("{value:?}")
}

fn symbol_array_attr_source(values: &[String]) -> String {
    let values = values
        .iter()
        .map(|value| format!("@{value}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{values}]")
}

fn int_attr_source(value: usize) -> String {
    format!("{value} : i64")
}

fn bool_attr_source(value: bool) -> &'static str {
    if value {
        "true"
    } else {
        "false"
    }
}
