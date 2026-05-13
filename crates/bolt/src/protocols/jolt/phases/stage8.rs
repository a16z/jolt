use melior::ir::operation::{OperationLike, OperationRef};
use melior::ir::Value;

use crate::ir::{BoltModule, Compute, Party, Protocol};
use crate::mlir::{verify_module, MeliorContext, MlirError};
use crate::schema::{verify_protocol_schema, SchemaError};

use super::super::oracles;
use super::super::params::JoltProtocolParams;
use super::lowering::lower_party_to_compute;

const EVALUATION_POINT_SOURCE_SYMBOL: &str = "stage8.evaluation.point_source";

pub fn build_stage8_protocol<'c>(
    context: &'c MeliorContext,
    params: &JoltProtocolParams,
) -> Result<BoltModule<'c, Protocol>, MlirError> {
    let module = context.new_module::<Protocol>("jolt.stage8", None);
    oracles::append_foundation_ops(context, &module, params)?;
    oracles::append_committed_oracles(context, &module, params)?;
    context.append_op_with_owned_attrs(
        &module,
        "protocol.params",
        Some("jolt.params"),
        &params.attrs(),
    )?;
    context.append_op(
        &module,
        "protocol.boundary",
        Some("jolt.stage8"),
        &[("roles", r#"["prover", "verifier"]"#)],
    )?;

    let fs = context.append_typed_op(
        &module,
        "transcript.state",
        Some("fs_after_stage7"),
        &[("scheme", "@blake2b_transcript")],
        &[],
        &["!transcript.state_type"],
    )?;
    let state = result(fs, 0, "transcript.state")?;
    let _stage = context.append_typed_op(
        &module,
        "piop.stage",
        Some("stage8"),
        &[
            ("name", r#""evaluation_proof""#),
            ("order", "8 : i64"),
            ("roles", r#"["prover", "verifier"]"#),
        ],
        &[],
        &["!piop.stage_type"],
    )?;

    let _point_source = append_opening_input(
        context,
        &module,
        Stage8OpeningInputSpec {
            symbol: EVALUATION_POINT_SOURCE_SYMBOL,
            source_stage: "stage7",
            source_claim: "stage7.input.stage6.booleanity.InstructionRa_0",
            oracle: "InstructionRa_0",
            point_arity: params.log_t + params.log_k_chunk,
        },
    )?;
    let mut claims = Vec::new();
    let mut claim_symbols = Vec::new();
    append_evaluation_claim(
        context,
        &module,
        params,
        &mut claims,
        &mut claim_symbols,
        Stage8EvaluationClaimSpec {
            oracle: "RamInc",
            source_stage: "stage6",
            source_claim: "stage6.inc_claim_reduction.eval.RamInc",
        },
    )?;
    append_evaluation_claim(
        context,
        &module,
        params,
        &mut claims,
        &mut claim_symbols,
        Stage8EvaluationClaimSpec {
            oracle: "RdInc",
            source_stage: "stage6",
            source_claim: "stage6.inc_claim_reduction.eval.RdInc",
        },
    )?;
    for index in 0..params.instruction_d {
        append_evaluation_claim(
            context,
            &module,
            params,
            &mut claims,
            &mut claim_symbols,
            Stage8EvaluationClaimSpec {
                oracle: &format!("InstructionRa_{index}"),
                source_stage: "stage7",
                source_claim: &format!(
                    "stage7.hamming_weight_claim_reduction.eval.InstructionRa_{index}"
                ),
            },
        )?;
    }
    for index in 0..params.bytecode_d {
        append_evaluation_claim(
            context,
            &module,
            params,
            &mut claims,
            &mut claim_symbols,
            Stage8EvaluationClaimSpec {
                oracle: &format!("BytecodeRa_{index}"),
                source_stage: "stage7",
                source_claim: &format!(
                    "stage7.hamming_weight_claim_reduction.eval.BytecodeRa_{index}"
                ),
            },
        )?;
    }
    for index in 0..params.ram_d {
        append_evaluation_claim(
            context,
            &module,
            params,
            &mut claims,
            &mut claim_symbols,
            Stage8EvaluationClaimSpec {
                oracle: &format!("RamRa_{index}"),
                source_stage: "stage7",
                source_claim: &format!("stage7.hamming_weight_claim_reduction.eval.RamRa_{index}"),
            },
        )?;
    }

    let opening_batch = context.append_typed_op(
        &module,
        "pcs.opening_batch",
        Some("stage8.evaluation.openings"),
        &[
            ("proof_slot", "@stage8.evaluation"),
            ("policy", r#""jolt_stage8_joint_rlc""#),
            ("count", &int_attr(claims.len())),
            ("ordered_claims", &symbol_array_attr(&claim_symbols)),
        ],
        &claims,
        &["!pcs.opening_batch_type"],
    )?;
    let opening_batch = result(opening_batch, 0, "pcs.opening_batch")?;
    let _state = context.append_typed_op(
        &module,
        "pcs.batch_open",
        Some("stage8.evaluation.proof"),
        &[
            ("pcs", "@dory"),
            ("proof_slot", "@stage8.evaluation"),
            ("transcript_label", r#""rlc_claims""#),
        ],
        &[state, opening_batch],
        &["!transcript.state_type", "!pcs.opening_proof_type"],
    )?;

    verify_module(&module)?;
    verify_protocol_schema(&module)?;
    Ok(module)
}

pub fn lower_stage8_to_compute<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Party>,
) -> Result<BoltModule<'c, Compute>, MlirError> {
    lower_party_to_compute(context, module, "jolt.stage8", "jolt.stage8", "stage8")
}

fn append_evaluation_claim<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    params: &JoltProtocolParams,
    claims: &mut Vec<Value<'c, 'a>>,
    claim_symbols: &mut Vec<String>,
    spec: Stage8EvaluationClaimSpec<'_>,
) -> Result<(), MlirError> {
    let input_symbol = format!("stage8.input.{}.{}", spec.source_stage, spec.oracle);
    let opening_input = append_opening_input(
        context,
        module,
        Stage8OpeningInputSpec {
            symbol: &input_symbol,
            source_stage: spec.source_stage,
            source_claim: spec.source_claim,
            oracle: spec.oracle,
            point_arity: params.log_t + params.log_k_chunk,
        },
    )?;
    let opening_symbol = format!("stage8.evaluation.opening.{}", spec.oracle);
    let opening = context.append_typed_op(
        module,
        "pcs.opening_claim",
        Some(&opening_symbol),
        &[
            ("oracle", &format!("@{}", spec.oracle)),
            ("family", "@jolt.main_witness_polys"),
            ("domain", "@jolt.main_witness_commit_domain"),
            ("point_arity", &int_attr(params.log_t + params.log_k_chunk)),
        ],
        &[opening_input.point, opening_input.eval],
        &["!pcs.opening_claim_type"],
    )?;
    claims.push(result(opening, 0, "pcs.opening_claim")?);
    claim_symbols.push(opening_symbol);
    Ok(())
}

fn append_opening_input<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    spec: Stage8OpeningInputSpec<'_>,
) -> Result<Stage8OpeningInput<'c, 'a>, MlirError> {
    let op = context.append_typed_op(
        module,
        "piop.opening_input",
        Some(spec.symbol),
        &[
            ("source_stage", &format!("@{}", spec.source_stage)),
            ("source_claim", &format!("@{}", spec.source_claim)),
            ("oracle", &format!("@{}", spec.oracle)),
            ("domain", "@jolt.main_witness_commit_domain"),
            ("point_arity", &int_attr(spec.point_arity)),
            ("claim_kind", r#""committed""#),
        ],
        &[],
        &["!poly.point", "!field.scalar", "!piop.opening_claim_type"],
    )?;
    Ok(Stage8OpeningInput {
        point: result(op, 0, "piop.opening_input")?,
        eval: result(op, 1, "piop.opening_input")?,
    })
}

#[derive(Clone, Copy)]
struct Stage8OpeningInputSpec<'a> {
    symbol: &'a str,
    source_stage: &'a str,
    source_claim: &'a str,
    oracle: &'a str,
    point_arity: usize,
}

#[derive(Clone, Copy)]
struct Stage8EvaluationClaimSpec<'a> {
    oracle: &'a str,
    source_stage: &'a str,
    source_claim: &'a str,
}

struct Stage8OpeningInput<'c, 'a> {
    point: Value<'c, 'a>,
    eval: Value<'c, 'a>,
}

fn result<'c, 'a>(
    operation: OperationRef<'c, 'a>,
    index: usize,
    op_name: &str,
) -> Result<Value<'c, 'a>, MlirError> {
    operation.result(index).map(Into::into).map_err(|_| {
        schema_error(format!(
            "{op_name} requires result {index}, got {} results",
            operation.result_count()
        ))
    })
}

fn symbol_array_attr(symbols: &[String]) -> String {
    let symbols = symbols
        .iter()
        .map(|symbol| format!("@{symbol}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{symbols}]")
}

fn int_attr(value: usize) -> String {
    format!("{value} : i64")
}

fn schema_error(message: impl Into<String>) -> MlirError {
    SchemaError::new(message).into()
}
