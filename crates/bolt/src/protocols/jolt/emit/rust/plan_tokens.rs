use std::collections::BTreeSet;

use crate::emit::rust::EmitError;
use crate::ir::Role;

pub(super) fn role_program_step_kind_expr(
    stage_type_prefix: &str,
    role: &Role,
    kind: &str,
) -> Result<String, EmitError> {
    if role == &Role::Prover {
        return Ok(format!("{kind:?}"));
    }
    program_step_kind_expr(stage_type_prefix, kind)
}

pub(super) fn role_transcript_squeeze_kind_expr(
    stage_type_prefix: &str,
    role: &Role,
    kind: &str,
) -> Result<String, EmitError> {
    if role == &Role::Prover {
        return Ok(format!("{kind:?}"));
    }
    transcript_squeeze_kind_expr(stage_type_prefix, kind)
}

pub(super) fn role_claim_kind_expr(
    stage_type_prefix: &str,
    role: &Role,
    kind: &str,
) -> Result<String, EmitError> {
    if role == &Role::Prover {
        return Ok(format!("{kind:?}"));
    }
    claim_kind_expr(stage_type_prefix, kind)
}

pub(super) fn role_field_expr_kind_expr(
    stage_type_prefix: &str,
    role: &Role,
    formula: &str,
) -> Result<String, EmitError> {
    if role == &Role::Prover {
        return Ok(format!("{formula:?}"));
    }
    field_expr_kind_expr(stage_type_prefix, formula)
}

pub(super) fn rust_str_slice_expr(values: &[String]) -> String {
    if values.is_empty() {
        return "&[]".to_owned();
    }
    let values = values
        .iter()
        .map(|value| format!("{value:?}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("&[{values}]")
}

pub(super) fn rust_str(value: &str) -> String {
    format!("{value:?}")
}

pub(super) fn rust_option_str(value: Option<&str>) -> String {
    value.map_or_else(
        || "None".to_owned(),
        |value| format!("Some({})", rust_str(value)),
    )
}

pub(super) fn emit_str_array(name: &str, values: &[String]) -> String {
    if values.is_empty() {
        return format!("pub const {name}: &[&str] = &[];\n\n");
    }
    if let [value] = values {
        return format!("pub const {name}: &[&str] = &[{}];\n\n", rust_str(value));
    }
    let entries = values
        .iter()
        .map(|value| format!("    {},", rust_str(value)))
        .collect::<Vec<_>>()
        .join("\n");
    format!("pub const {name}: &[&str] = &[\n{entries}\n];\n\n")
}

pub(super) fn emit_usize_array(name: &str, values: &[usize]) -> String {
    let entries = values
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    format!("pub const {name}: &[usize] = &[{entries}];\n\n")
}

pub(super) fn intern_str_array(
    source: &mut String,
    arrays: &mut Vec<(Vec<String>, String)>,
    name_prefix: &str,
    values: &[String],
) -> String {
    if let Some((_, name)) = arrays
        .iter()
        .find(|(existing, _)| existing.as_slice() == values)
    {
        return name.clone();
    }
    let name = format!("{name_prefix}_{}", arrays.len());
    source.push_str(&emit_str_array(&name, values));
    arrays.push((values.to_vec(), name.clone()));
    name
}

pub(super) fn require_supported_symbol(
    kind: &str,
    actual: &str,
    expected: &str,
) -> Result<(), EmitError> {
    if actual == expected {
        Ok(())
    } else {
        Err(EmitError::new(format!(
            "unsupported {kind} @{actual}; expected @{expected}"
        )))
    }
}

pub(super) fn verify_count(
    kind: &str,
    symbol: &str,
    expected: usize,
    actual: usize,
) -> Result<(), EmitError> {
    if expected == actual {
        Ok(())
    } else {
        Err(EmitError::new(format!(
            "{kind} @{symbol} count mismatch: expected {expected}, got {actual}"
        )))
    }
}

pub(super) fn symbols<'a>(values: impl Iterator<Item = &'a String>) -> BTreeSet<String> {
    values.cloned().collect()
}

pub(super) fn role_relation_kind_expr(
    stage_type_prefix: &str,
    role: &Role,
    relation: &str,
) -> Result<String, EmitError> {
    if role == &Role::Prover {
        return Ok(format!("{relation:?}"));
    }
    relation_kind_expr(stage_type_prefix, relation)
}

pub(super) fn role_optional_relation_kind_expr(
    stage_type_prefix: &str,
    role: &Role,
    relation: Option<&str>,
) -> Result<String, EmitError> {
    relation
        .map(|relation| role_relation_kind_expr(stage_type_prefix, role, relation))
        .transpose()
        .map(|relation| {
            relation.map_or_else(|| "None".to_owned(), |relation| format!("Some({relation})"))
        })
}

pub(super) fn role_opening_equality_mode_expr(
    stage_type_prefix: &str,
    role: &Role,
    mode: &str,
) -> Result<String, EmitError> {
    if role == &Role::Prover {
        return Ok(format!("{mode:?}"));
    }
    opening_equality_mode_expr(stage_type_prefix, mode)
}

fn program_step_kind_expr(stage_type_prefix: &str, kind: &str) -> Result<String, EmitError> {
    let variant = match kind {
        "transcript_squeeze" => "TranscriptSqueeze",
        "transcript_absorb_bytes" => "TranscriptAbsorbBytes",
        "sumcheck_driver" => "SumcheckDriver",
        _ => return Err(unsupported("program step kind", kind)),
    };
    Ok(format!("{stage_type_prefix}ProgramStepKind::{variant}"))
}

fn transcript_squeeze_kind_expr(stage_type_prefix: &str, kind: &str) -> Result<String, EmitError> {
    let variant = match kind {
        "challenge_scalar" => "ChallengeScalar",
        "challenge_vector" => "ChallengeVector",
        "scalar" => "Scalar",
        _ => return Err(unsupported("transcript squeeze kind", kind)),
    };
    Ok(format!(
        "{stage_type_prefix}TranscriptSqueezeKind::{variant}"
    ))
}

pub(super) fn claim_kind_expr(stage_type_prefix: &str, kind: &str) -> Result<String, EmitError> {
    let variant = match kind {
        "committed" => "Committed",
        "virtual" => "Virtual",
        _ => return Err(unsupported("opening claim kind", kind)),
    };
    Ok(format!("{stage_type_prefix}ClaimKind::{variant}"))
}

fn relation_kind_expr(stage_type_prefix: &str, relation: &str) -> Result<String, EmitError> {
    let variant = match relation {
        "jolt.stage1.outer.uniskip" => "Stage1OuterUniskip",
        "jolt.stage1.outer.remaining" => "Stage1OuterRemaining",
        "jolt.stage2.product_virtual.uniskip" => "Stage2ProductVirtualUniskip",
        "jolt.stage2.ram.read_write" => "Stage2RamReadWrite",
        "jolt.stage2.product_virtual.remainder" => "Stage2ProductVirtualRemainder",
        "jolt.stage2.instruction_lookup.claim_reduction" => "Stage2InstructionLookupClaimReduction",
        "jolt.stage2.ram.raf_evaluation" => "Stage2RamRafEvaluation",
        "jolt.stage2.ram.output_check" => "Stage2RamOutputCheck",
        "jolt.stage2.batched" => "Stage2Batched",
        "jolt.stage3.spartan_shift" => "Stage3SpartanShift",
        "jolt.stage3.instruction_input" => "Stage3InstructionInput",
        "jolt.stage3.registers_claim_reduction" => "Stage3RegistersClaimReduction",
        "jolt.stage3.batched" => "Stage3Batched",
        "jolt.stage4.registers_read_write" => "Stage4RegistersReadWrite",
        "jolt.stage4.ram_val_check" => "Stage4RamValCheck",
        "jolt.stage4.batched" => "Stage4Batched",
        "jolt.stage5.instruction_read_raf" => "Stage5InstructionReadRaf",
        "jolt.stage5.ram_ra_claim_reduction" => "Stage5RamRaClaimReduction",
        "jolt.stage5.registers_val_evaluation" => "Stage5RegistersValEvaluation",
        "jolt.stage5.batched" => "Stage5Batched",
        "jolt.stage6.bytecode_read_raf" => "Stage6BytecodeReadRaf",
        "jolt.stage6.booleanity" => "Stage6Booleanity",
        "jolt.stage6.hamming_booleanity" => "Stage6HammingBooleanity",
        "jolt.stage6.ram_ra_virtual" => "Stage6RamRaVirtual",
        "jolt.stage6.instruction_ra_virtual" => "Stage6InstructionRaVirtual",
        "jolt.stage6.inc_claim_reduction" => "Stage6IncClaimReduction",
        "jolt.stage6.batched" => "Stage6Batched",
        "jolt.stage7.hamming_weight_claim_reduction" => "Stage7HammingWeightClaimReduction",
        "jolt.stage7.batched" => "Stage7Batched",
        _ => return Err(unsupported("relation", relation)),
    };
    Ok(format!("{stage_type_prefix}RelationKind::{variant}"))
}

fn field_expr_kind_expr(stage_type_prefix: &str, formula: &str) -> Result<String, EmitError> {
    let variant = match formula {
        "opening_eval" => "OpeningEval".to_owned(),
        "field.add" => "Add".to_owned(),
        "field.sub" => "Sub".to_owned(),
        "field.mul" => "Mul".to_owned(),
        "field.neg" => "Neg".to_owned(),
        formula if formula.starts_with("field.pow:") => {
            let Some(exponent) = formula.strip_prefix("field.pow:") else {
                return Err(unsupported("field expression formula", formula));
            };
            let exponent = exponent
                .parse::<usize>()
                .map_err(|_| unsupported("field expression formula", formula))?;
            format!("Pow({exponent})")
        }
        formula if formula.starts_with("poly.lagrange_basis_eval:") => {
            let Some(spec) = formula.strip_prefix("poly.lagrange_basis_eval:") else {
                return Err(unsupported("field expression formula", formula));
            };
            let parts = spec.split(':').collect::<Vec<_>>();
            if parts.len() != 3 {
                return Err(unsupported("field expression formula", formula));
            }
            let domain_start = parts[0]
                .parse::<i64>()
                .map_err(|_| unsupported("field expression formula", formula))?;
            let domain_size = parts[1]
                .parse::<usize>()
                .map_err(|_| unsupported("field expression formula", formula))?;
            let index = parts[2]
                .parse::<usize>()
                .map_err(|_| unsupported("field expression formula", formula))?;
            format!("LagrangeBasisEval({domain_start}, {domain_size}, {index})")
        }
        _ => return Err(unsupported("field expression formula", formula)),
    };
    Ok(format!("{stage_type_prefix}FieldExprKind::{variant}"))
}

pub(super) fn pcs_proof_mode_expr(
    stage_type_prefix: &str,
    mode: &str,
) -> Result<String, EmitError> {
    let variant = match mode {
        "open" => "Open",
        "verify" => "Verify",
        _ => return Err(unsupported("PCS proof mode", mode)),
    };
    Ok(format!("{stage_type_prefix}PcsProofMode::{variant}"))
}

fn opening_equality_mode_expr(stage_type_prefix: &str, mode: &str) -> Result<String, EmitError> {
    let variant = match mode {
        "point_and_eval" => "PointAndEval",
        _ => return Err(unsupported("opening equality mode", mode)),
    };
    Ok(format!("{stage_type_prefix}OpeningEqualityMode::{variant}"))
}

fn unsupported(kind: &str, value: &str) -> EmitError {
    EmitError::new(format!("unsupported {kind} `{value}`"))
}
