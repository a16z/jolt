use std::collections::BTreeSet;
use std::error::Error;
use std::fmt::{self, Display, Formatter};

use melior::ir::block::BlockLike;
use melior::ir::operation::OperationLike;
use melior::ir::operation::OperationResult;
use melior::ir::{Attribute, OperationRef};

use crate::ir::{
    string_attribute_value, symbol_attribute_value, BoltModule, Compute, Concrete, Cpu, Party,
    Protocol, Role,
};
use crate::mlir::MlirError;
use crate::pass::{verify_concrete_transcript, VerifyError};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SchemaError {
    message: String,
}

impl SchemaError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for SchemaError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for SchemaError {}

impl From<SchemaError> for MlirError {
    fn from(error: SchemaError) -> Self {
        Self::Schema {
            message: error.to_string(),
        }
    }
}

impl From<VerifyError> for SchemaError {
    fn from(error: VerifyError) -> Self {
        Self::new(error.to_string())
    }
}

pub fn verify_protocol_schema(module: &BoltModule<'_, Protocol>) -> Result<(), SchemaError> {
    verify_schema(module, ModulePhase::Protocol)
}

pub fn verify_concrete_schema(module: &BoltModule<'_, Concrete>) -> Result<(), SchemaError> {
    verify_schema(module, ModulePhase::Concrete)?;
    verify_concrete_transcript(module)?;
    Ok(())
}

pub fn verify_party_schema(module: &BoltModule<'_, Party>) -> Result<(), SchemaError> {
    verify_schema(module, ModulePhase::Party)?;
    verify_concrete_transcript(module)?;
    Ok(())
}

pub fn verify_compute_schema(module: &BoltModule<'_, Compute>) -> Result<(), SchemaError> {
    verify_schema(module, ModulePhase::Compute)
}

pub fn verify_cpu_schema(module: &BoltModule<'_, Cpu>) -> Result<(), SchemaError> {
    verify_schema(module, ModulePhase::Cpu)
}

#[derive(Clone, Copy)]
enum ModulePhase {
    Protocol,
    Concrete,
    Party,
    Compute,
    Cpu,
}

fn verify_schema<P>(module: &BoltModule<'_, P>, phase: ModulePhase) -> Result<(), SchemaError>
where
    P: crate::ir::Phase,
{
    let phase_attr = module
        .as_mlir_module()
        .as_operation()
        .attribute("bolt.phase")
        .ok()
        .and_then(string_attribute_value)
        .ok_or_else(|| SchemaError::new("module missing required attr `bolt.phase`"))?;
    if phase_attr != P::NAME {
        return Err(SchemaError::new(format!(
            "module phase `{phase_attr}` does not match expected `{}`",
            P::NAME
        )));
    }

    let mut kernel_symbols = BTreeSet::new();
    let mut kernel_refs = Vec::new();
    let role = module.role();
    let mut operation = module.as_mlir_module().body().first_operation();
    while let Some(op) = operation {
        operation = op.next_in_block();
        validate_op(op, phase)?;
        if matches!(role, Some(Role::Verifier))
            && matches!(phase, ModulePhase::Compute | ModulePhase::Cpu)
        {
            validate_verifier_lowering_op(op)?;
        }
        match operation_name(op).as_str() {
            "compute.kernel" | "cpu.kernel" => {
                let _ = kernel_symbols.insert(string_attr(op, "sym_name")?);
            }
            "compute.sumcheck_kernel_claim"
            | "compute.sumcheck_kernel_driver"
            | "cpu.sumcheck_claim"
            | "cpu.sumcheck_driver" => {
                kernel_refs.push(symbol_attr(op, "kernel")?);
            }
            _ => {}
        }
    }

    if matches!(phase, ModulePhase::Compute | ModulePhase::Cpu) {
        for kernel in kernel_refs {
            if !kernel_symbols.contains(&kernel) {
                return Err(SchemaError::new(format!(
                    "kernel reference @{kernel} has no matching kernel definition"
                )));
            }
        }
    }

    Ok(())
}

fn validate_verifier_lowering_op(operation: OperationRef<'_, '_>) -> Result<(), SchemaError> {
    let name = operation_name(operation);
    match name.as_str() {
        "compute.kernel"
        | "compute.sumcheck_claim"
        | "compute.sumcheck_driver"
        | "compute.sumcheck_kernel_claim"
        | "compute.sumcheck_kernel_driver"
        | "compute.generate_oracle"
        | "compute.generate_oracle_family"
        | "cpu.kernel"
        | "cpu.sumcheck_claim"
        | "cpu.sumcheck_driver" => Err(SchemaError::new(format!(
            "verifier lowering must use verifier-specific ops, got `{name}`"
        ))),
        _ => Ok(()),
    }
}

fn validate_op(operation: OperationRef<'_, '_>, _phase: ModulePhase) -> Result<(), SchemaError> {
    let name = operation_name(operation);
    match name.as_str() {
        "field.define" => require_attrs(operation, &["sym_name", "modulus_bits", "role"]),
        "field.constant" => {
            require_attrs(operation, &["sym_name", "field", "value"])?;
            require_shape(operation, 0, 1)
        }
        "field.challenge_extract" => {
            require_attrs(operation, &["sym_name", "source", "index"])?;
            require_shape(operation, 1, 1)
        }
        "field.expr" => {
            require_attrs(operation, &["sym_name", "kind", "formula", "operands"])?;
            require_min_shape(operation, 0, 1)
        }
        "hash.function" => require_attrs(operation, &["sym_name", "algorithm"]),
        "transcript.scheme" => require_attrs(operation, &["sym_name", "hash"]),
        "pcs.scheme" => require_attrs(operation, &["sym_name", "field"]),
        "poly.domain" => require_attrs(operation, &["sym_name", "field", "log_size"]),
        "poly.point_slice" => {
            require_attrs(operation, &["sym_name", "source", "offset", "length"])?;
            require_shape(operation, 1, 1)
        }
        "poly.point_concat" => {
            require_attrs(operation, &["sym_name", "layout", "arity"])?;
            require_min_shape(operation, 1, 1)
        }
        "protocol.params" => require_attrs(operation, &["sym_name", "field", "pcs", "transcript"]),
        "protocol.boundary" => require_attrs(operation, &["sym_name", "roles"]),
        "piop.oracle" => require_attrs(
            operation,
            &[
                "sym_name",
                "field",
                "domain",
                "commit_domain",
                "visibility",
                "layout",
            ],
        ),
        "piop.oracle_family" => require_attrs(
            operation,
            &[
                "sym_name",
                "ordered_oracles",
                "visibility",
                "count",
                "domain",
            ],
        ),
        "commit.publish_batch" => {
            require_attrs(operation, &["sym_name", "oracle_family", "label"])?;
            require_shape(operation, 0, 1)
        }
        "commit.publish_optional" => {
            require_attrs(operation, &["sym_name", "oracle", "label", "skip_policy"])?;
            require_shape(operation, 0, 1)
        }
        "pcs.commit_batch" => {
            require_attrs(operation, &["sym_name", "scheme"])?;
            require_shape(operation, 1, 0)
        }
        "transcript.absorb" | "transcript.absorb_optional" => {
            require_attrs(operation, &["sym_name", "label"])?;
            require_shape(operation, 2, 1)
        }
        "transcript.squeeze" => {
            require_attrs(operation, &["sym_name", "label", "kind", "count"])?;
            require_shape(operation, 1, 2)
        }
        "transcript.state" => {
            require_attrs(operation, &["sym_name", "scheme"])?;
            require_shape(operation, 0, 1)
        }
        "piop.stage" => {
            require_attrs(operation, &["sym_name", "name", "order", "roles"])?;
            require_shape(operation, 0, 1)
        }
        "piop.relation" => require_attrs(
            operation,
            &[
                "sym_name",
                "kind",
                "domain",
                "num_rounds",
                "degree",
                "output_count",
            ],
        ),
        "piop.sumcheck_claim" => {
            require_attrs(
                operation,
                &[
                    "sym_name",
                    "stage",
                    "domain",
                    "num_rounds",
                    "degree",
                    "claim",
                    "relation",
                ],
            )?;
            require_min_shape(operation, 1, 1)
        }
        "piop.opening_input" => {
            require_attrs(
                operation,
                &[
                    "sym_name",
                    "source_stage",
                    "source_claim",
                    "oracle",
                    "domain",
                    "point_arity",
                    "claim_kind",
                ],
            )?;
            require_shape(operation, 0, 3)
        }
        "piop.sumcheck_batch" => {
            require_attrs(
                operation,
                &[
                    "sym_name",
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
            require_min_shape(operation, 1, 1)?;
            require_counted_operands(operation, 1, "ordered_claims")
        }
        "piop.sumcheck" => {
            require_attrs(
                operation,
                &[
                    "sym_name",
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
            require_shape(operation, 2, 4)
        }
        "piop.sumcheck_eval" => {
            require_attrs(
                operation,
                &["sym_name", "source", "name", "index", "oracle"],
            )?;
            require_shape(operation, 1, 1)
        }
        "piop.sumcheck_instance_result" => {
            require_attrs(
                operation,
                &[
                    "sym_name",
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
            require_shape(operation, 2, 2)
        }
        "piop.opening_claim" => {
            require_attrs(
                operation,
                &["sym_name", "oracle", "domain", "point_arity", "claim_kind"],
            )?;
            require_shape(operation, 2, 1)
        }
        "piop.opening_batch" => {
            require_attrs(
                operation,
                &[
                    "sym_name",
                    "stage",
                    "proof_slot",
                    "policy",
                    "count",
                    "ordered_claims",
                ],
            )?;
            require_min_shape(operation, 0, 1)?;
            require_counted_operands(operation, 0, "ordered_claims")
        }
        "party.function" => require_attrs(operation, &["sym_name", "source", "role"]),
        "compute.params" => require_attrs(operation, &["sym_name", "field", "pcs", "transcript"]),
        "compute.function" => require_attrs(operation, &["sym_name", "source"]),
        "compute.relation" => require_attrs(
            operation,
            &[
                "sym_name",
                "kind",
                "domain",
                "num_rounds",
                "degree",
                "output_count",
            ],
        ),
        "compute.kernel" => require_attrs(
            operation,
            &["sym_name", "relation", "kind", "backend", "abi"],
        ),
        "compute.oracle_dense_trace" => {
            require_attrs(
                operation,
                &[
                    "sym_name", "oracle", "source", "domain", "num_vars", "padding",
                ],
            )?;
            require_shape(operation, 0, 1)
        }
        "compute.oracle_one_hot_chunk" => {
            require_attrs(
                operation,
                &[
                    "sym_name",
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
                ],
            )?;
            require_shape(operation, 0, 1)
        }
        "compute.oracle_optional_advice" => {
            require_attrs(
                operation,
                &[
                    "sym_name",
                    "oracle",
                    "source",
                    "domain",
                    "num_vars",
                    "skip_policy",
                ],
            )?;
            require_shape(operation, 0, 1)
        }
        "compute.oracle_ref" => {
            require_attrs(operation, &["sym_name", "oracle", "domain", "num_vars"])?;
            require_shape(operation, 0, 1)
        }
        "compute.oracle_family_init" => {
            require_attrs(operation, &["sym_name", "family", "count"])?;
            require_shape(operation, 0, 1)
        }
        "compute.oracle_family_append" => {
            require_attrs(operation, &["sym_name", "family", "oracle", "index"])?;
            require_shape(operation, 2, 1)
        }
        "compute.pcs_commit_batch" | "compute.pcs_receive_batch" => {
            require_attrs(
                operation,
                &[
                    "sym_name",
                    "artifact",
                    "pcs",
                    "oracle_family",
                    "ordered_oracles",
                    "label",
                    "domain",
                    "num_vars",
                    "count",
                ],
            )?;
            require_shape(operation, 1, 1)
        }
        "compute.pcs_commit_optional" | "compute.pcs_receive_optional" => {
            require_attrs(
                operation,
                &[
                    "sym_name",
                    "artifact",
                    "pcs",
                    "oracle",
                    "label",
                    "domain",
                    "num_vars",
                    "skip_policy",
                ],
            )?;
            require_shape(operation, 1, 1)
        }
        "compute.transcript_init" => {
            require_attrs(operation, &["sym_name", "scheme"])?;
            require_shape(operation, 0, 1)
        }
        "compute.transcript_absorb" => {
            require_attrs(operation, &["sym_name", "label", "optional"])?;
            require_shape(operation, 2, 1)
        }
        "compute.transcript_squeeze" => {
            require_attrs(operation, &["sym_name", "label", "kind", "count"])?;
            require_shape(operation, 1, 2)
        }
        "compute.opening_input" => {
            require_attrs(
                operation,
                &[
                    "sym_name",
                    "source_stage",
                    "source_claim",
                    "oracle",
                    "domain",
                    "point_arity",
                    "claim_kind",
                ],
            )?;
            require_shape(operation, 0, 3)
        }
        "compute.point_slice" => {
            require_attrs(operation, &["sym_name", "source", "offset", "length"])?;
            require_shape(operation, 1, 1)
        }
        "compute.point_concat" => {
            require_attrs(operation, &["sym_name", "layout", "arity"])?;
            require_min_shape(operation, 1, 1)
        }
        "compute.field_constant" => {
            require_attrs(operation, &["sym_name", "field", "value"])?;
            require_shape(operation, 0, 1)
        }
        "compute.challenge_extract" => {
            require_attrs(operation, &["sym_name", "source", "index"])?;
            require_shape(operation, 1, 1)
        }
        "compute.field_expr" => {
            require_attrs(operation, &["sym_name", "kind", "formula", "operands"])?;
            require_min_shape(operation, 0, 1)
        }
        "compute.sumcheck_claim" => {
            require_attrs(
                operation,
                &[
                    "sym_name",
                    "stage",
                    "domain",
                    "num_rounds",
                    "degree",
                    "claim",
                    "relation",
                ],
            )?;
            require_min_shape(operation, 1, 1)
        }
        "compute.sumcheck_kernel_claim" => {
            require_attrs(
                operation,
                &[
                    "sym_name",
                    "stage",
                    "domain",
                    "num_rounds",
                    "degree",
                    "claim",
                    "kernel",
                ],
            )?;
            require_min_shape(operation, 1, 1)
        }
        "compute.sumcheck_verify_claim" => {
            require_attrs(
                operation,
                &[
                    "sym_name",
                    "stage",
                    "domain",
                    "num_rounds",
                    "degree",
                    "claim",
                    "relation",
                ],
            )?;
            require_min_shape(operation, 1, 1)
        }
        "compute.sumcheck_batch" => {
            require_attrs(
                operation,
                &[
                    "sym_name",
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
            require_min_shape(operation, 0, 1)?;
            require_counted_operands(operation, 0, "ordered_claims")
        }
        "compute.sumcheck_driver" => {
            require_attrs(
                operation,
                &[
                    "sym_name",
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
            require_shape(operation, 2, 4)
        }
        "compute.sumcheck_kernel_driver" => {
            require_attrs(
                operation,
                &[
                    "sym_name",
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
            require_shape(operation, 2, 4)
        }
        "compute.sumcheck_verify" => {
            require_attrs(
                operation,
                &[
                    "sym_name",
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
            require_shape(operation, 2, 4)
        }
        "compute.sumcheck_eval" => {
            require_attrs(
                operation,
                &["sym_name", "source", "name", "index", "oracle"],
            )?;
            require_shape(operation, 1, 1)
        }
        "compute.sumcheck_instance_result" => {
            require_attrs(
                operation,
                &[
                    "sym_name",
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
            require_shape(operation, 2, 2)
        }
        "compute.opening_claim" => {
            require_attrs(
                operation,
                &["sym_name", "oracle", "domain", "point_arity", "claim_kind"],
            )?;
            require_shape(operation, 2, 1)
        }
        "compute.opening_batch" => {
            require_attrs(
                operation,
                &[
                    "sym_name",
                    "stage",
                    "proof_slot",
                    "policy",
                    "count",
                    "ordered_claims",
                ],
            )?;
            require_min_shape(operation, 0, 1)?;
            require_counted_operands(operation, 0, "ordered_claims")
        }
        "compute.pcs_opening_claim" => {
            require_attrs(
                operation,
                &["sym_name", "oracle", "family", "domain", "point_arity"],
            )?;
            require_shape(operation, 2, 1)
        }
        "compute.pcs_opening_batch" => {
            require_attrs(
                operation,
                &[
                    "sym_name",
                    "proof_slot",
                    "policy",
                    "count",
                    "ordered_claims",
                ],
            )?;
            require_min_shape(operation, 0, 1)?;
            require_counted_operands(operation, 0, "ordered_claims")
        }
        "compute.pcs_batch_open" | "compute.pcs_batch_verify" => {
            require_attrs(
                operation,
                &["sym_name", "pcs", "proof_slot", "transcript_label"],
            )?;
            require_shape(operation, 2, 2)
        }
        "cpu.params" => require_attrs(operation, &["sym_name", "field", "pcs", "transcript"]),
        "cpu.function" => require_attrs(operation, &["sym_name", "source"]),
        "cpu.oracle_dense_trace" => {
            require_attrs(
                operation,
                &[
                    "sym_name", "oracle", "source", "domain", "num_vars", "padding",
                ],
            )?;
            require_shape(operation, 0, 1)
        }
        "cpu.oracle_one_hot_chunk" => {
            require_attrs(
                operation,
                &[
                    "sym_name",
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
                ],
            )?;
            require_shape(operation, 0, 1)
        }
        "cpu.oracle_optional_advice" => {
            require_attrs(
                operation,
                &[
                    "sym_name",
                    "oracle",
                    "source",
                    "domain",
                    "num_vars",
                    "skip_policy",
                ],
            )?;
            require_shape(operation, 0, 1)
        }
        "cpu.oracle_ref" => {
            require_attrs(operation, &["sym_name", "oracle", "domain", "num_vars"])?;
            require_shape(operation, 0, 1)
        }
        "cpu.oracle_family_init" => {
            require_attrs(operation, &["sym_name", "family", "count"])?;
            require_shape(operation, 0, 1)
        }
        "cpu.oracle_family_append" => {
            require_attrs(operation, &["sym_name", "family", "oracle", "index"])?;
            require_shape(operation, 2, 1)
        }
        "cpu.pcs_commit_batch" | "cpu.pcs_receive_batch" => {
            require_attrs(
                operation,
                &[
                    "sym_name",
                    "artifact",
                    "pcs",
                    "oracle_family",
                    "ordered_oracles",
                    "label",
                    "domain",
                    "num_vars",
                    "count",
                ],
            )?;
            require_shape(operation, 1, 1)
        }
        "cpu.pcs_commit_optional" | "cpu.pcs_receive_optional" => {
            require_attrs(
                operation,
                &[
                    "sym_name",
                    "artifact",
                    "pcs",
                    "oracle",
                    "label",
                    "domain",
                    "num_vars",
                    "skip_policy",
                ],
            )?;
            require_shape(operation, 1, 1)
        }
        "cpu.transcript_init" => {
            require_attrs(operation, &["sym_name", "scheme"])?;
            require_shape(operation, 0, 1)
        }
        "cpu.transcript_absorb" => {
            require_attrs(operation, &["sym_name", "label", "optional"])?;
            require_shape(operation, 2, 1)
        }
        "cpu.transcript_squeeze" => {
            require_attrs(operation, &["sym_name", "label", "kind", "count"])?;
            require_shape(operation, 1, 2)
        }
        "cpu.opening_input" => {
            require_attrs(
                operation,
                &[
                    "sym_name",
                    "source_stage",
                    "source_claim",
                    "oracle",
                    "domain",
                    "point_arity",
                    "claim_kind",
                ],
            )?;
            require_shape(operation, 0, 3)
        }
        "cpu.point_slice" => {
            require_attrs(operation, &["sym_name", "source", "offset", "length"])?;
            require_shape(operation, 1, 1)
        }
        "cpu.point_concat" => {
            require_attrs(operation, &["sym_name", "layout", "arity"])?;
            require_min_shape(operation, 1, 1)
        }
        "cpu.field_constant" => {
            require_attrs(operation, &["sym_name", "field", "value"])?;
            require_shape(operation, 0, 1)
        }
        "cpu.challenge_extract" => {
            require_attrs(operation, &["sym_name", "source", "index"])?;
            require_shape(operation, 1, 1)
        }
        "cpu.field_expr" => {
            require_attrs(operation, &["sym_name", "kind", "formula", "operands"])?;
            require_min_shape(operation, 0, 1)
        }
        "cpu.kernel" => require_attrs(
            operation,
            &["sym_name", "relation", "kind", "backend", "abi"],
        ),
        "cpu.sumcheck_claim" => {
            require_attrs(
                operation,
                &[
                    "sym_name",
                    "stage",
                    "domain",
                    "num_rounds",
                    "degree",
                    "claim",
                    "kernel",
                ],
            )?;
            require_min_shape(operation, 1, 1)
        }
        "cpu.sumcheck_verify_claim" => {
            require_attrs(
                operation,
                &[
                    "sym_name",
                    "stage",
                    "domain",
                    "num_rounds",
                    "degree",
                    "claim",
                    "relation",
                ],
            )?;
            require_min_shape(operation, 1, 1)
        }
        "cpu.sumcheck_batch" => {
            require_attrs(
                operation,
                &[
                    "sym_name",
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
            require_min_shape(operation, 0, 1)?;
            require_counted_operands(operation, 0, "ordered_claims")
        }
        "cpu.sumcheck_driver" => {
            require_attrs(
                operation,
                &[
                    "sym_name",
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
            require_shape(operation, 2, 4)
        }
        "cpu.sumcheck_verify" => {
            require_attrs(
                operation,
                &[
                    "sym_name",
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
            require_shape(operation, 2, 4)
        }
        "cpu.sumcheck_eval" => {
            require_attrs(
                operation,
                &["sym_name", "source", "name", "index", "oracle"],
            )?;
            require_shape(operation, 1, 1)
        }
        "cpu.sumcheck_instance_result" => {
            require_attrs(
                operation,
                &[
                    "sym_name",
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
            require_shape(operation, 2, 2)
        }
        "cpu.opening_claim" => {
            require_attrs(
                operation,
                &["sym_name", "oracle", "domain", "point_arity", "claim_kind"],
            )?;
            require_shape(operation, 2, 1)
        }
        "cpu.opening_batch" => {
            require_attrs(
                operation,
                &[
                    "sym_name",
                    "stage",
                    "proof_slot",
                    "policy",
                    "count",
                    "ordered_claims",
                ],
            )?;
            require_min_shape(operation, 0, 1)?;
            require_counted_operands(operation, 0, "ordered_claims")
        }
        "cpu.pcs_opening_claim" => {
            require_attrs(
                operation,
                &["sym_name", "oracle", "family", "domain", "point_arity"],
            )?;
            require_shape(operation, 2, 1)
        }
        "cpu.pcs_opening_batch" => {
            require_attrs(
                operation,
                &[
                    "sym_name",
                    "proof_slot",
                    "policy",
                    "count",
                    "ordered_claims",
                ],
            )?;
            require_min_shape(operation, 0, 1)?;
            require_counted_operands(operation, 0, "ordered_claims")
        }
        "cpu.pcs_batch_open" | "cpu.pcs_batch_verify" => {
            require_attrs(
                operation,
                &["sym_name", "pcs", "proof_slot", "transcript_label"],
            )?;
            require_shape(operation, 2, 2)
        }
        "pcs.opening_claim" => {
            require_attrs(
                operation,
                &["sym_name", "oracle", "family", "domain", "point_arity"],
            )?;
            require_shape(operation, 2, 1)
        }
        "pcs.opening_batch" => {
            require_attrs(
                operation,
                &[
                    "sym_name",
                    "proof_slot",
                    "policy",
                    "count",
                    "ordered_claims",
                ],
            )?;
            require_min_shape(operation, 0, 1)?;
            require_counted_operands(operation, 0, "ordered_claims")
        }
        "pcs.batch_open" | "pcs.batch_verify" => {
            require_attrs(
                operation,
                &["sym_name", "pcs", "proof_slot", "transcript_label"],
            )?;
            require_shape(operation, 2, 2)
        }
        _ if is_bolt_dialect_op(&name) => Err(SchemaError::new(format!(
            "unknown Bolt op `{name}` in schema verifier"
        ))),
        _ => Ok(()),
    }
}

fn require_shape(
    operation: OperationRef<'_, '_>,
    operands: usize,
    results: usize,
) -> Result<(), SchemaError> {
    if operation.operand_count() != operands {
        return Err(SchemaError::new(format!(
            "{} expected {operands} operands, got {}",
            operation_name(operation),
            operation.operand_count()
        )));
    }
    if operation.result_count() != results {
        return Err(SchemaError::new(format!(
            "{} expected {results} results, got {}",
            operation_name(operation),
            operation.result_count()
        )));
    }
    Ok(())
}

fn require_min_shape(
    operation: OperationRef<'_, '_>,
    min_operands: usize,
    results: usize,
) -> Result<(), SchemaError> {
    if operation.operand_count() < min_operands {
        return Err(SchemaError::new(format!(
            "{} expected at least {min_operands} operands, got {}",
            operation_name(operation),
            operation.operand_count()
        )));
    }
    if operation.result_count() != results {
        return Err(SchemaError::new(format!(
            "{} expected {results} results, got {}",
            operation_name(operation),
            operation.result_count()
        )));
    }
    Ok(())
}

fn require_counted_operands(
    operation: OperationRef<'_, '_>,
    fixed_operands: usize,
    ordered_attr: &str,
) -> Result<(), SchemaError> {
    let count = int_attr(operation, "count")?;
    let dynamic_count = operation.operand_count().saturating_sub(fixed_operands);
    if count != dynamic_count {
        return Err(SchemaError::new(format!(
            "{} attr `count` expected {dynamic_count}, got {count}",
            operation_name(operation)
        )));
    }
    let ordered = symbol_array_attr(operation, ordered_attr)?;
    if ordered.len() != count {
        return Err(SchemaError::new(format!(
            "{} attr `{ordered_attr}` length {} does not match count {count}",
            operation_name(operation),
            ordered.len()
        )));
    }
    for (index, expected) in ordered.iter().enumerate() {
        let operand_index = fixed_operands + index;
        let actual = operand_owner_symbol(operation, operand_index)?;
        if &actual != expected {
            return Err(SchemaError::new(format!(
                "{} operand {operand_index} expected @{expected}, got @{actual}",
                operation_name(operation)
            )));
        }
    }
    Ok(())
}

pub(crate) fn require_attrs(
    operation: OperationRef<'_, '_>,
    attrs: &[&str],
) -> Result<(), SchemaError> {
    for attr in attrs {
        if !operation.has_attribute(attr) {
            return Err(SchemaError::new(format!(
                "{} missing required attr `{attr}`",
                operation_name(operation)
            )));
        }
    }
    Ok(())
}

pub(crate) fn operand_owner_symbol(
    operation: OperationRef<'_, '_>,
    index: usize,
) -> Result<String, SchemaError> {
    let operand = operation.operand(index).map_err(|_| {
        SchemaError::new(format!(
            "{} missing required operand {index}",
            operation_name(operation)
        ))
    })?;
    let owner = OperationResult::try_from(operand).map_err(|_| {
        SchemaError::new(format!(
            "{} operand {index} must be an op result",
            operation_name(operation)
        ))
    })?;
    owner
        .owner()
        .attribute("sym_name")
        .ok()
        .and_then(string_attribute_value)
        .ok_or_else(|| {
            SchemaError::new(format!(
                "{} operand {index} owner missing sym_name",
                operation_name(operation)
            ))
        })
}

pub(crate) fn require_symbol_attr_eq(
    operation: OperationRef<'_, '_>,
    attr: &str,
    expected: &str,
) -> Result<(), SchemaError> {
    let actual = symbol_attr(operation, attr)?;
    if actual == expected {
        Ok(())
    } else {
        Err(SchemaError::new(format!(
            "{} attr `{attr}` expected @{expected}, got @{actual}",
            operation_name(operation)
        )))
    }
}

pub(crate) fn find_symbol<'c, P>(
    module: &'c BoltModule<'_, P>,
    symbol: &str,
) -> Option<OperationRef<'c, 'c>>
where
    P: crate::ir::Phase,
{
    let mut operation = module.as_mlir_module().body().first_operation();
    while let Some(op) = operation {
        operation = op.next_in_block();
        if op
            .attribute("sym_name")
            .ok()
            .and_then(string_attribute_value)
            .as_deref()
            == Some(symbol)
        {
            return Some(op);
        }
    }
    None
}

pub(crate) fn symbol_attr(
    operation: OperationRef<'_, '_>,
    attr: &str,
) -> Result<String, SchemaError> {
    operation
        .attribute(attr)
        .ok()
        .and_then(symbol_attribute_value)
        .ok_or_else(|| attr_error(operation, attr, "symbol"))
}

fn string_attr(operation: OperationRef<'_, '_>, attr: &str) -> Result<String, SchemaError> {
    operation
        .attribute(attr)
        .ok()
        .and_then(string_attribute_value)
        .ok_or_else(|| attr_error(operation, attr, "string"))
}

pub(crate) fn symbol_array_attr(
    operation: OperationRef<'_, '_>,
    attr: &str,
) -> Result<Vec<String>, SchemaError> {
    let attribute = operation
        .attribute(attr)
        .map(|attribute| attribute.to_string())
        .ok()
        .ok_or_else(|| attr_error(operation, attr, "symbol array"))?;
    parse_symbol_array(&attribute).ok_or_else(|| attr_error(operation, attr, "symbol array"))
}

fn parse_symbol_array(attribute: &str) -> Option<Vec<String>> {
    let inner = attribute.strip_prefix('[')?.strip_suffix(']')?.trim();
    if inner.is_empty() {
        return Some(Vec::new());
    }
    inner
        .split(',')
        .map(|item| item.trim().strip_prefix('@').map(ToOwned::to_owned))
        .collect()
}

pub(crate) fn int_attr(operation: OperationRef<'_, '_>, attr: &str) -> Result<usize, SchemaError> {
    operation
        .attribute(attr)
        .map(parse_integer_attr)
        .ok()
        .flatten()
        .ok_or_else(|| attr_error(operation, attr, "integer"))
}

fn parse_integer_attr(attribute: Attribute<'_>) -> Option<usize> {
    attribute
        .to_string()
        .split_whitespace()
        .next()
        .and_then(|value| value.parse().ok())
}

fn attr_error(operation: OperationRef<'_, '_>, attr: &str, expected: &str) -> SchemaError {
    SchemaError::new(format!(
        "{} attr `{attr}` is not a {expected}",
        operation_name(operation)
    ))
}

pub(crate) fn operation_name(operation: OperationRef<'_, '_>) -> String {
    operation
        .name()
        .as_string_ref()
        .as_str()
        .unwrap_or("<invalid-operation-name>")
        .to_owned()
}

pub(crate) fn missing_module_op(name: &str) -> SchemaError {
    SchemaError::new(format!("module missing required op `{name}`"))
}

pub(crate) fn missing_symbol(symbol: &str) -> SchemaError {
    SchemaError::new(format!("module missing required symbol @{symbol}"))
}

fn is_bolt_dialect_op(name: &str) -> bool {
    matches!(
        name.split_once('.').map(|(dialect, _)| dialect),
        Some(
            "field"
                | "poly"
                | "hash"
                | "transcript"
                | "commit"
                | "pcs"
                | "protocol"
                | "piop"
                | "party"
                | "compute"
                | "cpu"
        )
    )
}
