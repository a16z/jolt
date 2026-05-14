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

const STRUCTURED_POLYNOMIAL_EVAL_ATTRS: &[&str] = &[
    "sym_name",
    "polynomial",
    "x_point_segment",
    "x_point_length",
    "x_point_order",
    "y_point_segment",
    "y_point_length",
    "y_point_order",
];

const SUMCHECK_OUTPUT_EVAL_FAMILY_ATTRS: &[&str] = &[
    "sym_name",
    "power_stride",
    "value_term_offsets",
    "shared_term_offsets",
    "item_term_offsets",
    "evals",
    "shared_terms",
    "item_terms",
];

const SUMCHECK_OUTPUT_PRODUCT_FAMILY_ATTRS: &[&str] = &[
    "sym_name",
    "term_gamma_power_offsets",
    "term_eval_counts",
    "term_factor_counts",
    "evals",
    "factors",
];

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
        "field.const" => {
            require_attrs(operation, &["sym_name", "field", "value"])?;
            require_shape(operation, 0, 1)
        }
        "field.zero" | "field.one" => {
            require_attrs(operation, &["sym_name", "field"])?;
            require_shape(operation, 0, 1)
        }
        "field.add" | "field.sub" | "field.mul" => {
            require_attrs(operation, &["sym_name"])?;
            require_shape(operation, 2, 1)
        }
        "field.neg" => {
            require_attrs(operation, &["sym_name"])?;
            require_shape(operation, 1, 1)
        }
        "field.pow" => {
            require_attrs(operation, &["sym_name", "exponent"])?;
            require_shape(operation, 1, 1)
        }
        "hash.function" => require_attrs(operation, &["sym_name", "algorithm"]),
        "transcript.scheme" => require_attrs(operation, &["sym_name", "hash"]),
        "pcs.scheme" => require_attrs(operation, &["sym_name", "field"]),
        "poly.domain" => require_attrs(operation, &["sym_name", "field", "log_size"]),
        "poly.point_slice" => {
            require_attrs(operation, &["sym_name", "source", "offset", "length"])?;
            require_shape(operation, 1, 1)
        }
        "poly.point_zero" => {
            require_attrs(operation, &["sym_name", "field", "arity"])?;
            require_shape(operation, 0, 1)
        }
        "poly.point_concat" => {
            require_attrs(operation, &["sym_name", "layout", "arity"])?;
            require_min_shape(operation, 1, 1)
        }
        "poly.lagrange_basis_eval" => {
            require_attrs(
                operation,
                &["sym_name", "domain_start", "domain_size", "index"],
            )?;
            require_shape(operation, 1, 1)
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
        "transcript.absorb_bytes" => {
            require_attrs(operation, &["sym_name", "label", "payload"])?;
            require_shape(operation, 1, 1)
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
        "piop.structured_polynomial_eval" => {
            require_attrs(operation, STRUCTURED_POLYNOMIAL_EVAL_ATTRS)?;
            require_shape(operation, 2, 1)?;
            require_structured_polynomial_eval(operation)
        }
        "piop.sumcheck_output_eval_family" => {
            require_attrs(operation, SUMCHECK_OUTPUT_EVAL_FAMILY_ATTRS)?;
            require_min_shape(operation, 1, 1)?;
            require_sumcheck_output_eval_family(operation)
        }
        "piop.sumcheck_output_product_family" => {
            require_attrs(operation, SUMCHECK_OUTPUT_PRODUCT_FAMILY_ATTRS)?;
            require_min_shape(operation, 1, 1)?;
            require_sumcheck_output_product_family(operation)
        }
        "piop.sumcheck_output_claim" => {
            require_attrs(
                operation,
                &["sym_name", "stage", "relation", "count", "polynomial_evals"],
            )?;
            require_min_shape(operation, 1, 0)?;
            require_sumcheck_output_claim(operation)
        }
        "piop.opening_claim" => {
            require_attrs(
                operation,
                &["sym_name", "oracle", "domain", "point_arity", "claim_kind"],
            )?;
            require_shape(operation, 2, 1)
        }
        "piop.opening_claim_equal" => {
            require_attrs(operation, &["sym_name", "mode"])?;
            require_shape(operation, 2, 0)?;
            require_opening_claim_equality(operation)
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
        "compute.transcript_absorb_bytes" => {
            require_attrs(operation, &["sym_name", "label", "payload"])?;
            require_shape(operation, 1, 1)
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
        "compute.point_zero" => {
            require_attrs(operation, &["sym_name", "field", "arity"])?;
            require_shape(operation, 0, 1)
        }
        "compute.point_concat" => {
            require_attrs(operation, &["sym_name", "layout", "arity"])?;
            require_min_shape(operation, 1, 1)
        }
        "compute.field_const" => {
            require_attrs(operation, &["sym_name", "field", "value"])?;
            require_shape(operation, 0, 1)
        }
        "compute.field_zero" | "compute.field_one" => {
            require_attrs(operation, &["sym_name", "field"])?;
            require_shape(operation, 0, 1)
        }
        "compute.field_add" | "compute.field_sub" | "compute.field_mul" => {
            require_attrs(operation, &["sym_name"])?;
            require_shape(operation, 2, 1)
        }
        "compute.field_neg" => {
            require_attrs(operation, &["sym_name"])?;
            require_shape(operation, 1, 1)
        }
        "compute.field_pow" => {
            require_attrs(operation, &["sym_name", "exponent"])?;
            require_shape(operation, 1, 1)
        }
        "compute.poly_lagrange_basis_eval" => {
            require_attrs(
                operation,
                &["sym_name", "domain_start", "domain_size", "index"],
            )?;
            require_shape(operation, 1, 1)
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
        "compute.structured_polynomial_eval" => {
            require_attrs(operation, STRUCTURED_POLYNOMIAL_EVAL_ATTRS)?;
            require_shape(operation, 2, 1)?;
            require_structured_polynomial_eval(operation)
        }
        "compute.sumcheck_output_eval_family" => {
            require_attrs(operation, SUMCHECK_OUTPUT_EVAL_FAMILY_ATTRS)?;
            require_min_shape(operation, 1, 1)?;
            require_sumcheck_output_eval_family(operation)
        }
        "compute.sumcheck_output_product_family" => {
            require_attrs(operation, SUMCHECK_OUTPUT_PRODUCT_FAMILY_ATTRS)?;
            require_min_shape(operation, 1, 1)?;
            require_sumcheck_output_product_family(operation)
        }
        "compute.sumcheck_output_claim" => {
            require_attrs(
                operation,
                &["sym_name", "stage", "relation", "count", "polynomial_evals"],
            )?;
            require_min_shape(operation, 1, 0)?;
            require_sumcheck_output_claim(operation)
        }
        "compute.opening_claim" => {
            require_attrs(
                operation,
                &["sym_name", "oracle", "domain", "point_arity", "claim_kind"],
            )?;
            require_shape(operation, 2, 1)
        }
        "compute.opening_claim_equal" => {
            require_attrs(operation, &["sym_name", "mode"])?;
            require_shape(operation, 2, 0)?;
            require_opening_claim_equality(operation)
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
        "cpu.transcript_absorb_bytes" => {
            require_attrs(operation, &["sym_name", "label", "payload"])?;
            require_shape(operation, 1, 1)
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
        "cpu.point_zero" => {
            require_attrs(operation, &["sym_name", "field", "arity"])?;
            require_shape(operation, 0, 1)
        }
        "cpu.point_concat" => {
            require_attrs(operation, &["sym_name", "layout", "arity"])?;
            require_min_shape(operation, 1, 1)
        }
        "cpu.field_const" => {
            require_attrs(operation, &["sym_name", "field", "value"])?;
            require_shape(operation, 0, 1)
        }
        "cpu.field_zero" | "cpu.field_one" => {
            require_attrs(operation, &["sym_name", "field"])?;
            require_shape(operation, 0, 1)
        }
        "cpu.field_add" | "cpu.field_sub" | "cpu.field_mul" => {
            require_attrs(operation, &["sym_name"])?;
            require_shape(operation, 2, 1)
        }
        "cpu.field_neg" => {
            require_attrs(operation, &["sym_name"])?;
            require_shape(operation, 1, 1)
        }
        "cpu.field_pow" => {
            require_attrs(operation, &["sym_name", "exponent"])?;
            require_shape(operation, 1, 1)
        }
        "cpu.poly_lagrange_basis_eval" => {
            require_attrs(
                operation,
                &["sym_name", "domain_start", "domain_size", "index"],
            )?;
            require_shape(operation, 1, 1)
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
        "cpu.structured_polynomial_eval" => {
            require_attrs(operation, STRUCTURED_POLYNOMIAL_EVAL_ATTRS)?;
            require_shape(operation, 2, 1)?;
            require_structured_polynomial_eval(operation)
        }
        "cpu.sumcheck_output_eval_family" => {
            require_attrs(operation, SUMCHECK_OUTPUT_EVAL_FAMILY_ATTRS)?;
            require_min_shape(operation, 1, 1)?;
            require_sumcheck_output_eval_family(operation)
        }
        "cpu.sumcheck_output_product_family" => {
            require_attrs(operation, SUMCHECK_OUTPUT_PRODUCT_FAMILY_ATTRS)?;
            require_min_shape(operation, 1, 1)?;
            require_sumcheck_output_product_family(operation)
        }
        "cpu.sumcheck_output_claim" => {
            require_attrs(
                operation,
                &["sym_name", "stage", "relation", "count", "polynomial_evals"],
            )?;
            require_min_shape(operation, 1, 0)?;
            require_sumcheck_output_claim(operation)
        }
        "cpu.opening_claim" => {
            require_attrs(
                operation,
                &["sym_name", "oracle", "domain", "point_arity", "claim_kind"],
            )?;
            require_shape(operation, 2, 1)
        }
        "cpu.opening_claim_equal" => {
            require_attrs(operation, &["sym_name", "mode"])?;
            require_shape(operation, 2, 0)?;
            require_opening_claim_equality(operation)
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

#[derive(Clone, Debug, PartialEq, Eq)]
struct OpeningClaimMetadata {
    owner: String,
    oracle: String,
    domain: String,
    point_arity: usize,
    claim_kind: String,
}

fn require_opening_claim_equality(operation: OperationRef<'_, '_>) -> Result<(), SchemaError> {
    let mode = string_attr(operation, "mode")?;
    if mode != "point_and_eval" {
        return Err(SchemaError::new(format!(
            "{} attr `mode` expected \"point_and_eval\", got \"{mode}\"",
            operation_name(operation)
        )));
    }

    let left = opening_claim_metadata(operation, 0)?;
    let right = opening_claim_metadata(operation, 1)?;
    if left.oracle != right.oracle
        || left.domain != right.domain
        || left.point_arity != right.point_arity
        || left.claim_kind != right.claim_kind
    {
        return Err(SchemaError::new(format!(
            "{} compares incompatible claims @{} and @{}",
            operation_name(operation),
            left.owner,
            right.owner
        )));
    }
    Ok(())
}

fn require_structured_polynomial_eval(operation: OperationRef<'_, '_>) -> Result<(), SchemaError> {
    let polynomial = string_attr(operation, "polynomial")?;
    if !matches!(polynomial.as_str(), "eq" | "eq_plus_one" | "lt") {
        return Err(SchemaError::new(format!(
            "{} attr `polynomial` has unsupported structured polynomial `{polynomial}`",
            operation_name(operation)
        )));
    }
    require_structured_polynomial_point_attrs(operation, "x_point")?;
    require_structured_polynomial_point_attrs(operation, "y_point")?;
    Ok(())
}

fn require_structured_polynomial_point_attrs(
    operation: OperationRef<'_, '_>,
    prefix: &str,
) -> Result<(), SchemaError> {
    let segment_attr = format!("{prefix}_segment");
    let segment = string_attr(operation, &segment_attr)?;
    if !matches!(segment.as_str(), "full" | "prefix" | "suffix") {
        return Err(SchemaError::new(format!(
            "{} attr `{segment_attr}` has unsupported output point segment `{segment}`",
            operation_name(operation)
        )));
    }
    let length_attr = format!("{prefix}_length");
    let length = string_attr(operation, &length_attr)?;
    if !matches!(length.as_str(), "full" | "x_point" | "y_point") {
        return Err(SchemaError::new(format!(
            "{} attr `{length_attr}` has unsupported output point length `{length}`",
            operation_name(operation)
        )));
    }
    if segment == "full" && length != "full" {
        return Err(SchemaError::new(format!(
            "{} output point `{prefix}` uses segment `full` but length `{length}`; full segments must use length `full`",
            operation_name(operation)
        )));
    }
    let order_attr = format!("{prefix}_order");
    let order = string_attr(operation, &order_attr)?;
    if !matches!(order.as_str(), "as_is" | "reverse") {
        return Err(SchemaError::new(format!(
            "{} attr `{order_attr}` has unsupported output point order `{order}`",
            operation_name(operation)
        )));
    }
    Ok(())
}

fn require_sumcheck_output_eval_family(operation: OperationRef<'_, '_>) -> Result<(), SchemaError> {
    let evals = symbol_array_attr(operation, "evals")?;
    let shared_terms = symbol_array_attr(operation, "shared_terms")?;
    let item_terms = symbol_array_attr(operation, "item_terms")?;
    let shared_offsets = int_array_attr(operation, "shared_term_offsets")?;
    let item_offsets = int_array_attr(operation, "item_term_offsets")?;
    if shared_terms.len() != shared_offsets.len() {
        return Err(SchemaError::new(format!(
            "{} attr `shared_terms` length {} does not match shared_term_offsets length {}",
            operation_name(operation),
            shared_terms.len(),
            shared_offsets.len()
        )));
    }
    let expected_item_terms = item_offsets.len() * evals.len();
    if item_terms.len() != expected_item_terms {
        return Err(SchemaError::new(format!(
            "{} attr `item_terms` length {} does not match item_term_offsets length {} times evals length {}",
            operation_name(operation),
            item_terms.len(),
            item_offsets.len(),
            evals.len()
        )));
    }
    let expected_operands = 1 + evals.len() + shared_terms.len() + item_terms.len();
    if operation.operand_count() != expected_operands {
        return Err(SchemaError::new(format!(
            "{} expected {expected_operands} operands, got {}",
            operation_name(operation),
            operation.operand_count()
        )));
    }
    let expected_symbols = evals
        .iter()
        .chain(shared_terms.iter())
        .chain(item_terms.iter())
        .collect::<Vec<_>>();
    for (index, expected) in expected_symbols.iter().enumerate() {
        let operand_index = index + 1;
        let actual = operand_owner_symbol(operation, operand_index)?;
        if &actual != *expected {
            return Err(SchemaError::new(format!(
                "{} operand {operand_index} expected @{expected}, got @{actual}",
                operation_name(operation)
            )));
        }
    }
    Ok(())
}

fn require_sumcheck_output_product_family(
    operation: OperationRef<'_, '_>,
) -> Result<(), SchemaError> {
    let evals = symbol_array_attr(operation, "evals")?;
    let factors = symbol_array_attr(operation, "factors")?;
    let term_gamma_power_offsets = int_array_attr(operation, "term_gamma_power_offsets")?;
    let term_eval_counts = int_array_attr(operation, "term_eval_counts")?;
    let term_factor_counts = int_array_attr(operation, "term_factor_counts")?;
    if term_eval_counts.len() != term_gamma_power_offsets.len() {
        return Err(SchemaError::new(format!(
            "{} attr `term_eval_counts` length {} does not match term_gamma_power_offsets length {}",
            operation_name(operation),
            term_eval_counts.len(),
            term_gamma_power_offsets.len()
        )));
    }
    if term_factor_counts.len() != term_gamma_power_offsets.len() {
        return Err(SchemaError::new(format!(
            "{} attr `term_factor_counts` length {} does not match term_gamma_power_offsets length {}",
            operation_name(operation),
            term_factor_counts.len(),
            term_gamma_power_offsets.len()
        )));
    }
    for ((index, eval_count), factor_count) in term_eval_counts
        .iter()
        .enumerate()
        .zip(term_factor_counts.iter())
    {
        if *eval_count == 0 && *factor_count == 0 {
            return Err(SchemaError::new(format!(
                "{} product-family term {index} is empty",
                operation_name(operation)
            )));
        }
    }
    let expected_evals: usize = term_eval_counts.iter().sum();
    if evals.len() != expected_evals {
        return Err(SchemaError::new(format!(
            "{} attr `evals` length {} does not match sum(term_eval_counts) {}",
            operation_name(operation),
            evals.len(),
            expected_evals
        )));
    }
    let expected_factors: usize = term_factor_counts.iter().sum();
    if factors.len() != expected_factors {
        return Err(SchemaError::new(format!(
            "{} attr `factors` length {} does not match sum(term_factor_counts) {}",
            operation_name(operation),
            factors.len(),
            expected_factors
        )));
    }
    let expected_operands = 1 + evals.len() + factors.len();
    if operation.operand_count() != expected_operands {
        return Err(SchemaError::new(format!(
            "{} expected {expected_operands} operands, got {}",
            operation_name(operation),
            operation.operand_count()
        )));
    }
    let expected_symbols = evals.iter().chain(factors.iter()).collect::<Vec<_>>();
    for (index, expected) in expected_symbols.iter().enumerate() {
        let operand_index = index + 1;
        let actual = operand_owner_symbol(operation, operand_index)?;
        if &actual != *expected {
            return Err(SchemaError::new(format!(
                "{} operand {operand_index} expected @{expected}, got @{actual}",
                operation_name(operation)
            )));
        }
    }
    Ok(())
}

fn require_sumcheck_output_claim(operation: OperationRef<'_, '_>) -> Result<(), SchemaError> {
    let count = int_attr(operation, "count")?;
    let polynomial_evals = symbol_array_attr(operation, "polynomial_evals")?;
    if polynomial_evals.len() != count {
        return Err(SchemaError::new(format!(
            "{} attr `polynomial_evals` length {} does not match count {count}",
            operation_name(operation),
            polynomial_evals.len()
        )));
    }
    let dynamic_count = operation.operand_count().saturating_sub(1);
    if dynamic_count != count {
        return Err(SchemaError::new(format!(
            "{} attr `count` expected {dynamic_count}, got {count}",
            operation_name(operation)
        )));
    }
    for (index, expected) in polynomial_evals.iter().enumerate() {
        let operand_index = index + 1;
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

fn opening_claim_metadata(
    equality_op: OperationRef<'_, '_>,
    operand_index: usize,
) -> Result<OpeningClaimMetadata, SchemaError> {
    let operand = equality_op.operand(operand_index).map_err(|_| {
        SchemaError::new(format!(
            "{} missing required operand {operand_index}",
            operation_name(equality_op)
        ))
    })?;
    let owner = OperationResult::try_from(operand).map_err(|_| {
        SchemaError::new(format!(
            "{} operand {operand_index} must be an op result",
            operation_name(equality_op)
        ))
    })?;
    let operation = owner.owner();
    let result_number = owner.result_number();
    let expected_result = match operation_name(operation).as_str() {
        "piop.opening_input" | "compute.opening_input" | "cpu.opening_input" => 2,
        "piop.opening_claim" | "compute.opening_claim" | "cpu.opening_claim" => 0,
        name => {
            return Err(SchemaError::new(format!(
                "{} operand {operand_index} must be an opening claim, got result from `{name}`",
                operation_name(equality_op)
            )));
        }
    };
    if result_number != expected_result {
        return Err(SchemaError::new(format!(
            "{} operand {operand_index} must use opening claim result {expected_result}, got result {result_number}",
            operation_name(equality_op)
        )));
    }

    Ok(OpeningClaimMetadata {
        owner: string_attr(operation, "sym_name")?,
        oracle: symbol_attr(operation, "oracle")?,
        domain: symbol_attr(operation, "domain")?,
        point_arity: int_attr(operation, "point_arity")?,
        claim_kind: string_attr(operation, "claim_kind")?,
    })
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

fn int_array_attr(operation: OperationRef<'_, '_>, attr: &str) -> Result<Vec<usize>, SchemaError> {
    let attribute = operation
        .attribute(attr)
        .map(|attribute| attribute.to_string())
        .ok()
        .ok_or_else(|| attr_error(operation, attr, "integer array"))?;
    parse_int_array(&attribute).ok_or_else(|| attr_error(operation, attr, "integer array"))
}

fn parse_int_array(attribute: &str) -> Option<Vec<usize>> {
    let inner = attribute.strip_prefix('[')?.strip_suffix(']')?.trim();
    if inner.is_empty() {
        return Some(Vec::new());
    }
    inner
        .split(',')
        .map(|item| item.trim().parse().ok())
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
