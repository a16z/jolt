use melior::ir::operation::OperationRef;

use crate::ir::{Compute, Role};
use crate::mlir::MlirError;
use crate::schema::operation_name;

use super::super::super::super::support::{
    PcsDialect, PcsLoweringRole, PcsOpFamily, COMPUTE_OPENING_BATCH_OPENING_RESULT_TYPES,
    COMPUTE_OPENING_BATCH_RESULT_TYPES, COMPUTE_OPENING_CLAIM_RESULT_TYPES,
};

pub(super) struct PartyToComputePcsDialect;

impl PcsDialect for PartyToComputePcsDialect {
    type Phase = Compute;

    const CLAIM_RESULT_TYPES: &'static [&'static str] = COMPUTE_OPENING_CLAIM_RESULT_TYPES;
    const BATCH_RESULT_TYPES: &'static [&'static str] = COMPUTE_OPENING_BATCH_RESULT_TYPES;
    const BATCH_OPENING_RESULT_TYPES: &'static [&'static str] =
        COMPUTE_OPENING_BATCH_OPENING_RESULT_TYPES;

    fn classify(source_name: &str) -> Option<PcsOpFamily> {
        match source_name {
            "pcs.opening_claim" => Some(PcsOpFamily::Claim),
            "pcs.opening_batch" => Some(PcsOpFamily::Batch),
            "pcs.batch_open" | "pcs.batch_verify" => Some(PcsOpFamily::BatchOpening),
            _ => None,
        }
    }

    fn target_op_name(
        operation: OperationRef<'_, '_>,
        role: PcsLoweringRole<'_>,
    ) -> Result<String, MlirError> {
        let target_name = match operation_name(operation).as_str() {
            "pcs.opening_claim" => "compute.pcs_opening_claim",
            "pcs.opening_batch" => "compute.pcs_opening_batch",
            "pcs.batch_open" | "pcs.batch_verify" => {
                batch_opening_target_op(role.required_for(operation)?)
            }
            source_name => return Ok(source_name.to_owned()),
        };
        Ok(target_name.to_owned())
    }
}

fn batch_opening_target_op(role: &Role) -> &'static str {
    match role {
        Role::Prover => "compute.pcs_batch_open",
        Role::Verifier => "compute.pcs_batch_verify",
    }
}
