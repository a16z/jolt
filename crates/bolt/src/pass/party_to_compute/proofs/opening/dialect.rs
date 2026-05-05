use melior::ir::operation::OperationRef;

use crate::ir::Compute;
use crate::schema::operation_name;

use super::super::super::super::support::{
    OpeningDialect, OpeningOpFamily, COMPUTE_OPENING_BATCH_RESULT_TYPES,
    COMPUTE_OPENING_CLAIM_RESULT_TYPES,
};

pub(super) struct PartyToComputeOpeningDialect;

impl OpeningDialect for PartyToComputeOpeningDialect {
    type Phase = Compute;

    const CLAIM_RESULT_TYPES: &'static [&'static str] = COMPUTE_OPENING_CLAIM_RESULT_TYPES;
    const BATCH_RESULT_TYPES: &'static [&'static str] = COMPUTE_OPENING_BATCH_RESULT_TYPES;

    fn classify(source_name: &str) -> Option<OpeningOpFamily> {
        match source_name {
            "piop.opening_claim" => Some(OpeningOpFamily::Claim),
            "piop.opening_claim_equal" => Some(OpeningOpFamily::ClaimEqual),
            "piop.opening_batch" => Some(OpeningOpFamily::Batch),
            _ => None,
        }
    }

    fn target_op_name(operation: OperationRef<'_, '_>) -> String {
        match operation_name(operation).as_str() {
            "piop.opening_claim" => "compute.opening_claim".to_owned(),
            "piop.opening_claim_equal" => "compute.opening_claim_equal".to_owned(),
            "piop.opening_batch" => "compute.opening_batch".to_owned(),
            source_name => source_name.to_owned(),
        }
    }
}
