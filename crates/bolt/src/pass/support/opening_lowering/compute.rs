use super::family::OpeningOpFamily;

pub(in crate::pass) fn classify_compute_opening_op(source_name: &str) -> Option<OpeningOpFamily> {
    match source_name {
        "compute.opening_claim" => Some(OpeningOpFamily::Claim),
        "compute.opening_claim_equal" => Some(OpeningOpFamily::ClaimEqual),
        "compute.opening_batch" => Some(OpeningOpFamily::Batch),
        _ => None,
    }
}
