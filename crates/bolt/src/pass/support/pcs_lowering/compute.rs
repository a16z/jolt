use super::family::PcsOpFamily;

pub(in crate::pass) fn classify_compute_pcs_op(source_name: &str) -> Option<PcsOpFamily> {
    match source_name {
        "compute.pcs_opening_claim" => Some(PcsOpFamily::Claim),
        "compute.pcs_opening_batch" => Some(PcsOpFamily::Batch),
        "compute.pcs_batch_open" | "compute.pcs_batch_verify" => Some(PcsOpFamily::BatchOpening),
        _ => None,
    }
}
