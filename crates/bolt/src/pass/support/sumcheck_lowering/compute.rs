use super::family::SumcheckValueFamily;

pub(in crate::pass) fn classify_compute_sumcheck_value_op(
    source_name: &str,
) -> Option<SumcheckValueFamily> {
    match source_name {
        "compute.sumcheck_eval" => Some(SumcheckValueFamily::Eval),
        "compute.sumcheck_instance_result" => Some(SumcheckValueFamily::InstanceResult),
        _ => None,
    }
}
