use melior::ir::operation::OperationRef;

use crate::schema::operation_name;

pub(in crate::pass::compute_to_cpu::commitment::pcs) fn batch_op_name(
    operation: OperationRef<'_, '_>,
) -> &'static str {
    match operation_name(operation).as_str() {
        "compute.pcs_commit_batch" => "cpu.pcs_commit_batch",
        "compute.pcs_receive_batch" => "cpu.pcs_receive_batch",
        _ => unreachable!(),
    }
}

pub(in crate::pass::compute_to_cpu::commitment::pcs) fn optional_op_name(
    operation: OperationRef<'_, '_>,
) -> &'static str {
    match operation_name(operation).as_str() {
        "compute.pcs_commit_optional" => "cpu.pcs_commit_optional",
        "compute.pcs_receive_optional" => "cpu.pcs_receive_optional",
        _ => unreachable!(),
    }
}
