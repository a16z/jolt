use std::collections::BTreeMap;

use melior::ir::operation::OperationRef;
use melior::ir::Value;

use crate::ir::{BoltModule, Compute};
use crate::mlir::{MeliorContext, MlirError};
use crate::pass::kernel_resolution::registry::{ensure_compute_kernel, KernelRegistry};
use crate::pass::support::{
    append_and_map_result_count, lowered_operands, string_attr, symbol_attr,
};

use super::attrs::kernel_attrs;
use super::shape::KernelSumcheckShape;

pub(super) fn lower_kernel_sumcheck<'c, 'a, R>(
    context: &'c MeliorContext,
    kernelized: &'a BoltModule<'c, Compute>,
    value_map: &mut BTreeMap<String, Value<'c, 'a>>,
    kernels: &mut BTreeMap<String, String>,
    kernel_registry: &mut R,
    op: OperationRef<'_, '_>,
    shape: KernelSumcheckShape,
) -> Result<(), MlirError>
where
    R: KernelRegistry,
{
    let relation = symbol_attr(op, "relation")?;
    let kernel = ensure_compute_kernel(context, kernelized, kernels, &relation, kernel_registry)?;
    let operands = lowered_operands(op, value_map, 0)?;
    let attrs = kernel_attrs(op, shape.source_attrs, &kernel)?;
    let symbol = string_attr(op, "sym_name")?;
    append_and_map_result_count(
        context,
        kernelized,
        value_map,
        op,
        shape.target_op,
        &symbol,
        &attrs,
        &operands,
        shape.result_types,
        shape.result_count,
    )
}
