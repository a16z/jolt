use std::collections::BTreeMap;

use crate::ir::{BoltModule, Compute};
use crate::mlir::{MeliorContext, MlirError};

use super::KernelRegistry;

pub(in crate::pass::kernel_resolution) fn ensure_compute_kernel<'c, R>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Compute>,
    kernels: &mut BTreeMap<String, String>,
    relation: &str,
    kernel_registry: &mut R,
) -> Result<String, MlirError>
where
    R: KernelRegistry,
{
    if let Some(kernel) = kernels.get(relation) {
        return Ok(kernel.clone());
    }
    let spec = kernel_registry.kernel_spec(relation)?;
    context.append_op_with_owned_attrs(
        module,
        "compute.kernel",
        Some(&spec.symbol),
        &spec.compute_kernel_attrs(relation),
    )?;
    let inserted = kernels.insert(relation.to_owned(), spec.symbol.clone());
    debug_assert!(inserted.is_none());
    Ok(spec.symbol)
}
