mod attrs;
mod lowering;
mod shape;

use std::collections::BTreeMap;

use melior::ir::operation::OperationRef;
use melior::ir::Value;

use crate::ir::{BoltModule, Compute};
use crate::mlir::{MeliorContext, MlirError};

use crate::pass::kernel_resolution::registry::KernelRegistry;
use lowering::lower_kernel_sumcheck;
use shape::{KERNEL_CLAIM_SHAPE, KERNEL_DRIVER_SHAPE};

pub(super) fn lower_claim<'c, 'a, R>(
    context: &'c MeliorContext,
    kernelized: &'a BoltModule<'c, Compute>,
    value_map: &mut BTreeMap<String, Value<'c, 'a>>,
    kernels: &mut BTreeMap<String, String>,
    kernel_registry: &mut R,
    op: OperationRef<'_, '_>,
) -> Result<(), MlirError>
where
    R: KernelRegistry,
{
    lower_kernel_sumcheck(
        context,
        kernelized,
        value_map,
        kernels,
        kernel_registry,
        op,
        KERNEL_CLAIM_SHAPE,
    )
}

pub(super) fn lower_driver<'c, 'a, R>(
    context: &'c MeliorContext,
    kernelized: &'a BoltModule<'c, Compute>,
    value_map: &mut BTreeMap<String, Value<'c, 'a>>,
    kernels: &mut BTreeMap<String, String>,
    kernel_registry: &mut R,
    op: OperationRef<'_, '_>,
) -> Result<(), MlirError>
where
    R: KernelRegistry,
{
    lower_kernel_sumcheck(
        context,
        kernelized,
        value_map,
        kernels,
        kernel_registry,
        op,
        KERNEL_DRIVER_SHAPE,
    )
}
