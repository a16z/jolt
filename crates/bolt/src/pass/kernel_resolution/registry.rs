mod ensure;
mod spec;

use crate::mlir::MlirError;

pub(super) use ensure::ensure_compute_kernel;
pub use spec::ComputeKernelSpec;

pub trait KernelRegistry {
    fn kernel_spec(&mut self, relation: &str) -> Result<ComputeKernelSpec, MlirError>;
}

impl<F> KernelRegistry for F
where
    F: FnMut(&str) -> Result<ComputeKernelSpec, MlirError>,
{
    fn kernel_spec(&mut self, relation: &str) -> Result<ComputeKernelSpec, MlirError> {
        self(relation)
    }
}
