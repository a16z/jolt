use std::sync::Arc;

use jolt_field::Field;
use jolt_witness::backend::cuda::DeviceTrace;
use jolt_witness::JoltWitnessPlane;

use super::common::context::CudaKernelContext;
use super::common::device_columns::witness_identity;
use crate::{KernelError, ProofSession};

#[tracing::instrument(
    skip_all,
    name = "cuda_collect_rows",
    fields(bundle = core::any::type_name::<B>(), cycles)
)]
pub(crate) fn collect_rows<F, B>(
    witness: &dyn JoltWitnessPlane<F>,
    cycles: usize,
) -> Result<Vec<B>, jolt_witness::WitnessError>
where
    F: Field,
    B: jolt_witness::WitnessBundle + Copy + Send + Sync,
{
    crate::optimized::support::collect_rows::<F, B>(witness, cycles)
}

pub(crate) struct ResidentTrace {
    source: usize,
    cycles: usize,
    trace: Arc<DeviceTrace>,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for ResidentTrace {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(
            allocative::Key::new("device_trace"),
            self.trace.device_bytes(),
        );
        visitor.exit();
    }
}

pub(crate) fn session_device_trace<F: Field>(
    context: &CudaKernelContext,
    session: &mut ProofSession,
    witness: &dyn JoltWitnessPlane<F>,
    cycles: usize,
) -> Result<Arc<DeviceTrace>, KernelError<F>> {
    let identity = witness_identity(witness);
    if let Some(resident) = session.state::<ResidentTrace>() {
        if resident.source == identity && resident.cycles == cycles {
            return Ok(Arc::clone(&resident.trace));
        }
    }
    let rows = witness.rows().ok_or(KernelError::Unsupported {
        reason: "the CUDA backend needs a slice-backed trace source to build its device residency",
    })?;
    let trace = tracing::info_span!("cuda_witness_residency", cycles).in_scope(|| {
        DeviceTrace::upload(
            Arc::clone(context.stream()),
            rows,
            cycles,
            witness.program_preprocessing(),
        )
    })?;
    let trace = Arc::new(trace);
    session.park(ResidentTrace {
        source: identity,
        cycles,
        trace: Arc::clone(&trace),
    });
    Ok(trace)
}
