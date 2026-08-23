use std::sync::{Arc, OnceLock};

use jolt_field::Field;
use jolt_witness::backend::cuda::{DeviceAtomColumns, DeviceTrace};
use jolt_witness::JoltWitnessPlane;

use super::common::context::CudaKernelContext;
use super::common::device_columns::witness_identity;
use super::common::devices::CycleWindow;
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
    base: usize,
    trace: Arc<DeviceTrace>,
    atoms: OnceLock<Arc<DeviceAtomColumns>>,
}

#[derive(Default)]
pub(crate) struct ResidentTraces {
    devices: Vec<Option<ResidentTrace>>,
}

impl ResidentTraces {
    fn get(&self, ordinal: usize) -> Option<&ResidentTrace> {
        self.devices.get(ordinal)?.as_ref()
    }

    fn park(&mut self, ordinal: usize, resident: ResidentTrace) {
        if self.devices.len() <= ordinal {
            self.devices.resize_with(ordinal + 1, || None);
        }
        if let Some(slot) = self.devices.get_mut(ordinal) {
            *slot = Some(resident);
        }
    }
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for ResidentTraces {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        for resident in self.devices.iter().flatten() {
            visitor.visit_simple(
                allocative::Key::new("device_trace"),
                resident.trace.device_bytes(),
            );
            visitor.visit_simple(
                allocative::Key::new("device_atom_columns"),
                resident.atoms.get().map_or(0, |atoms| atoms.device_bytes()),
            );
        }
        visitor.exit();
    }
}

pub(crate) fn session_device_trace<F: Field>(
    context: &CudaKernelContext,
    session: &mut ProofSession,
    witness: &dyn JoltWitnessPlane<F>,
    cycles: usize,
) -> Result<Arc<DeviceTrace>, KernelError<F>> {
    session_device_trace_window(
        context,
        session,
        witness,
        cycles,
        &CycleWindow {
            start: 0,
            len: cycles,
        },
    )
}

pub(crate) fn session_device_trace_window<F: Field>(
    context: &CudaKernelContext,
    session: &mut ProofSession,
    witness: &dyn JoltWitnessPlane<F>,
    cycles: usize,
    window: &CycleWindow,
) -> Result<Arc<DeviceTrace>, KernelError<F>> {
    let identity = witness_identity(witness);
    let ordinal = context.ordinal();
    if let Some(resident) = session
        .state::<ResidentTraces>()
        .and_then(|traces| traces.get(ordinal))
    {
        if resident.source == identity
            && resident.cycles == window.len
            && resident.base == window.start
        {
            return Ok(Arc::clone(&resident.trace));
        }
    }
    let rows = witness.rows().ok_or(KernelError::Unsupported {
        reason: "the CUDA backend needs a slice-backed trace source to build its device residency",
    })?;
    let trace = tracing::info_span!(
        "cuda_witness_residency",
        cycles = window.len,
        device = ordinal
    )
    .in_scope(|| {
        DeviceTrace::upload_window(
            Arc::clone(context.stream()),
            rows,
            cycles,
            window.start,
            window.len,
            witness.program_preprocessing(),
        )
    })?;
    let trace = Arc::new(trace);
    park_device_trace(session, ordinal, identity, window, Arc::clone(&trace));
    Ok(trace)
}

pub(crate) fn park_device_trace(
    session: &mut ProofSession,
    ordinal: usize,
    source: usize,
    window: &CycleWindow,
    trace: Arc<DeviceTrace>,
) {
    session.state_or_insert_with(ResidentTraces::default).park(
        ordinal,
        ResidentTrace {
            source,
            cycles: window.len,
            base: window.start,
            trace,
            atoms: OnceLock::new(),
        },
    );
}

pub(crate) fn session_window_residency<F: Field>(
    context: &CudaKernelContext,
    session: &mut ProofSession,
    witness: &dyn JoltWitnessPlane<F>,
    cycles: usize,
    window: &CycleWindow,
) -> Result<(Arc<DeviceTrace>, Arc<DeviceAtomColumns>), KernelError<F>> {
    if window.start == 0 {
        return Ok((
            session_device_trace(context, session, witness, cycles)?,
            session_atom_columns(context, session, witness, cycles)?,
        ));
    }
    let resident = window.residency(cycles);
    Ok((
        session_device_trace_window(context, session, witness, cycles, &resident)?,
        session_atom_columns_window(context, session, witness, cycles, &resident)?,
    ))
}

pub(crate) fn session_atom_columns<F: Field>(
    context: &CudaKernelContext,
    session: &mut ProofSession,
    witness: &dyn JoltWitnessPlane<F>,
    cycles: usize,
) -> Result<Arc<DeviceAtomColumns>, KernelError<F>> {
    session_atom_columns_window(
        context,
        session,
        witness,
        cycles,
        &CycleWindow {
            start: 0,
            len: cycles,
        },
    )
}

pub(crate) fn session_atom_columns_window<F: Field>(
    context: &CudaKernelContext,
    session: &mut ProofSession,
    witness: &dyn JoltWitnessPlane<F>,
    cycles: usize,
    window: &CycleWindow,
) -> Result<Arc<DeviceAtomColumns>, KernelError<F>> {
    let trace = session_device_trace_window(context, session, witness, cycles, window)?;
    let resident = session
        .state::<ResidentTraces>()
        .and_then(|traces| traces.get(context.ordinal()))
        .ok_or(KernelError::InvariantViolation {
            reason: "the device residency was parked without an atom-column cache",
        })?;
    if let Some(columns) = resident.atoms.get() {
        return Ok(Arc::clone(columns));
    }
    let columns = Arc::new(trace.atom_columns()?);
    let _ = resident.atoms.set(Arc::clone(&columns));
    Ok(columns)
}
