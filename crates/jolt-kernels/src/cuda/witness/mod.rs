//! Proof-session ownership of packed trace windows and lazily extracted atoms.
//!
//! Optimized kernels share host rows or recollect compact bundles. CUDA retains
//! the uploaded window and its atom columns instead. `ResidentTraces::get`
//! checks witness identity, device, window base, and covered length before reuse.
//! A window includes its successor row when one exists, so next-row flags agree
//! at device boundaries; consumers process only the window's logical cycles.

#[cfg(feature = "allocative")]
use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::Key;
#[cfg(feature = "allocative")]
use allocative::Visitor;
use jolt_program::preprocess::JoltProgramPreprocessing;
use jolt_witness::RowSource;
use jolt_witness::WitnessBundle;
use jolt_witness::WitnessError;
use std::sync::{Arc, OnceLock};

use jolt_field::JoltField;
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
) -> Result<Vec<B>, WitnessError>
where
    F: JoltField,
    B: WitnessBundle + Copy + Send + Sync,
{
    crate::optimized::support::collect_rows::<B>(witness, cycles)
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
    devices: Vec<Vec<ResidentTrace>>,
}

impl ResidentTraces {
    fn get(&self, ordinal: usize, source: usize, window: &CycleWindow) -> Option<&ResidentTrace> {
        self.devices.get(ordinal)?.iter().find(|resident| {
            resident.source == source
                && resident.base == window.start
                && resident.cycles >= window.len
        })
    }

    fn park(&mut self, ordinal: usize, resident: ResidentTrace) {
        if self.devices.len() <= ordinal {
            self.devices.resize_with(ordinal + 1, Vec::new);
        }
        if let Some(slot) = self.devices.get_mut(ordinal) {
            slot.retain(|held| held.source == resident.source && held.base != resident.base);
            slot.push(resident);
        }
    }
}

#[cfg(feature = "allocative")]
impl Allocative for ResidentTraces {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        for resident in self.devices.iter().flatten() {
            visitor.visit_simple(Key::new("device_trace"), resident.trace.device_bytes());
            visitor.visit_simple(
                Key::new("device_atom_columns"),
                resident.atoms.get().map_or(0, |atoms| atoms.device_bytes()),
            );
        }
        visitor.exit();
    }
}

#[cfg(test)]
pub(crate) fn session_device_trace<F: JoltField>(
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

pub(crate) fn session_device_trace_window<F: JoltField>(
    context: &CudaKernelContext,
    session: &mut ProofSession,
    witness: &dyn JoltWitnessPlane<F>,
    cycles: usize,
    window: &CycleWindow,
) -> Result<Arc<DeviceTrace>, KernelError<F>> {
    session_row_trace_window(
        context,
        session,
        witness,
        witness.program_preprocessing(),
        cycles,
        window,
    )
}

pub(crate) fn session_row_trace_window<F: JoltField>(
    context: &CudaKernelContext,
    session: &mut ProofSession,
    witness: &dyn RowSource,
    preprocessing: &JoltProgramPreprocessing,
    _cycles: usize,
    window: &CycleWindow,
) -> Result<Arc<DeviceTrace>, KernelError<F>> {
    let identity = witness_identity(witness);
    let ordinal = context.ordinal();
    if let Some(resident) = session
        .state::<ResidentTraces>()
        .and_then(|traces| traces.get(ordinal, identity, window))
    {
        return Ok(Arc::clone(&resident.trace));
    }
    let trace = tracing::info_span!(
        "cuda_witness_residency",
        cycles = window.len,
        device = ordinal
    )
    .in_scope(|| {
        DeviceTrace::upload_source(
            Arc::clone(context.stream()),
            witness,
            window.start..window.start + window.len,
            preprocessing,
        )
    })?;
    let trace = Arc::new(trace);
    park_device_trace(session, ordinal, identity, window, Arc::clone(&trace));
    Ok(trace)
}

pub(crate) fn session_resident_trace(
    session: &ProofSession,
    ordinal: usize,
    source: usize,
    window: &CycleWindow,
) -> Option<Arc<DeviceTrace>> {
    session
        .state::<ResidentTraces>()
        .and_then(|traces| traces.get(ordinal, source, window))
        .map(|resident| Arc::clone(&resident.trace))
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

pub(crate) fn session_window_residency<F: JoltField>(
    context: &CudaKernelContext,
    session: &mut ProofSession,
    witness: &dyn JoltWitnessPlane<F>,
    cycles: usize,
    window: &CycleWindow,
) -> Result<(Arc<DeviceTrace>, Arc<DeviceAtomColumns>), KernelError<F>> {
    let resident = window.residency(cycles);
    Ok((
        session_device_trace_window(context, session, witness, cycles, &resident)?,
        session_atom_columns_window(context, session, witness, cycles, &resident)?,
    ))
}

#[cfg(test)]
pub(crate) fn session_atom_columns<F: JoltField>(
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

pub(crate) fn session_atom_columns_window<F: JoltField>(
    context: &CudaKernelContext,
    session: &mut ProofSession,
    witness: &dyn JoltWitnessPlane<F>,
    cycles: usize,
    window: &CycleWindow,
) -> Result<Arc<DeviceAtomColumns>, KernelError<F>> {
    let trace = session_device_trace_window(context, session, witness, cycles, window)?;
    let identity = witness_identity(witness);
    let resident = session
        .state::<ResidentTraces>()
        .and_then(|traces| traces.get(context.ordinal(), identity, window))
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

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations and fixture errors fail loudly"
)]
mod tests {
    use std::sync::Arc;

    use jolt_claims::protocols::jolt::JoltOneHotConfig;
    use jolt_field::Fr;
    use jolt_witness::backend::cuda::FLAG_BIT_NEXT_IS_NOOP;
    use jolt_witness::witnesses::{NextIsNoop, RamInc, RdInc};
    use jolt_witness::{FixedBackend, WitnessBundle, WitnessError};

    use super::{session_window_residency, witness_identity, ResidentTraces};
    use crate::cuda::common::context::current_device;
    use crate::cuda::common::devices::witness_windows;
    use crate::cuda::common::testing::{with_r1cs_witness, FixedPlane};
    use crate::cuda::{
        context_for, device_count, enter_device, shared_context, warm_shared_witness,
    };
    use crate::optimized::support::collect_rows;
    use crate::{KernelError, ProofSession};

    const ONE_HOT: JoltOneHotConfig = JoltOneHotConfig {
        log_k_chunk: 8,
        lookups_ra_virtual_log_k_chunk: 32,
    };
    const RAM_K: usize = 1 << 10;

    #[derive(Clone, Copy, WitnessBundle)]
    struct WarmupRow {
        rd_inc: RdInc,
        ram_inc: RamInc,
        next_is_noop: NextIsNoop,
    }

    #[test]
    fn warmup_populates_and_reuses_windows_matching_host_rows() {
        if shared_context().is_none() {
            return;
        }
        let selected = device_count() - 1;
        let _device = enter_device(selected);
        for log_t in [0, 6, 13] {
            let cycles = 1usize << log_t;
            let windows = witness_windows(cycles);
            with_r1cs_witness(log_t, RAM_K, ONE_HOT, 23, |witness| {
                let rows = collect_rows::<WarmupRow>(witness, cycles).expect("host rows");
                let source = witness_identity(witness);
                let mut session = ProofSession::default();
                warm_shared_witness::<Fr>(&mut session, witness, log_t).expect("warm-up");
                assert_eq!(current_device(), selected);
                let residents = session.state::<ResidentTraces>().expect("trace cache");
                assert_eq!(residents.devices.len(), windows.len());
                let mut allocations = Vec::new();
                for (ordinal, window) in windows.iter().enumerate() {
                    let context = context_for(ordinal).expect("window device");
                    let resident_window = window.residency(cycles);
                    assert_eq!(residents.devices[ordinal].len(), 1);
                    let resident = residents
                        .get(ordinal, source, &resident_window)
                        .expect("warmed window");
                    assert_eq!(resident.trace.cycles(), resident_window.len);
                    let atoms = resident.atoms.get().expect("eager atom extraction");
                    let host = &rows[resident_window.start..resident_window.end()];
                    assert_eq!(
                        context.download_u64(&atoms.rd_inc).expect("rd increments"),
                        host.iter()
                            .flat_map(|row| [row.rd_inc.0 as u64, (row.rd_inc.0 >> 64) as u64])
                            .collect::<Vec<_>>(),
                    );
                    assert_eq!(
                        context
                            .download_u64(&atoms.ram_inc)
                            .expect("RAM increments"),
                        host.iter()
                            .flat_map(|row| [row.ram_inc.0 as u64, (row.ram_inc.0 >> 64) as u64])
                            .collect::<Vec<_>>(),
                    );
                    let flags = context.download_u32(&atoms.flags).expect("flags");
                    assert_eq!(flags.len(), resident_window.len);
                    for (flag, row) in flags[..window.len].iter().zip(host) {
                        assert_eq!(flag & (1 << FLAG_BIT_NEXT_IS_NOOP) != 0, row.next_is_noop.0,);
                    }
                    allocations.push((Arc::clone(&resident.trace), Arc::clone(atoms)));
                }
                warm_shared_witness::<Fr>(&mut session, witness, log_t).expect("repeat warm-up");
                assert_eq!(current_device(), selected);
                for (ordinal, (window, (trace, atoms))) in
                    windows.iter().zip(allocations).enumerate()
                {
                    let context = context_for(ordinal).expect("window device");
                    let (served_trace, served_atoms) = session_window_residency::<Fr>(
                        context,
                        &mut session,
                        witness,
                        cycles,
                        window,
                    )
                    .expect("relation residency");
                    assert!(Arc::ptr_eq(&trace, &served_trace));
                    assert!(Arc::ptr_eq(&atoms, &served_atoms));
                }
            });
        }
    }

    #[test]
    fn warmup_is_scoped_to_the_session_and_witness() {
        if shared_context().is_none() {
            return;
        }
        let log_t = 6;
        let cycles = 1usize << log_t;
        let window = &witness_windows(cycles)[0];
        let context = context_for(0).expect("first device");
        with_r1cs_witness(log_t, RAM_K, ONE_HOT, 23, |first| {
            let mut session = ProofSession::default();
            warm_shared_witness::<Fr>(&mut session, first, log_t).expect("first warm-up");
            let (first_trace, first_atoms) =
                session_window_residency::<Fr>(context, &mut session, first, cycles, window)
                    .expect("first residency");

            let mut other_session = ProofSession::default();
            warm_shared_witness::<Fr>(&mut other_session, first, log_t)
                .expect("other session warm-up");
            let (other_trace, other_atoms) =
                session_window_residency::<Fr>(context, &mut other_session, first, cycles, window)
                    .expect("other session residency");
            assert!(!Arc::ptr_eq(&first_trace, &other_trace));
            assert!(!Arc::ptr_eq(&first_atoms, &other_atoms));

            with_r1cs_witness(log_t, RAM_K, ONE_HOT, 47, |second| {
                warm_shared_witness::<Fr>(&mut session, second, log_t)
                    .expect("second witness warm-up");
                let residents = session.state::<ResidentTraces>().expect("trace cache");
                assert!(residents.get(0, witness_identity(first), window).is_none());
                assert_eq!(residents.devices[0].len(), 1);
                let resident = residents
                    .get(0, witness_identity(second), window)
                    .expect("second witness residency");
                assert!(!Arc::ptr_eq(&first_trace, &resident.trace));
                assert!(!Arc::ptr_eq(
                    &first_atoms,
                    resident.atoms.get().expect("second witness atoms"),
                ));
            });
        });
    }

    #[test]
    fn warmup_propagates_row_errors_and_restores_the_selected_device() {
        if shared_context().is_none() {
            return;
        }
        let selected = device_count() - 1;
        let _device = enter_device(selected);
        let witness = FixedPlane::with_log_t(FixedBackend::default(), "warm-up row error", Some(6));
        let mut session = ProofSession::default();
        assert!(matches!(
            warm_shared_witness::<Fr>(&mut session, &witness, 6),
            Err(KernelError::Witness(WitnessError::InvalidWitnessData {
                label: "warm-up row error",
                ..
            })),
        ));
        assert_eq!(current_device(), selected);
        assert!(session.state::<ResidentTraces>().is_none());
        with_r1cs_witness(6, RAM_K, ONE_HOT, 23, |witness| {
            warm_shared_witness::<Fr>(&mut session, witness, 6).expect("retry with rows");
            assert_eq!(current_device(), selected);
            assert!(session.state::<ResidentTraces>().is_some());
        });
    }
}
