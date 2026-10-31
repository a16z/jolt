//! Proof-session ownership of packed trace windows and lazily extracted atoms.
//!
//! Optimized kernels share host rows or recollect compact bundles. CUDA retains
//! the uploaded window and caches execution, register/increment, and bytecode
//! atom groups. `ResidentTraces::get` checks witness identity, device, window base,
//! and covered length before reuse.
//! A window includes its successor row when one exists, so next-row flags agree
//! at device boundaries; consumers process only the window's logical cycles.
//!
//! Windows below 2^25 cycles extract all groups in one pass: staging them saved
//! little peak memory and added later-stage latency at 2^24 on one/two L4s.
//! Larger windows extract groups on demand, following optimized's narrower
//! collection scope. At 2^25 on one L4 and 2^26 on two, this reduced peak live
//! memory for clear proofs with comparable whole-proof time, despite extra work
//! in later stages. Explicit warm-up materializes every group.
//!
//! Keeping the cache through the joint opening avoids deallocation work on that
//! stage's critical path. For SHA-2-chain at 2^24 on one and two L4s, retiring it
//! after building the opening views slowed stage 8 even with background
//! destruction, without reducing the peak live allocation reached earlier.

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
use jolt_witness::backend::cuda::{
    DeviceAtomColumns, DeviceBytecodeColumns, DeviceExecutionColumns, DeviceMemoryColumns,
    DeviceTrace,
};
use jolt_witness::JoltWitnessPlane;

use super::common::context::CudaKernelContext;
use super::common::device_columns::witness_identity;
use super::common::devices::CycleWindow;
use crate::{KernelError, ProofSession};

const STAGED_ATOM_MIN_CYCLES: usize = 1 << 25;

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
    execution: OnceLock<Arc<DeviceExecutionColumns>>,
    memory: OnceLock<Arc<DeviceMemoryColumns>>,
    bytecode: OnceLock<Arc<DeviceBytecodeColumns>>,
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
                resident
                    .execution
                    .get()
                    .map_or(0, |columns| columns.device_bytes())
                    + resident
                        .memory
                        .get()
                        .map_or(0, |columns| columns.device_bytes())
                    + resident
                        .bytecode
                        .get()
                        .map_or(0, |columns| columns.device_bytes()),
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
            execution: OnceLock::new(),
            memory: OnceLock::new(),
            bytecode: OnceLock::new(),
        },
    );
}

trait AtomGroup: Sized {
    fn cache(resident: &ResidentTrace) -> &OnceLock<Arc<Self>>;
    fn extract(trace: &DeviceTrace) -> Result<Self, WitnessError>;
}

macro_rules! atom_group {
    ($type:ty, $field:ident, $extract:ident, $window:ident, $columns:ident $(, $full:ident)?) => {
        impl AtomGroup for $type {
            fn cache(resident: &ResidentTrace) -> &OnceLock<Arc<Self>> {
                &resident.$field
            }

            fn extract(trace: &DeviceTrace) -> Result<Self, WitnessError> {
                trace.$extract()
            }
        }

        pub(crate) fn $window<F: JoltField>(
            context: &CudaKernelContext,
            session: &mut ProofSession,
            witness: &dyn JoltWitnessPlane<F>,
            cycles: usize,
            window: &CycleWindow,
        ) -> Result<(Arc<DeviceTrace>, Arc<$type>), KernelError<F>> {
            let resident = window.residency(cycles);
            Ok((
                session_device_trace_window(context, session, witness, cycles, &resident)?,
                $columns(context, session, witness, cycles, &resident)?,
            ))
        }

        pub(crate) fn $columns<F: JoltField>(
            context: &CudaKernelContext,
            session: &mut ProofSession,
            witness: &dyn JoltWitnessPlane<F>,
            cycles: usize,
            window: &CycleWindow,
        ) -> Result<Arc<$type>, KernelError<F>> {
            session_columns_window::<F, $type>(context, session, witness, cycles, window)
        }

        $(#[cfg(test)]
        pub(crate) fn $full<F: JoltField>(
            context: &CudaKernelContext,
            session: &mut ProofSession,
            witness: &dyn JoltWitnessPlane<F>,
            cycles: usize,
        ) -> Result<Arc<$type>, KernelError<F>> {
            $columns(
                context,
                session,
                witness,
                cycles,
                &CycleWindow { start: 0, len: cycles },
            )
        })?
    };
}

atom_group!(
    DeviceExecutionColumns,
    execution,
    execution_columns,
    session_execution_window,
    session_execution_columns_window,
    session_execution_columns
);
atom_group!(
    DeviceMemoryColumns,
    memory,
    memory_columns,
    session_memory_window,
    session_memory_columns_window,
    session_memory_columns
);
#[cfg(not(feature = "field-inline"))]
atom_group!(
    DeviceBytecodeColumns,
    bytecode,
    bytecode_columns,
    session_bytecode_window,
    session_bytecode_columns_window
);

fn session_columns_window<F: JoltField, C: AtomGroup>(
    context: &CudaKernelContext,
    session: &mut ProofSession,
    witness: &dyn JoltWitnessPlane<F>,
    cycles: usize,
    window: &CycleWindow,
) -> Result<Arc<C>, KernelError<F>> {
    let _ = session_device_trace_window(context, session, witness, cycles, window)?;
    let identity = witness_identity(witness);
    let resident = session
        .state::<ResidentTraces>()
        .and_then(|traces| traces.get(context.ordinal(), identity, window))
        .ok_or(KernelError::InvariantViolation {
            reason: "the device residency was parked without an atom-column cache",
        })?;
    if C::cache(resident).get().is_none() && resident.cycles < STAGED_ATOM_MIN_CYCLES {
        resident.warm()?;
    }
    Ok(resident.columns::<C>()?)
}

impl ResidentTrace {
    fn columns<C: AtomGroup>(&self) -> Result<Arc<C>, WitnessError> {
        if let Some(columns) = C::cache(self).get() {
            return Ok(Arc::clone(columns));
        }
        let columns = Arc::new(C::extract(&self.trace)?);
        let _ = C::cache(self).set(Arc::clone(&columns));
        Ok(columns)
    }

    fn warm(&self) -> Result<(), WitnessError> {
        if self.execution.get().is_none()
            && self.memory.get().is_none()
            && self.bytecode.get().is_none()
        {
            let DeviceAtomColumns {
                execution,
                memory,
                bytecode,
            } = self.trace.atom_columns()?;
            let _ = self.execution.set(execution);
            let _ = self.memory.set(memory);
            let _ = self.bytecode.set(bytecode);
        } else {
            if self.execution.get().is_none() {
                let _ = self
                    .execution
                    .set(Arc::new(self.trace.execution_columns()?));
            }
            if self.memory.get().is_none() {
                let _ = self.memory.set(Arc::new(self.trace.memory_columns()?));
            }
            if self.bytecode.get().is_none() {
                let _ = self.bytecode.set(Arc::new(self.trace.bytecode_columns()?));
            }
        }
        Ok(())
    }
}

pub(crate) fn warm_window<F: JoltField>(
    context: &CudaKernelContext,
    session: &mut ProofSession,
    witness: &dyn JoltWitnessPlane<F>,
    cycles: usize,
    window: &CycleWindow,
) -> Result<(), KernelError<F>> {
    let window = window.residency(cycles);
    let _ = session_device_trace_window(context, session, witness, cycles, &window)?;
    let resident = session
        .state::<ResidentTraces>()
        .and_then(|traces| traces.get(context.ordinal(), witness_identity(witness), &window))
        .ok_or(KernelError::InvariantViolation {
            reason: "missing warm-up residency",
        })?;
    resident.warm()?;
    Ok(())
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
    use jolt_witness::backend::cuda::{DeviceExecutionColumns, FLAG_BIT_NEXT_IS_NOOP};
    use jolt_witness::witnesses::{NextIsNoop, RamInc, RdInc};
    use jolt_witness::{FixedBackend, WitnessBundle, WitnessError};

    use super::{
        session_device_trace_window, session_execution_window, witness_identity, ResidentTraces,
    };
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
                    let atoms = resident
                        .execution
                        .get()
                        .expect("eager execution extraction");
                    let memory = resident.memory.get().expect("eager memory extraction");
                    assert!(resident.bytecode.get().is_some());
                    let host = &rows[resident_window.start..resident_window.end()];
                    assert_eq!(
                        context.download_u64(&memory.rd_inc).expect("rd increments"),
                        host.iter()
                            .flat_map(|row| [row.rd_inc.0 as u64, (row.rd_inc.0 >> 64) as u64])
                            .collect::<Vec<_>>(),
                    );
                    assert_eq!(
                        context
                            .download_u64(&memory.ram_inc)
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
                    let (served_trace, served_atoms) = session_execution_window::<Fr>(
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
    fn warming_partial_residency_preserves_existing_columns() {
        let Some(context) = context_for(0) else {
            return;
        };
        let log_t = 6;
        let cycles = 1 << log_t;
        let window = witness_windows(cycles)[0].residency(cycles);
        with_r1cs_witness(log_t, RAM_K, ONE_HOT, 23, |witness| {
            let mut session = ProofSession::default();
            let source = witness_identity(witness);
            let _ =
                session_device_trace_window::<Fr>(context, &mut session, witness, cycles, &window)
                    .expect("packed window");
            let execution = {
                let resident = session
                    .state::<ResidentTraces>()
                    .expect("cache")
                    .get(0, source, &window)
                    .expect("window");
                let execution = resident
                    .columns::<DeviceExecutionColumns>()
                    .expect("execution group");
                assert!(resident.memory.get().is_none());
                assert!(resident.bytecode.get().is_none());
                execution
            };
            warm_shared_witness::<Fr>(&mut session, witness, log_t).expect("complete warm-up");
            let resident = session
                .state::<ResidentTraces>()
                .expect("cache")
                .get(0, source, &window)
                .expect("window");
            assert!(Arc::ptr_eq(
                &execution,
                resident.execution.get().expect("execution cache")
            ));
            let memory = Arc::clone(resident.memory.get().expect("memory cache"));
            let bytecode = Arc::clone(resident.bytecode.get().expect("bytecode cache"));
            warm_shared_witness::<Fr>(&mut session, witness, log_t).expect("repeat warm-up");
            let resident = session
                .state::<ResidentTraces>()
                .expect("cache")
                .get(0, source, &window)
                .expect("window");
            assert!(Arc::ptr_eq(
                &execution,
                resident.execution.get().expect("execution cache")
            ));
            assert!(Arc::ptr_eq(
                &memory,
                resident.memory.get().expect("memory cache")
            ));
            assert!(Arc::ptr_eq(
                &bytecode,
                resident.bytecode.get().expect("bytecode cache")
            ));
        });
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
                session_execution_window::<Fr>(context, &mut session, first, cycles, window)
                    .expect("first residency");

            let mut other_session = ProofSession::default();
            warm_shared_witness::<Fr>(&mut other_session, first, log_t)
                .expect("other session warm-up");
            let (other_trace, other_atoms) =
                session_execution_window::<Fr>(context, &mut other_session, first, cycles, window)
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
                    resident.execution.get().expect("second witness atoms"),
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
