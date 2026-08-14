use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

use jolt_field::AkitaField;
use jolt_sumcheck::SumcheckError;
use jolt_verifier::stages::stage4::registers_read_write_checking::RegistersReadWriteChecking;
use jolt_witness::{JoltWitnessPlane, OwnedRows};

use super::backend::MetalBackend;
use crate::optimized::registers_read_write::{
    OptimizedRegistersReadWrite, PreparedRegisterEntries,
};
use crate::{KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegistersReadWriteMetalConfig {
    pub precompute_cutoff_elements: usize,
}

impl Default for RegistersReadWriteMetalConfig {
    fn default() -> Self {
        Self {
            precompute_cutoff_elements: usize::MAX,
        }
    }
}

struct RegistersReadWritePrefetchSource {
    cycles: usize,
    physical_rows: usize,
    rows: OwnedRows,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for RegistersReadWritePrefetchSource {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        visitor.enter_self_sized::<Self>().exit();
    }
}

enum RegistersReadWritePrefetchState {
    Running,
    Ready {
        result: Result<PreparedRegisterEntries<AkitaField>, String>,
        service_ns: u64,
    },
    Taken,
}

struct RegistersReadWritePrefetch {
    cycles: usize,
    physical_rows: usize,
    state: Arc<Mutex<RegistersReadWritePrefetchState>>,
    handle: Option<JoinHandle<()>>,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for RegistersReadWritePrefetch {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        if let Ok(state) = self.state.try_lock() {
            if let RegistersReadWritePrefetchState::Ready {
                result: Ok(prepared),
                ..
            } = &*state
            {
                visitor.visit_simple(
                    allocative::Key::new("precomputed_sparse_state"),
                    prepared.heap_bytes(),
                );
            }
        }
        visitor.exit();
    }
}

impl Drop for RegistersReadWritePrefetch {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

struct JoinedRegistersReadWritePrefetch {
    prepared: PreparedRegisterEntries<AkitaField>,
    completed_before_join: bool,
    wait_ns: u64,
    service_ns: u64,
}

impl RegistersReadWritePrefetch {
    fn spawn(source: RegistersReadWritePrefetchSource) -> Result<Self, KernelError<AkitaField>> {
        let cycles = source.cycles;
        let physical_rows = source.physical_rows;
        let worker_threads = rayon::current_num_threads();
        let state = Arc::new(Mutex::new(RegistersReadWritePrefetchState::Running));
        let worker_state = Arc::clone(&state);
        let handle = std::thread::Builder::new()
            .name("jolt-registers-rw-prefetch".to_owned())
            .spawn(move || {
                let span = tracing::info_span!(
                    "MetalRegistersReadWrite::prefetch_service",
                    cycles,
                    physical_rows,
                    random_access = true,
                    source_extraction_passes = 1usize,
                    entries = tracing::field::Empty,
                    entry_chunks = tracing::field::Empty,
                    heap_bytes = tracing::field::Empty,
                    service_ns = tracing::field::Empty,
                    complete = tracing::field::Empty,
                );
                let _entered = span.enter();
                let started = Instant::now();
                let result = rayon::ThreadPoolBuilder::new()
                    .num_threads(worker_threads)
                    .thread_name(|index| format!("jolt-registers-rw-{index}"))
                    .build()
                    .map_err(|error| error.to_string())
                    .and_then(|pool| {
                        pool.install(|| {
                            OptimizedRegistersReadWrite::precompute_owned(source.rows, cycles)
                                .map_err(|error| error.to_string())
                        })
                    });
                let service_ns = duration_nanos(started.elapsed());
                let _ = span.record(
                    "entries",
                    result.as_ref().map_or(0, PreparedRegisterEntries::entries),
                );
                let _ = span.record(
                    "entry_chunks",
                    result
                        .as_ref()
                        .map_or(0, PreparedRegisterEntries::entry_chunks),
                );
                let _ = span.record(
                    "heap_bytes",
                    result
                        .as_ref()
                        .map_or(0, PreparedRegisterEntries::heap_bytes),
                );
                let _ = span.record("service_ns", service_ns);
                let _ = span.record("complete", result.is_ok());
                if let Ok(mut state) = worker_state.lock() {
                    *state = RegistersReadWritePrefetchState::Ready { result, service_ns };
                }
            })
            .map_err(metal_prepare_error)?;
        Ok(Self {
            cycles,
            physical_rows,
            state,
            handle: Some(handle),
        })
    }

    fn join(mut self) -> Result<JoinedRegistersReadWritePrefetch, KernelError<AkitaField>> {
        let completed_before_join = self
            .handle
            .as_ref()
            .is_some_and(std::thread::JoinHandle::is_finished);
        let started = Instant::now();
        let handle = self.handle.take().ok_or(KernelError::InvariantViolation {
            reason: "registers read-write prefetch handle was already consumed",
        })?;
        handle.join().map_err(|_| KernelError::InvariantViolation {
            reason: "registers read-write prefetch worker panicked",
        })?;
        let wait_ns = duration_nanos(started.elapsed());
        let mut state = self
            .state
            .lock()
            .map_err(|_| metal_prepare_error("registers read-write prefetch state was poisoned"))?;
        let RegistersReadWritePrefetchState::Ready { result, service_ns } =
            std::mem::replace(&mut *state, RegistersReadWritePrefetchState::Taken)
        else {
            return Err(KernelError::InvariantViolation {
                reason: "registers read-write prefetch completed without publishing its state",
            });
        };
        let prepared = result.map_err(metal_prepare_error)?;
        Ok(JoinedRegistersReadWritePrefetch {
            prepared,
            completed_before_join,
            wait_ns,
            service_ns,
        })
    }
}

impl MetalBackend {
    pub(super) fn publish_registers_read_write_prefetch_source(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        cycles: usize,
    ) -> Result<(), KernelError<AkitaField>> {
        let cutoff = self.config.registers_read_write.precompute_cutoff_elements;
        if cutoff == usize::MAX || cycles < cutoff {
            return Ok(());
        }
        if session
            .state::<RegistersReadWritePrefetchSource>()
            .is_some()
            || session.state::<RegistersReadWritePrefetch>().is_some()
        {
            return Err(KernelError::InvariantViolation {
                reason: "registers read-write prefetch source was published more than once",
            });
        }
        let Some(rows) = witness.owned_rows().filter(|rows| cycles <= rows.cycles()) else {
            return Ok(());
        };
        let physical_rows = rows.physical_rows().min(cycles);
        let span = tracing::info_span!(
            "MetalRegistersReadWrite::prefetch_source_publish",
            cycles,
            physical_rows,
            random_access = true,
            complete = true,
        );
        let _entered = span.enter();
        session.park(RegistersReadWritePrefetchSource {
            cycles,
            physical_rows,
            rows,
        });
        Ok(())
    }
}

impl PrepareKernel<AkitaField, RegistersReadWriteChecking<AkitaField>> for MetalBackend {
    fn prefetch(&self, session: &mut ProofSession) -> Result<(), KernelError<AkitaField>> {
        let Some(source) = session.take::<RegistersReadWritePrefetchSource>() else {
            return Ok(());
        };
        if session.state::<RegistersReadWritePrefetch>().is_some() {
            return Err(KernelError::InvariantViolation {
                reason: "registers read-write prefetch was submitted more than once",
            });
        }
        let span = tracing::info_span!(
            "MetalRegistersReadWrite::prefetch_submit",
            cycles = source.cycles,
            physical_rows = source.physical_rows,
            worker_threads = rayon::current_num_threads(),
            complete = tracing::field::Empty,
        );
        let _entered = span.enter();
        let prefetch = RegistersReadWritePrefetch::spawn(source)?;
        let _ = span.record("complete", true);
        session.park(prefetch);
        Ok(())
    }

    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        inputs: ProverInputs<'_, AkitaField, RegistersReadWriteChecking<AkitaField>>,
    ) -> Result<
        Box<dyn SumcheckKernel<AkitaField, Relation = RegistersReadWriteChecking<AkitaField>>>,
        KernelError<AkitaField>,
    > {
        let dimensions = inputs.relation.register_dimensions();
        let cycles = 1usize.checked_shl(dimensions.log_t() as u32).ok_or(
            KernelError::InvariantViolation {
                reason: "registers read-write trace domain overflows usize",
            },
        )?;
        let cutoff = self.config.registers_read_write.precompute_cutoff_elements;
        let requested = cutoff != usize::MAX && cycles >= cutoff;
        let prefetch = requested
            .then(|| session.take::<RegistersReadWritePrefetch>())
            .flatten();
        let route = tracing::info_span!(
            "MetalRegistersReadWrite::route",
            cycles,
            requested = if requested {
                "prefetched_cpu"
            } else {
                "optimized_cpu"
            },
            realized_route = tracing::field::Empty,
            fallback_reason = tracing::field::Empty,
        );
        let _entered = route.enter();

        let Some(prefetch) = prefetch else {
            let _ = route.record("realized_route", "optimized_cpu");
            let _ = route.record(
                "fallback_reason",
                if requested {
                    "prefetch_missing"
                } else {
                    "below_cutoff"
                },
            );
            let prepared = OptimizedRegistersReadWrite.prepare(session, witness, inputs)?;
            super::instruction_read_raf::start_instruction_read_raf_scatter(session)?;
            return Ok(prepared);
        };
        if prefetch.cycles != cycles {
            return Err(KernelError::InvariantViolation {
                reason: "registers read-write prefetch has stale geometry",
            });
        }
        let physical_rows = prefetch.physical_rows;
        let join_span = tracing::info_span!(
            "MetalRegistersReadWrite::prefetch_join",
            cycles,
            physical_rows,
            completed_before_join = tracing::field::Empty,
            wait_ns = tracing::field::Empty,
            service_ns = tracing::field::Empty,
            entries = tracing::field::Empty,
            entry_chunks = tracing::field::Empty,
            heap_bytes = tracing::field::Empty,
            complete = tracing::field::Empty,
        );
        let _join_entered = join_span.enter();
        let joined = prefetch.join()?;
        let _ = join_span.record("completed_before_join", joined.completed_before_join);
        let _ = join_span.record("wait_ns", joined.wait_ns);
        let _ = join_span.record("service_ns", joined.service_ns);
        let _ = join_span.record("entries", joined.prepared.entries());
        let _ = join_span.record("entry_chunks", joined.prepared.entry_chunks());
        let _ = join_span.record("heap_bytes", joined.prepared.heap_bytes());
        let _ = join_span.record("complete", true);
        drop(_join_entered);
        if joined.prepared.cycles() != cycles {
            return Err(KernelError::InvariantViolation {
                reason: "prefetched registers read-write state has stale geometry",
            });
        }
        let _ = route.record("realized_route", "prefetched_cpu");
        let _ = route.record("fallback_reason", "none");
        let prepared =
            OptimizedRegistersReadWrite::prepare_precomputed(session, inputs, joined.prepared)?;
        super::instruction_read_raf::start_instruction_read_raf_scatter(session)?;
        Ok(prepared)
    }
}

fn metal_prepare_error(error: impl ToString) -> KernelError<AkitaField> {
    SumcheckError::ComputeBackend {
        backend: "metal",
        message: error.to_string(),
    }
    .into()
}

fn duration_nanos(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}
