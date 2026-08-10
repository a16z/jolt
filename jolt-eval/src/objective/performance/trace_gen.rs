use common::constants::RAM_START_ADDRESS;
use jolt_program::execution::{JoltProgram, TraceInputs};
use jolt_prover_legacy::host::Program;
use tracer::TracerBackend;

use crate::guests::GuestConfig;
use crate::objective::Objective;

/// Everything needed to call `ExecutionBackend::trace`, built once so that
/// guest compilation and `JoltProgram` construction (ELF decode + expansion)
/// are excluded from the measurement.
pub struct TraceGenSetup {
    pub program: JoltProgram,
    pub inputs: TraceInputs,
    /// Row count of the resulting trace (for Criterion `Throughput::Elements`).
    pub trace_len: usize,
}

/// Builds the guest and constructs the modular-seam artifacts without tracing.
pub fn build_trace_setup<G: GuestConfig>(guest: &G) -> (JoltProgram, TraceInputs) {
    let mut host_program = Program::new(guest.package());
    let mut memory_config = guest.memory_config();
    host_program.set_memory_config(memory_config);
    let program = host_program
        .jolt_program()
        .expect("failed to build Jolt program");
    memory_config.program_size = Some(program.program_end - RAM_START_ADDRESS);
    let inputs = TraceInputs::new(guest.input(), Vec::new(), Vec::new(), memory_config);
    (program, inputs)
}

/// Benchmarks trace generation (eager `ExecutionBackend::trace`) for a guest
/// program. Backend-generic: variants beyond `reference` are added as
/// Criterion ids within the same bench target.
pub struct TraceGenObjective<G: GuestConfig> {
    guest: G,
    name: String,
}

impl<G: GuestConfig> Default for TraceGenObjective<G> {
    fn default() -> Self {
        Self::new(G::default())
    }
}

impl<G: GuestConfig> TraceGenObjective<G> {
    pub fn new(guest: G) -> Self {
        let name = format!("trace_gen_{}", guest.label());
        Self { guest, name }
    }

    /// One eager trace with the AOT x86-64 backend (record mode).
    ///
    /// Only present on x86_64 Linux; elsewhere `NativeBackend` is the
    /// interpreter and this arm would duplicate `run_reference`.
    ///
    /// The backend is a parameter (not constructed here) so its compile
    /// cache survives across bench iterations: a fresh backend would pay
    /// the one-time AOT compile inside the measured region, which for the
    /// small guests is on the order of the fast pass itself. Callers warm
    /// the cache with one un-timed run (the phase3_baseline pattern).
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    pub fn run_x86(
        &self,
        backend: &mut jolt_tracer_x86::X86TracerBackend,
        setup: &TraceGenSetup,
    ) -> usize {
        let output = setup
            .program
            .trace_with(backend, setup.inputs.clone())
            .expect("x86 trace failed");
        std::hint::black_box(output.trace.rows().len())
    }

    /// One fast (non-recording) pass with the AOT x86-64 backend. See
    /// [`Self::run_x86`] for why the backend is a parameter.
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    pub fn run_x86_fast(
        &self,
        backend: &mut jolt_tracer_x86::X86TracerBackend,
        setup: &TraceGenSetup,
    ) -> usize {
        let output = backend
            .fast_run(&setup.program, setup.inputs.clone())
            .expect("x86 fast run failed");
        std::hint::black_box(output.trace_len)
    }

    /// One eager trace with the reference interpreter backend.
    pub fn run_reference(&self, setup: &TraceGenSetup) -> usize {
        let mut backend = TracerBackend::new();
        let output = setup
            .program
            .trace_with(&mut backend, setup.inputs.clone())
            .expect("reference trace failed");
        std::hint::black_box(output.trace.rows().len())
    }

    /// The same trace without the `Cycle` to `TraceRow` conversion, i.e. the
    /// raw `tracer::trace` call that `TracerBackend::trace` wraps.
    ///
    /// WHY this exists: the conversion pass dominates the seam (measured at
    /// 74% for fibonacci and 86% for sha2-chain), and a backend that emits
    /// `TraceRow` directly skips it entirely. Reporting only the seam total
    /// would let a backend bank that share as if it were codegen speedup, so
    /// AC8/AC9 ratios are only interpretable against both numbers: `reference`
    /// bounds the end-to-end win, `reference_raw` isolates the part that is
    /// actually attributable to how rows are produced.
    pub fn run_reference_raw(&self, setup: &TraceGenSetup) -> usize {
        raw_trace_cycles(&setup.program, &setup.inputs)
    }
}

/// Raw `tracer::trace`: the call `TracerBackend::trace` wraps, without the
/// `Cycle` to `TraceRow` conversion that follows it. Shared by the Criterion
/// `reference_raw` id and the `trace-gen-baseline` harness so both time the
/// same thing.
pub fn raw_trace_cycles(program: &JoltProgram, inputs: &TraceInputs) -> usize {
    let (_lazy, cycles, _memory, _device, _advice) = tracer::trace(
        program.elf_bytes(),
        None,
        &inputs.inputs,
        &inputs.untrusted_advice,
        &inputs.trusted_advice,
        &inputs.memory_config,
        None,
    );
    std::hint::black_box(cycles.len())
}

impl<G: GuestConfig + 'static> Objective for TraceGenObjective<G> {
    type Setup = TraceGenSetup;

    fn name(&self) -> &str {
        self.name.as_str()
    }

    fn setup(&self) -> TraceGenSetup {
        let (program, inputs) = build_trace_setup(&self.guest);
        // One un-timed trace to learn the row count for throughput reporting.
        let mut backend = TracerBackend::new();
        let output = program
            .trace_with(&mut backend, inputs.clone())
            .expect("reference trace failed during setup");
        let trace_len = output.trace.rows().len();
        TraceGenSetup {
            program,
            inputs,
            trace_len,
        }
    }

    fn run(&self, setup: TraceGenSetup) {
        self.run_reference(&setup);
    }

    fn units(&self) -> Option<&str> {
        Some("s")
    }
}

#[cfg(test)]
mod tests {
    use crate::guests::{Fibonacci, Sha2Chain};

    use super::*;

    #[test]
    fn names() {
        assert_eq!(
            TraceGenObjective::new(Fibonacci(400000)).name(),
            "trace_gen_fibonacci_400000"
        );
        assert_eq!(
            TraceGenObjective::new(Sha2Chain::profiling_default()).name(),
            "trace_gen_sha2_chain_4446"
        );
    }
}
