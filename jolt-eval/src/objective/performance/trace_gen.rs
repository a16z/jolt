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

    /// One eager trace with the reference interpreter backend.
    pub fn run_reference(&self, setup: &TraceGenSetup) -> usize {
        let mut backend = TracerBackend::new();
        let output = setup
            .program
            .trace_with(&mut backend, setup.inputs.clone())
            .expect("reference trace failed");
        std::hint::black_box(output.trace.rows().len())
    }
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
