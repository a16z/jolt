use jolt_core::host::Program;

use crate::guests::{self, GuestConfig, GuestProgram, ProverPreprocessing};
use crate::objective::Objective;

/// Per-iteration state: everything needed to call `prove`.
pub struct ProverTimeSetup {
    pub program: GuestProgram,
    pub prover_pp: ProverPreprocessing,
    pub input: Vec<u8>,
}

/// Benchmarks end-to-end prover time for a guest program.
///
/// Setup compiles the guest, traces to determine trace length,
/// and preprocesses. Each iteration calls `prove`.
#[derive(Default)]
pub struct ProverTimeObjective<G: GuestConfig> {
    guest: G,
    name: String,
}

impl<G: GuestConfig> ProverTimeObjective<G> {
    pub fn new(guest: G) -> Self {
        let name = format!("{} prover time", guest.bench_name());
        Self { guest, name }
    }
}

impl<G: GuestConfig + 'static> Objective for ProverTimeObjective<G> {
    type Setup = ProverTimeSetup;

    fn name(&self) -> &str {
        self.name.as_str()
    }

    fn setup(&self) -> ProverTimeSetup {
        let mut mc = self.guest.memory_config();
        let input = self.guest.input();

        // Compile
        let target_dir = "/tmp/jolt-eval-bench-targets";
        let mut host_program = Program::new(self.guest.package());
        host_program.set_memory_config(mc);
        host_program.build(target_dir);
        let elf_bytes = host_program
            .get_elf_contents()
            .expect("guest ELF not found after build");

        // Decode to get program_size, trace to get trace length
        let (_bytecode, _memory_init, program_size, _e_entry) =
            jolt_core::guest::program::decode(&elf_bytes);
        mc.program_size = Some(program_size);

        let program = GuestProgram::new(&elf_bytes, &mc);
        let (_lazy_trace, trace, _memory, _io) = program.trace(&input, &[], &[]);
        let max_trace_length = (trace.len() + 1).next_power_of_two();
        drop(trace);

        let prover_pp = guests::prover_preprocessing(&program, max_trace_length);

        ProverTimeSetup {
            program,
            prover_pp,
            input,
        }
    }

    fn run(&self, setup: ProverTimeSetup) {
        let (_proof, _io) = guests::prove(&setup.program, &setup.prover_pp, &setup.input);
        std::hint::black_box(());
    }

    fn units(&self) -> Option<&str> {
        Some("s")
    }
}

#[cfg(test)]
mod tests {
    use crate::guests::Fibonacci;

    use super::*;

    #[test]
    fn fibonacci_config() {
        let g = Fibonacci(100);
        assert_eq!(g.package(), "fibonacci-guest");
        assert!(!g.input().is_empty());
        assert_eq!(g.bench_name(), "prover_time_fibonacci_100");
    }
}
