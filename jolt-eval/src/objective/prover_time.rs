use common::constants::{DEFAULT_MAX_TRUSTED_ADVICE_SIZE, DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE};
use common::jolt_device::MemoryConfig;
use jolt_core::host::Program;

use crate::guests;

use super::PerfObjective;

/// Trait for configuring which guest program to benchmark.
pub trait GuestConfig: Default + Send + Sync {
    /// Cargo package name (e.g. "fibonacci-guest").
    fn package(&self) -> &str;

    fn memory_config(&self) -> MemoryConfig {
        // Default memory config
        MemoryConfig {
            max_input_size: 4096,
            max_output_size: 4096,
            max_untrusted_advice_size: DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE,
            max_trusted_advice_size: DEFAULT_MAX_TRUSTED_ADVICE_SIZE,
            stack_size: 4096,
            heap_size: 32768,
            program_size: None,
        }
    }

    /// Serialized program input (postcard-encoded).
    fn input(&self) -> Vec<u8>;

    /// Display name for the benchmark.
    fn bench_name(&self) -> String;
}

/// Per-iteration state: everything needed to call `prove`.
pub struct ProverTimeSetup {
    pub program: guests::GuestProgram,
    pub prover_pp: guests::ProverPreprocessing,
    pub input: Vec<u8>,
}

/// Benchmarks end-to-end prover time for a guest program.
///
/// Setup compiles the guest, traces to determine trace length,
/// and preprocesses. Each iteration calls `prove`.
#[derive(Default)]
pub struct ProverTimeObjective<G: GuestConfig> {
    guest: G,
}

impl<G: GuestConfig> ProverTimeObjective<G> {
    pub fn new(guest: G) -> Self {
        Self { guest }
    }
}

impl<G: GuestConfig + 'static> PerfObjective for ProverTimeObjective<G> {
    type Setup = ProverTimeSetup;

    fn name(&self) -> &str {
        // Leak a string so we can return &str from a computed name.
        // This is fine — there are only a handful of objectives.
        let name = self.guest.bench_name();
        Box::leak(name.into_boxed_str())
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

        let program = guests::GuestProgram::new(&elf_bytes, &mc);
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
}

// ── Guest configurations ────────────────────────────────────────────

/// Fibonacci guest: computes fib(n).
pub struct Fibonacci(pub u32);

impl Default for Fibonacci {
    fn default() -> Self {
        Self(100)
    }
}

impl GuestConfig for Fibonacci {
    fn package(&self) -> &str {
        "fibonacci-guest"
    }
    fn input(&self) -> Vec<u8> {
        postcard::to_stdvec(&self.0).unwrap()
    }
    fn bench_name(&self) -> String {
        format!("prover_time_fibonacci_{}", self.0)
    }
}

/// Muldiv guest: computes a * b / c.
pub struct Muldiv(pub u32, pub u32, pub u32);

impl Default for Muldiv {
    fn default() -> Self {
        Self(12031293, 17, 92)
    }
}

impl GuestConfig for Muldiv {
    fn package(&self) -> &str {
        "muldiv-guest"
    }
    fn input(&self) -> Vec<u8> {
        postcard::to_stdvec(&(self.0, self.1, self.2)).unwrap()
    }
    fn bench_name(&self) -> String {
        "prover_time_muldiv".to_string()
    }
}

/// SHA-2 guest: computes sha256 of input bytes.
pub struct Sha2(pub Vec<u8>);

impl Default for Sha2 {
    fn default() -> Self {
        Self(vec![5u8; 32])
    }
}

impl GuestConfig for Sha2 {
    fn package(&self) -> &str {
        "sha2-guest"
    }
    fn input(&self) -> Vec<u8> {
        postcard::to_stdvec(&self.0).unwrap()
    }
    fn bench_name(&self) -> String {
        "prover_time_sha2".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fibonacci_config_serializes() {
        let g = Fibonacci(100);
        assert_eq!(g.package(), "fibonacci-guest");
        assert!(!g.input().is_empty());
        assert_eq!(g.bench_name(), "prover_time_fibonacci_100");
    }

    #[test]
    fn muldiv_config_serializes() {
        let g = Muldiv::default();
        assert_eq!(g.package(), "muldiv-guest");
        assert!(!g.input().is_empty());
    }
}
