use std::sync::Arc;

use common::constants::{DEFAULT_MAX_TRUSTED_ADVICE_SIZE, DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE};
use common::jolt_device::MemoryConfig;

use crate::TestCase;

/// A known guest program that jolt-eval can compile and run.
pub struct GuestSpec {
    /// Cargo package name of the guest crate (e.g. "muldiv-guest").
    pub package: &'static str,
    /// Short name used in CLI `--guest` flags.
    pub name: &'static str,
    pub heap_size: u64,
    pub stack_size: u64,
    pub max_input_size: u64,
    pub max_output_size: u64,
    pub max_trace_length: usize,
    /// Default inputs to serialize and pass to the guest program.
    pub default_inputs: fn() -> Vec<u8>,
}

impl GuestSpec {
    pub fn memory_config(&self) -> MemoryConfig {
        MemoryConfig {
            max_input_size: self.max_input_size,
            max_output_size: self.max_output_size,
            max_untrusted_advice_size: DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE,
            max_trusted_advice_size: DEFAULT_MAX_TRUSTED_ADVICE_SIZE,
            stack_size: self.stack_size,
            heap_size: self.heap_size,
            program_size: None,
        }
    }

    /// Compile the guest and return a `TestCase`.
    ///
    /// Invokes the `jolt` CLI to cross-compile the guest crate to
    /// RISC-V, then wraps the resulting ELF bytes in a `TestCase`.
    pub fn compile(&self, target_dir: &str) -> TestCase {
        let mut program = jolt_core::host::Program::new(self.package);
        program.set_memory_config(self.memory_config());
        program.build(target_dir);
        let elf_bytes = program
            .get_elf_contents()
            .expect("guest ELF not found after build");
        TestCase {
            elf_contents: elf_bytes,
            memory_config: self.memory_config(),
            max_trace_length: self.max_trace_length,
        }
    }
}

/// The fixed catalog of guest programs available for evaluation.
///
/// Modeled after the benchmark suite in `jolt-core/benches/e2e_profiling.rs`.
/// Each entry carries the memory config and default inputs extracted from
/// the `#[jolt::provable(...)]` attributes in the guest crate.
pub static GUESTS: &[GuestSpec] = &[
    GuestSpec {
        package: "muldiv-guest",
        name: "muldiv",
        heap_size: 32768,
        stack_size: 4096,
        max_input_size: 4096,
        max_output_size: 4096,
        max_trace_length: 65536,
        default_inputs: || postcard::to_stdvec(&(12031293u32, 17u32, 92u32)).unwrap(),
    },
    GuestSpec {
        package: "fibonacci-guest",
        name: "fibonacci",
        heap_size: 32768,
        stack_size: 4096,
        max_input_size: 4096,
        max_output_size: 4096,
        max_trace_length: 65536,
        default_inputs: || postcard::to_stdvec(&100u32).unwrap(),
    },
    GuestSpec {
        package: "sha2-guest",
        name: "sha2",
        heap_size: 32768,
        stack_size: 4096,
        max_input_size: 4096,
        max_output_size: 4096,
        max_trace_length: 65536,
        default_inputs: || postcard::to_stdvec(&vec![5u8; 32]).unwrap(),
    },
    GuestSpec {
        package: "sha3-guest",
        name: "sha3",
        heap_size: 32768,
        stack_size: 4096,
        max_input_size: 4096,
        max_output_size: 4096,
        max_trace_length: 65536,
        default_inputs: || postcard::to_stdvec(&vec![5u8; 32]).unwrap(),
    },
    GuestSpec {
        package: "collatz-guest",
        name: "collatz",
        heap_size: 32768,
        stack_size: 4096,
        max_input_size: 4096,
        max_output_size: 4096,
        max_trace_length: 1048576,
        default_inputs: || postcard::to_stdvec(&19u32).unwrap(),
    },
    GuestSpec {
        package: "alloc-guest",
        name: "alloc",
        heap_size: 32768,
        stack_size: 4096,
        max_input_size: 4096,
        max_output_size: 4096,
        max_trace_length: 65536,
        default_inputs: Vec::new,
    },
];

/// Look up a guest by its short name.
pub fn find_guest(name: &str) -> Option<&'static GuestSpec> {
    GUESTS.iter().find(|g| g.name == name)
}

/// Return the short names of all known guests.
pub fn guest_names() -> Vec<&'static str> {
    GUESTS.iter().map(|g| g.name).collect()
}

/// Resolve a `TestCase` from either `--guest <name>` or `--elf <path>`.
///
/// If `guest` is `Some`, compiles the named guest. If `elf` is `Some`,
/// reads the ELF from disk with a default memory config. Exits the
/// process with a helpful message if neither is provided.
pub fn resolve_test_case(
    guest: Option<&str>,
    elf: Option<&str>,
    max_trace_length_override: Option<usize>,
) -> (Arc<TestCase>, Vec<u8>) {
    if let Some(name) = guest {
        let spec = find_guest(name).unwrap_or_else(|| {
            eprintln!(
                "Unknown guest '{name}'. Available: {}",
                guest_names().join(", ")
            );
            std::process::exit(1);
        });
        let mut tc = spec.compile("/tmp/jolt-guest-targets");
        if let Some(mtl) = max_trace_length_override {
            tc.max_trace_length = mtl;
        }
        let inputs = (spec.default_inputs)();
        (Arc::new(tc), inputs)
    } else if let Some(path) = elf {
        let elf_bytes = std::fs::read(path).unwrap_or_else(|e| {
            eprintln!("Failed to read ELF {path}: {e}");
            std::process::exit(1);
        });
        let tc = TestCase {
            elf_contents: elf_bytes,
            memory_config: MemoryConfig {
                max_input_size: 4096,
                max_output_size: 4096,
                max_untrusted_advice_size: DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE,
                max_trusted_advice_size: DEFAULT_MAX_TRUSTED_ADVICE_SIZE,
                stack_size: 65536,
                heap_size: 32768,
                program_size: None,
            },
            max_trace_length: max_trace_length_override.unwrap_or(65536),
        };
        (Arc::new(tc), vec![])
    } else {
        eprintln!(
            "Provide either --guest <name> or --elf <path>.\n\
             Available guests: {}",
            guest_names().join(", ")
        );
        std::process::exit(1);
    }
}
