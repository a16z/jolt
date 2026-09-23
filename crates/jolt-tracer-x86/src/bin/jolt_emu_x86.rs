//! ACT4 (riscv-arch-test) runner for the AOT x86-64 backend — the x86
//! counterpart of the tracer's `jolt-emu`.
//!
//! `tests/arch-tests/run.sh` invokes the emulator with a single positional
//! ELF path and judges the test by exit status alone. The arch tests halt by
//! storing a result word to the `tohost` symbol and then spinning in `j .`
//! (see `tests/arch-tests/jolt/rvmodel_macros.h`), so the backend's PC-stall
//! termination ends the run naturally and the result can be read out of
//! final memory — no mid-execution HTIF hook is needed.
//!
//! Exit codes:
//! * 0 — the test passed (`tohost` payload decodes to endcode 0)
//! * 1 — the test ran and failed
//! * 2 — usage/IO error
//! * 3 — the Jolt pipeline or the transpiler rejected the program (decode,
//!   expansion, or an unimplemented row kind). Distinct from 1 so skip lists
//!   can be regenerated mechanically and the two categories stay
//!   distinguishable, per AC4's "skipped and listed" requirement.
//! * 4 — a runtime fault (out-of-bounds access, bad jump target, helper
//!   error)

#![expect(clippy::print_stderr)]

use std::process::ExitCode;

/// Native implementation; the binary exists on every target so build and
/// packaging scripts do not need per-target conditionals.
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
mod native {

    use std::collections::HashMap;
    use std::process::ExitCode;

    use common::constants::RAM_START_ADDRESS;
    use common::jolt_device::MemoryConfig;
    use jolt_program::execution::TraceInputs;
    use jolt_tracer_x86::X86TracerBackend;

    const EXIT_FAILED: u8 = 1;
    const EXIT_USAGE: u8 = 2;
    const EXIT_REJECTED: u8 = 3;
    const EXIT_FAULT: u8 = 4;

    pub fn run() -> ExitCode {
        let mut args = std::env::args().skip(1);
        let Some(elf_path) = args.next() else {
            eprintln!(
                "usage: jolt-emu-x86 <elf> [--signature <path>] [--signature-granularity <n>]"
            );
            return ExitCode::from(EXIT_USAGE);
        };

        let mut signature_path: Option<String> = None;
        let mut granularity: usize = 4;
        while let Some(flag) = args.next() {
            match flag.as_str() {
                "-s" | "--signature" => signature_path = args.next(),
                "--signature-granularity" => {
                    granularity = args
                        .next()
                        .and_then(|value| value.parse().ok())
                        .unwrap_or(4);
                }
                other => {
                    eprintln!("unknown argument: {other}");
                    return ExitCode::from(EXIT_USAGE);
                }
            }
        }

        let elf = match std::fs::read(&elf_path) {
            Ok(bytes) => bytes,
            Err(error) => {
                eprintln!("failed to read {elf_path}: {error}");
                return ExitCode::from(EXIT_USAGE);
            }
        };

        // Symbol addresses come from the ELF directly; the Jolt program image
        // discards the symbol table.
        let symbols = symbol_map(&elf);
        let tohost = symbols.get("tohost").copied().unwrap_or(0);

        let program = match jolt_program::execution::build_jolt_program(&elf) {
            Ok(program) => program,
            Err(error) => {
                eprintln!("{elf_path}: pipeline rejected the program: {error:?}");
                return ExitCode::from(EXIT_REJECTED);
            }
        };
        let memory_config = MemoryConfig {
            program_size: Some(program.program_end - RAM_START_ADDRESS),
            ..Default::default()
        };
        let inputs = TraceInputs::new(Vec::new(), Vec::new(), Vec::new(), memory_config);

        let mut backend = X86TracerBackend::new();
        let output = match backend.fast_run(&program, inputs) {
            Ok(output) => output,
            Err(error) => {
                // Compile-time rejection and runtime faults both surface as
                // TraceError; distinguish them for the skip-list tooling.
                let message = format!("{error:?}");
                eprintln!("{elf_path}: {message}");
                return ExitCode::from(if message.contains("unsupported instruction kind") {
                    EXIT_REJECTED
                } else {
                    EXIT_FAULT
                });
            }
        };

        // Final memory arrives as nonzero (address, byte) pairs, RAM-relative.
        let memory: HashMap<u64, u8> = output.final_memory.bytes.into_iter().collect();
        let read_byte = |address: u64| -> u8 {
            address
                .checked_sub(RAM_START_ADDRESS)
                .and_then(|offset| memory.get(&offset).copied())
                .unwrap_or(0)
        };

        if let Some(path) = signature_path {
            if let Err(error) = write_signature(
                &path,
                symbols.get("begin_signature").copied().unwrap_or(0),
                symbols.get("end_signature").copied().unwrap_or(0),
                granularity.max(1),
                read_byte,
            ) {
                eprintln!("failed to write signature to {path}: {error}");
                return ExitCode::from(EXIT_USAGE);
            }
        }

        if tohost == 0 {
            // No `tohost` symbol at all: nothing to judge.
            eprintln!("{elf_path}: no tohost symbol; cannot determine the result");
            return ExitCode::from(EXIT_FAILED);
        }

        let mut value = 0u64;
        for i in 0..8 {
            value |= u64::from(read_byte(tohost + i)) << (i * 8);
        }
        if value == 0 {
            // The guest stalled without executing a halt macro. `jolt-emu`
            // reports success here (PC-stall return path); for an arch test that
            // is a silent pass, so fail loudly instead.
            eprintln!("{elf_path}: terminated without writing tohost");
            return ExitCode::from(EXIT_FAILED);
        }

        // HTIF encoding, as decoded by Emulator::run_test.
        let device = (value >> 56) & 0xFF;
        let payload = value & 0xFFFF_FFFF_FFFF;
        if device != 0 || payload & 1 != 1 {
            eprintln!("{elf_path}: unexpected tohost value {value:#x}");
            return ExitCode::from(EXIT_FAILED);
        }
        let endcode = payload >> 1;
        if endcode == 0 {
            ExitCode::SUCCESS
        } else {
            eprintln!("{elf_path}: failed with endcode {endcode}");
            ExitCode::from(EXIT_FAILED)
        }
    }

    /// Signature bytes, `granularity` per line, big-endian within a group —
    /// byte-for-byte the format `Emulator::write_signature` produces.
    fn write_signature(
        path: &str,
        begin: u64,
        end: u64,
        granularity: usize,
        read_byte: impl Fn(u64) -> u8,
    ) -> std::io::Result<()> {
        use std::fmt::Write as _;

        let mut out = String::new();
        let mut address = begin;
        while address < end {
            for offset in (0..granularity as u64).rev() {
                let _ = write!(out, "{:02x}", read_byte(address + offset));
            }
            out.push('\n');
            address += granularity as u64;
        }
        std::fs::write(path, out)
    }

    /// `name -> address` for the ELF's symbols, mirroring the tracer's map
    /// construction (NOTYPE and FUNC entries).
    fn symbol_map(elf: &[u8]) -> HashMap<String, u64> {
        use tracer::emulator::elf_analyzer::ElfAnalyzer;

        let analyzer = ElfAnalyzer::new(elf);
        if !analyzer.validate() {
            return HashMap::new();
        }
        let header = analyzer.read_header();
        let section_headers = analyzer.read_section_headers(&header);
        let mut symbol_tables = Vec::new();
        let mut string_tables = Vec::new();
        for section in &section_headers {
            match section.sh_type {
                2 => symbol_tables.push(section),
                3 => string_tables.push(section),
                _ => {}
            }
        }
        let Some(string_table) = string_tables.first() else {
            return HashMap::new();
        };
        let entries = analyzer.read_symbol_entries(&header, &symbol_tables);
        analyzer
            .create_symbol_map(&entries, string_table)
            .into_iter()
            .collect()
    }
}

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
fn main() -> ExitCode {
    native::run()
}

#[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
fn main() -> ExitCode {
    eprintln!("jolt-emu-x86 requires x86_64 Linux; use jolt-emu on this target");
    ExitCode::from(2u8)
}
