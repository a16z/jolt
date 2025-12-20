use crate::field::JoltField;
use crate::guest;
use crate::host::analyze::ProgramSummary;
#[cfg(not(target_arch = "wasm32"))]
use crate::host::toolchain::{install_no_std_toolchain, install_toolchain};
use crate::host::TOOLCHAIN_VERSION;
use crate::host::{Program, DEFAULT_TARGET_DIR, LINKER_SCRIPT_TEMPLATE};
use common::constants::{
    DEFAULT_MAX_INPUT_SIZE, DEFAULT_MAX_OUTPUT_SIZE, DEFAULT_MAX_TRUSTED_ADVICE_SIZE,
    DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE, DEFAULT_MEMORY_SIZE, DEFAULT_STACK_SIZE, RAM_START_ADDRESS,
    STACK_CANARY_SIZE,
};
use common::jolt_device::{JoltDevice, MemoryConfig};
use std::fs::File;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::Command;
use std::str::FromStr;
use std::{fs, io};
use tracer::emulator::memory::Memory;
use tracer::instruction::{Cycle, Instruction};
use tracer::LazyTraceIterator;
use tracing::info;

impl Program {
    pub fn new(guest: &str) -> Self {
        Self {
            guest: guest.to_string(),
            func: None,
            memory_size: DEFAULT_MEMORY_SIZE,
            stack_size: DEFAULT_STACK_SIZE,
            max_input_size: DEFAULT_MAX_INPUT_SIZE,
            max_untrusted_advice_size: DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE,
            max_trusted_advice_size: DEFAULT_MAX_TRUSTED_ADVICE_SIZE,
            max_output_size: DEFAULT_MAX_OUTPUT_SIZE,
            std: false,
            elf: None,
        }
    }

    pub fn set_std(&mut self, std: bool) {
        self.std = std;
    }

    pub fn set_func(&mut self, func: &str) {
        self.func = Some(func.to_string())
    }

    pub fn set_memory_config(&mut self, memory_config: MemoryConfig) {
        self.set_memory_size(memory_config.memory_size);
        self.set_stack_size(memory_config.stack_size);
        self.set_max_input_size(memory_config.max_input_size);
        self.set_max_trusted_advice_size(memory_config.max_trusted_advice_size);
        self.set_max_untrusted_advice_size(memory_config.max_untrusted_advice_size);
        self.set_max_output_size(memory_config.max_output_size);
    }

    pub fn set_memory_size(&mut self, len: u64) {
        self.memory_size = len;
    }

    pub fn set_stack_size(&mut self, len: u64) {
        self.stack_size = len;
    }

    pub fn set_max_input_size(&mut self, size: u64) {
        self.max_input_size = size;
    }

    pub fn set_max_trusted_advice_size(&mut self, size: u64) {
        self.max_trusted_advice_size = size;
    }

    pub fn set_max_untrusted_advice_size(&mut self, size: u64) {
        self.max_untrusted_advice_size = size;
    }

    pub fn set_max_output_size(&mut self, size: u64) {
        self.max_output_size = size;
    }

    pub fn build(&mut self, target_dir: &str) {
        self.build_with_channel(target_dir, "stable");
    }

    #[tracing::instrument(skip_all, name = "Program::build")]
    pub fn build_with_channel(&mut self, target_dir: &str, _channel: &str) {
        if self.elf.is_none() {
            // Use cargo-jolt to build the guest program
            // CARGO_JOLT_PATH can be set to override (for development/testing)
            let cargo_jolt_path = std::env::var("CARGO_JOLT_PATH")
                .unwrap_or_else(|_| "cargo-jolt".to_string());

            // Build base arguments for cargo-jolt
            // cargo-jolt is invoked as: cargo-jolt jolt build -p <package> --release [--std] -- --target-dir <dir> --features guest
            let mut args = vec![
                "jolt".to_string(),
                "build".to_string(),
                "-p".to_string(),
                self.guest.clone(),
                "--release".to_string(),
            ];

            // Add --std flag if std mode is enabled
            if self.std {
                args.push("--std".to_string());
            }

            // Create per-guest target directory (isolates builds)
            let guest_target_dir = format!(
                "{}/{}-{}",
                target_dir,
                self.guest,
                self.func.as_ref().unwrap_or(&"".to_string())
            );

            // Add separator for cargo passthrough args
            args.push("--".to_string());

            // Pass --target-dir to cargo (not cargo-jolt)
            args.push("--target-dir".to_string());
            args.push(guest_target_dir.clone());

            // Always pass --features guest to enable the guest feature on the example package
            // (this is separate from the jolt-sdk features specified in the example's Cargo.toml)
            args.push("--features".to_string());
            args.push("guest".to_string());

            let cmd_line = compose_command_line(&cargo_jolt_path, &[], &args.iter().map(|s| s.as_str()).collect::<Vec<_>>());
            info!("\n{cmd_line}");

            let mut cmd = Command::new(&cargo_jolt_path);
            cmd.args(&args);

            // Pass JOLT_FUNC_NAME if a specific function is set (for guest packages with multiple provable functions)
            if let Some(func) = &self.func {
                cmd.env("JOLT_FUNC_NAME", func);
            }

            let output = cmd.output()
                .expect("failed to run cargo-jolt - make sure it's installed (cargo install --path crates/bolt)");

            if !output.status.success() {
                io::stderr().write_all(&output.stderr).unwrap();
                let output_msg = format!("::build command: \n{cmd_line}\n");
                io::stderr().write_all(output_msg.as_bytes()).unwrap();
                panic!("failed to compile guest with cargo-jolt");
            }

            // Determine the ELF path based on std mode
            let target_triple = if self.std {
                "riscv64imac-zero-linux-musl"
            } else {
                "riscv64imac-unknown-none-elf"
            };

            // ELF is built to guest_target_dir with standard cargo layout
            let elf_path = PathBuf::from(&guest_target_dir)
                .join(target_triple)
                .join("release")
                .join(&self.guest);

            // Verify the ELF exists
            if !elf_path.exists() {
                panic!("Built ELF not found at expected location: {}", elf_path.display());
            }

            // Store the main ELF path
            self.elf = Some(elf_path.clone());

            info!("Built guest binary with cargo-jolt: {}", elf_path.display());
        }
    }

    pub fn get_elf_contents(&self) -> Option<Vec<u8>> {
        if let Some(elf) = &self.elf {
            let mut elf_file =
                File::open(elf).unwrap_or_else(|_| panic!("could not open elf file: {elf:?}"));
            let mut elf_contents = Vec::new();
            elf_file.read_to_end(&mut elf_contents).unwrap();
            Some(elf_contents)
        } else {
            None
        }
    }

    pub fn decode(&mut self) -> (Vec<Instruction>, Vec<(u64, u8)>, u64) {
        self.build(DEFAULT_TARGET_DIR);
        let elf = self.elf.as_ref().unwrap();
        let mut elf_file =
            File::open(elf).unwrap_or_else(|_| panic!("could not open elf file: {elf:?}"));
        let mut elf_contents = Vec::new();
        elf_file.read_to_end(&mut elf_contents).unwrap();
        guest::program::decode(&elf_contents)
    }

    // TODO(moodlezoup): Make this generic over InstructionSet
    #[tracing::instrument(skip_all, name = "Program::trace")]
    pub fn trace(
        &mut self,
        inputs: &[u8],
        untrusted_advice: &[u8],
        trusted_advice: &[u8],
    ) -> (LazyTraceIterator, Vec<Cycle>, Memory, JoltDevice) {
        self.build(DEFAULT_TARGET_DIR);
        let elf = self.elf.as_ref().unwrap();
        let mut elf_file =
            File::open(elf).unwrap_or_else(|_| panic!("could not open elf file: {elf:?}"));
        let mut elf_contents = Vec::new();
        elf_file.read_to_end(&mut elf_contents).unwrap();
        let (_, _, program_end, _) = tracer::decode(&elf_contents);
        let program_size = program_end - RAM_START_ADDRESS;

        let memory_config = MemoryConfig {
            memory_size: self.memory_size,
            stack_size: self.stack_size,
            max_input_size: self.max_input_size,
            max_untrusted_advice_size: self.max_untrusted_advice_size,
            max_trusted_advice_size: self.max_trusted_advice_size,
            max_output_size: self.max_output_size,
            program_size: Some(program_size),
        };

        guest::program::trace(
            &elf_contents,
            self.elf.as_ref(),
            inputs,
            untrusted_advice,
            trusted_advice,
            &memory_config,
        )
    }

    #[tracing::instrument(skip_all, name = "Program::trace_to_file")]
    pub fn trace_to_file(
        &mut self,
        inputs: &[u8],
        untrusted_advice: &[u8],
        trusted_advice: &[u8],
        trace_file: &PathBuf,
    ) -> (Memory, JoltDevice) {
        self.build(DEFAULT_TARGET_DIR);
        let elf = self.elf.as_ref().unwrap();
        let mut elf_file =
            File::open(elf).unwrap_or_else(|_| panic!("could not open elf file: {elf:?}"));
        let mut elf_contents = Vec::new();
        elf_file.read_to_end(&mut elf_contents).unwrap();
        let (_, _, program_end, _) = tracer::decode(&elf_contents);
        let program_size = program_end - RAM_START_ADDRESS;
        let memory_config = MemoryConfig {
            memory_size: self.memory_size,
            stack_size: self.stack_size,
            max_input_size: self.max_input_size,
            max_untrusted_advice_size: self.max_untrusted_advice_size,
            max_trusted_advice_size: self.max_trusted_advice_size,
            max_output_size: self.max_output_size,
            program_size: Some(program_size),
        };

        tracer::trace_to_file(
            &elf_contents,
            self.elf.as_ref(),
            inputs,
            untrusted_advice,
            trusted_advice,
            &memory_config,
            trace_file,
        )
    }

    pub fn trace_analyze<F: JoltField>(
        mut self,
        inputs: &[u8],
        untrusted_advice: &[u8],
        trusted_advice: &[u8],
    ) -> ProgramSummary {
        let (bytecode, init_memory_state, _) = self.decode();
        let (_, trace, _, io_device) = self.trace(inputs, untrusted_advice, trusted_advice);

        ProgramSummary {
            trace,
            bytecode,
            memory_init: init_memory_state,
            io_device,
        }
    }

    // save_linker and linker_path are no longer needed when using cargo-jolt
    // cargo-jolt handles linker script generation
    // Keeping these methods for backward compatibility but they're unused
    fn save_linker(&self) {
        // No-op: cargo-jolt handles linker scripts
    }

    fn linker_path(&self) -> String {
        // No-op: cargo-jolt handles linker scripts
        String::new()
    }
}

fn compose_command_line(program: &str, envs: &[(&str, String)], args: &[&str]) -> String {
    fn has_ctrl(s: &str) -> bool {
        s.chars()
            .any(|c| c.is_control() && !matches!(c, '\t' | '\n' | '\r'))
    }

    // ANSI-C ($'...') quoting for when control chars are present.
    fn quote_ansi_c(s: &str) -> String {
        use std::fmt::Write as _;
        let mut out = String::with_capacity(s.len() + 3);
        out.push_str("$'");
        for c in s.chars() {
            match c {
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                '\\' => out.push_str("\\\\"),
                '\'' => out.push_str("\\'"),
                c if c.is_control() => {
                    let _ = write!(out, "\\x{:02x}", c as u32);
                }
                _ => out.push(c),
            }
        }
        out.push('\'');
        out
    }

    // Safe POSIX-style single-quote quoting (no expansions).
    fn sh_quote(s: &str) -> String {
        const SAFE: &str =
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_@%+=:,./-";
        if !s.is_empty() && s.chars().all(|c| SAFE.contains(c)) {
            s.to_string()
        } else {
            let mut out = String::with_capacity(s.len() + 2);
            out.push('\'');
            for ch in s.chars() {
                if ch == '\'' {
                    out.push_str("'\\''");
                } else {
                    out.push(ch);
                }
            }
            out.push('\'');
            out
        }
    }

    let mut parts = Vec::new();

    if !envs.is_empty() {
        parts.push("env".to_string());
        for &(k, ref v) in envs {
            let v = v.as_str();
            let q = if has_ctrl(v) {
                quote_ansi_c(v)
            } else {
                sh_quote(v)
            };
            parts.push(format!("{k}={q}"));
        }
    }

    parts.push(sh_quote(program));
    parts.extend(args.iter().map(|&a| {
        if has_ctrl(a) {
            quote_ansi_c(a)
        } else {
            sh_quote(a)
        }
    }));

    parts.join(" ")
}
