use crate::field::JoltField;
use crate::guest;
use crate::host::analyze::ProgramSummary;
use crate::host::{Program, DEFAULT_TARGET_DIR};
use common::constants::{
    DEFAULT_HEAP_SIZE, DEFAULT_MAX_INPUT_SIZE, DEFAULT_MAX_OUTPUT_SIZE,
    DEFAULT_MAX_TRUSTED_ADVICE_SIZE, DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE, DEFAULT_STACK_SIZE,
    RAM_START_ADDRESS,
};
use common::jolt_device::{JoltDevice, MemoryConfig};
use std::fs::File;
use std::io;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::Command;
use tracer::emulator::memory::Memory;
use tracer::instruction::{Cycle, Instruction};
use tracer::LazyTraceIterator;
use tracing::info;

impl Program {
    pub fn new(guest: &str) -> Self {
        Self {
            guest: guest.to_string(),
            func: None,
            profile: None,
            heap_size: DEFAULT_HEAP_SIZE,
            stack_size: DEFAULT_STACK_SIZE,
            max_input_size: DEFAULT_MAX_INPUT_SIZE,
            max_untrusted_advice_size: DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE,
            max_trusted_advice_size: DEFAULT_MAX_TRUSTED_ADVICE_SIZE,
            max_output_size: DEFAULT_MAX_OUTPUT_SIZE,
            std: false,
            backtrace: Some("off".to_string()), // Default to off for minimal size
            elf: None,
            elf_compute_advice: None,
        }
    }

    pub fn set_std(&mut self, std: bool) {
        self.std = std;
    }

    pub fn set_func(&mut self, func: &str) {
        self.func = Some(func.to_string())
    }

    /// Set the cargo profile used to compile the guest.
    ///
    /// If unset, guest builds default to `--release`.
    /// If set, guest builds use `--profile <name>`.
    pub fn set_profile(&mut self, profile: &str) {
        self.profile = Some(profile.to_string());
    }

    /// Set backtrace mode for the guest build.
    ///
    /// This adds --backtrace <mode> to the cargo-jolt CLI.
    /// Valid modes: "off", "dwarf", "frame-pointers".
    pub fn set_backtrace(&mut self, mode: &str) {
        self.backtrace = Some(mode.to_string());
    }

    pub fn set_memory_config(&mut self, memory_config: MemoryConfig) {
        self.set_heap_size(memory_config.heap_size);
        self.set_stack_size(memory_config.stack_size);
        self.set_max_input_size(memory_config.max_input_size);
        self.set_max_trusted_advice_size(memory_config.max_trusted_advice_size);
        self.set_max_untrusted_advice_size(memory_config.max_untrusted_advice_size);
        self.set_max_output_size(memory_config.max_output_size);
    }

    pub fn set_heap_size(&mut self, len: u64) {
        self.heap_size = len;
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
        self.build_with_features(target_dir, &[]);
    }

    #[tracing::instrument(skip_all, name = "Program::build_with_features")]
    pub fn build_with_features(&mut self, target_dir: &str, extra_features: &[&str]) {
        if self.elf.is_none() {
            // Use jolt CLI to build the guest program
            // Fallback order: JOLT_PATH → 'jolt' → 'cargo jolt'
            let (jolt_cmd, mut args) = if let Ok(path) = std::env::var("JOLT_PATH") {
                // Use explicit JOLT_PATH
                (path, vec!["build".to_string()])
            } else if Command::new("jolt").arg("--version").output().is_ok() {
                // Use installed jolt
                ("jolt".to_string(), vec!["build".to_string()])
            } else {
                // Use cargo alias (with local patches)
                (
                    "cargo".to_string(),
                    vec!["jolt".to_string(), "build".to_string()],
                )
            };

            // Add package argument
            args.push("-p".to_string());
            args.push(self.guest.clone());

            // Add --mode std flag if std mode is enabled
            if self.std {
                args.push("--mode".to_string());
                args.push("std".to_string());
            }

            // Add --backtrace <mode> flag if backtrace is configured
            if let Some(mode) = &self.backtrace {
                args.push("--backtrace".to_string());
                args.push(mode.to_string());
            }

            // Pass memory layout parameters to cargo-jolt
            args.push("--stack-size".to_string());
            args.push(self.stack_size.to_string());
            args.push("--heap-size".to_string());
            args.push(self.heap_size.to_string());

            // Add suffix to target dir if building with compute_advice feature
            let guest_target_dir = if extra_features.contains(&"compute_advice") {
                format!(
                    "{}/{}-{}-compute-advice",
                    target_dir,
                    self.guest,
                    self.func.as_ref().unwrap_or(&"".to_string())
                )
            } else {
                format!(
                    "{}/{}-{}",
                    target_dir,
                    self.guest,
                    self.func.as_ref().unwrap_or(&"".to_string())
                )
            };

            // Add separator for cargo passthrough args
            args.push("--".to_string());

            // Cargo profile selection. Default to `--release` for backwards compatibility.
            // If a profile is set, pass `--profile <name>` instead.
            if let Some(profile) = &self.profile {
                args.push("--profile".to_string());
                args.push(profile.clone());
            } else {
                // --release goes after -- as a cargo argument
                args.push("--release".to_string());
            }

            // Pass --target-dir to cargo (not cargo-jolt)
            args.push("--target-dir".to_string());
            args.push(guest_target_dir.clone());

            // Always pass --features guest to enable the guest feature on the example package
            // (this is separate from the jolt-sdk features specified in the example's Cargo.toml)
            args.push("--features".to_string());
            let mut features = vec!["guest".to_string()];
            features.extend(extra_features.iter().map(|s| s.to_string()));
            args.push(features.join(","));

            let cmd_line = compose_command_line(
                &jolt_cmd,
                &[],
                &args.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            );
            info!("\n{cmd_line}");

            let mut cmd = Command::new(&jolt_cmd);
            cmd.args(&args);

            // Pass JOLT_FUNC_NAME if a specific function is set (for guest packages with multiple provable functions)
            if let Some(func) = &self.func {
                cmd.env("JOLT_FUNC_NAME", func);
            }

            let output = cmd
                .output()
                .expect("failed to run jolt - make sure it's installed (cargo install --path .)");

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                if stderr.contains("does not contain this feature: compute_advice") {
                    info!("guest does not support compute_advice feature");
                    return;
                } else {
                    io::stderr().write_all(&output.stderr).unwrap();
                    let output_msg = format!("::build command: \n{cmd_line}\n");
                    io::stderr().write_all(output_msg.as_bytes()).unwrap();
                    panic!("failed to compile guest with jolt");
                }
            }

            // Determine the ELF path based on std mode
            let target_triple = if self.std {
                "riscv64imac-zero-linux-musl"
            } else {
                "riscv64imac-unknown-none-elf"
            };

            // ELF is built to guest_target_dir with standard cargo layout.
            // Note: output directory includes the selected cargo profile (default: "release").
            let out_profile = self.profile.as_deref().unwrap_or("release");
            let elf_path = PathBuf::from(&guest_target_dir)
                .join(target_triple)
                .join(out_profile)
                .join(&self.guest);

            // Verify the ELF exists
            if !elf_path.exists() {
                panic!(
                    "Built ELF not found at expected location: {}",
                    elf_path.display()
                );
            }

            // If extra_features contains "compute_advice", store in elf_compute_advice
            // Otherwise store in elf
            if extra_features.contains(&"compute_advice") {
                self.elf_compute_advice = Some(elf_path.clone());
                info!("Built compute_advice guest binary: {}", elf_path.display());
            } else {
                self.elf = Some(elf_path.clone());
                info!("Built guest binary with jolt: {}", elf_path.display());
            }
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

    pub fn get_elf_compute_advice_contents(&self) -> Option<Vec<u8>> {
        if let Some(elf) = &self.elf_compute_advice {
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
            heap_size: self.heap_size,
            stack_size: self.stack_size,
            max_input_size: self.max_input_size,
            max_untrusted_advice_size: self.max_untrusted_advice_size,
            max_trusted_advice_size: self.max_trusted_advice_size,
            max_output_size: self.max_output_size,
            program_size: Some(program_size),
        };

        let (lazy_trace, trace, memory, jolt_device, _advice_tape) = guest::program::trace(
            &elf_contents,
            self.elf.as_ref(),
            inputs,
            untrusted_advice,
            trusted_advice,
            &memory_config,
            None,
        );
        (lazy_trace, trace, memory, jolt_device)
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
            heap_size: self.heap_size,
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
