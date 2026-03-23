//! Guest program building, decoding, and tracing.

use common::constants::{
    DEFAULT_HEAP_SIZE, DEFAULT_MAX_INPUT_SIZE, DEFAULT_MAX_OUTPUT_SIZE,
    DEFAULT_MAX_TRUSTED_ADVICE_SIZE, DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE, DEFAULT_STACK_SIZE,
    RAM_START_ADDRESS,
};
use common::jolt_device::{JoltDevice, MemoryConfig};
use std::io;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use tracer::emulator::memory::Memory;
use tracer::instruction::{Cycle, Instruction};
use tracer::utils::virtual_registers::VirtualRegisterAllocator;
use tracer::LazyTraceIterator;
use tracing::info;

use crate::analyze::ProgramSummary;
use crate::{Program, DEFAULT_TARGET_DIR};

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
            backtrace: Some("off".to_string()),
            elf: None,
            elf_compute_advice: None,
        }
    }

    pub fn set_std(&mut self, std: bool) {
        self.std = std;
    }

    pub fn set_func(&mut self, func: &str) {
        self.func = Some(func.to_string());
    }

    /// Set the cargo profile used to compile the guest.
    ///
    /// If unset, guest builds default to `--release`.
    pub fn set_profile(&mut self, profile: &str) {
        self.profile = Some(profile.to_string());
    }

    /// Set backtrace mode for the guest build.
    ///
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
        if self.elf.is_some() {
            return;
        }

        let jolt_cmd = std::env::var("JOLT_PATH").unwrap_or_else(|_| "jolt".to_string());
        let mut args = vec!["build".to_string()];

        args.push("-p".to_string());
        args.push(self.guest.clone());

        if self.std {
            args.push("--mode".to_string());
            args.push("std".to_string());
        }

        if let Some(mode) = &self.backtrace {
            args.push("--backtrace".to_string());
            args.push(mode.clone());
        }

        args.push("--stack-size".to_string());
        args.push(self.stack_size.to_string());
        args.push("--heap-size".to_string());
        args.push(self.heap_size.to_string());

        let guest_target_dir = if extra_features.contains(&"compute_advice") {
            format!(
                "{}/{}-{}-compute-advice",
                target_dir,
                self.guest,
                self.func.as_deref().unwrap_or("")
            )
        } else {
            format!(
                "{}/{}-{}",
                target_dir,
                self.guest,
                self.func.as_deref().unwrap_or("")
            )
        };

        args.push("--".to_string());

        if let Some(profile) = &self.profile {
            args.push("--profile".to_string());
            args.push(profile.clone());
        } else {
            args.push("--release".to_string());
        }

        args.push("--target-dir".to_string());
        args.push(guest_target_dir.clone());

        args.push("--features".to_string());
        let mut features = vec!["guest".to_string()];
        features.extend(extra_features.iter().map(|&s| s.to_string()));
        args.push(features.join(","));

        let cmd_line = compose_command_line(
            &jolt_cmd,
            &[],
            &args.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        );
        info!("\n{cmd_line}");

        let mut cmd = Command::new(&jolt_cmd);
        let _ = cmd.args(&args);

        if let Some(func) = &self.func {
            let _ = cmd.env("JOLT_FUNC_NAME", func);
        }

        let output = cmd
            .output()
            .expect("failed to run jolt - make sure it's installed (cargo install --path .)");

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            if stderr.contains("does not contain this feature: compute_advice") {
                info!("guest does not support compute_advice feature");
                return;
            }
            io::stderr().write_all(&output.stderr).unwrap();
            let output_msg = format!("::build command: \n{cmd_line}\n");
            io::stderr().write_all(output_msg.as_bytes()).unwrap();
            panic!("failed to compile guest with jolt");
        }

        let target_triple = if self.std {
            "riscv64imac-zero-linux-musl"
        } else {
            "riscv64imac-unknown-none-elf"
        };

        let out_profile = self.profile.as_deref().unwrap_or("release");
        let elf_path = PathBuf::from(&guest_target_dir)
            .join(target_triple)
            .join(out_profile)
            .join(&self.guest);

        assert!(
            elf_path.exists(),
            "Built ELF not found at expected location: {}",
            elf_path.display()
        );

        if extra_features.contains(&"compute_advice") {
            self.elf_compute_advice = Some(elf_path.clone());
            info!("Built compute_advice guest binary: {}", elf_path.display());
        } else {
            self.elf = Some(elf_path.clone());
            info!("Built guest binary with jolt: {}", elf_path.display());
        }
    }

    pub fn get_elf_contents(&self) -> Option<Vec<u8>> {
        self.elf.as_ref().map(|elf| {
            std::fs::read(elf)
                .unwrap_or_else(|_| panic!("could not read elf file: {}", elf.display()))
        })
    }

    pub fn get_elf_compute_advice_contents(&self) -> Option<Vec<u8>> {
        self.elf_compute_advice.as_ref().map(|elf| {
            std::fs::read(elf)
                .unwrap_or_else(|_| panic!("could not read elf file: {}", elf.display()))
        })
    }

    fn read_elf(&self) -> Vec<u8> {
        let elf = self
            .elf
            .as_ref()
            .expect("ELF not built yet — call build() first");
        std::fs::read(elf).unwrap_or_else(|_| panic!("could not read elf file: {}", elf.display()))
    }

    fn memory_config(&self, program_size: u64) -> MemoryConfig {
        MemoryConfig {
            heap_size: self.heap_size,
            stack_size: self.stack_size,
            max_input_size: self.max_input_size,
            max_untrusted_advice_size: self.max_untrusted_advice_size,
            max_trusted_advice_size: self.max_trusted_advice_size,
            max_output_size: self.max_output_size,
            program_size: Some(program_size),
        }
    }

    pub fn decode(&mut self) -> (Vec<Instruction>, Vec<(u64, u8)>, u64) {
        self.build(DEFAULT_TARGET_DIR);
        decode(&self.read_elf())
    }

    #[tracing::instrument(skip_all, name = "Program::trace")]
    pub fn trace(
        &mut self,
        inputs: &[u8],
        untrusted_advice: &[u8],
        trusted_advice: &[u8],
    ) -> (LazyTraceIterator, Vec<Cycle>, Memory, JoltDevice) {
        self.build(DEFAULT_TARGET_DIR);
        let elf_contents = self.read_elf();
        let (_, _, program_end, _, _) = tracer::decode(&elf_contents);
        let program_size = program_end - RAM_START_ADDRESS;
        let memory_config = self.memory_config(program_size);

        let (lazy_trace, trace_vec, memory, jolt_device, _advice_tape) = trace(
            &elf_contents,
            self.elf.as_ref(),
            inputs,
            untrusted_advice,
            trusted_advice,
            &memory_config,
            None,
        );
        (lazy_trace, trace_vec, memory, jolt_device)
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
        let elf_contents = self.read_elf();
        let (_, _, program_end, _, _) = tracer::decode(&elf_contents);
        let program_size = program_end - RAM_START_ADDRESS;
        let memory_config = self.memory_config(program_size);

        trace_to_file(
            &elf_contents,
            self.elf.as_ref(),
            inputs,
            untrusted_advice,
            trusted_advice,
            &memory_config,
            trace_file,
        )
    }

    pub fn trace_analyze(
        mut self,
        inputs: &[u8],
        untrusted_advice: &[u8],
        trusted_advice: &[u8],
    ) -> ProgramSummary {
        let (bytecode, init_memory_state, _) = self.decode();
        let (_, trace_vec, _, io_device) = self.trace(inputs, untrusted_advice, trusted_advice);

        ProgramSummary {
            trace: trace_vec,
            bytecode,
            memory_init: init_memory_state,
            io_device,
        }
    }
}

/// Decode an ELF into instructions and memory initialization data.
///
/// Expands virtual instruction sequences (inline sequences) as part of decoding.
///
/// Returns `(instructions, memory_init_bytes, program_size)`.
pub fn decode(elf: &[u8]) -> (Vec<Instruction>, Vec<(u64, u8)>, u64) {
    let (mut instructions, raw_bytes, program_end, _, xlen) = tracer::decode(elf);
    let program_size = program_end - RAM_START_ADDRESS;
    let allocator = VirtualRegisterAllocator::default();

    instructions = instructions
        .into_iter()
        .flat_map(|instr: Instruction| instr.inline_sequence(&allocator, xlen))
        .collect();

    (instructions, raw_bytes, program_size)
}

/// Trace a guest program, returning the full execution trace.
#[allow(clippy::type_complexity)]
pub fn trace(
    elf_contents: &[u8],
    elf_path: Option<&PathBuf>,
    inputs: &[u8],
    untrusted_advice: &[u8],
    trusted_advice: &[u8],
    memory_config: &MemoryConfig,
    advice_tape: Option<tracer::AdviceTape>,
) -> (
    LazyTraceIterator,
    Vec<Cycle>,
    Memory,
    JoltDevice,
    tracer::AdviceTape,
) {
    tracer::trace(
        elf_contents,
        elf_path,
        inputs,
        untrusted_advice,
        trusted_advice,
        memory_config,
        advice_tape,
    )
}

/// Trace a guest program and write the trace to a file.
pub fn trace_to_file(
    elf_contents: &[u8],
    elf_path: Option<&PathBuf>,
    inputs: &[u8],
    untrusted_advice: &[u8],
    trusted_advice: &[u8],
    memory_config: &MemoryConfig,
    trace_file: &PathBuf,
) -> (Memory, JoltDevice) {
    tracer::trace_to_file(
        elf_contents,
        elf_path,
        inputs,
        untrusted_advice,
        trusted_advice,
        memory_config,
        trace_file,
    )
}

fn compose_command_line(program: &str, envs: &[(&str, String)], args: &[&str]) -> String {
    fn has_ctrl(s: &str) -> bool {
        s.chars()
            .any(|c| c.is_control() && !matches!(c, '\t' | '\n' | '\r'))
    }

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
