//! Guest program building, decoding, and tracing.

use std::io;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use tracing::info;

use common::constants::{
    DEFAULT_HEAP_SIZE, DEFAULT_MAX_INPUT_SIZE, DEFAULT_MAX_OUTPUT_SIZE,
    DEFAULT_MAX_TRUSTED_ADVICE_SIZE, DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE, DEFAULT_STACK_SIZE,
    RAM_START_ADDRESS,
};
use common::jolt_device::{JoltDevice, MemoryConfig};
use tracer::emulator::memory::Memory;
use tracer::instruction::{Cycle, Instruction};
use tracer::utils::virtual_registers::VirtualRegisterAllocator;
use tracer::LazyTraceIterator;

use crate::analyze::ProgramSummary;
use crate::{Program, DEFAULT_TARGET_DIR};

impl Program {
    /// Create a new `Program` targeting the given guest crate name.
    ///
    /// All memory sizes are initialized to the defaults from `common::constants`.
    /// The guest is compiled with `--release` unless overridden via [`Self::set_profile`].
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

    /// Enable or disable linking against the Rust standard library.
    pub fn set_std(&mut self, std: bool) -> &mut Self {
        self.std = std;
        self
    }

    /// Set the guest function name (passed as `JOLT_FUNC_NAME` env var during build).
    pub fn set_func(&mut self, func: &str) -> &mut Self {
        self.func = Some(func.to_string());
        self
    }

    /// Set the cargo profile used to compile the guest.
    ///
    /// If unset, guest builds default to `--release`.
    pub fn set_profile(&mut self, profile: &str) -> &mut Self {
        self.profile = Some(profile.to_string());
        self
    }

    /// Set backtrace mode for the guest build.
    ///
    /// Valid modes: "off", "dwarf", "frame-pointers".
    pub fn set_backtrace(&mut self, mode: &str) -> &mut Self {
        self.backtrace = Some(mode.to_string());
        self
    }

    /// Apply all fields from a [`MemoryConfig`] to this program's settings.
    pub fn set_memory_config(&mut self, memory_config: MemoryConfig) -> &mut Self {
        self.heap_size = memory_config.heap_size;
        self.stack_size = memory_config.stack_size;
        self.max_input_size = memory_config.max_input_size;
        self.max_trusted_advice_size = memory_config.max_trusted_advice_size;
        self.max_untrusted_advice_size = memory_config.max_untrusted_advice_size;
        self.max_output_size = memory_config.max_output_size;
        self
    }

    /// Set the guest heap size in bytes.
    pub fn set_heap_size(&mut self, size: u64) -> &mut Self {
        self.heap_size = size;
        self
    }

    /// Set the guest stack size in bytes.
    pub fn set_stack_size(&mut self, size: u64) -> &mut Self {
        self.stack_size = size;
        self
    }

    /// Set the maximum input buffer size in bytes.
    pub fn set_max_input_size(&mut self, size: u64) -> &mut Self {
        self.max_input_size = size;
        self
    }

    /// Set the maximum trusted advice buffer size in bytes.
    pub fn set_max_trusted_advice_size(&mut self, size: u64) -> &mut Self {
        self.max_trusted_advice_size = size;
        self
    }

    /// Set the maximum untrusted advice buffer size in bytes.
    pub fn set_max_untrusted_advice_size(&mut self, size: u64) -> &mut Self {
        self.max_untrusted_advice_size = size;
        self
    }

    /// Set the maximum output buffer size in bytes.
    pub fn set_max_output_size(&mut self, size: u64) -> &mut Self {
        self.max_output_size = size;
        self
    }

    /// Compile the guest program with default features.
    ///
    /// Uses [`Self::build_with_features`] with an empty feature set.
    /// No-op if the ELF has already been built.
    pub fn build(&mut self, target_dir: &str) {
        self.build_with_features(target_dir, &[]);
    }

    /// Compile the guest program via the `jolt` CLI with the given extra Cargo features.
    ///
    /// No-op if the ELF has already been built (unless `extra_features` contains
    /// `"compute_advice"`, which produces a separate ELF).
    #[tracing::instrument(skip_all, name = "Program::build_with_features")]
    pub fn build_with_features(&mut self, target_dir: &str, extra_features: &[&str]) {
        if self.elf.is_some() {
            return;
        }

        let jolt_cmd = std::env::var("JOLT_PATH").unwrap_or_else(|_| "jolt".to_string());
        let is_compute_advice = extra_features.contains(&"compute_advice");
        let guest_target_dir = self.guest_target_dir(target_dir, is_compute_advice);
        let args = self.build_args(extra_features, &guest_target_dir);

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

        let elf_path = self.resolve_elf_path(&guest_target_dir);
        assert!(
            elf_path.exists(),
            "Built ELF not found at expected location: {}",
            elf_path.display()
        );

        if is_compute_advice {
            info!("Built compute_advice guest binary: {}", elf_path.display());
            self.elf_compute_advice = Some(elf_path);
        } else {
            info!("Built guest binary with jolt: {}", elf_path.display());
            self.elf = Some(elf_path);
        }
    }

    /// Returns the contents of the built guest ELF, or `None` if not yet built.
    pub fn get_elf_contents(&self) -> Option<Vec<u8>> {
        self.elf.as_ref().map(|path| read_elf_at(path))
    }

    /// Returns the contents of the built compute-advice ELF, or `None` if not yet built.
    pub fn get_elf_compute_advice_contents(&self) -> Option<Vec<u8>> {
        self.elf_compute_advice
            .as_ref()
            .map(|path| read_elf_at(path))
    }

    /// Compile (if needed) and decode the guest ELF into instructions and memory init data.
    ///
    /// Returns `(instructions, memory_init_bytes, program_size)`.
    pub fn decode(&mut self) -> (Vec<Instruction>, Vec<(u64, u8)>, u64) {
        self.build(DEFAULT_TARGET_DIR);
        decode(&self.read_elf())
    }

    /// Compile (if needed) and trace the guest program with the given I/O buffers.
    ///
    /// Returns the lazy trace iterator, materialized trace, final memory state, and I/O device.
    #[tracing::instrument(skip_all, name = "Program::trace")]
    pub fn trace(
        &mut self,
        inputs: &[u8],
        untrusted_advice: &[u8],
        trusted_advice: &[u8],
    ) -> (LazyTraceIterator, Vec<Cycle>, Memory, JoltDevice) {
        self.build(DEFAULT_TARGET_DIR);
        let elf_contents = self.read_elf();
        let program_size = compute_program_size(&elf_contents);
        let memory_config = self.memory_config(program_size);

        let (lazy_trace, trace_vec, memory, jolt_device, _advice_tape) = tracer::trace(
            &elf_contents,
            self.elf.as_ref().map(|p| p as &PathBuf),
            inputs,
            untrusted_advice,
            trusted_advice,
            &memory_config,
            None,
        );
        (lazy_trace, trace_vec, memory, jolt_device)
    }

    /// Compile (if needed) and trace the guest program, writing the trace to a file.
    ///
    /// Returns the final memory state and I/O device (the trace itself is written to `trace_file`).
    #[tracing::instrument(skip_all, name = "Program::trace_to_file")]
    pub fn trace_to_file(
        &mut self,
        inputs: &[u8],
        untrusted_advice: &[u8],
        trusted_advice: &[u8],
        trace_file: &Path,
    ) -> (Memory, JoltDevice) {
        self.build(DEFAULT_TARGET_DIR);
        let elf_contents = self.read_elf();
        let program_size = compute_program_size(&elf_contents);
        let memory_config = self.memory_config(program_size);

        let trace_pathbuf = trace_file.to_path_buf();
        tracer::trace_to_file(
            &elf_contents,
            self.elf.as_ref().map(|p| p as &PathBuf),
            inputs,
            untrusted_advice,
            trusted_advice,
            &memory_config,
            &trace_pathbuf,
        )
    }

    /// Compile, decode, and trace the guest, returning a [`ProgramSummary`] for analysis.
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

    fn read_elf(&self) -> Vec<u8> {
        self.get_elf_contents()
            .expect("ELF not built yet — call build() first")
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

    fn guest_target_dir(&self, target_dir: &str, is_compute_advice: bool) -> String {
        let func_suffix = self.func.as_deref().unwrap_or("");
        if is_compute_advice {
            format!("{target_dir}/{}-{func_suffix}-compute-advice", self.guest)
        } else {
            format!("{target_dir}/{}-{func_suffix}", self.guest)
        }
    }

    fn build_args(&self, extra_features: &[&str], guest_target_dir: &str) -> Vec<String> {
        let mut args = Vec::with_capacity(16);
        args.push("build".to_string());

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

        args.push("--".to_string());

        if let Some(profile) = &self.profile {
            args.push("--profile".to_string());
            args.push(profile.clone());
        } else {
            args.push("--release".to_string());
        }

        args.push("--target-dir".to_string());
        args.push(guest_target_dir.to_string());

        args.push("--features".to_string());
        let mut features = vec!["guest".to_string()];
        features.extend(extra_features.iter().map(|&s| s.to_string()));
        args.push(features.join(","));

        args
    }

    fn resolve_elf_path(&self, guest_target_dir: &str) -> PathBuf {
        let target_triple = if self.std {
            "riscv64imac-zero-linux-musl"
        } else {
            "riscv64imac-unknown-none-elf"
        };
        let out_profile = self.profile.as_deref().unwrap_or("release");

        PathBuf::from(guest_target_dir)
            .join(target_triple)
            .join(out_profile)
            .join(&self.guest)
    }
}

fn read_elf_at(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|_| panic!("could not read elf file: {}", path.display()))
}

fn compute_program_size(elf_contents: &[u8]) -> u64 {
    let (_, _, program_end, _, _) = tracer::decode(elf_contents);
    program_end - RAM_START_ADDRESS
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Program;

    #[test]
    fn compose_command_line_simple() {
        let result = compose_command_line("jolt", &[], &["build", "-p", "guest"]);
        assert_eq!(result, "jolt build -p guest");
    }

    #[test]
    fn compose_command_line_empty_args() {
        let result = compose_command_line("jolt", &[], &[]);
        assert_eq!(result, "jolt");
    }

    #[test]
    fn compose_command_line_quotes_spaces() {
        let result = compose_command_line("jolt", &[], &["--path", "my dir/file"]);
        assert_eq!(result, "jolt --path 'my dir/file'");
    }

    #[test]
    fn compose_command_line_quotes_empty_arg() {
        let result = compose_command_line("jolt", &[], &[""]);
        assert_eq!(result, "jolt ''");
    }

    #[test]
    fn compose_command_line_escapes_single_quotes() {
        let result = compose_command_line("jolt", &[], &["it's"]);
        assert_eq!(result, "jolt 'it'\\''s'");
    }

    #[test]
    fn compose_command_line_escapes_control_chars() {
        let result = compose_command_line("jolt", &[], &["a\x01b"]);
        assert_eq!(result, "jolt $'a\\x01b'");
    }

    #[test]
    fn compose_command_line_with_envs() {
        let envs = vec![("FOO", "bar".to_string())];
        let result = compose_command_line("jolt", &envs, &["build"]);
        assert_eq!(result, "env FOO=bar jolt build");
    }

    #[test]
    fn compose_command_line_env_with_spaces() {
        let envs = vec![("MY_VAR", "hello world".to_string())];
        let result = compose_command_line("jolt", &envs, &[]);
        assert_eq!(result, "env MY_VAR='hello world' jolt");
    }

    #[test]
    fn compose_command_line_env_with_control_chars() {
        let envs = vec![("VAR", "val\x02ue".to_string())];
        let result = compose_command_line("jolt", &envs, &["run"]);
        assert_eq!(result, "env VAR=$'val\\x02ue' jolt run");
    }

    #[test]
    fn compose_command_line_preserves_safe_special_chars() {
        let result = compose_command_line("jolt", &[], &["/tmp/path_@:,.+-/file"]);
        assert_eq!(result, "jolt /tmp/path_@:,.+-/file");
    }

    #[test]
    fn builder_new_defaults() {
        let p = Program::new("test-guest");
        assert_eq!(p.guest, "test-guest");
        assert!(p.elf.is_none());
        assert!(p.elf_compute_advice.is_none());
        assert!(!p.std);
        assert_eq!(p.heap_size, DEFAULT_HEAP_SIZE);
        assert_eq!(p.stack_size, DEFAULT_STACK_SIZE);
    }

    #[test]
    fn builder_chaining() {
        let mut p = Program::new("guest");
        let _ = p
            .set_std(true)
            .set_func("main")
            .set_heap_size(1024)
            .set_stack_size(2048)
            .set_max_input_size(512)
            .set_max_output_size(256)
            .set_max_trusted_advice_size(128)
            .set_max_untrusted_advice_size(64);

        assert!(p.std);
        assert_eq!(p.func.as_deref(), Some("main"));
        assert_eq!(p.heap_size, 1024);
        assert_eq!(p.stack_size, 2048);
        assert_eq!(p.max_input_size, 512);
        assert_eq!(p.max_output_size, 256);
        assert_eq!(p.max_trusted_advice_size, 128);
        assert_eq!(p.max_untrusted_advice_size, 64);
    }

    #[test]
    fn builder_set_memory_config() {
        let config = MemoryConfig {
            heap_size: 100,
            stack_size: 200,
            max_input_size: 300,
            max_untrusted_advice_size: 400,
            max_trusted_advice_size: 500,
            max_output_size: 600,
            program_size: None,
        };
        let mut p = Program::new("guest");
        let _ = p.set_memory_config(config);

        assert_eq!(p.heap_size, 100);
        assert_eq!(p.stack_size, 200);
        assert_eq!(p.max_input_size, 300);
        assert_eq!(p.max_untrusted_advice_size, 400);
        assert_eq!(p.max_trusted_advice_size, 500);
        assert_eq!(p.max_output_size, 600);
    }

    #[test]
    fn builder_set_profile_and_backtrace() {
        let mut p = Program::new("guest");
        let _ = p.set_profile("dev").set_backtrace("dwarf");
        assert_eq!(p.profile.as_deref(), Some("dev"));
        assert_eq!(p.backtrace.as_deref(), Some("dwarf"));
    }

    #[test]
    fn elf_path_accessors_none_before_build() {
        let p = Program::new("guest");
        assert!(p.elf_path().is_none());
        assert!(p.elf_compute_advice_path().is_none());
    }

    #[test]
    fn guest_target_dir_regular() {
        let p = Program::new("myguest");
        let dir = p.guest_target_dir("/tmp/targets", false);
        assert_eq!(dir, "/tmp/targets/myguest-");
    }

    #[test]
    fn guest_target_dir_compute_advice() {
        let mut p = Program::new("myguest");
        let _ = p.set_func("entry");
        let dir = p.guest_target_dir("/tmp/targets", true);
        assert_eq!(dir, "/tmp/targets/myguest-entry-compute-advice");
    }

    #[test]
    fn resolve_elf_path_release() {
        let p = Program::new("myguest");
        let path = p.resolve_elf_path("/tmp/targets/myguest-");
        assert_eq!(
            path,
            PathBuf::from("/tmp/targets/myguest-/riscv64imac-unknown-none-elf/release/myguest")
        );
    }

    #[test]
    fn resolve_elf_path_std_custom_profile() {
        let mut p = Program::new("myguest");
        let _ = p.set_std(true).set_profile("dev");
        let path = p.resolve_elf_path("/tmp/dir");
        assert_eq!(
            path,
            PathBuf::from("/tmp/dir/riscv64imac-zero-linux-musl/dev/myguest")
        );
    }

    #[test]
    fn build_args_default() {
        let p = Program::new("myguest");
        let args = p.build_args(&[], "/tmp/target-dir");
        assert!(args.contains(&"build".to_string()));
        assert!(args.contains(&"-p".to_string()));
        assert!(args.contains(&"myguest".to_string()));
        assert!(args.contains(&"--release".to_string()));
        assert!(args.contains(&"/tmp/target-dir".to_string()));
        assert!(args.contains(&"guest".to_string()));
    }

    #[test]
    fn build_args_with_features() {
        let p = Program::new("myguest");
        let args = p.build_args(&["compute_advice", "extra"], "/tmp/dir");
        let features_arg = args
            .iter()
            .skip_while(|a| a.as_str() != "--features")
            .nth(1)
            .unwrap();
        assert_eq!(features_arg, "guest,compute_advice,extra");
    }

    #[test]
    fn build_args_std_mode() {
        let mut p = Program::new("myguest");
        let _ = p.set_std(true);
        let args = p.build_args(&[], "/tmp/dir");
        assert!(args.contains(&"--mode".to_string()));
        assert!(args.contains(&"std".to_string()));
    }

    #[test]
    fn build_args_custom_profile() {
        let mut p = Program::new("myguest");
        let _ = p.set_profile("dev");
        let args = p.build_args(&[], "/tmp/dir");
        assert!(args.contains(&"--profile".to_string()));
        assert!(args.contains(&"dev".to_string()));
        assert!(!args.contains(&"--release".to_string()));
    }

    #[test]
    fn program_debug_impl() {
        let p = Program::new("test");
        let debug_str = format!("{p:?}");
        assert!(debug_str.contains("test"));
        assert!(debug_str.contains("Program"));
    }
}
