use crate::field::JoltField;
use crate::guest;
use crate::host::analyze::ProgramSummary;
#[cfg(not(target_arch = "wasm32"))]
use crate::host::toolchain::{install_no_std_toolchain, install_toolchain};
use crate::host::TOOLCHAIN_VERSION;
use crate::host::{Program, DEFAULT_TARGET_DIR, LINKER_SCRIPT_TEMPLATE};
use common::constants::{
    DEFAULT_MAX_INPUT_SIZE, DEFAULT_MAX_OUTPUT_SIZE, DEFAULT_MEMORY_SIZE, DEFAULT_STACK_SIZE,
    EMULATOR_MEMORY_CAPACITY, RAM_START_ADDRESS, STACK_CANARY_SIZE,
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
use tracing::info;

impl Program {
    pub fn new(guest: &str) -> Self {
        Self {
            guest: guest.to_string(),
            func: None,
            memory_size: DEFAULT_MEMORY_SIZE,
            stack_size: DEFAULT_STACK_SIZE,
            max_input_size: DEFAULT_MAX_INPUT_SIZE,
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

    pub fn set_max_output_size(&mut self, size: u64) {
        self.max_output_size = size;
    }

    pub fn build(&mut self, target_dir: &str) {
        self.build_with_channel(target_dir, "stable");
    }

    #[tracing::instrument(skip_all, name = "Program::build")]
    pub fn build_with_channel(&mut self, target_dir: &str, channel: &str) {
        if self.elf.is_none() {
            #[cfg(not(target_arch = "wasm32"))]
            install_toolchain().unwrap();
            #[cfg(not(target_arch = "wasm32"))]
            install_no_std_toolchain().unwrap();

            self.save_linker();

            let rust_flags = [
                "-C",
                &format!("link-arg=-T{}", self.linker_path()),
                "-C",
                "passes=lower-atomic",
                "-C",
                "panic=abort",
                "-C",
                "strip=symbols",
                "-C",
                "opt-level=z",
                "--cfg",
                "getrandom_backend=\"custom\"",
            ];

            let target_triple = if self.std {
                "riscv64imac-jolt-zkvm-elf"
            } else {
                "riscv64imac-unknown-none-elf"
            };

            let mut envs = vec![("CARGO_ENCODED_RUSTFLAGS", rust_flags.join("\x1f"))];

            if self.std {
                envs.push((
                    "RUSTUP_TOOLCHAIN",
                    format!("{channel}-jolt-{TOOLCHAIN_VERSION}"),
                ));
            }

            if let Some(func) = &self.func {
                envs.push(("JOLT_FUNC_NAME", func.to_string()));
            }

            let target = format!(
                "{}/{}-{}",
                target_dir,
                self.guest,
                self.func.as_ref().unwrap_or(&"".to_string())
            );

            let cc_env_var = format!("CC_{target_triple}");
            let cc_value = std::env::var(&cc_env_var).unwrap_or_else(|_| {
                #[cfg(target_os = "linux")]
                {
                    "riscv64-unknown-elf-gcc".to_string()
                }
                #[cfg(not(target_os = "linux"))]
                {
                    // Default fallback for other platforms
                    "".to_string()
                }
            });
            envs.push((&cc_env_var, cc_value));

            let cc_env_var = format!("CFLAGS_{target_triple}");
            let cc_value = std::env::var(&cc_env_var).unwrap_or_else(|_| {
                #[cfg(target_os = "linux")]
                {
                    "-mcmodel=medany".to_string()
                }
                #[cfg(not(target_os = "linux"))]
                {
                    // Default fallback for other platforms
                    "".to_string()
                }
            });
            envs.push((&cc_env_var, cc_value));

            let args = [
                "build",
                "--release",
                "--features",
                "guest",
                "-p",
                &self.guest,
                "--target-dir",
                &target,
                "--target",
                target_triple,
            ];

            let cmd_line = compose_command_line("cargo", &envs, &args);
            info!("\n{cmd_line}");

            let output = Command::new("cargo")
                .envs(envs.clone())
                .args(args)
                .output()
                .expect("failed to build guest");

            if !output.status.success() {
                io::stderr().write_all(&output.stderr).unwrap();
                let output_msg = format!("::build command: \n{cmd_line}\n");
                io::stderr().write_all(output_msg.as_bytes()).unwrap();
                panic!("failed to compile guest");
            }

            let elf = format!("{}/{}/release/{}", target, target_triple, self.guest);
            self.elf = Some(PathBuf::from_str(&elf).unwrap());
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
    pub fn trace(&mut self, inputs: &[u8]) -> (Vec<Cycle>, Memory, JoltDevice) {
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
            max_output_size: self.max_output_size,
            program_size: Some(program_size),
        };
        guest::program::trace(&elf_contents, inputs, &memory_config)
    }

    #[tracing::instrument(skip_all, name = "Program::trace_to_file")]
    pub fn trace_to_file(&mut self, inputs: &[u8], trace_file: &PathBuf) -> (Memory, JoltDevice) {
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
            max_output_size: self.max_output_size,
            program_size: Some(program_size),
        };
        tracer::trace_to_file(&elf_contents, inputs, &memory_config, trace_file)
    }

    pub fn trace_analyze<F: JoltField>(mut self, inputs: &[u8]) -> ProgramSummary {
        let (bytecode, init_memory_state, _) = self.decode();
        let (trace, _, io_device) = self.trace(inputs);

        ProgramSummary {
            trace,
            bytecode,
            memory_init: init_memory_state,
            io_device,
        }
    }

    fn save_linker(&self) {
        let linker_path = PathBuf::from_str(&self.linker_path()).unwrap();
        if let Some(parent) = linker_path.parent() {
            fs::create_dir_all(parent).expect("could not create linker file");
        }

        let linker_script = LINKER_SCRIPT_TEMPLATE
            .replace("{EMULATOR_MEMORY}", &EMULATOR_MEMORY_CAPACITY.to_string())
            .replace("{STACK_CANARY}", &STACK_CANARY_SIZE.to_string())
            .replace("{MEMORY_SIZE}", &self.memory_size.to_string())
            .replace("{STACK_SIZE}", &self.stack_size.to_string());

        let mut file = File::create(linker_path).expect("could not create linker file");
        file.write_all(linker_script.as_bytes())
            .expect("could not save linker");
    }

    fn linker_path(&self) -> String {
        format!("/tmp/jolt-guest-linkers/{}.ld", self.guest)
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
