mod build_wasm;

use std::{
    fs::{self, File},
    io::Write,
    path::PathBuf,
    process::{exit, Command},
};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use log::{debug, info};
use rand::prelude::SliceRandom;
use sysinfo::System;

use build_wasm::{build_wasm, modify_cargo_toml};
use zeroos_build::cmds::{BuildArgs, StdMode};
use zeroos_build::spec::TargetRenderOptions;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Linker script template embedded at compile time.
/// This linker script is for Jolt zkVM guests.
static LINKER_TEMPLATE: &str = include_str!("linker.ld.template");

#[derive(Parser)]
#[command(version = version(), about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: JoltCommand,
}

#[derive(Subcommand)]
enum JoltCommand {
    /// Creates a new Jolt project with the specified name
    New {
        /// Project name
        name: String,
        /// Whether to generate WASM compatible files
        #[arg(short, long)]
        wasm: bool,
    },

    /// Build a guest program for Jolt zkVM
    Build(JoltBuildArgs),

    /// Run an ELF binary on the Jolt emulator
    Run(RunArgs),

    /// Generate target specs or linker scripts
    #[command(subcommand)]
    Generate(GenerateCmd),

    /// Handles preprocessing and generates WASM compatible files
    BuildWasm,
}

#[derive(clap::Subcommand, Debug)]
enum GenerateCmd {
    /// Generate a custom target specification JSON file
    Target(JoltGenerateTargetArgs),

    /// Generate a linker script with custom memory layout
    Linker(JoltGenerateLinkerArgs),
}

#[derive(clap::Args, Debug)]
struct JoltBuildArgs {
    #[command(flatten)]
    base: BuildArgs,
}

#[derive(clap::Args, Debug)]
struct RunArgs {
    /// Path to the ELF binary to run
    #[arg(value_name = "BINARY")]
    binary: PathBuf,

    /// Path to jolt-emu binary (defaults to searching PATH, then common locations)
    #[arg(long, env = "JOLT_EMU_PATH")]
    jolt_emu: Option<PathBuf>,

    /// Additional arguments to pass to jolt-emu
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    pub emu_args: Vec<String>,
}

#[derive(clap::Args, Debug)]
struct JoltGenerateTargetArgs {
    #[command(flatten)]
    base: zeroos_build::cmds::GenerateTargetArgs,

    #[arg(long, short = 'o')]
    output: Option<PathBuf>,
}

#[derive(clap::Args, Debug)]
struct JoltGenerateLinkerArgs {
    #[command(flatten)]
    base: zeroos_build::cmds::GenerateLinkerArgs,

    #[arg(long, short = 'o', default_value = "linker.ld")]
    output: PathBuf,
}

fn version() -> &'static str {
    concat!(
        env!("CARGO_PKG_VERSION"),
        " (",
        env!("GIT_SHORT_HASH"),
        " ",
        env!("GIT_DATE"),
        ")"
    )
}

fn main() {
    env_logger::Builder::from_default_env()
        .format_timestamp(None)
        .format_module_path(false)
        .init();

    debug!("jolt starting");

    let cli = Cli::parse();
    let result = match cli.command {
        JoltCommand::New { name, wasm } => {
            create_project(name, wasm);
            Ok(())
        }
        JoltCommand::Build(args) => build_command(args),
        JoltCommand::Run(args) => run_command(args),
        JoltCommand::Generate(gen_cmd) => match gen_cmd {
            GenerateCmd::Target(args) => generate_target_command(args),
            GenerateCmd::Linker(args) => generate_linker_command(args),
        },
        JoltCommand::BuildWasm => {
            build_wasm();
            Ok(())
        }
    };

    if let Err(e) = result {
        eprintln!("Error: {e:#}");
        exit(1);
    }
}

// ============================================================================
// Build command (from cargo-jolt)
// ============================================================================

/// Resolve the guest optimization level from `JOLT_GUEST_OPT` env var (default: "3").
fn guest_opt_flag() -> String {
    let level = std::env::var("JOLT_GUEST_OPT").unwrap_or_else(|_| "3".to_string());
    match level.as_str() {
        "0" | "1" | "2" | "3" | "s" | "z" => {}
        _ => panic!(
            "Invalid JOLT_GUEST_OPT value: {level}. Allowed values are 0, 1, 2, 3, s, z"
        ),
    }
    format!("-Copt-level={level}")
}

fn build_command(args: JoltBuildArgs) -> Result<()> {
    debug!("build_command: {args:?}");

    let workspace_root = zeroos_build::cmds::find_workspace_root()?;
    debug!("workspace_root: {}", workspace_root.display());

    // Use the embedded linker template (compiled into the binary)
    let linker_tpl = LINKER_TEMPLATE.to_string();

    let fully = args.base.mode == StdMode::Std || args.base.fully;

    let toolchain_paths = if args.base.mode == StdMode::Std || fully {
        let tc_cfg = zeroos_build::toolchain::ToolchainConfig::default();
        let install_cfg = zeroos_build::toolchain::InstallConfig::default();
        let paths = zeroos_build::toolchain::get_or_install_or_build_toolchain(
            args.base.musl_lib_path.clone(),
            args.base.gcc_lib_path.clone(),
            &tc_cfg,
            &install_cfg,
            fully,
        )
        .map(|p| (p.musl_lib, p.gcc_lib))
        .map_err(|e| anyhow::anyhow!("Toolchain setup failed: {e}"))?;
        Some(paths)
    } else {
        None
    };

    let opt_flag = guest_opt_flag();
    let jolt_rustflags: &[&str] = &[
        "-Cpasses=lower-atomic",
        "-Cpanic=abort",
        &opt_flag,
        // Disable LLVM's MachineOutliner: it generates a faulty calling pattern on RISC-V
        // that creates infinite loops (auipc t1,0; jr offset(t1); ... jr t1 returns to auipc).
        "-Cllvm-args=-enable-machine-outliner=never",
        "--cfg=getrandom_backend=\"custom\"",
    ];

    zeroos_build::cmds::build_binary_with_rustflags(
        &workspace_root,
        &args.base,
        toolchain_paths,
        Some(linker_tpl),
        Some(jolt_rustflags),
    )?;

    Ok(())
}

// ============================================================================
// Run command (from cargo-jolt)
// ============================================================================

fn find_jolt_emu() -> Option<PathBuf> {
    // First check if jolt-emu is in PATH
    if let Ok(output) = Command::new("which").arg("jolt-emu").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Some(PathBuf::from(path));
            }
        }
    }

    // Check common locations relative to the jolt repository
    let common_paths = [
        // Relative to current working directory (if in jolt repo)
        "target/release/jolt-emu",
        "target/debug/jolt-emu",
        // Common sibling directory layout
        "../jolt/target/release/jolt-emu",
        "../jolt/target/debug/jolt-emu",
    ];

    for path in &common_paths {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p.canonicalize().unwrap_or(p));
        }
    }

    None
}

fn run_command(args: RunArgs) -> Result<()> {
    if !args.binary.exists() {
        anyhow::bail!("Binary not found: {}", args.binary.display());
    }

    let jolt_emu = args.jolt_emu.or_else(find_jolt_emu).ok_or_else(|| {
        anyhow::anyhow!(
            "jolt-emu not found. Please specify --jolt-emu or set JOLT_EMU_PATH environment variable"
        )
    })?;

    debug!("Running binary: {}", args.binary.display());
    debug!("Using jolt-emu: {}", jolt_emu.display());

    println!("Running on Jolt emulator...\n");

    let mut cmd = Command::new(&jolt_emu);
    cmd.arg(&args.binary);
    cmd.args(&args.emu_args);

    let args_vec: Vec<String> = cmd
        .get_args()
        .map(|s| s.to_string_lossy().to_string())
        .collect();
    let cmd_str = format!("{} {}", jolt_emu.display(), args_vec.join(" "));
    debug!("Command: {cmd_str}");

    let status = cmd
        .status()
        .with_context(|| format!("Failed to execute jolt-emu at {}", jolt_emu.display()))?;

    if !status.success() {
        exit(status.code().unwrap_or(1));
    }

    Ok(())
}

// ============================================================================
// Generate commands (from cargo-jolt)
// ============================================================================

fn generate_target_command(cli_args: JoltGenerateTargetArgs) -> Result<()> {
    use zeroos_build::cmds::generate_target_spec;
    use zeroos_build::spec::{load_target_profile, parse_target_triple};

    let target_triple = if let Some(profile_name) = &cli_args.base.profile {
        load_target_profile(profile_name)
            .ok_or_else(|| anyhow::anyhow!("Unknown profile: {profile_name}"))?
            .config
            .target_triple()
    } else if let Some(target) = &cli_args.base.target {
        parse_target_triple(target)
            .ok_or_else(|| anyhow::anyhow!("Cannot parse target triple: {target}"))?
            .target_triple()
    } else {
        return Err(anyhow::anyhow!("Either --profile or --target is required"));
    };

    let json_content = generate_target_spec(&cli_args.base, TargetRenderOptions::default())
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    let output_path = cli_args
        .output
        .unwrap_or_else(|| PathBuf::from(format!("{target_triple}.json")));

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create output directory: {}", parent.display()))?;
    }

    fs::write(&output_path, &json_content)
        .with_context(|| format!("Failed to write target spec to {}", output_path.display()))?;

    info!("Generated target spec: {}", output_path.display());
    info!("Target triple: {target_triple}");

    Ok(())
}

fn generate_linker_command(cli_args: JoltGenerateLinkerArgs) -> Result<()> {
    use zeroos_build::cmds::generate_linker_script;

    let result = generate_linker_script(&cli_args.base)?;

    if let Some(parent) = cli_args.output.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create output directory: {}", parent.display()))?;
    }

    fs::write(&cli_args.output, &result.script_content).with_context(|| {
        format!(
            "Failed to write linker script to {}",
            cli_args.output.display()
        )
    })?;

    info!("Generated linker script: {}", cli_args.output.display());

    Ok(())
}

// ============================================================================
// Project scaffolding (original jolt new)
// ============================================================================

fn create_project(name: String, wasm: bool) {
    create_folder_structure(&name).expect("could not create directory");
    create_host_files(&name).expect("file creation failed");
    create_guest_files(&name).expect("file creation failed");
    if wasm {
        modify_cargo_toml(&name).expect("Failed to update Cargo.toml");
    }
    display_welcome();
}

fn create_folder_structure(name: &str) -> eyre::Result<()> {
    fs::create_dir(name)?;
    fs::create_dir(format!("{name}/src"))?;
    fs::create_dir(format!("{name}/guest"))?;
    fs::create_dir(format!("{name}/guest/src"))?;

    Ok(())
}

fn create_host_files(name: &str) -> eyre::Result<()> {
    let mut toolchain_file = File::create(format!("{name}/rust-toolchain.toml"))?;
    toolchain_file.write_all(RUST_TOOLCHAIN.as_bytes())?;

    let mut gitignore_file = File::create(format!("{name}/.gitignore"))?;
    gitignore_file.write_all(GITIGNORE.as_bytes())?;

    let cargo_file_contents = HOST_CARGO_TEMPLATE.replace("{NAME}", name);
    let mut cargo_file = File::create(format!("{name}/Cargo.toml"))?;
    cargo_file.write_all(cargo_file_contents.as_bytes())?;

    let mut main_file = File::create(format!("{name}/src/main.rs"))?;
    main_file.write_all(HOST_MAIN.as_bytes())?;

    Ok(())
}

fn create_guest_files(name: &str) -> eyre::Result<()> {
    let mut cargo_file = File::create(format!("{name}/guest/Cargo.toml"))?;
    cargo_file.write_all(GUEST_CARGO.as_bytes())?;

    let mut lib_file = File::create(format!("{name}/guest/src/lib.rs"))?;
    lib_file.write_all(GUEST_LIB.as_bytes())?;

    let mut main_file = File::create(format!("{name}/guest/src/main.rs"))?;
    main_file.write_all(GUEST_MAIN.as_bytes())?;

    Ok(())
}

fn display_welcome() {
    display_greeting();
    println!("{}", "-".repeat(80));
    display_sysinfo();
}

fn display_greeting() {
    let jolt_logo_ascii = include_str!("ascii/jolt_ascii.ans");
    println!("\n\n\n\n");
    println!("{jolt_logo_ascii}");
    println!("\n\n\n\n");

    let prompts = [
        "The most Snarky zkVM. Watch out for the lasso.",
        "Buckle your seat belt.",
        "zkVMs are compressors.",
        "Never dupe your network's compute.",
        "You look great today.",
        "Satiate your cores.",
        "The multilinear one.",
        "Transforming network architectures since 2025.",
        "We heard you like sumcheck.",
        "Reed and Solomon were quite the chaps.",
        "Techno optimistic Jolt.",
        "zk is a misnomer.",
        "Twice as fast as Apollo 11.",
        "Mason's favorite zkVM.",
        "Sumcheck Is All You Need",
        "Lasso-ing RISC-V instructions since 2024.",
        "Read. Write. Jolt.",
        "Jolt is not financial advice. Jolt is a zkVM.",
    ];
    let prompt = prompts.choose(&mut rand::thread_rng()).unwrap();
    println!("\x1B[1mWelcome to Jolt.\x1B[0m");
    println!("\x1B[3m{prompt}\x1B[0m");
}

fn display_sysinfo() {
    let mut sys = System::new_all();

    sys.refresh_all();

    println!(
        "OS:             {}",
        System::name().unwrap_or("UNKNOWN".to_string())
    );
    println!(
        "version:        {}",
        System::os_version().unwrap_or("UNKNOWN".to_string())
    );
    println!(
        "Host:           {}",
        System::host_name().unwrap_or("UNKNOWN".to_string())
    );
    println!("CPUs:           {}", sys.cpus().len());
    println!(
        "RAM:            {:.2} GB",
        sys.total_memory() as f64 / 1_000_000_000.0
    );
}

const RUST_TOOLCHAIN: &str = include_str!("../rust-toolchain.toml");

const HOST_CARGO_TEMPLATE: &str = r#"[package]
name = "{NAME}"
version = "0.1.0"
edition = "2021"

[workspace]
members = ["guest"]

[profile.release]
debug = 1
codegen-units = 1
lto = "fat"

[dependencies]
jolt-sdk = { git = "https://github.com/a16z/jolt", features = ["host"] }
guest = { path = "./guest" }
tracing = "0.1"
tracing-subscriber = "0.3"

[patch.crates-io]
ark-ff = { git = "https://github.com/a16z/arkworks-algebra", branch = "dev/twist-shout" }
ark-ec = { git = "https://github.com/a16z/arkworks-algebra", branch = "dev/twist-shout" }
jolt-optimizations = { git = "https://github.com/a16z/arkworks-algebra", branch = "dev/twist-shout" }
ark-serialize = { git = "https://github.com/a16z/arkworks-algebra", branch = "dev/twist-shout" }
ark-bn254 = { git = "https://github.com/a16z/arkworks-algebra", branch = "dev/twist-shout" }
allocative = { git = "https://github.com/facebookexperimental/allocative", rev = "85b773d85d526d068ce94724ff7a7b81203fc95e" }
"#;

const HOST_MAIN: &str = r#"use tracing::info;

pub fn main() {
    tracing_subscriber::fmt::init();

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_fib(target_dir);

    let prover_preprocessing = guest::preprocess_prover_fib(&mut program);
    let verifier_preprocessing = guest::verifier_preprocessing_from_prover_fib(&prover_preprocessing);

    let prove_fib = guest::build_prover_fib(program, prover_preprocessing);
    let verify_fib = guest::build_verifier_fib(verifier_preprocessing);

    let (output, proof, io_device) = prove_fib(50);
    let is_valid = verify_fib(50, output, io_device.panic, proof);

    info!("output: {output}");
    info!("valid: {is_valid}");
}
"#;

const GITIGNORE: &str = "target";

const GUEST_CARGO: &str = r#"[package]
name = "guest"
version = "0.1.0"
edition = "2021"

[features]
guest = []

[dependencies]
jolt = { package = "jolt-sdk", git = "https://github.com/a16z/jolt" }
"#;

const GUEST_LIB: &str = r#"#![cfg_attr(feature = "guest", no_std)]

#[jolt::provable(heap_size = 32768, max_trace_length = 65536)]
fn fib(n: u32) -> u128 {
    let mut a: u128 = 0;
    let mut b: u128 = 1;
    let mut sum: u128;
    for _ in 1..n {
        sum = a + b;
        a = b;
        b = sum;
    }

    b
}
"#;

const GUEST_MAIN: &str = r#"#![cfg_attr(feature = "guest", no_std)]
#![no_main]

#[allow(unused_imports)]
use guest::*;
"#;
