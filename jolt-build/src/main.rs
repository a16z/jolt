use anyhow::{Context, Result};
use clap::Parser;
use log::{debug, info};
use std::fs;
use std::path::PathBuf;
use std::process::{exit, Command};

use zeroos_build::cmds::{BuildArgs, StdMode};
use zeroos_build::spec::TargetRenderOptions;

/// Linker script template embedded at compile time.
/// This linker script is for Jolt zkVM guests.
static LINKER_TEMPLATE: &str = include_str!("linker.ld.template");

#[derive(Parser)]
#[command(name = "cargo-jolt")]
#[command(bin_name = "cargo")]
#[command(about = "Build for Jolt zkVM", version, long_about = None)]
enum Cli {
    #[command(name = "jolt", subcommand)]
    Jolt(JoltCmd),
}

#[derive(clap::Subcommand, Debug)]
enum JoltCmd {
    Build(JoltBuildArgs),

    Run(RunArgs),

    #[command(subcommand)]
    Generate(GenerateCmd),
}

#[derive(clap::Subcommand, Debug)]
enum GenerateCmd {
    Target(JoltGenerateTargetArgs),

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

fn main() {
    env_logger::Builder::from_default_env()
        .format_timestamp(None)
        .format_module_path(false)
        .init();

    debug!("cargo-jolt starting");

    if let Err(e) = run() {
        eprintln!("Error: {:#}", e);
        exit(1);
    }
}

fn run() -> Result<()> {
    let Cli::Jolt(cmd) = Cli::parse();

    match cmd {
        JoltCmd::Build(args) => build_command(args),
        JoltCmd::Run(args) => run_command(args),
        JoltCmd::Generate(gen_cmd) => match gen_cmd {
            GenerateCmd::Target(args) => generate_target_command(args),
            GenerateCmd::Linker(args) => generate_linker_command(args),
        },
    }
}

/// Jolt-specific rustflags for zkVM environment
const JOLT_RUSTFLAGS: &[&str] = &[
    // lower-atomic: Jolt prover doesn't support LR/SC atomic instructions
    "-Cpasses=lower-atomic",
    // panic=abort: Required for zkVM
    "-Cpanic=abort",
    // Optimize for size (helps with register allocation for inline asm)
    "-Copt-level=z",
    // Disable LLVM's MachineOutliner: it generates a faulty calling pattern on RISC-V
    // that creates infinite loops (auipc t1,0; jr offset(t1); ... jr t1 returns to auipc).
    "-Cllvm-args=-enable-machine-outliner=never",
    // Use jolt-platform's custom getrandom implementation
    "--cfg=getrandom_backend=\"custom\"",
];

fn build_command(args: JoltBuildArgs) -> Result<()> {
    debug!("build_command: {:?}", args);

    let workspace_root = zeroos_build::cmds::find_workspace_root()?;
    debug!("workspace_root: {}", workspace_root.display());

    // Use the embedded linker template (compiled into the binary)
    let linker_tpl = LINKER_TEMPLATE.to_string();

    let fully = args.base.mode == StdMode::Std || args.base.fully;

    let toolchain_paths = if args.base.mode == StdMode::Std || fully {
        Some(zeroos_build::cmds::get_or_build_toolchain(
            args.base.musl_lib_path.clone(),
            args.base.gcc_lib_path.clone(),
            fully,
        )?)
    } else {
        None
    };

    zeroos_build::cmds::build_binary_with_rustflags(
        &workspace_root,
        &args.base,
        toolchain_paths,
        Some(linker_tpl),
        Some(JOLT_RUSTFLAGS),
    )?;

    Ok(())
}

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
    debug!("Command: {}", cmd_str);

    let status = cmd
        .status()
        .with_context(|| format!("Failed to execute jolt-emu at {}", jolt_emu.display()))?;

    if !status.success() {
        exit(status.code().unwrap_or(1));
    }

    Ok(())
}

fn generate_target_command(cli_args: JoltGenerateTargetArgs) -> Result<()> {
    use zeroos_build::cmds::generate_target_spec;
    use zeroos_build::spec::{load_target_profile, parse_target_triple};

    let target_triple = if let Some(profile_name) = &cli_args.base.profile {
        load_target_profile(profile_name)
            .ok_or_else(|| anyhow::anyhow!("Unknown profile: {}", profile_name))?
            .config
            .target_triple()
    } else if let Some(target) = &cli_args.base.target {
        parse_target_triple(target)
            .ok_or_else(|| anyhow::anyhow!("Cannot parse target triple: {}", target))?
            .target_triple()
    } else {
        return Err(anyhow::anyhow!("Either --profile or --target is required"));
    };

    let json_content = generate_target_spec(&cli_args.base, TargetRenderOptions::default())
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    let output_path = cli_args
        .output
        .unwrap_or_else(|| PathBuf::from(format!("{}.json", target_triple)));

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create output directory: {}", parent.display()))?;
    }

    fs::write(&output_path, &json_content)
        .with_context(|| format!("Failed to write target spec to {}", output_path.display()))?;

    info!("Generated target spec: {}", output_path.display());
    info!("Target triple: {}", target_triple);

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
