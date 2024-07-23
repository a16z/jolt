mod build_wasm;

use std::{
    fs::{self, File},
    io::Write,
};

use clap::{Parser, Subcommand};
use eyre::Result;
use rand::prelude::SliceRandom;
use sysinfo::System;

use build_wasm::{build_wasm, modify_cargo_toml};
use jolt_core::host::toolchain;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Creates a new Jolt project with the specified name
    New {
        /// Project name
        name: String,
        /// Whether to generate WASM compatible files
        #[arg(short, long)]
        wasm: bool,
    },
    /// Installs the required RISC-V toolchains for Rust
    InstallToolchain,
    /// Handles preprocessing and generates WASM compatible files
    BuildWasm,
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::New { name, wasm } => create_project(name, wasm),
        Command::InstallToolchain => install_toolchain(),
        Command::BuildWasm => build_wasm(),
    }
}

fn create_project(name: String, wasm: bool) {
    create_folder_structure(&name).expect("could not create directory");
    create_host_files(&name).expect("file creation failed");
    create_guest_files(&name).expect("file creation failed");
    if wasm {
        modify_cargo_toml(&name).expect("Failed to update Cargo.toml");
    }
}

fn install_toolchain() {
    if let Err(err) = toolchain::install_toolchain() {
        panic!("toolchain install failed: {}", err);
    }
    display_welcome();
}

fn create_folder_structure(name: &str) -> Result<()> {
    fs::create_dir(name)?;
    fs::create_dir(format!("{}/src", name))?;
    fs::create_dir(format!("{}/guest", name))?;
    fs::create_dir(format!("{}/guest/src", name))?;

    Ok(())
}

fn create_host_files(name: &str) -> Result<()> {
    let mut toolchain_file = File::create(format!("{}/rust-toolchain.toml", name))?;
    toolchain_file.write_all(RUST_TOOLCHAIN.as_bytes())?;

    let mut gitignore_file = File::create(format!("{}/.gitignore", name))?;
    gitignore_file.write_all(GITIGNORE.as_bytes())?;

    let cargo_file_contents = HOST_CARGO_TEMPLATE.replace("{NAME}", name);
    let mut cargo_file = File::create(format!("{}/Cargo.toml", name))?;
    cargo_file.write_all(cargo_file_contents.as_bytes())?;

    let mut main_file = File::create(format!("{}/src/main.rs", name))?;
    main_file.write_all(HOST_MAIN.as_bytes())?;

    Ok(())
}

fn create_guest_files(name: &str) -> Result<()> {
    let mut cargo_file = File::create(format!("{}/guest/Cargo.toml", name))?;
    cargo_file.write_all(GUEST_CARGO.as_bytes())?;

    let mut lib_file = File::create(format!("{}/guest/src/lib.rs", name))?;
    lib_file.write_all(GUEST_LIB.as_bytes())?;

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
    println!("{}", jolt_logo_ascii);
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
        "Lasso-ing RV32 instructions since 2024.",
        "Read. Write. Jolt.",
        "Jolt is not financial advice. Jolt is a zkVM.",
    ];
    let prompt = prompts.choose(&mut rand::thread_rng()).unwrap();
    println!("\x1B[1mWelcome to Jolt.\x1B[0m");
    println!("\x1B[3m{}\x1B[0m", prompt);
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
jolt = { package = "jolt-sdk", git = "https://github.com/a16z/jolt", features = ["host"] }
guest = { path = "./guest" }

[patch.crates-io]
ark-ff = { git = "https://github.com/a16z/arkworks-algebra", branch = "optimize/field-from-u64" }
ark-ec = { git = "https://github.com/a16z/arkworks-algebra", branch = "optimize/field-from-u64" }
ark-serialize = { git = "https://github.com/a16z/arkworks-algebra", branch = "optimize/field-from-u64" }
"#;

const HOST_MAIN: &str = r#"pub fn main() {
    let (prove_fib, verify_fib) = guest::build_fib();

    let (output, proof) = prove_fib(50);
    let is_valid = verify_fib(proof);

    println!("output: {}", output);
    println!("valid: {}", is_valid);
}
"#;

const GITIGNORE: &str = "target";

const GUEST_CARGO: &str = r#"[package]
name = "guest"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "guest"
path = "./src/lib.rs"

[features]
guest = []

[dependencies]
jolt = { package = "jolt-sdk", git = "https://github.com/a16z/jolt" }
"#;

const GUEST_LIB: &str = r#"#![cfg_attr(feature = "guest", no_std)]
#![no_main]

#[jolt::provable]
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
