use std::{fs::{self, File}, io::Write};

use clap::{Parser, Subcommand};
use eyre::Result;

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
    },
    InstallToolchain,
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::New { name } => create_project(name),
        Command::InstallToolchain => install_toolchain(),
    }
}

fn create_project(name: String) {
    create_folder_structure(&name).expect("could not create directory");
    create_host_files(&name).expect("file creation failed");
    create_guest_files(&name).expect("file creation failed");
}

fn install_toolchain() {
   std::process::Command::new("rustup")
       .args(["target", "add", "riscv32i-unknown-none-elf"])
       .output()
       .expect("could not install toolchain");

}

fn create_folder_structure(name: &str) -> Result<()> {
    fs::create_dir(name)?;
    fs::create_dir(format!("{}/src", name))?;
    fs::create_dir(format!("{}/guest", name))?;
    fs::create_dir(format!("{}/guest/src", name))?;

    Ok(())
}

fn create_host_files(name: &str) -> Result<()> {
    let mut toolchain_file = File::create(format!("{}/rust-toolchain", name))?;
    toolchain_file.write("nightly-2023-09-22".as_bytes())?;

    let cargo_file_contents = HOST_CARGO_TEMPLATE.replace("{NAME}", name);
    let mut cargo_file = File::create(format!("{}/Cargo.toml", name))?;
    cargo_file.write(cargo_file_contents.as_bytes())?;

    let mut main_file = File::create(format!("{}/src/main.rs", name))?;
    main_file.write(HOST_MAIN.as_bytes())?;

    Ok(())
}

fn create_guest_files(name: &str) -> Result<()> {
    let mut cargo_file = File::create(format!("{}/guest/Cargo.toml", name))?;
    cargo_file.write(GUEST_CARGO.as_bytes())?;

    let mut lib_file = File::create(format!("{}/guest/src/lib.rs", name))?;
    lib_file.write(GUEST_LIB.as_bytes())?;

    Ok(())
}

const HOST_CARGO_TEMPLATE: &str = r#"
[package]
name = "{NAME}"
version = "0.1.0"
edition = "2021"

[workspace]
members = ["guest"]

[dependencies]
jolt = { git = "https://github.com/a16z/Lasso", branch = "jolt", features = ["std"] }
guest = { path = "./guest" }

hex = "0.4.3"
sha3 = { version = "0.10.8", default-features = false }
"#;

const HOST_MAIN: &str = r#"
pub fn main() {
    let (prove_fib, verify_fib) = guest::build_fib();

    let (output, proof) = prove_fib(50);
    let is_valid = verify_fib(proof);

    println!("output: {}", output);
    println!("valid: {}", is_valid);
}
"#;

const GUEST_CARGO: &str = r#"
[package]
name = "guest"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "guest"
path = "./src/lib.rs"

[features]
guest = []

[dependencies]
jolt = { git = "https://github.com/a16z/Lasso", branch = "jolt" }
"#;

const GUEST_LIB: &str = r#"
#![cfg_attr(feature = "guest", no_std)]
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
