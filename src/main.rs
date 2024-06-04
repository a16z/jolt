use std::{
    fs::{self, File},
    io::Write,
};

use clap::{Parser, Subcommand};
use eyre::Result;
use rand::prelude::SliceRandom;
use sysinfo::System;

use jolt_core::host::{toolchain, ELFInstruction, Program};
use rmp_serde::Serializer;
use serde::{Deserialize, Serialize};
use syn::{Attribute, ItemFn};
use toml_edit::{value, Array, DocumentMut, Item, Value};

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
    /// Installs the required RISC-V toolchains for Rust
    InstallToolchain,
    /// Handles preprocessing and generates WASM compatible files
    CreateWasm,
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::New { name } => create_project(name),
        Command::InstallToolchain => install_toolchain(),
        Command::CreateWasm => create_wasm(),
    }
}

fn create_project(name: String) {
    create_folder_structure(&name).expect("could not create directory");
    create_host_files(&name).expect("file creation failed");
    create_guest_files(&name).expect("file creation failed");
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

#[derive(Serialize, Deserialize)]
struct DecodedData {
    bytecode: Vec<ELFInstruction>,
    memory_init: Vec<(u64, u8)>,
}

fn preprocess_and_save(func_name: &str, output_file: &str) {
    println!("Preprocessing {}...", func_name);
    let mut program = Program::new("guest");
    program.set_func(func_name);
    program.set_std(false);
    // Set memory and stack sizes to 10MB and 4KB respectively
    // TODO: Make these configurable
    program.set_memory_size(10485760u64);
    program.set_stack_size(4096u64);
    program.set_max_input_size(4096u64);
    program.set_max_output_size(4096u64);

    let (bytecode, memory_init) = program.decode();
    let decoded_data = DecodedData {
        bytecode,
        memory_init,
    };

    let mut buf = Vec::new();
    decoded_data
        .serialize(&mut Serializer::new(&mut buf))
        .unwrap();

    let target_dir = "target/wasm32-unknown-unknown/release";
    fs::create_dir_all(target_dir).unwrap();

    let output_path = format!("{}/{}", target_dir, output_file);
    let mut file = File::create(&output_path).unwrap();
    file.write_all(&buf).unwrap();

    println!("Decoded data for {} saved to {}", func_name, output_file);
}

fn extract_provable_functions() -> Vec<String> {
    println!("Extracting provable functions from \"guest/src/lib.rs\"...",);
    let content = fs::read_to_string("guest/src/lib.rs").expect("Unable to read file");
    let syntax: syn::File = syn::parse_file(&content).expect("Unable to parse file");

    syntax
        .items
        .iter()
        .filter_map(|item| {
            if let syn::Item::Fn(ItemFn { attrs, sig, .. }) = item {
                for attr in attrs {
                    if is_provable(&attr) {
                        println!("Found provable function: {}", sig.ident);
                        return Some(sig.ident.to_string());
                    }
                }
            }
            None
        })
        .collect()
}

fn is_provable(attr: &Attribute) -> bool {
    if let Some(first_segment) = attr.path().segments.first() {
        if first_segment.ident == "jolt" {
            if let Some(second_segment) = attr.path().segments.iter().nth(1) {
                if second_segment.ident == "provable" {
                    return true;
                }
            }
        }
    }

    false
}

fn get_project_name() -> Option<String> {
    let cargo_toml_path = "Cargo.toml";
    let content = fs::read_to_string(cargo_toml_path).ok()?;
    let doc = content.parse::<DocumentMut>().ok()?;
    doc["package"]["name"].as_str().map(|s| s.replace("-", "_"))
}

fn generate_lib_rs_content() -> String {
    let content = r#"
use jolt::{tracer::ELFInstruction, Jolt, Proof, RV32IJoltVM};
use rmp_serde::Deserializer;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use web_sys::console;

#[derive(Serialize, Deserialize)]
pub struct DecodedData {
    bytecode: Vec<ELFInstruction>,
    memory_init: Vec<(u64, u8)>,
}

fn deserialize_from_bin<'a, T: Deserialize<'a>>(
    data: &'a [u8],
) -> Result<T, rmp_serde::decode::Error> {
    let mut de = Deserializer::new(data);
    Deserialize::deserialize(&mut de)
}

#[wasm_bindgen]
pub fn verify_proof(preprocessing_data: &[u8], proof_bytes: &[u8]) -> bool {
    console::log_1(&"Verifying proof...".into());

    let decoded_preprocessing_data: DecodedData = match deserialize_from_bin(preprocessing_data) {
        Ok(data) => data,
        Err(e) => {
            console::log_1(&format!("Failed to decode proof_data: {:?}", e).into());
            return false;
        }
    };

    let proof = match Proof::deserialize_from_bytes(proof_bytes) {
        Ok(p) => p,
        Err(e) => {
            console::log_1(&format!("Failed to deserialize proof from bytes: {:?}", e).into());
            return false;
        }
    };

    let preprocessing = RV32IJoltVM::preprocess(
        decoded_preprocessing_data.bytecode,
        decoded_preprocessing_data.memory_init,
        1 << 20,
        1 << 20,
        1 << 24,
    );

    let result = RV32IJoltVM::verify(preprocessing, proof.proof, proof.commitments);

    match result {
        Ok(_is_valid) => {
            console::log_1(&"Proof successfully VERIFIED".into());
            true
        }
        Err(e) => {
            console::log_1(&format!("Verification error: {:?}", e).into());
            false
        }
    }
}
"#;

    content.to_string()
}

fn write_lib_rs() -> Result<()> {
    let lib_rs_path = "src/lib.rs";
    let lib_rs_content = generate_lib_rs_content();

    // Ensure the src directory exists
    fs::create_dir_all("src")?;

    let mut file = File::create(lib_rs_path)?;
    file.write_all(lib_rs_content.as_bytes())?;
    Ok(())
}

fn create_index_html(func_names: Vec<String>) -> std::io::Result<()> {
    let mut html_content = String::from(
        r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Jolt x WASMn</title>
</head>
<body>
    <h1>Jolt x WASM</h1>
"#,
    );

    for func_name in &func_names {
        html_content.push_str(&format!(
            r#"
    <div style="margin-bottom: 10px;">
        <input type="file" id="proofFile_{}" />
        <button id="verifyButton_{}">Verify Proof for {}-Function</button>
    </div>
"#,
            func_name, func_name, func_name
        ));
    }

    html_content.push_str(&format!(
        r#"
    <script type="module">
        import init, {{ verify_proof }} from './out/{}.js';

        async function run() {{
            await init();
"#,
        get_project_name().unwrap()
    ));

    for func_name in &func_names {
        html_content.push_str(&format!(
            r#"
            document.getElementById('verifyButton_{}').addEventListener('click', async () => {{
                const fileInput = document.getElementById('proofFile_{}');
                if (fileInput.files.length === 0) {{
                    alert("Please select a proof file first.");
                    return;
                }}

                const file = fileInput.files[0];
                const reader = new FileReader();

                reader.onload = async (event) => {{
                    const proofArrayBuffer = event.target.result;
                    const proofData = new Uint8Array(proofArrayBuffer);
                    console.log(proofData);

                    // Fetch preprocessing data and prepare wasm binary to json conversion
                    const response = await fetch('target/wasm32-unknown-unknown/release/{}_wasm.bin')
                    const wasmBinary = await response.arrayBuffer();
                    const wasmData = new Uint8Array(wasmBinary);

                    const result = verify_proof(wasmData, proofData);
                    console.log(result);
                    alert(result ? "Proof is valid!" : "Proof is invalid.");
                }};

                reader.readAsArrayBuffer(file);
            }});
"#,
            func_name, func_name, func_name
        ));
    }

    html_content.push_str(
        r#"
        }

        run();
    </script>
</body>
</html>
"#,
    );

    let mut file = File::create("index.html")?;
    file.write_all(html_content.as_bytes())?;
    Ok(())
}

fn modify_cargo_toml() -> Result<()> {
    let cargo_toml_path = "Cargo.toml";
    let content = fs::read_to_string(cargo_toml_path)?;
    let mut doc = content.parse::<DocumentMut>()?;

    if !doc.contains_key("lib") {
        doc["lib"] = toml_edit::table();
    }

    let lib_section = doc["lib"].as_table_mut().unwrap();
    if !lib_section.contains_key("crate-type") {
        let mut array = Array::new();
        array.push("cdylib");
        // better way to do this?
        lib_section["crate-type"] = Item::Value(toml_edit::Value::Array(array));
    }

    // shouldnt be the case because the project is created with jolt new ...
    if !doc.contains_key("dependencies") {
        doc["dependencies"] = toml_edit::table();
    }

    let dependencies = doc["dependencies"].as_table_mut().unwrap();
    dependencies.insert("wasm-bindgen", toml_edit::value("0.2.73"));
    dependencies.insert("serde", {
        let mut table = toml_edit::InlineTable::new();
        table.insert("version", Value::from("1.0"));
        let mut features_array = toml_edit::Array::new();
        features_array.push("derive");
        table.insert("features", Value::Array(features_array));
        Item::Value(Value::InlineTable(table))
    });
    dependencies.insert("serde_json", toml_edit::value("1.0"));
    dependencies.insert("serde-wasm-bindgen", toml_edit::value("=0.6.5"));
    dependencies.insert("canonical", toml_edit::value("0.7.1"));
    dependencies.insert("ark-serialize", toml_edit::value("0.4.2"));
    dependencies.insert("web-sys", {
        let mut table = toml_edit::InlineTable::new();
        table.insert("version", Value::from("0.3"));
        let mut features_array = toml_edit::Array::new();
        features_array.push("console");
        table.insert("features", Value::Array(features_array));
        Item::Value(Value::InlineTable(table))
    });
    dependencies.insert("rmp-serde", toml_edit::value("1.3.0"));

    fs::write(cargo_toml_path, doc.to_string())?;
    Ok(())
}

fn create_wasm() {
    // write an example lib.rs file
    if let Err(e) = write_lib_rs() {
        eprintln!("Failed to write src/lib.rs: {}", e);
        return;
    }

    // Update Cargo.toml
    if let Err(e) = modify_cargo_toml() {
        eprintln!("Failed to update Cargo.toml: {}", e);
        return;
    }

    println!("Creating wasm files...");
    let func_names = extract_provable_functions();

    // TODO: any better solution for this?
    for func_name in func_names.clone() {
        let output_file = format!("{}_{}.bin", &func_name, "wasm");
        preprocess_and_save(&func_name, &output_file);
    }

    if let Err(e) = create_index_html(func_names) {
        eprintln!("Failed to create index.html: {}", e);
    }

    // Build the project for the wasm32 target
    let build_status = std::process::Command::new("cargo")
        .args(&["build", "--release", "--target", "wasm32-unknown-unknown"])
        .status()
        .expect("Failed to build the project for wasm32 target");

    if !build_status.success() {
        eprintln!("cargo build failed");
        return;
    }

    // Check if wasm-bindgen is installed
    let wasm_bindgen_installed = std::process::Command::new("wasm-bindgen")
        .arg("--version")
        .status()
        .is_ok();

    if !wasm_bindgen_installed {
        //TODO: or install it automatically?!
        eprintln!("wasm-bindgen is not installed. Please install it with `cargo install wasm-bindgen-cli`.");
        return;
    }

    // Get the project name vecause we need it for the wasm-bindgen command
    let project_name = match get_project_name() {
        Some(name) => name,
        None => {
            eprintln!("Failed to get the project name from Cargo.toml");
            return;
        }
    };

    // then run wasm-bindgen
    let wasm_file = format!(
        "target/wasm32-unknown-unknown/release/{}.wasm",
        project_name
    );
    let wasm_bindgen_status = std::process::Command::new("wasm-bindgen")
        .args(&["--target", "web", "--out-dir", "./out", &wasm_file])
        .status()
        .expect("Failed to run wasm-bindgen");

    if !wasm_bindgen_status.success() {
        eprintln!("wasm-bindgen failed");
    }
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
