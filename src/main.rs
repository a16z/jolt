use std::{
    fs::{self, File},
    io::Write,
    path::Path,
};

use clap::{Parser, Subcommand};
use eyre::Result;
use rand::prelude::SliceRandom;
use sysinfo::System;

use jolt_core::host::{toolchain, ELFInstruction, Program};
use rmp_serde::Serializer;
use serde::{Deserialize, Serialize};
use syn::{Attribute, ItemFn};
use toml_edit::{value, Array, DocumentMut, InlineTable, Item, Table, Value};

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

#[derive(Serialize, Deserialize)]
struct DecodedData {
    bytecode: Vec<ELFInstruction>,
    memory_init: Vec<(u64, u8)>,
}

fn preprocess_and_save(func_name: &str) -> Result<()> {
    let mut program = Program::new("guest");
    program.set_func(func_name);
    program.set_std(false);
    // Set memory and stack sizes to 10MB and 4KB respectively for now (standard values in jolt-core)
    // TODO: Make these configurable / dynamic depending on how set by the user
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
    decoded_data.serialize(&mut Serializer::new(&mut buf))?;

    let target_dir = Path::new("target/wasm32-unknown-unknown/release");
    fs::create_dir_all(target_dir)?;

    let output_path = target_dir.join(format!("preprocessed_{}.bin", func_name));
    let mut file = File::create(output_path)?;
    file.write_all(&buf)?;
    Ok(())
}

fn extract_provable_functions() -> Vec<String> {
    let content = fs::read_to_string("guest/src/lib.rs").expect("Unable to read file");
    let syntax: syn::File = syn::parse_file(&content).expect("Unable to parse file");

    syntax
        .items
        .iter()
        .filter_map(|item| {
            if let syn::Item::Fn(ItemFn { attrs, sig, .. }) = item {
                if attrs.iter().any(is_provable) {
                    return Some(sig.ident.to_string());
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
    let content = fs::read_to_string("Cargo.toml").ok()?;
    let doc = content.parse::<DocumentMut>().ok()?;
    // replace "-" with "_" to make it a valid identifier
    doc["package"]["name"].as_str().map(|s| {
        s.chars()
            .map(|c| if c == '-' { '_' } else { c })
            .collect::<String>()
    })
}

fn create_index_html(func_names: Vec<String>) -> Result<()> {
    let func_names_with_verify_prefix: Vec<String> = func_names
        .iter()
        .map(|name| format!("verify_{}", name))
        .collect();

    let mut html_content = String::from(HTML_HEAD);

    for func_name in &func_names {
        html_content.push_str(&format!(
            r#"
    <div style="margin-bottom: 10px;">
        <input type="file" id="proofFile_{0}" />
        <button id="verifyButton_{0}">Verify Proof for {0}-Function</button>
    </div>
"#,
            func_name
        ));
    }

    html_content.push_str(&format!(
        r#"
    <script type="module">
        import init, {{ {} }} from './pkg/{}.js';

        async function run() {{
            await init();
"#,
        func_names_with_verify_prefix.join(", "),
        get_project_name().unwrap()
    ));

    for func_name in &func_names {
        html_content.push_str(&format!(
            r#"
            document.getElementById('verifyButton_{0}').addEventListener('click', async () => {{
                const fileInput = document.getElementById('proofFile_{0}');
                if (fileInput.files.length === 0) {{
                    alert("Please select a proof file first.");
                    return;
                }}

                const file = fileInput.files[0];
                const reader = new FileReader();

                reader.onload = async (event) => {{
                    const proofArrayBuffer = event.target.result;
                    const proofData = new Uint8Array(proofArrayBuffer);

                    // Fetch preprocessing data and prepare wasm binary to json conversion
                    const response = await fetch('target/wasm32-unknown-unknown/release/preprocessed_{0}.bin')
                    const wasmBinary = await response.arrayBuffer();
                    const wasmData = new Uint8Array(wasmBinary);

                    const result = verify_{0}(wasmData, proofData);
                    alert(result ? "Proof is valid!" : "Proof is invalid.");
                }};

                reader.readAsArrayBuffer(file);
            }});
"#,
            func_name
        ));
    }

    html_content.push_str(HTML_TAIL);

    let mut file = File::create("index.html")?;
    file.write_all(html_content.as_bytes())?;
    Ok(())
}

fn modify_cargo_toml(name: &str) -> Result<()> {
    fn add_dependencies(dependencies: &mut Table) {
        dependencies.insert("wasm-bindgen", toml_edit::value("0.2.73"));
        dependencies.insert("serde", {
            let mut table = InlineTable::new();
            table.insert("version", Value::from("1.0"));
            let mut features_array = Array::new();
            features_array.push("derive");
            table.insert("features", Value::Array(features_array));
            Item::Value(Value::InlineTable(table))
        });
        dependencies.insert("serde_json", toml_edit::value("1.0"));
        dependencies.insert("serde-wasm-bindgen", toml_edit::value("=0.6.5"));
        dependencies.insert("rmp-serde", toml_edit::value("1.3.0"));
    }

    {
        // first we need to edit the Cargo.toml file in the root directory
        let cargo_toml_path = format!("{}/Cargo.toml", name);
        let content = fs::read_to_string(&cargo_toml_path)?;
        let mut doc = content.parse::<DocumentMut>()?;
        if !doc.contains_key("lib") {
            doc["lib"] = toml_edit::table();
        }

        let lib_section = doc["lib"].as_table_mut().unwrap();

        // add lib section with cdylib crate-type if it doesn't exist
        if !lib_section.contains_key("crate-type") {
            let mut array = Array::new();
            array.push("cdylib");
            lib_section["crate-type"] = Item::Value(toml_edit::Value::Array(array));
            lib_section["path"] = value("guest/src/lib.rs");
        }
        let dependencies = doc["dependencies"].as_table_mut().unwrap();
        add_dependencies(dependencies);

        fs::write(cargo_toml_path, doc.to_string())?;
    }

    // then we need to edit the Cargo.toml file in the guest directory
    {
        let cargo_toml_path = format!("{}/guest/Cargo.toml", name);
        let content = fs::read_to_string(&cargo_toml_path)?;
        let mut doc = content.parse::<DocumentMut>()?;

        if !doc
            .as_table()
            .get("target")
            .and_then(|target| target.get("cfg(target_arch = \"wasm32\")"))
            .and_then(|cfg| cfg.get("dependencies"))
            .map_or(false, |dependencies| dependencies.is_table())
        {
            let mut toml_str = doc.to_string();
            toml_str.push_str("\n[target.'cfg(target_arch = \"wasm32\")'.dependencies]\n");
            doc = toml_str.parse::<DocumentMut>()?;

            let mut table = Table::new();
            add_dependencies(&mut table);

            doc["target"]["cfg(target_arch = \"wasm32\")"]["dependencies"] = Item::Table(table);
            fs::write(cargo_toml_path, doc.to_string())?;
        }
    }
    Ok(())
}

fn build_wasm() {
    println!("Building the project with wasm-pack...");
    let func_names = extract_provable_functions();
    for func_name in func_names.clone() {
        preprocess_and_save(&func_name).expect("Failed to preprocess functions");
    }

    create_index_html(func_names).expect("Failed to create example index.html");

    // todo implement test if wasm-pack is installed

    std::process::Command::new("wasm-pack")
        .args(["build", "--release", "--target", "web"])
        .output()
        .expect("Failed to build the project with wasm-pack");
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

const HTML_HEAD: &str = r#"
<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Jolt x WASMn</title>
    </head>
    <body>
        <h1>Jolt x WASM</h1>
"#;

const HTML_TAIL: &str = r#"
            }

            run();
        </script>
    </body>
</html>
"#;
