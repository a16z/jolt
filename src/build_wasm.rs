use common::attributes::{parse_attributes, Attributes};
use common::jolt_device::{MemoryConfig, MemoryLayout};

use std::{
    fs::{self, File},
    io::Write,
    path::Path,
};

use ark_bn254::Fr;
use eyre::Result;
use jolt_core::{
    host::Program,
    poly::commitment::dory::DoryCommitmentScheme,
    zkvm::{
        prover::JoltProverPreprocessing,
        verifier::{JoltSharedPreprocessing, JoltVerifierPreprocessing},
        Serializable,
    },
};
use syn::{Attribute, ItemFn, Meta, PathSegment};
use toml_edit::{value, Array, DocumentMut, Item, Table};

struct FunctionAttributes {
    pub func_name: String,
    pub attributes: Attributes,
}

fn preprocess_and_save(func_name: &str, attributes: &Attributes, is_std: bool) -> Result<()> {
    let mut program = Program::new("guest");

    program.set_func(func_name);
    program.set_std(is_std);
    program.set_heap_size(attributes.heap_size);
    program.set_stack_size(attributes.stack_size);
    program.set_max_input_size(attributes.max_input_size);
    program.set_max_output_size(attributes.max_output_size);

    let (bytecode, memory_init, program_size) = program.decode();

    let memory_config = MemoryConfig {
        max_input_size: attributes.max_input_size,
        max_trusted_advice_size: attributes.max_trusted_advice_size,
        max_untrusted_advice_size: attributes.max_untrusted_advice_size,
        max_output_size: attributes.max_output_size,
        stack_size: attributes.stack_size,
        heap_size: attributes.heap_size,
        program_size: Some(program_size),
    };
    let memory_layout = MemoryLayout::new(&memory_config);

    let shared = JoltSharedPreprocessing::new(
        bytecode,
        memory_layout,
        memory_init,
        attributes.max_trace_length as usize,
    );

    let prover_preprocessing =
        JoltProverPreprocessing::<Fr, DoryCommitmentScheme>::new(shared.clone());
    let verifier_setup = prover_preprocessing.generators.to_verifier_setup();
    let verifier_preprocessing =
        JoltVerifierPreprocessing::<Fr, DoryCommitmentScheme>::new(shared, verifier_setup);

    let verifier_bytes = verifier_preprocessing.serialize_to_bytes()?;

    let target_dir = Path::new("target/wasm32-unknown-unknown/release");
    fs::create_dir_all(target_dir)?;

    let verifier_path = target_dir.join(format!("preprocessed_{func_name}_verifier.bin"));
    let mut file = File::create(verifier_path)?;
    file.write_all(&verifier_bytes)?;

    let elf_bytes = program
        .get_elf_contents()
        .expect("ELF not found after decode");
    let elf_path = target_dir.join(format!("{func_name}.elf"));
    let mut file = File::create(elf_path)?;
    file.write_all(&elf_bytes)?;

    Ok(())
}

fn extract_provable_functions() -> Vec<FunctionAttributes> {
    let guest_path = Path::new("guest/src/lib.rs");
    let content = fs::read_to_string(guest_path)
        .unwrap_or_else(|_| panic!("Unable to read file: {guest_path:?}"));
    let syntax: syn::File = syn::parse_file(&content)
        .unwrap_or_else(|_| panic!("Unable to parse file: {guest_path:?}"));

    syntax
        .items
        .iter()
        .filter_map(|item| {
            if let syn::Item::Fn(ItemFn { attrs, sig, .. }) = item {
                if let Some(provable_attr) = attrs.iter().find(|attr| is_provable(attr)) {
                    let meta = provable_attr.parse_meta().expect("Unable to parse meta");
                    if let Meta::List(meta_list) = meta {
                        let attributes =
                            parse_attributes(&meta_list.nested.iter().cloned().collect());
                        return Some(FunctionAttributes {
                            func_name: sig.ident.to_string(),
                            attributes,
                        });
                    }
                }
            }
            None
        })
        .collect()
}

fn is_provable(attr: &Attribute) -> bool {
    if attr.path.segments.len() == 2 {
        let segments: Vec<&PathSegment> = attr.path.segments.iter().collect();
        if let [first, second] = segments.as_slice() {
            return first.ident == "jolt" && second.ident == "provable";
        }
    }
    false
}

fn get_project_name() -> Option<String> {
    let content = fs::read_to_string("Cargo.toml").ok()?;
    let doc = content.parse::<DocumentMut>().ok()?;
    doc["package"]["name"].as_str().map(|s| {
        s.chars()
            .map(|c| if c == '-' { '_' } else { c })
            .collect::<String>()
    })
}

fn is_std() -> Option<bool> {
    let content = fs::read_to_string("guest/Cargo.toml").expect("Failed to read Cargo.toml");
    let doc = content
        .parse::<DocumentMut>()
        .expect("Failed to parse Cargo.toml");

    let dependencies = doc["dependencies"]["jolt"].as_inline_table()?;
    let package = dependencies.get("package")?.as_str()?;
    if package != "jolt-sdk" {
        return None;
    }

    Some(
        dependencies
            .get("features")
            .and_then(|v| v.as_array())
            .is_some_and(|features| features.iter().any(|f| f.as_str() == Some("guest-std"))),
    )
}

fn create_index_html(func_names: &[String]) -> Result<()> {
    let func_names_with_verify_prefix: Vec<String> = func_names
        .iter()
        .map(|name| format!("verify_{name}"))
        .collect();

    let mut html_content = String::from(HTML_HEAD);

    for func_name in func_names {
        html_content.push_str(&format!(
            r#"
    <div style="margin-bottom: 10px;">
        <label>{func_name}</label><br/>
        <input type="file" id="proofFile_{func_name}" />
        <input type="file" id="ioFile_{func_name}" />
        <button id="verifyButton_{func_name}">Verify</button>
    </div>
"#
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

    for func_name in func_names {
        html_content.push_str(&format!(
            r#"
            document.getElementById('verifyButton_{func_name}').addEventListener('click', async () => {{
                const proofInput = document.getElementById('proofFile_{func_name}');
                const ioInput = document.getElementById('ioFile_{func_name}');
                if (proofInput.files.length === 0 || ioInput.files.length === 0) {{
                    alert("Please select proof and I/O files.");
                    return;
                }}

                const proofData = new Uint8Array(await proofInput.files[0].arrayBuffer());
                const ioData = new Uint8Array(await ioInput.files[0].arrayBuffer());

                const ppResp = await fetch('target/wasm32-unknown-unknown/release/preprocessed_{func_name}_verifier.bin');
                const ppData = new Uint8Array(await ppResp.arrayBuffer());

                const result = verify_{func_name}(ppData, proofData, ioData);
                alert(result ? "Proof is valid!" : "Proof is invalid.");
            }});
"#
        ));
    }

    html_content.push_str(HTML_TAIL);

    let mut file = File::create("index.html")?;
    file.write_all(html_content.as_bytes())?;
    Ok(())
}

fn generate_wasm_verify_rs(func_names: &[String]) -> Result<()> {
    let src_dir = Path::new("src");
    fs::create_dir_all(src_dir)?;

    let mut code = String::new();
    code.push_str(
        r#"use wasm_bindgen::prelude::*;
use jolt_sdk::{
    F, PCS, JoltDevice, JoltVerifierPreprocessing, RV64IMACProof, RV64IMACVerifier, Serializable,
};

type VerifierPreprocessing = JoltVerifierPreprocessing<F, PCS>;
"#,
    );

    for func_name in func_names {
        code.push_str(&format!(
            r#"
#[wasm_bindgen]
pub fn verify_{func_name}(preprocessing_data: &[u8], proof_data: &[u8], io_data: &[u8]) -> bool {{
    let preprocessing = match VerifierPreprocessing::deserialize_from_bytes(preprocessing_data) {{
        Ok(p) => p,
        Err(_) => return false,
    }};
    let proof = match RV64IMACProof::deserialize_from_bytes(proof_data) {{
        Ok(p) => p,
        Err(_) => return false,
    }};
    let program_io = match JoltDevice::deserialize_from_bytes(io_data) {{
        Ok(d) => d,
        Err(_) => return false,
    }};
    let verifier = match RV64IMACVerifier::new(&preprocessing, proof, program_io, None, None) {{
        Ok(v) => v,
        Err(_) => return false,
    }};
    verifier.verify().is_ok()
}}
"#
        ));
    }

    let path = src_dir.join("wasm_verify.rs");
    let mut file = File::create(path)?;
    file.write_all(code.as_bytes())?;
    Ok(())
}

pub fn modify_cargo_toml(name: &str) -> Result<()> {
    fn insert_if_absent(dependencies: &mut Table, key: &str, value: Item) {
        if !dependencies.contains_key(key) {
            dependencies.insert(key, value);
        }
    }

    fn add_wasm_dependencies(dependencies: &mut Table) {
        insert_if_absent(dependencies, "wasm-bindgen", toml_edit::value("0.2.73"));
    }

    {
        let cargo_toml_path = format!("{name}/Cargo.toml");
        let content = fs::read_to_string(&cargo_toml_path)?;
        let mut doc = content.parse::<DocumentMut>()?;
        if !doc.contains_key("lib") {
            doc["lib"] = toml_edit::table();
        }

        let lib_section = doc["lib"].as_table_mut().unwrap();

        if let Some(array) = lib_section
            .get_mut("crate-type")
            .and_then(|v| v.as_array_mut())
        {
            if !array.iter().any(|v| v.as_str() == Some("cdylib")) {
                array.push("cdylib");
            }
        } else {
            let mut array = Array::new();
            array.push("cdylib");
            lib_section["crate-type"] = Item::Value(toml_edit::Value::Array(array));
        }
        lib_section["path"] = value("src/wasm_verify.rs");
        let dependencies = doc["dependencies"].as_table_mut().unwrap();
        add_wasm_dependencies(dependencies);

        fs::write(cargo_toml_path, doc.to_string())?;
    }

    {
        let cargo_toml_path = format!("{name}/guest/Cargo.toml");
        let content = fs::read_to_string(&cargo_toml_path)?;
        let mut doc = content.parse::<DocumentMut>()?;

        if !doc
            .as_table()
            .get("target")
            .and_then(|target| target.get("cfg(target_arch = \"wasm32\")"))
            .and_then(|cfg| cfg.get("dependencies"))
            .is_some_and(|dependencies| dependencies.is_table())
        {
            let mut toml_str = doc.to_string();
            toml_str.push_str("\n[target.'cfg(target_arch = \"wasm32\")'.dependencies]\n");
            doc = toml_str.parse::<DocumentMut>()?;

            let mut table = Table::new();
            add_wasm_dependencies(&mut table);

            doc["target"]["cfg(target_arch = \"wasm32\")"]["dependencies"] = Item::Table(table);
            fs::write(cargo_toml_path, doc.to_string())?;
        }
    }
    Ok(())
}

pub fn build_wasm() {
    println!("Building the project with wasm-pack...");
    let functions = extract_provable_functions();
    let function_names: Vec<String> = functions.iter().map(|f| f.func_name.clone()).collect();
    let is_std = is_std().expect("Failed to check if std feature is enabled");
    for function in &functions {
        preprocess_and_save(&function.func_name, &function.attributes, is_std)
            .expect("Failed to preprocess functions");
    }

    generate_wasm_verify_rs(&function_names).expect("Failed to generate wasm_verify.rs");
    create_index_html(&function_names).expect("Failed to create index.html");

    modify_cargo_toml(".").expect("Failed to update Cargo.toml for WASM build");

    let output = std::process::Command::new("wasm-pack")
        .args(["build", "--release", "--target", "web"])
        .output()
        .expect("Failed to execute wasm-pack command");

    if !output.status.success() {
        eprintln!("Error: Failed to build the project with wasm-pack");
        eprintln!("stdout: {}", String::from_utf8_lossy(&output.stdout));
        eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));
        panic!("wasm-pack build failed");
    }
}

const HTML_HEAD: &str = r#"
<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Jolt x WASM</title>
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
