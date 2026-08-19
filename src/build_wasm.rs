use std::{
    fs::{self, File},
    io::Write,
    path::Path,
};

use eyre::Result;
use syn::{Attribute, ItemFn, PathSegment};
use toml_edit::{value, Array, DocumentMut, Item, Table};

fn extract_provable_functions() -> Vec<String> {
    let guest_path = Path::new("guest/src/lib.rs");
    let content = fs::read_to_string(guest_path)
        .unwrap_or_else(|_| panic!("Unable to read file: {guest_path:?}"));
    functions_from_guest_source(&content)
}

fn functions_from_guest_source(content: &str) -> Vec<String> {
    let syntax: syn::File =
        syn::parse_file(content).unwrap_or_else(|_| panic!("Unable to parse guest source"));

    syntax
        .items
        .iter()
        .filter_map(|item| {
            let syn::Item::Fn(ItemFn { attrs, sig, .. }) = item else {
                return None;
            };
            if attrs.iter().any(is_provable) {
                Some(sig.ident.to_string())
            } else {
                None
            }
        })
        .collect()
}

fn is_provable(attr: &Attribute) -> bool {
    if attr.path().segments.len() == 2 {
        let segments: Vec<&PathSegment> = attr.path().segments.iter().collect();
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

fn create_index_html(func_names: &[String]) -> Result<()> {
    let project_name = get_project_name().unwrap();
    let html_content = index_html_source(func_names, &project_name);
    let mut file = File::create("index.html")?;
    file.write_all(html_content.as_bytes())?;
    Ok(())
}

fn index_html_source(func_names: &[String], project_name: &str) -> String {
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
        <input type="file" id="trustedAdviceFile_{func_name}" />
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
        project_name
    ));

    for func_name in func_names {
        html_content.push_str(&format!(
            r#"
            document.getElementById('verifyButton_{func_name}').addEventListener('click', async () => {{
                const proofInput = document.getElementById('proofFile_{func_name}');
                const ioInput = document.getElementById('ioFile_{func_name}');
                const trustedInput = document.getElementById('trustedAdviceFile_{func_name}');
                if (proofInput.files.length === 0 || ioInput.files.length === 0) {{
                    alert("Please select proof and I/O files.");
                    return;
                }}

                const proofData = new Uint8Array(await proofInput.files[0].arrayBuffer());
                const ioData = new Uint8Array(await ioInput.files[0].arrayBuffer());
                const trustedData = trustedInput.files.length === 0
                    ? new Uint8Array()
                    : new Uint8Array(await trustedInput.files[0].arrayBuffer());

                const ppResp = await fetch('target/wasm32-unknown-unknown/release/preprocessed_{func_name}_verifier.bin');
                const ppData = new Uint8Array(await ppResp.arrayBuffer());

                const result = verify_{func_name}(ppData, proofData, ioData, trustedData);
                alert(result ? "Proof is valid!" : "Proof is invalid.");
            }});
"#
        ));
    }

    html_content.push_str(HTML_TAIL);
    html_content
}

fn generate_wasm_verify_rs(func_names: &[String]) -> Result<()> {
    let src_dir = Path::new("src");
    fs::create_dir_all(src_dir)?;

    let path = src_dir.join("wasm_verify.rs");
    let mut file = File::create(path)?;
    file.write_all(wasm_verify_source(func_names).as_bytes())?;
    Ok(())
}

fn wasm_verify_source(func_names: &[String]) -> String {
    let mut code = String::from(
        r#"use wasm_bindgen::prelude::*;
use jolt_sdk::{
    deserialize_verifier_object, JoltDevice, JoltVerifierPreprocessing, RV64IMACProof,
};
"#,
    );

    for func_name in func_names {
        code.push_str(&format!(
            r#"
#[wasm_bindgen]
pub fn verify_{func_name}(
    preprocessing_data: &[u8],
    proof_data: &[u8],
    io_data: &[u8],
    trusted_advice_commitment_bytes: &[u8],
) -> bool {{
    let preprocessing: JoltVerifierPreprocessing = match deserialize_verifier_object(preprocessing_data) {{
        Ok(p) => p,
        Err(_) => return false,
    }};
    let proof: RV64IMACProof = match deserialize_verifier_object(proof_data) {{
        Ok(p) => p,
        Err(_) => return false,
    }};
    let program_io: JoltDevice = match deserialize_verifier_object(io_data) {{
        Ok(d) => d,
        Err(_) => return false,
    }};
    let trusted_advice_commitment: Option<jolt_sdk::VerifierTrustedAdviceCommitment> =
        if trusted_advice_commitment_bytes.is_empty() {{
            None
        }} else {{
            match deserialize_verifier_object(trusted_advice_commitment_bytes) {{
                Ok(commitment) => commitment,
                Err(_) => return false,
            }}
        }};
    jolt_sdk::jolt_verifier::verify::<
        jolt_sdk::VerifierField,
        jolt_sdk::VerifierPCS,
        jolt_sdk::VerifierVC,
        jolt_sdk::VerifierTranscript,
    >(&preprocessing, &program_io, &proof, trusted_advice_commitment.as_ref()).is_ok()
}}
"#
        ));
    }

    code
}

fn wasm_preprocess_source(func_names: &[String]) -> String {
    let mut code = String::from(
        r#"fn main() {
    let target_dir = "/tmp/jolt-guest-targets";
    let out_dir = std::path::Path::new("target/wasm32-unknown-unknown/release");
    std::fs::create_dir_all(out_dir).expect("create wasm preprocessing directory");
"#,
    );

    for func_name in func_names {
        code.push_str(&format!(
            r#"
    {{
        let mut program = guest::compile_{func_name}(target_dir);
        let shared = guest::preprocess_shared_{func_name}(&mut program)
            .unwrap_or_else(|err| panic!("shared preprocessing failed for {func_name}: {{err}}"));
        let prover = guest::preprocess_prover_{func_name}(shared);
        let verifier = guest::verifier_preprocessing_from_prover_{func_name}(&prover);
        let bytes = jolt_sdk::serialize_verifier_object(&verifier)
            .unwrap_or_else(|err| panic!("serialize preprocessing failed for {func_name}: {{err}}"));
        std::fs::write(out_dir.join("preprocessed_{func_name}_verifier.bin"), &bytes)
            .unwrap_or_else(|err| panic!("write preprocessing failed for {func_name}: {{err}}"));
    }}
"#
        ));
    }

    code.push_str("}\n");
    code
}

fn run_project_native_preprocess(func_names: &[String]) {
    let bin_dir = Path::new("src/bin");
    fs::create_dir_all(bin_dir).expect("Failed to create src/bin for WASM preprocessing");
    let bin_path = bin_dir.join("jolt_wasm_preprocess.rs");
    fs::write(&bin_path, wasm_preprocess_source(func_names))
        .expect("Failed to write WASM preprocessing helper");

    let output = std::process::Command::new("cargo")
        .args(["run", "--release", "--bin", "jolt_wasm_preprocess"])
        .output()
        .expect("Failed to execute cargo run for WASM preprocessing");

    let _ = fs::remove_file(&bin_path);
    if fs::read_dir(bin_dir)
        .ok()
        .is_some_and(|mut entries| entries.next().is_none())
    {
        let _ = fs::remove_dir(bin_dir);
    }

    if !output.status.success() {
        eprintln!("Error: project-native WASM preprocessing failed");
        eprintln!("stdout: {}", String::from_utf8_lossy(&output.stdout));
        eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));
        panic!("WASM preprocessing failed");
    }

    for func_name in func_names {
        let path = Path::new("target/wasm32-unknown-unknown/release")
            .join(format!("preprocessed_{func_name}_verifier.bin"));
        let metadata = fs::metadata(&path).unwrap_or_else(|_| {
            panic!(
                "WASM preprocessing did not produce {}: the command cannot package missing or stale preprocessing",
                path.display()
            )
        });
        if metadata.len() == 0 {
            panic!(
                "WASM preprocessing produced an empty artifact at {}",
                path.display()
            );
        }
    }
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
    run_project_native_preprocess(&functions);

    generate_wasm_verify_rs(&functions).expect("Failed to generate wasm_verify.rs");
    create_index_html(&functions).expect("Failed to create index.html");

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
