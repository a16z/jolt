use common::attributes::{parse_attributes, Attributes};
use common::jolt_device::{MemoryConfig, MemoryLayout};

use std::{
    fs::{self, File},
    io::Write,
    path::Path,
};

use ark_bn254::Fr;
use eyre::Result;
use jolt_prover_legacy::zkvm::proof::verifier_preprocessing_from_prover;
use jolt_prover_legacy::{
    curve::Bn254Curve,
    host::Program,
    poly::commitment::dory::DoryCommitmentScheme,
    zkvm::{
        preprocessing::JoltSharedPreprocessing, program::ProgramPreprocessing,
        prover::JoltProverPreprocessing,
    },
};
use jolt_sdk::serialize_verifier_object;
use syn::{punctuated::Punctuated, Attribute, ItemFn, Meta, PathSegment, Token};
use toml_edit::{value, Array, DocumentMut, Item, Table};

struct FunctionAttributes {
    pub func_name: String,
    pub attributes: Attributes,
    pub has_trusted_advice: bool,
}

fn memory_config_from_attributes(attributes: &Attributes) -> MemoryConfig {
    MemoryConfig {
        max_input_size: attributes.max_input_size,
        max_trusted_advice_size: attributes.max_trusted_advice_size,
        max_untrusted_advice_size: attributes.max_untrusted_advice_size,
        max_output_size: attributes.max_output_size,
        stack_size: attributes.stack_size,
        heap_size: attributes.heap_size,
        program_size: None,
    }
}

fn preprocess_and_save(func_name: &str, attributes: &Attributes, is_std: bool) -> Result<()> {
    let mut host_program = Program::new("guest");

    host_program.set_func(func_name);
    host_program.set_std(is_std);
    host_program.set_memory_config(memory_config_from_attributes(attributes));
    if let Some(profile) = &attributes.profile {
        host_program.set_profile(profile);
    }
    if let Some(backtrace) = &attributes.backtrace {
        host_program.set_backtrace(backtrace);
    }

    let (bytecode, memory_init, program_size, e_entry) = host_program.decode();

    let mut memory_config = memory_config_from_attributes(attributes);
    memory_config.program_size = Some(program_size);
    let memory_layout = MemoryLayout::new(&memory_config);

    let preprocessed_program = ProgramPreprocessing::preprocess(bytecode, memory_init, e_entry)?;
    let shared = JoltSharedPreprocessing::new(
        preprocessed_program,
        memory_layout,
        attributes.max_trace_length as usize,
    );

    let prover_preprocessing =
        JoltProverPreprocessing::<Fr, Bn254Curve, DoryCommitmentScheme>::new(shared);
    let verifier_preprocessing = verifier_preprocessing_from_prover(&prover_preprocessing);

    let verifier_bytes = serialize_verifier_object(&verifier_preprocessing)?;

    let target_dir = Path::new("target/wasm32-unknown-unknown/release");
    fs::create_dir_all(target_dir)?;

    let verifier_path = target_dir.join(format!("preprocessed_{func_name}_verifier.bin"));
    let mut file = File::create(verifier_path)?;
    file.write_all(&verifier_bytes)?;

    let elf_bytes = host_program
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
    functions_from_guest_source(&content)
}

fn functions_from_guest_source(content: &str) -> Vec<FunctionAttributes> {
    let syntax: syn::File =
        syn::parse_file(content).unwrap_or_else(|_| panic!("Unable to parse guest source"));

    syntax
        .items
        .iter()
        .filter_map(|item| {
            let syn::Item::Fn(ItemFn { attrs, sig, .. }) = item else {
                return None;
            };
            let provable_attr = attrs.iter().find(|attr| is_provable(attr))?;
            let attributes = attributes_from_provable(provable_attr)?;
            Some(FunctionAttributes {
                func_name: sig.ident.to_string(),
                attributes,
                has_trusted_advice: has_trusted_advice_arg(sig),
            })
        })
        .collect()
}

fn attributes_from_provable(attr: &Attribute) -> Option<Attributes> {
    match &attr.meta {
        Meta::List(meta_list) => {
            let parsed: Punctuated<Meta, Token![,]> = meta_list
                .parse_args_with(Punctuated::parse_terminated)
                .expect("Unable to parse attribute args");
            Some(parse_attributes(&parsed))
        }
        Meta::Path(_) => Some(parse_attributes(&Punctuated::new())),
        _ => None,
    }
}

fn has_trusted_advice_arg(sig: &syn::Signature) -> bool {
    sig.inputs.iter().any(|arg| match arg {
        syn::FnArg::Typed(pat_type) => is_trusted_advice_type(&pat_type.ty),
        syn::FnArg::Receiver(_) => false,
    })
}

fn is_trusted_advice_type(ty: &syn::Type) -> bool {
    if let syn::Type::Path(type_path) = ty {
        return type_path
            .path
            .segments
            .last()
            .is_some_and(|segment| segment.ident == "TrustedAdvice");
    }
    false
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

fn crate_dependency_has_feature(doc: &DocumentMut, crate_name: &str, feature: &str) -> bool {
    let Some(dependencies) = doc.get("dependencies") else {
        return false;
    };
    let Some(dep) = dependencies.get(crate_name) else {
        return false;
    };
    let features = if let Some(table) = dep.as_inline_table() {
        table.get("features").and_then(|value| value.as_array())
    } else if let Some(table) = dep.as_table() {
        table.get("features").and_then(|item| item.as_array())
    } else {
        None
    };
    features.is_some_and(|features| features.iter().any(|value| value.as_str() == Some(feature)))
}

fn project_enables_zk() -> bool {
    fs::read_to_string("Cargo.toml")
        .ok()
        .and_then(|content| content.parse::<DocumentMut>().ok())
        .is_some_and(|doc| crate_dependency_has_feature(&doc, "jolt-sdk", "zk"))
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

fn create_index_html(functions: &[FunctionAttributes]) -> Result<()> {
    let project_name = get_project_name().unwrap();
    let html_content = index_html_source(functions, &project_name);
    let mut file = File::create("index.html")?;
    file.write_all(html_content.as_bytes())?;
    Ok(())
}

fn index_html_source(functions: &[FunctionAttributes], project_name: &str) -> String {
    let func_names_with_verify_prefix: Vec<String> = functions
        .iter()
        .map(|function| format!("verify_{}", function.func_name))
        .collect();

    let mut html_content = String::from(HTML_HEAD);

    for function in functions {
        let func_name = &function.func_name;
        let trusted_advice_input = if function.has_trusted_advice {
            format!(
                r#"
        <input type="file" id="trustedAdviceFile_{func_name}" />"#
            )
        } else {
            String::new()
        };
        html_content.push_str(&format!(
            r#"
    <div style="margin-bottom: 10px;">
        <label>{func_name}</label><br/>
        <input type="file" id="proofFile_{func_name}" />
        <input type="file" id="ioFile_{func_name}" />{trusted_advice_input}
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

    for function in functions {
        let func_name = &function.func_name;
        if function.has_trusted_advice {
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
        } else {
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
    }

    html_content.push_str(HTML_TAIL);
    html_content
}

fn generate_wasm_verify_rs(functions: &[FunctionAttributes]) -> Result<()> {
    let src_dir = Path::new("src");
    fs::create_dir_all(src_dir)?;

    let path = src_dir.join("wasm_verify.rs");
    let mut file = File::create(path)?;
    file.write_all(wasm_verify_source(functions).as_bytes())?;
    Ok(())
}

fn wasm_verify_source(functions: &[FunctionAttributes]) -> String {
    let mut code = String::from(
        r#"use wasm_bindgen::prelude::*;
use jolt_sdk::{
    deserialize_verifier_object, JoltDevice, JoltVerifierPreprocessing, RV64IMACProof,
};
"#,
    );

    for function in functions {
        let func_name = &function.func_name;
        if function.has_trusted_advice {
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
        } else {
            code.push_str(&format!(
                r#"
#[wasm_bindgen]
pub fn verify_{func_name}(preprocessing_data: &[u8], proof_data: &[u8], io_data: &[u8]) -> bool {{
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
    jolt_sdk::jolt_verifier::verify::<
        jolt_sdk::VerifierField,
        jolt_sdk::VerifierPCS,
        jolt_sdk::VerifierVC,
        jolt_sdk::VerifierTranscript,
    >(&preprocessing, &program_io, &proof, None).is_ok()
}}
"#
            ));
        }
    }

    code
}

pub fn modify_cargo_toml(name: &str, set_lib_path: bool) -> Result<()> {
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
        if set_lib_path {
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
        }
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
    let is_std = is_std().expect("Failed to check if std feature is enabled");
    let project_zk = project_enables_zk();
    let cli_zk = cfg!(feature = "zk");
    if project_zk != cli_zk {
        eprintln!(
            "warning: skipping CLI preprocessing because this jolt binary was built with zk={cli_zk} \
             but the project enables zk={project_zk}. Serialize verifier preprocessing from the project's native prover instead."
        );
    } else {
        for function in &functions {
            preprocess_and_save(&function.func_name, &function.attributes, is_std)
                .expect("Failed to preprocess functions");
        }
    }

    generate_wasm_verify_rs(&functions).expect("Failed to generate wasm_verify.rs");
    create_index_html(&functions).expect("Failed to create index.html");

    modify_cargo_toml(".", true).expect("Failed to update Cargo.toml for WASM build");

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

#[cfg(test)]
mod tests {
    use super::*;
    use common::constants::{DEFAULT_MAX_TRUSTED_ADVICE_SIZE, DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE};

    fn host_cargo_toml(features: &str) -> String {
        format!(
            r#"[package]
name = "sample-project"
version = "0.1.0"
edition = "2021"

[dependencies]
jolt-sdk = {{ git = "https://github.com/a16z/jolt", features = [{features}] }}
guest = {{ path = "./guest" }}
"#
        )
    }

    const GUEST_CARGO_TOML: &str = r#"[package]
name = "guest"
version = "0.1.0"
edition = "2021"

[dependencies]
jolt = { package = "jolt-sdk", git = "https://github.com/a16z/jolt" }
"#;

    fn function(name: &str, has_trusted_advice: bool) -> FunctionAttributes {
        FunctionAttributes {
            func_name: name.to_string(),
            attributes: parse_attributes(&Punctuated::new()),
            has_trusted_advice,
        }
    }

    #[test]
    fn wasm_verifier_keeps_three_arg_signature_without_trusted_advice() {
        let generated = wasm_verify_source(&[function("fib", false)]);
        syn::parse_file(&generated).expect("generated WASM verifier must parse");
        assert!(generated.contains("pub fn verify_fib(preprocessing_data: &[u8], proof_data: &[u8], io_data: &[u8]) -> bool"));
        assert!(!generated.contains("trusted_advice_commitment_bytes"));
        assert!(generated.contains("&proof, None"));
    }

    #[test]
    fn wasm_verifier_deserializes_trusted_advice_commitment() {
        let generated = wasm_verify_source(&[function("merkle_tree", true)]);
        syn::parse_file(&generated).expect("generated WASM verifier must parse");
        assert!(generated.contains("trusted_advice_commitment_bytes: &[u8]"));
        assert!(generated.contains("Option<jolt_sdk::VerifierTrustedAdviceCommitment>"));
        assert!(generated.contains("trusted_advice_commitment.as_ref()"));
        assert!(generated.contains("if trusted_advice_commitment_bytes.is_empty()"));
    }

    #[test]
    fn guest_source_detects_trusted_advice_and_bare_provable() {
        let functions = functions_from_guest_source(
            r#"
            #[jolt::provable]
            fn fib(n: u32) -> u128 { n as u128 }

            #[jolt::provable(max_trusted_advice_size = 12345)]
            fn merkle(leaf: jolt::TrustedAdvice<[u8; 32]>) -> bool { true }
            "#,
        );
        assert_eq!(functions.len(), 2);
        assert_eq!(functions[0].func_name, "fib");
        assert!(!functions[0].has_trusted_advice);
        assert_eq!(functions[1].func_name, "merkle");
        assert!(functions[1].has_trusted_advice);
        assert_eq!(functions[1].attributes.max_trusted_advice_size, 12345);
    }

    #[test]
    fn memory_config_forwards_advice_sizes() {
        let attributes = parse_attributes(&syn::parse_quote!(
            max_trusted_advice_size = 12345,
            max_untrusted_advice_size = 23456
        ));
        let config = memory_config_from_attributes(&attributes);
        assert_eq!(config.max_trusted_advice_size, 12345);
        assert_eq!(config.max_untrusted_advice_size, 23456);
        assert_ne!(
            config.max_trusted_advice_size,
            DEFAULT_MAX_TRUSTED_ADVICE_SIZE
        );
        assert_ne!(
            config.max_untrusted_advice_size,
            DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE
        );
    }

    #[test]
    fn index_html_passes_trusted_advice_bytes() {
        let html = index_html_source(&[function("merkle_tree", true)], "sample_project");
        assert!(html.contains("trustedAdviceFile_merkle_tree"));
        assert!(html.contains("verify_merkle_tree(ppData, proofData, ioData, trustedData)"));
    }

    #[test]
    fn index_html_keeps_three_arg_call_without_trusted_advice() {
        let html = index_html_source(&[function("fib", false)], "sample_project");
        assert!(!html.contains("trustedAdviceFile_fib"));
        assert!(html.contains("verify_fib(ppData, proofData, ioData);"));
        assert!(!html.contains("trustedData"));
    }

    #[test]
    fn project_zk_feature_is_detected_from_host_manifest() {
        let zk_doc: DocumentMut = host_cargo_toml(r#""host", "zk""#).parse().unwrap();
        let clear_doc: DocumentMut = host_cargo_toml(r#""host""#).parse().unwrap();
        assert!(crate_dependency_has_feature(&zk_doc, "jolt-sdk", "zk"));
        assert!(!crate_dependency_has_feature(&clear_doc, "jolt-sdk", "zk"));
    }

    #[test]
    fn new_wasm_does_not_point_lib_at_missing_source() {
        let dir = std::env::temp_dir().join(format!(
            "jolt-build-wasm-new-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(dir.join("guest")).unwrap();
        fs::write(dir.join("Cargo.toml"), host_cargo_toml(r#""host""#)).unwrap();
        fs::write(dir.join("guest/Cargo.toml"), GUEST_CARGO_TOML).unwrap();

        modify_cargo_toml(dir.to_str().unwrap(), false).unwrap();

        let host = fs::read_to_string(dir.join("Cargo.toml")).unwrap();
        assert!(host.contains("wasm-bindgen"));
        assert!(!host.contains("wasm_verify.rs"));
        assert!(!host.contains("[lib]"));

        modify_cargo_toml(dir.to_str().unwrap(), true).unwrap();
        let host = fs::read_to_string(dir.join("Cargo.toml")).unwrap();
        assert!(host.contains("src/wasm_verify.rs"));

        fs::remove_dir_all(&dir).unwrap();
    }
}
