use common::attributes::{parse_attributes, Attributes};

use std::{
    fs::{self, File},
    io::Write,
    path::Path,
};

use eyre::Result;
use jolt_core::host::{ELFInstruction, Program};
use rmp_serde::Serializer;
use serde::{Deserialize, Serialize};
use syn::{Attribute, ItemFn, Meta, PathSegment};
use toml_edit::{value, Array, DocumentMut, InlineTable, Item, Table, Value};

#[derive(Serialize, Deserialize)]
struct DecodedData {
    bytecode: Vec<ELFInstruction>,
    memory_init: Vec<(u64, u8)>,
}

struct FunctionAttributes {
    pub func_name: String,
    pub attributes: Attributes,
}

fn preprocess_and_save(func_name: &str, attributes: &Attributes, is_std: bool) -> Result<()> {
    let mut program = Program::new("guest");

    program.set_func(func_name);
    program.set_std(is_std);
    program.set_memory_size(attributes.memory_size);
    program.set_stack_size(attributes.stack_size);
    program.set_max_input_size(attributes.max_input_size);
    program.set_max_output_size(attributes.max_output_size);

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

fn extract_provable_functions() -> Vec<FunctionAttributes> {
    let content = fs::read_to_string("guest/src/lib.rs").expect("Unable to read file");
    let syntax: syn::File = syn::parse_file(&content).expect("Unable to parse file");

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
    // replace "-" with "_" to make it a valid identifier
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

    let git = dependencies
        .get("git")
        .expect("Failed to get git-path")
        .as_str()
        .expect("Failed to get git-path as string");

    if package == "jolt-sdk" && git == "https://github.com/a16z/jolt" {
        return Some(
            dependencies
                .get("features")
                .and_then(|v| v.as_array())
                .map_or(false, |features| {
                    features.iter().any(|f| f.as_str() == Some("guest-std"))
                }),
        );
    }
    None
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

pub fn modify_cargo_toml(name: &str) -> Result<()> {
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

pub fn build_wasm() {
    println!("Building the project with wasm-pack...");
    let functions = extract_provable_functions();
    let function_names: Vec<String> = functions.iter().map(|f| f.func_name.clone()).collect();
    let is_std = is_std().expect("Failed to check if std feature is enabled");
    for function in functions {
        preprocess_and_save(&function.func_name, &function.attributes, is_std)
            .expect("Failed to preprocess functions");
    }

    create_index_html(function_names).expect("Failed to create example index.html");

    // todo implement test if wasm-pack is installed
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
