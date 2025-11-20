use std::env;
use std::path::PathBuf;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let package_name = env::var("CARGO_PKG_NAME").unwrap();
    let output_file = target_dir()
        .join(format!("{}.h", package_name))
        .display()
        .to_string();

    match cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_language(cbindgen::Language::C)
        .with_include_guard("JOLT_FFI_H")
        .with_cpp_compat(true)
        .with_documentation(true)
        .with_parse_deps(false)
        .generate()
    {
        Ok(bindings) => {
            bindings.write_to_file(&output_file);
            println!("cargo:warning=Generated C header at: {}", output_file);
        }
        Err(e) => {
            println!("cargo:warning=Unable to generate bindings: {:?}", e);
            println!("cargo:warning=Continuing without header generation");
        }
    }

    println!("cargo:rerun-if-changed=src/lib.rs");
}

fn target_dir() -> PathBuf {
    let out_dir = env::var("OUT_DIR").unwrap();
    let mut target_dir = PathBuf::from(out_dir);

    // Navigate up from OUT_DIR to the target directory
    // OUT_DIR is typically target/<profile>/build/<crate>/out
    for _ in 0..3 {
        target_dir.pop();
    }

    target_dir
}
