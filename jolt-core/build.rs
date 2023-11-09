use std::process::Command;
use std::env;
use std::path::PathBuf;

fn main() {
    build_circom();
}

fn build_circom() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("could not find CARGO_MANIFEST_DIR"); // Absolute path to jolt-core
    let script_path = PathBuf::from(&manifest_dir).join("src/r1cs/scripts/compile_jolt.sh");
    let circom_path = PathBuf::from(&manifest_dir).join("src/r1cs/circuits/jolt.circom");
    let target_dir = env::var("OUT_DIR").expect("could not find OUT_DIR");
    let out_dir = PathBuf::from(&target_dir).join("circom");

    // Store in environment variable which binary can read
    let out_dir_str = out_dir.to_str().expect("failed to convert path to string");
    println!("cargo:rustc-env=CIRCUIT_DIR={out_dir_str}");

    let status = Command::new(script_path)
        .arg(&circom_path)
        .arg(&out_dir)
        .status()
        .expect("failed to build circom");
    assert!(status.success());
}