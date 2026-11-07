#![cfg(all(feature = "profiling", not(feature = "field-inline")))]
#![expect(clippy::unwrap_used, reason = "test fixture")]

use std::{fs, process::Command};

use jolt_host::Program;
use serde_json::Value;

#[test]
fn budgeted_profile_proves_through_the_limited_route() {
    let root = std::env::temp_dir().join(format!("jolt-profile-budget-{}", std::process::id()));
    fs::create_dir(&root).unwrap();
    let mut program = Program::new("fibonacci-guest");
    program.build(root.join("guest-target").to_str().unwrap());
    let elf = root.join("guest.elf");
    fs::write(&elf, program.elf.unwrap().bytes()).unwrap();
    let input = root.join("input.bin");
    fs::write(&input, postcard::to_stdvec(&100u32).unwrap()).unwrap();
    let metrics = root.join("result.json");
    let output = Command::new(env!("CARGO_BIN_EXE_jolt-prover"))
        .current_dir(&root)
        .env_remove("TRACER_PARALLEL")
        .env_remove("JOLT_TRACER_CAPACITY_ROWS")
        .args([
            "profile",
            "--name",
            "fibonacci",
            "--scale",
            "12",
            "--backend",
            "optimized",
            "--format",
            "none",
            "--memory-budget",
            "64MiB",
            "--guest-elf",
        ])
        .arg(&elf)
        .arg("--input-file")
        .arg(&input)
        .arg("--result-json")
        .arg(&metrics)
        .output()
        .unwrap();
    let result = fs::read(&metrics);
    fs::remove_dir_all(&root).unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: Value = serde_json::from_slice(&result.unwrap()).unwrap();
    let bytes: Vec<u8> = serde_json::from_value(result["public_output"].clone()).unwrap();
    let (answer, _): (u128, _) = postcard::take_from_bytes(&bytes).unwrap();
    assert_eq!(answer, 354_224_848_179_261_915_075);
}
