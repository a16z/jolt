use common::path::JoltPaths;

fn main() {
    let circuit_path = JoltPaths::circuit_path();
    let build_script_path = JoltPaths::circom_build_script_path();
    let circuit_artifacts_destination = JoltPaths::circuit_artifacts_path();
    let circom_build_status = std::process::Command::new(build_script_path)
        .arg(&circuit_path)
        .arg(&circuit_artifacts_destination)
        .output()
        .expect("Failed to build circom");
    if !circom_build_status.status.success() {
        println!("Failed to build circom: {}", circuit_path.display());
        std::process::exit(1);
    }

    // Currently it is impossible to automate the witness build because of a circular compile dependency.
    // 1. circom-witness-rs/build.rs depends on WITNESS_CPP (actually the .circom path), compiles circom, creates artifacts in working directory
    // 2. circom-witness-rs does a string replace on the cpp file
    // 3. circom-witness-rs compiles cpp using cxx bridge to make a rust binary
    // 4. circom-witness-rs::generate::build_witness() generates a graph.bin file
    // We only need the 4th artifact. Currently this can be created by generating all the other artifacts in the circom-witness-rs working directory
    // 1. Uncomment this line
    // 2. Set jolt-core/cargo.toml build-dependencies:
    //     [build-dependencies]
    //     witness = { git = "https://github.com/philsippl/circom-witness-rs", features = ["build-witness"]}
    // 3. WITNESS_CPP=/Users/sragsdale/Documents/Code/a16z/lasso-cp-2/jolt-core/src/r1cs/circuits/jolt_single_step.circom cargo build -p jolt-core
    // 4. mv jolt-core/graph.bin jolt-core/src/r1cs/graph.bin
    // 5. Commit
    // 6. Remove trash
    // witness::generate::build_witness();

    println!("cargo:rerun-if-changed={}", JoltPaths::circom_build_script_path().display());
    println!("cargo:rerun-if-changed={}", JoltPaths::circuit_path().display());
}