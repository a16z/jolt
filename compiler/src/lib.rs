use std::fs::File;
use std::io::{Read, Write};
use std::io::BufReader;
use std::io::BufWriter;
use std::path::PathBuf;

use common::constants::RAM_START_ADDRESS;
use common::path::JoltPaths;
use tracer::run_tracer_with_paths;

pub fn cached_compile_example(example_name: &str) {
    let elf_location = JoltPaths::elf_path(example_name);
    let trace_destination = JoltPaths::trace_path(example_name);
    let bytecode_destination = JoltPaths::bytecode_path(example_name);
    let witness_gen_location = JoltPaths::witness_generator_path(example_name);
    let r1cs_location = JoltPaths::r1cs_path(example_name);
    let circom_location = JoltPaths::compiled_circuit_path(example_name);

    if elf_location.exists() && trace_destination.exists() && bytecode_destination.exists() &&
       witness_gen_location.exists() && r1cs_location.exists() && circom_location.exists() {
        println!("{example_name} circuit cached");
    } else {
        compile_example(example_name);
    }
}

pub fn compile_example(example_name: &str) {
    // Cargo Build
    let cargo_build_status = std::process::Command::new("cargo")
        .args(&["build", "-p", example_name, "--release"])
        .output()
        .expect("Failed to execute command");
    if !cargo_build_status.status.success() {
        println!("Failed to build example: {}. Error: {}", example_name, String::from_utf8_lossy(&cargo_build_status.stderr));
        std::process::exit(1);
    }
    println!("Successfully built example: {}", example_name);

    // Trace
    let elf_location = JoltPaths::elf_path(example_name);
    let trace_destination = JoltPaths::trace_path(example_name);
    let bytecode_destination = JoltPaths::bytecode_path(example_name);
    let num_trace_rows = match run_tracer_with_paths(elf_location, trace_destination, bytecode_destination) {
        Ok((num_trace_rows, num_bytecode_rows)) => {
            println!("Successfully ran tracer on example: {}", example_name);
            println!("\t - Number of trace rows: {}", num_trace_rows);
            println!("\t - Number of bytecode rows: {}", num_bytecode_rows);
            num_trace_rows
        },
        Err(e) => {
            println!("Failed to run tracer on example: {}. Error: {}", example_name, e);
            std::process::exit(1);
        }
    };

    // Template
    // let circuit_template_location = JoltPaths::circuit_template_path();
    // let circuit_destination = JoltPaths::compiled_circuit_path(example_name);
    // execute_template(&circuit_template_location, &circuit_destination, num_trace_rows, RAM_START_ADDRESS as usize);

    // Circom build
    // let build_script_path = JoltPaths::circom_build_script_path();
    // let circuit_artifacts_destination = JoltPaths::circuit_artifacts_path();
    // let circom_build_status = std::process::Command::new(build_script_path)
    //     .arg(&circuit_destination)
    //     .arg(&circuit_artifacts_destination)
    //     .output()
    //     .expect("Failed to build circom");
    // if !circom_build_status.status.success() {
    //     println!("Failed to build circom: {}", example_name);
    //     std::process::exit(1);
    // }

    // println!("Successfully built circom {} -> {}", circuit_destination.display(), circuit_artifacts_destination.display())

}

pub fn execute_template(circuit_template_location: &PathBuf, circuit_destination: &PathBuf, num_steps: usize, prog_start: usize) {
    // Load the template
    let mut file = BufReader::new(File::open(circuit_template_location).expect("Could not open circuit template"));
    let mut contents = String::new();
    file.read_to_string(&mut contents).expect("Could not read circuit template");

    // Check and replace placeholders
    if contents.contains("<NUM_STEPS>") && contents.contains("<PROG_START_ADDR>") {
        contents = contents.replace("<NUM_STEPS>", &num_steps.to_string());
        contents = contents.replace("<PROG_START_ADDR>", &prog_start.to_string());
    } else {
        panic!("Could not find placeholders in the template");
    }

    // Write to destination
    let path = circuit_destination;
    std::fs::create_dir_all(path.parent().unwrap()).expect("Failed to create directories");
    let mut file = BufWriter::new(File::create(&path).expect("Could not create circuit destination file"));
    file.write_all(contents.as_bytes()).expect("Could not write to circuit destination file");
}
