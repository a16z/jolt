use clap::Parser;
use common::{path::JoltPaths, constants::RAM_START_ADDRESS};
use tracer::run_tracer_with_paths;
use compiler::execute_template;

#[derive(Parser)]
struct Opts {
    #[clap(short, long)]
    example: String,
}

fn main() {
    let opts: Opts = Opts::parse();
    println!("Example: {}", opts.example);

    // Cargo Build
    let cargo_build_status = std::process::Command::new("cargo")
        .args(&["build", "-p", &opts.example, "--release"])
        .output()
        .expect("Failed to execute command");
    if !cargo_build_status.status.success() {
        println!("Failed to build example: {}. Error: {}", opts.example, String::from_utf8_lossy(&cargo_build_status.stderr));
        std::process::exit(1);
    }
    println!("Successfully built example: {}", opts.example);

    // Trace
    let elf_location = JoltPaths::elf_path(&opts.example);
    let trace_destination = JoltPaths::trace_path(&opts.example);
    let bytecode_destination = JoltPaths::bytecode_path(&opts.example);
    let num_trace_rows = match run_tracer_with_paths(elf_location, trace_destination, bytecode_destination) {
        Ok((num_trace_rows, num_bytecode_rows)) => {
            println!("Successfully ran tracer on example: {}", opts.example);
            println!("\t - Number of trace rows: {}", num_trace_rows);
            println!("\t - Number of bytecode rows: {}", num_bytecode_rows);
            num_trace_rows
        },
        Err(e) => {
            println!("Failed to run tracer on example: {}. Error: {}", opts.example, e);
            std::process::exit(1);
        }
    };

    // Template
    let circuit_template_location = JoltPaths::circuit_template_path();
    let circuit_destination = JoltPaths::compiled_circuit_path(&opts.example);
    execute_template(&circuit_template_location, &circuit_destination, num_trace_rows, RAM_START_ADDRESS as usize);

    // Circom build
    let build_script_path = JoltPaths::circom_build_script_path();
    let circuit_artifacts_destination = JoltPaths::circuit_artifacts_path();
    let circom_build_status = std::process::Command::new(build_script_path)
        .arg(&circuit_destination)
        .arg(&circuit_artifacts_destination)
        .output()
        .expect("Failed to build circom");
    if !circom_build_status.status.success() {
        println!("Failed to build circom: {}", opts.example);
        std::process::exit(1);
    }

    println!("Successfully built circom {} -> {}", circuit_destination.display(), circuit_artifacts_destination.display())
}