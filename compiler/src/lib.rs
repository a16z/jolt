use common::path::JoltPaths;
use tracer::run_tracer_with_paths;

pub fn cached_compile_example(example_name: &str) {
    let elf_location = JoltPaths::elf_path(example_name);
    let trace_destination = JoltPaths::trace_path(example_name);
    let bytecode_destination = JoltPaths::bytecode_path(example_name);

    // if elf_location.exists() && trace_destination.exists() && bytecode_destination.exists() {
    //     println!("{example_name} circuit cached");
    // } else {
        compile_example(example_name);
    // }
}

pub fn compile_example(example_name: &str) {
    // Cargo Build
    let cargo_build_status = std::process::Command::new("cargo")
        .args(&["build", "--profile", "guest", "-p", example_name])
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
    match run_tracer_with_paths(elf_location, trace_destination, bytecode_destination) {
        Ok((num_trace_rows, num_bytecode_rows)) => {
            println!("Successfully ran tracer on example: {}", example_name);
            println!("\t - Number of trace rows: {}", num_trace_rows);
            println!("\t - Number of bytecode rows: {}", num_bytecode_rows);
        },
        Err(e) => {
            println!("Failed to run tracer on example: {}. Error: {}", example_name, e);
            std::process::exit(1);
        }
    };
}