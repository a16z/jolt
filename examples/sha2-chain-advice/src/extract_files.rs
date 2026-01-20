use jolt_sdk::postcard;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

/// Script to extract ELF, JSON config, and binary input files for the sha2-chain example.
/// This produces the files needed by the tracer/jolt-cpp:
/// - program/program.elf  
/// - program/program.json
/// - inputs/input.bin
/// - inputs/trusted_advice.bin  
/// - inputs/untrusted_advice.bin
fn main() {
    // Create output directories
    let output_base = PathBuf::from("./sha2-chain-export");
    let program_dir = output_base.join("program");
    let inputs_dir = output_base.join("inputs");
    
    fs::create_dir_all(&program_dir).expect("Failed to create program directory");
    fs::create_dir_all(&inputs_dir).expect("Failed to create inputs directory");

    // Compile the guest program
    let target_dir = "/tmp/jolt-guest-targets";
    let program = guest::compile_sha2_chain(target_dir);
    
    // Get the ELF path and copy it
    let elf_path = program.elf.as_ref().expect("ELF not compiled");
    let elf_dest = program_dir.join("program.elf");
    fs::copy(elf_path, &elf_dest).expect("Failed to copy ELF file");
    println!("Copied ELF to: {}", elf_dest.display());

    // Get memory config and create JSON
    let memory_config = guest::memory_config_sha2_chain();
    // max_trace_length must match the value in guest/src/lib.rs #[jolt::provable] attribute
    let max_trace_length: u64 = 16777216; // 2^24
    let compressed_chunk_size: u64 = 512;
    let json_config = serde_json::json!({
        "compressed_chunk_size": compressed_chunk_size,
        "max_trace_length": max_trace_length,
        "mem_conf": {
            "max_untrusted_advice_size": memory_config.max_untrusted_advice_size,
            "max_trusted_advice_size": memory_config.max_trusted_advice_size,
            "max_input_size": memory_config.max_input_size,
            "max_output_size": memory_config.max_output_size,
            "stack_size": memory_config.stack_size,
            "memory_size": memory_config.memory_size
        }
    });
    
    let json_dest = program_dir.join("program.json");
    let mut json_file = fs::File::create(&json_dest).expect("Failed to create JSON file");
    json_file
        .write_all(serde_json::to_string_pretty(&json_config).unwrap().as_bytes())
        .expect("Failed to write JSON file");
    println!("Created config JSON at: {}", json_dest.display());

    // Create input binary files using postcard serialization (same as Jolt SDK)
    // These match the inputs from main.rs
    let input = [5u8; 32];
    let iters = 1000u32;

    // Input: serialized tuple of (input, iters)
    let input_tuple = (input, iters);
    let input_bytes = postcard::to_stdvec(&input_tuple).expect("Failed to serialize input");
    let input_dest = inputs_dir.join("input.bin");
    fs::write(&input_dest, &input_bytes).expect("Failed to write input.bin");
    println!("Created input.bin ({} bytes) at: {}", input_bytes.len(), input_dest.display());

    // Trusted advice: 16384 bytes (2048*8)
    let trusted_data = [7u8; 16384];
    let trusted_advice_bytes =
        postcard::to_stdvec(&jolt_sdk::TrustedAdvice::new(trusted_data.as_slice()))
            .expect("Failed to serialize trusted_data");
    let trusted_dest = inputs_dir.join("trusted_advice.bin");
    fs::write(&trusted_dest, &trusted_advice_bytes).expect("Failed to write trusted_advice.bin");
    println!("Created trusted_advice.bin ({} bytes) at: {}", trusted_advice_bytes.len(), trusted_dest.display());

    // Untrusted advice: 8192 bytes (1024*8)
    let untrusted_data = [9u8; 8192];
    let untrusted_advice_bytes =
        postcard::to_stdvec(&jolt_sdk::UntrustedAdvice::new(untrusted_data.as_slice()))
            .expect("Failed to serialize untrusted_data");
    let untrusted_dest = inputs_dir.join("untrusted_advice.bin");
    fs::write(&untrusted_dest, &untrusted_advice_bytes).expect("Failed to write untrusted_advice.bin");
    println!("Created untrusted_advice.bin ({} bytes) at: {}", untrusted_advice_bytes.len(), untrusted_dest.display());

    println!("\n✅ Export complete!");
    println!("Program files in: {}", program_dir.display());
    println!("Input files in: {}", inputs_dir.display());
}
