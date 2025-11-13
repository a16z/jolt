use clap::Parser;
use eyre::Result;
use jolt_core::host;
use jolt_core::zkvm::prover::JoltProverPreprocessing;
use jolt_core::zkvm::Serializable;
use std::fs;
use std::path::PathBuf;

use ark_bn254::Fr;
use jolt_core::poly::commitment::dory::DoryCommitmentScheme;

type RV64IMACPreprocessing = JoltProverPreprocessing<Fr, DoryCommitmentScheme>;

/// Utility to prepare a Jolt guest program for use with the C FFI.
/// This builds the guest program, extracts the ELF, and generates preprocessing.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Name of the guest program to build (e.g., "fibonacci-guest")
    #[arg(short, long)]
    guest: String,

    /// Output path for the ELF file
    #[arg(short, long, default_value = "guest.elf")]
    elf_output: PathBuf,

    /// Output path for the preprocessing file
    #[arg(short, long, default_value = "preprocessing.bin")]
    preprocessing_output: PathBuf,

    /// Maximum trace length (power of 2)
    #[arg(short, long, default_value = "65536")]
    max_trace_length: usize,

    /// Target directory for cargo build
    #[arg(short, long, default_value = "target")]
    target_dir: String,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    println!("Jolt Guest Preparation Utility");
    println!("===============================\n");

    // Step 1: Build the guest program
    println!("Step 1: Building guest program '{}'...", args.guest);
    let mut program = host::Program::new(&args.guest);
    program.build(&args.target_dir);
    println!("  ✓ Guest program built successfully\n");

    // Step 2: Extract and save ELF
    println!("Step 2: Extracting ELF binary...");
    let elf_contents = program
        .get_elf_contents()
        .ok_or_else(|| eyre::eyre!("Failed to get ELF contents"))?;

    fs::write(&args.elf_output, &elf_contents)?;
    println!("  ✓ Saved ELF ({} bytes) to: {}",
        elf_contents.len(),
        args.elf_output.display()
    );
    println!();

    // Step 3: Generate preprocessing
    println!("Step 3: Generating preprocessing (this may take a while)...");

    // Decode the program to get bytecode and memory initialization
    let (bytecode, init_memory_state, _) = program.decode();

    // Trace once to get the memory layout
    let (_, _, _, io_device) = program.trace(&[], &[], &[]);

    println!("  - Bytecode size: {} instructions", bytecode.len());
    println!("  - Memory layout: {} bytes", io_device.memory_layout.memory_size);
    println!("  - Max trace length: {}", args.max_trace_length);

    let preprocessing = RV64IMACPreprocessing::gen(
        bytecode,
        io_device.memory_layout,
        init_memory_state,
        args.max_trace_length,
    );

    // Save preprocessing to file
    preprocessing.save_to_file(&args.preprocessing_output)?;

    let preprocessing_size = fs::metadata(&args.preprocessing_output)?.len();
    println!("  ✓ Saved preprocessing ({} bytes) to: {}",
        preprocessing_size,
        args.preprocessing_output.display()
    );
    println!();

    println!("Done! Generated files:");
    println!("  ELF:           {}", args.elf_output.display());
    println!("  Preprocessing: {}", args.preprocessing_output.display());
    println!();
    println!("You can now use these files with the C FFI:");
    println!("  ./jolt_example {} proof.bin {}",
        args.elf_output.display(),
        args.preprocessing_output.display()
    );

    Ok(())
}
