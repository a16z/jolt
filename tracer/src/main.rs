use std::fs::File;
use std::path::Path;
use std::process::exit;

use clap::Parser;

use common::jolt_device::{JoltDevice, MemoryConfig};
use tracer::emulator::{default_terminal::DefaultTerminal, Emulator};

/// RISC-V emulator for Jolt
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the ELF file to execute
    elf: String,

    /// Path to write the signature file
    #[arg(short, long)]
    signature: Option<String>,

    /// Signature granularity in bytes (must be a power of 2)
    #[arg(long, default_value = "4")]
    signature_granularity: usize,

    /// Execute the program in trace mode
    #[arg(short, long, value_name = "true|false")]
    trace: Option<bool>,

    /// Disassemble and print each instruction as it executes (like spike -d)
    #[arg(short, long)]
    disassemble: bool,
}

fn main() {
    let args = Args::parse();

    // Initialize tracing
    tracing_subscriber::fmt()
        .compact()
        .with_target(false)
        .with_file(false)
        .with_line_number(false)
        .with_thread_ids(false)
        .with_thread_names(false)
        .init();

    // Read the ELF file
    let elf_path = Path::new(&args.elf);
    let elf_content = std::fs::read(elf_path).expect("Failed to read ELF file");

    // Create and run the emulator
    let mut emulator = Emulator::new(Box::new(DefaultTerminal::default()));
    emulator.setup_program(&elf_content);

    // Setup JoltDevice for low memory access
    let mut memory_config = MemoryConfig::default();
    memory_config.program_size = Some(elf_content.len() as u64);
    emulator.get_mut_cpu().get_mut_mmu().jolt_device = Some(JoltDevice::new(&memory_config));

    emulator.run_test(args.trace.unwrap_or(false), args.disassemble);

    // If signature file is specified, write the signature with specified granularity
    if let Some(sig_path) = args.signature {
        let mut file = File::create(sig_path).expect("Failed to create signature file");
        if let Err(e) = emulator.write_signature(&mut file, args.signature_granularity) {
            eprintln!("Failed to write signature file: {e}");
            exit(1);
        }
    }
}
