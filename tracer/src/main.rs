use std::fs::File;
use std::path::Path;
use std::process::exit;

use clap::Parser;

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
}

fn main() {
    let args = Args::parse();

    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Read the ELF file
    let elf_path = Path::new(&args.elf);
    let elf_content = std::fs::read(elf_path).expect("Failed to read ELF file");

    // Create and run the emulator
    let mut emulator = Emulator::new(Box::new(DefaultTerminal::default()));
    emulator.setup_program(elf_content);
    emulator.run();

    // If signature file is specified, write the signature with 4-byte granularity
    if let Some(sig_path) = args.signature {
        let mut file = File::create(sig_path).expect("Failed to create signature file");
        if let Err(e) = emulator.write_signature(&mut file, 4) {
            eprintln!("Failed to write signature file: {e}");
            exit(1);
        }
    }
}
