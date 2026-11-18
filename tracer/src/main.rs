use std::fs::File;
use std::path::Path;
use std::process::exit;

use clap::Parser;
use common::constants::RAM_START_ADDRESS;
use common::jolt_device::{JoltDevice, MemoryConfig};

use tracer::emulator::{
    default_terminal::DefaultTerminal, elf_analyzer::ElfAnalyzer, tohost_address, Emulator,
    GUEST_PROFILE,
};

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

    /// Print disassembly of each instruction as it executes
    #[arg(short, long)]
    disassemble: bool,
}

#[expect(
    clippy::expect_used,
    reason = "the CLI reports an unreadable or undecodable ELF, or a signature file it cannot create, by panicking"
)]
fn main() {
    let args = Args::parse();

    tracing_subscriber::fmt()
        .compact()
        .with_target(false)
        .with_file(false)
        .with_line_number(false)
        .with_thread_ids(false)
        .with_thread_names(false)
        .init();

    let elf_path = Path::new(&args.elf);
    let elf_content = std::fs::read(elf_path).expect("Failed to read ELF file");

    let mut emulator = if is_riscv_test(&elf_content) {
        let mut emulator = Emulator::new(Box::new(DefaultTerminal::default()));
        emulator.setup_program(&elf_content);
        let memory_config = MemoryConfig {
            program_size: Some(elf_content.len() as u64),
            ..Default::default()
        };
        emulator.get_mut_cpu().get_mut_mmu().jolt_device = Some(JoltDevice::new(&memory_config));
        // Test ELFs (`tohost`) get the code guards of tracing for a proof too, so
        // ACT4 runs under the rules a Jolt guest is proved under.
        emulator.protect_code(&elf_content);
        emulator
    } else {
        // The ELF does not carry the guest's MemoryConfig, so jolt-emu lays
        // out the default configuration over the decoded image and, as
        // tracing for a proof does, attaches the device before loading. A
        // guest built with other I/O, stack or heap sizes therefore runs here
        // under a different layout than when proved.
        let image = jolt_program::image::decode_elf(&elf_content, GUEST_PROFILE)
            .expect("program ELF decoding failed");
        let memory_config = MemoryConfig {
            program_size: Some(image.program_end - RAM_START_ADDRESS),
            ..Default::default()
        };
        tracer::create_emulator(
            &elf_content,
            Some(&elf_path.to_path_buf()),
            &[],
            &[],
            &[],
            &memory_config,
            None,
        )
    };

    let endcode = emulator.run_test(args.trace.unwrap_or(false), args.disassemble);

    if let Some(sig_path) = args.signature {
        let mut file = File::create(sig_path).expect("Failed to create signature file");
        if let Err(e) = emulator.write_signature(&mut file, args.signature_granularity) {
            eprintln!("Failed to write signature file: {e}");
            exit(1);
        }
    }

    // Propagate pass/fail to the shell as the process exit status so the
    // ACT4 runner (tests/arch-tests/run.sh) can detect it reliably. We
    // collapse the HTIF endcode to 0 or 1 — on Unix `exit()` truncates to
    // the low 8 bits, so returning the raw endcode would map any multiple
    // of 256 to kernel exit 0 and silently look like a pass.
    exit(if endcode == 0 { 0 } else { 1 });
}

/// Whether `elf` is a riscv-tests ELF, by the rule of [`tohost_address`].
fn is_riscv_test(elf: &[u8]) -> bool {
    let analyzer = ElfAnalyzer::new(elf);
    if !analyzer.validate() {
        return false;
    }
    let header = analyzer.read_header();
    let section_headers = analyzer.read_section_headers(&header);
    tohost_address(&analyzer.read_symbol_map(&header, &section_headers)).is_some()
}
