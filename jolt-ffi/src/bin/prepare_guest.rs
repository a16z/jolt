use clap::Parser;
use eyre::Result;
use jolt_core::host::Program;
use jolt_core::zkvm::Serializable;
use std::fs;
use std::path::PathBuf;

enum GuestProgram {
    Fibonacci,
    Sha3Chain,
    EcdsaSign,
}

/// Utility to prepare a Jolt guest program for use with the C FFI.
/// This builds the guest program, extracts the ELF, and generates preprocessing.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Name of the guest program to build (e.g., "fibonacci", "sha3", "ecdsa")
    #[arg(short, long)]
    guest: String,

    /// Output path for the ELF file
    #[arg(short, long, default_value = "guest.elf")]
    elf_output: PathBuf,

    /// Output path for the preprocessing file
    #[arg(short, long, default_value = "preprocessing.bin")]
    preprocessing_output: PathBuf,

    /// Target directory for cargo build
    #[arg(short, long, default_value = "target")]
    target_dir: String,

    /// Quiet mode: minimal output
    #[arg(short, long)]
    quiet: bool,
}

fn extract_elf(program: &Program, elf_output: &PathBuf, quiet: bool) -> Result<()> {
    let elf_contents = program
        .get_elf_contents()
        .ok_or_else(|| eyre::eyre!("Failed to get ELF contents"))?;

    fs::write(elf_output, &elf_contents)?;
    if !quiet {
        println!(
            "  ✓ Saved ELF ({} bytes) to: {}",
            elf_contents.len(),
            elf_output.display()
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    // Initialize inline handlers (keccak256/SHA3, SHA2, etc.)
    let _ = jolt_inlines_keccak256::init_inlines();
    let _ = jolt_inlines_sha2::init_inlines();

    let args = Args::parse();

    if !args.quiet {
        println!("Jolt Guest Preparation Utility");
        println!("===============================\n");
    }

    // Determine which guest program to use
    let guest_program = if args.guest.contains("fib") {
        GuestProgram::Fibonacci
    } else if args.guest.contains("sha3") {
        GuestProgram::Sha3Chain
    } else if args.guest.contains("sign") || args.guest.contains("ecdsa") {
        GuestProgram::EcdsaSign
    } else {
        return Err(eyre::eyre!(
            "Unknown guest program: {}. Supported: fibonacci, sha3-chain, ecdsa-sign",
            args.guest
        ));
    };

    if !args.quiet {
        println!("Step 1: Building guest program '{}'...", args.guest);
    }

    // Determine target directory
    let target_dir = if args.target_dir == "target" {
        "/tmp/jolt-guest-targets"
    } else {
        &args.target_dir
    };

    // Compile guest program using macro-generated functions
    let preprocessing = match guest_program {
        GuestProgram::Fibonacci => {
            let mut program = fibonacci_guest::compile_fib(target_dir);
            if !args.quiet {
                println!("  ✓ Guest program built successfully\n");
                println!("Step 2: Extracting ELF binary...");
            }
            extract_elf(&program, &args.elf_output, args.quiet)?;
            if !args.quiet {
                println!();
                println!("Step 3: Generating preprocessing (this may take a while)...");
            }
            fibonacci_guest::preprocess_prover_fib(&mut program)
        }
        GuestProgram::Sha3Chain => {
            let mut program = sha3_chain_guest::compile_sha3_chain(target_dir);
            if !args.quiet {
                println!("  ✓ Guest program built successfully\n");
                println!("Step 2: Extracting ELF binary...");
            }
            extract_elf(&program, &args.elf_output, args.quiet)?;
            if !args.quiet {
                println!();
                println!("Step 3: Generating preprocessing (this may take a while)...");
            }
            sha3_chain_guest::preprocess_prover_sha3_chain(&mut program)
        }
        GuestProgram::EcdsaSign => {
            let mut program = ecdsa_sign_guest::compile_ecdsa_sign(target_dir);
            if !args.quiet {
                println!("  ✓ Guest program built successfully\n");
                println!("Step 2: Extracting ELF binary...");
            }
            extract_elf(&program, &args.elf_output, args.quiet)?;
            if !args.quiet {
                println!();
                println!("Step 3: Generating preprocessing (this may take a while)...");
            }
            ecdsa_sign_guest::preprocess_prover_ecdsa_sign(&mut program)
        }
    };

    // Save preprocessing to file
    preprocessing.save_to_file(&args.preprocessing_output)?;

    if args.quiet {
        // Quiet mode: just confirm success
        println!(
            "✓ Generated {} ({} bytes) and {} ({} bytes)",
            args.elf_output.file_name().unwrap().to_str().unwrap(),
            fs::metadata(&args.elf_output)?.len(),
            args.preprocessing_output
                .file_name()
                .unwrap()
                .to_str()
                .unwrap(),
            fs::metadata(&args.preprocessing_output)?.len()
        );
    } else {
        // Verbose mode: show all details
        let preprocessing_size = fs::metadata(&args.preprocessing_output)?.len();
        println!(
            "  ✓ Saved preprocessing ({} bytes) to: {}",
            preprocessing_size,
            args.preprocessing_output.display()
        );
        println!();

        println!("Done! Generated files:");
        println!("  ELF:           {}", args.elf_output.display());
        println!("  Preprocessing: {}", args.preprocessing_output.display());
        println!();
        println!("You can now use these files with the C FFI:");
        println!(
            "  ./jolt_example {} proof.bin {}",
            args.elf_output.display(),
            args.preprocessing_output.display()
        );
    }

    Ok(())
}
