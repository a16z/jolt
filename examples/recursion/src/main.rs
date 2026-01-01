use ark_serialize::CanonicalDeserialize;
use ark_serialize::CanonicalSerialize;
use clap::{Parser, Subcommand};
use jolt_sdk::{JoltDevice, MemoryConfig, RV64IMACProof, Serializable};
use std::cmp::PartialEq;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tracing::{error, info};

fn get_guest_src_dir() -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let guest_src_dir = manifest_dir.join("guest").join("src");

    guest_src_dir.canonicalize().unwrap_or(guest_src_dir)
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate proofs for guest programs
    Generate {
        /// Example to run (fibonacci or muldiv)
        #[arg(long, value_name = "EXAMPLE")]
        example: String,
        /// Working directory for output files
        #[arg(long, value_name = "DIRECTORY", default_value = "output")]
        workdir: PathBuf,
    },
    /// Verify proofs and optionally embed them
    Verify {
        /// Example to verify (fibonacci or muldiv)
        #[arg(long, value_name = "EXAMPLE")]
        example: String,
        /// Working directory containing proof files
        #[arg(long, value_name = "DIRECTORY", default_value = "output")]
        workdir: PathBuf,
        /// Embed proof data to specified directory
        #[arg(long, value_name = "DIRECTORY", num_args = 0..=1)]
        embed: Option<Option<PathBuf>>,
    },
    /// Trace the execution of guest programs without attempting to prove them
    Trace {
        /// Example to trace (fibonacci or muldiv)
        #[arg(long, value_name = "EXAMPLE")]
        example: String,
        /// Working directory containing proof files
        #[arg(long, value_name = "DIRECTORY", default_value = "output")]
        workdir: PathBuf,
        /// Embed proof data to specified directory
        #[arg(long, value_name = "DIRECTORY", num_args = 0..=1)]
        embed: Option<Option<PathBuf>>,
        /// Trace to disk instead of memory (redues memory usage)
        #[arg(short = 'd', long = "disk", default_value_t = false)]
        trace_to_file: bool,
    },
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum GuestProgram {
    Fibonacci,
    Muldiv,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RunConfig {
    Prove,
    Trace,
    TraceToFile,
}

impl GuestProgram {
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "fibonacci" => Some(GuestProgram::Fibonacci),
            "muldiv" => Some(GuestProgram::Muldiv),
            _ => None,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            GuestProgram::Fibonacci => "fibonacci-guest",
            GuestProgram::Muldiv => "muldiv-guest",
        }
    }

    fn func(&self) -> &'static str {
        match self {
            GuestProgram::Fibonacci => "fib",
            GuestProgram::Muldiv => "muldiv",
        }
    }

    fn inputs(&self) -> Vec<Vec<u8>> {
        match self {
            GuestProgram::Fibonacci => {
                vec![postcard::to_stdvec(&2u32).unwrap()]
            }
            GuestProgram::Muldiv => {
                vec![postcard::to_stdvec(&(10u32, 5u32, 2u32)).unwrap()]
            }
        }
    }

    fn get_memory_config(&self, use_embed: bool) -> MemoryConfig {
        match self {
            GuestProgram::Fibonacci => {
                if use_embed {
                    MemoryConfig {
                        max_input_size: 4096,
                        max_output_size: 4096,
                        max_untrusted_advice_size: 0,
                        max_trusted_advice_size: 0,
                        memory_size: 134217728,
                        stack_size: 33554432,
                        program_size: None,
                    }
                } else {
                    MemoryConfig {
                        max_input_size: 2000000,
                        max_output_size: 4096,
                        max_untrusted_advice_size: 0,
                        max_trusted_advice_size: 0,
                        memory_size: 33554432,
                        stack_size: 33554432,
                        program_size: None,
                    }
                }
            }
            GuestProgram::Muldiv => {
                if use_embed {
                    MemoryConfig {
                        max_input_size: 1024,
                        max_output_size: 4096,
                        max_untrusted_advice_size: 0,
                        max_trusted_advice_size: 0,
                        memory_size: 134217728,
                        stack_size: 33554432,
                        program_size: None,
                    }
                } else {
                    MemoryConfig {
                        max_input_size: 2000000,
                        max_output_size: 4096,
                        max_untrusted_advice_size: 0,
                        max_trusted_advice_size: 0,
                        memory_size: 33554432,
                        stack_size: 33554432,
                        program_size: None,
                    }
                }
            }
        }
    }

    fn get_max_trace_length(&self, use_embed: bool) -> usize {
        match self {
            GuestProgram::Fibonacci => {
                if use_embed {
                    67108864
                } else {
                    5000000
                }
            }
            GuestProgram::Muldiv => {
                if use_embed {
                    800000
                } else {
                    3000000
                }
            }
        }
    }
}

fn generate_provable_macro(guest: GuestProgram, use_embed: bool, output_dir: &Path) {
    let memory_config = guest.get_memory_config(use_embed);
    let max_trace_length = guest.get_max_trace_length(use_embed);

    let macro_content = format!(
        r#"macro_rules! provable_with_config {{
    ($item: item) => {{
        #[jolt::provable(
            max_input_size = {},
            max_output_size = {},
            max_untrusted_advice_size = {},
            max_trusted_advice_size = {},
            memory_size = {},
            stack_size = {},
            max_trace_length = {}
        )]
        $item
    }};
}}"#,
        memory_config.max_input_size,
        memory_config.max_output_size,
        memory_config.max_untrusted_advice_size,
        memory_config.max_trusted_advice_size,
        memory_config.memory_size,
        memory_config.stack_size,
        max_trace_length
    );

    let provable_macro_path = output_dir.join("provable_macro.rs");

    std::fs::create_dir_all(output_dir).unwrap();

    std::fs::write(&provable_macro_path, macro_content).unwrap();
    info!(
        "Generated {} with config: input={}, output={}, memory={}, stack={}, trace={}",
        provable_macro_path.display(),
        memory_config.max_input_size,
        memory_config.max_output_size,
        memory_config.memory_size,
        memory_config.stack_size,
        max_trace_length
    );
}

fn check_data_integrity(all_groups_data: &[u8]) -> (u32, u32) {
    info!("Checking data integrity...");

    let mut cursor = std::io::Cursor::new(all_groups_data);

    let verifier_preprocessing =
        jolt_sdk::JoltVerifierPreprocessing::<jolt_sdk::F, jolt_sdk::PCS>::deserialize_compressed(
            &mut cursor,
        )
        .unwrap();
    let verifier_bytes = verifier_preprocessing.serialize_to_bytes().unwrap();
    info!(
        "✓ Verifier preprocessing deserialized successfully ({} bytes)",
        verifier_bytes.len()
    );

    let n = u32::deserialize_compressed(&mut cursor).unwrap();
    info!("✓ Number of proofs deserialized: {n}");

    for i in 0..n {
        match RV64IMACProof::deserialize_compressed(&mut cursor) {
            Ok(_) => info!("✓ Proof {i} deserialized"),
            Err(e) => error!("✗ Failed to deserialize proof {i}: {e:?}"),
        }
        match JoltDevice::deserialize_compressed(&mut cursor) {
            Ok(_) => info!("✓ Device {i} deserialized"),
            Err(e) => error!("✗ Failed to deserialize device {i}: {e:?}"),
        }
    }

    let position = cursor.position() as usize;
    let all_data = cursor.into_inner();
    let remaining_data: Vec<u8> = all_data[position..].to_vec();
    info!("✓ Remaining data size: {} bytes", remaining_data.len());

    assert_eq!(
        remaining_data.len(),
        0,
        "Not all data was consumed during deserialization"
    );

    (n, remaining_data.len() as u32)
}

fn collect_guest_proofs(guest: GuestProgram, target_dir: &str, use_embed: bool) -> Vec<u8> {
    info!("Starting collect_guest_proofs for {}", guest.name());
    let max_trace_length = guest.get_max_trace_length(use_embed);

    // This should match the example being run, it can cause layout issues if the guest's macro and our assumption here differ
    let memory_config = MemoryConfig {
        memory_size: 32768u64,
        ..Default::default()
    };

    info!("Creating program...");
    let mut program = jolt_sdk::host::Program::new(guest.name());
    program.set_func(guest.func());
    program.set_std(false);
    program.set_memory_config(memory_config);
    info!("Building program...");
    program.build(target_dir);
    info!("Getting ELF contents...");
    let elf_contents = program.get_elf_contents().unwrap();
    info!("Creating guest program...");
    let mut guest_prog = jolt_sdk::guest::program::Program::new(&elf_contents, &memory_config);
    guest_prog.elf = program.elf;

    info!("Preprocessing guest prover...");
    let guest_prover_preprocessing =
        jolt_sdk::guest::prover::preprocess(&guest_prog, max_trace_length);
    info!("Preprocessing guest verifier...");
    let guest_verifier_preprocessing =
        jolt_sdk::JoltVerifierPreprocessing::from(&guest_prover_preprocessing);

    let inputs = guest.inputs();
    info!("Got inputs: {inputs:?}");

    let mut all_groups_data = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut all_groups_data);
    let mut total_prove_time = 0.0;

    guest_verifier_preprocessing
        .serialize_compressed(&mut cursor)
        .unwrap();

    let n = inputs.len() as u32;
    u32::serialize_compressed(&n, &mut cursor).unwrap();

    info!("Starting {} recursion with {}", guest.name(), n);

    for (i, input_bytes) in inputs.into_iter().enumerate() {
        info!("Processing input {i}: {:#?}", &input_bytes);

        let now = Instant::now();

        let mut output_bytes = vec![0; 4096];

        // Running tracing allows things like JOLT_BACKTRACE=1 to work properly
        info!("  Tracing...");
        guest_prog.memory_config.program_size = Some(
            guest_verifier_preprocessing
                .shared
                .memory_layout
                .program_size,
        );
        let (_, _, _, device_io) = guest_prog.trace(&input_bytes, &[], &[]);
        assert!(!device_io.panic, "Guest program panicked during tracing");

        info!("  Proving...");
        let (proof, io_device, _debug): (RV64IMACProof, _, _) = jolt_sdk::guest::prover::prove(
            &guest_prog,
            &input_bytes,
            &[],
            &[],
            None,
            None,
            &mut output_bytes,
            &guest_prover_preprocessing,
        );
        let prove_time = now.elapsed().as_secs_f64();
        total_prove_time += prove_time;
        info!(
            "  Input: {:?}, Prove time: {:.3}s",
            &input_bytes, prove_time
        );

        proof.serialize_compressed(&mut cursor).unwrap();
        io_device.serialize_compressed(&mut cursor).unwrap();

        info!("  Verifying...");
        let is_valid = jolt_sdk::guest::verifier::verify(
            &input_bytes,
            None,
            &output_bytes,
            proof,
            &guest_verifier_preprocessing,
        )
        .is_ok();
        info!("  Verification result: {is_valid}");
    }
    info!("Total prove time: {total_prove_time:.3}s");
    info!("Total data size: {} bytes", all_groups_data.len());
    all_groups_data
}

fn generate_embedded_bytes(guest: GuestProgram, all_groups_data: &[u8], output_dir: &Path) {
    info!(
        "Generating embedded bytes for {} guest program...",
        guest.name()
    );

    let (n, remaining_data_size) = check_data_integrity(all_groups_data);

    if remaining_data_size > 0 {
        info!("Warning: Remaining data is not empty ({remaining_data_size} bytes). This might indicate proofs are included.");
        info!("For embedded mode, only verifier preprocessing should be included.");
    }

    let mut output = String::new();
    output.push_str(&format!(
        "// Generated embedded bytes for {} recursion guest\n",
        guest.name()
    ));
    output.push_str("pub static EMBEDDED_BYTES: &[u8] = &[\n");

    for (i, byte) in all_groups_data.iter().enumerate() {
        if i > 0 && i % 16 == 0 {
            output.push('\n');
        }
        output.push_str(&format!("0x{byte:02x}, "));
    }

    output.push_str("\n];\n");
    output.push_str(&format!(
        "// Total embedded bytes: {}\n",
        all_groups_data.len()
    ));
    output.push_str(&format!("// Number of proofs: {n}\n"));

    std::fs::create_dir_all(output_dir).unwrap();

    let filename = output_dir.join("embedded_bytes.rs");
    std::fs::write(&filename, output).unwrap();
    info!("Embedded bytes written to {}", filename.display());
}

fn save_proof_data(guest: GuestProgram, all_groups_data: &[u8], workdir: &Path) {
    info!(
        "Saving proof data for {} to {}",
        guest.name(),
        workdir.display()
    );

    std::fs::create_dir_all(workdir).unwrap();

    let proof_file = workdir.join(format!("{}_proofs.bin", guest.name()));
    std::fs::write(&proof_file, all_groups_data).unwrap();

    info!("Proof data saved to {}", proof_file.display());
    info!("Total proof data size: {} bytes", all_groups_data.len());
}

fn load_proof_data(guest: GuestProgram, workdir: &Path) -> Vec<u8> {
    info!(
        "Loading proof data for {} from {}",
        guest.name(),
        workdir.display()
    );

    let proof_file = workdir.join(format!("{}_proofs.bin", guest.name()));

    if !proof_file.exists() {
        panic!("Proof file not found: {}", proof_file.display());
    }

    let proof_data = std::fs::read(&proof_file).unwrap();
    info!(
        "Loaded proof data from {} ({} bytes)",
        proof_file.display(),
        proof_data.len()
    );

    proof_data
}

fn generate_proofs(guest: GuestProgram, workdir: &Path) {
    info!("Generating proofs for {} guest program...", guest.name());

    let target_dir = "/tmp/jolt-guest-targets";

    // Collect guest proofs
    let all_groups_data = collect_guest_proofs(guest, target_dir, false);

    // Save proof data
    save_proof_data(guest, &all_groups_data, workdir);

    info!("Proof generation completed for {}", guest.name());
}

fn run_recursion_proof(
    guest: GuestProgram,
    run_config: RunConfig,
    input_bytes: Vec<u8>,
    memory_config: MemoryConfig,
    mut max_trace_length: usize,
) {
    let target_dir = "/tmp/jolt-guest-targets";

    let mut program = jolt_sdk::host::Program::new("recursion-guest");
    program.set_func("verify");
    program.set_std(true);
    program.set_memory_config(memory_config);
    program.build(target_dir);
    let elf_contents = program.get_elf_contents().unwrap();
    let mut recursion = jolt_sdk::guest::program::Program::new(&elf_contents, &memory_config);
    recursion.elf = program.elf;

    if run_config == RunConfig::Trace || run_config == RunConfig::TraceToFile {
        // shorten the max_trace_length for tracing only. Speeds up setup time for tracing purposes.
        max_trace_length = 0;
    }
    let recursion_prover_preprocessing =
        jolt_sdk::guest::prover::preprocess(&recursion, max_trace_length);
    let recursion_verifier_preprocessing =
        jolt_sdk::JoltVerifierPreprocessing::from(&recursion_prover_preprocessing);

    // update program_size in memory_config now that we know it
    recursion.memory_config.program_size = Some(
        recursion_verifier_preprocessing
            .shared
            .memory_layout
            .program_size,
    );

    let mut output_bytes = vec![
        0;
        recursion_verifier_preprocessing
            .shared
            .memory_layout
            .max_output_size as usize
    ];
    match run_config {
        RunConfig::Prove => {
            let (proof, _io_device, _debug): (RV64IMACProof, _, _) = jolt_sdk::guest::prover::prove(
                &recursion,
                &input_bytes,
                &[],
                &[],
                None,
                None,
                &mut output_bytes,
                &recursion_prover_preprocessing,
            );
            let is_valid = jolt_sdk::guest::verifier::verify(
                &input_bytes,
                None,
                &output_bytes,
                proof,
                &recursion_verifier_preprocessing,
            )
            .is_ok();
            let rv = postcard::from_bytes::<u32>(&output_bytes).unwrap();
            info!("  Recursion verification result: {rv}");
            info!("  Recursion verification result: {is_valid}");
        }
        RunConfig::Trace => {
            info!("  Trace-only mode: Skipping proof generation and verification.");
            let (_, _, _, io_device) = recursion.trace(&input_bytes, &[], &[]);
            let rv = postcard::from_bytes::<u32>(&io_device.outputs).unwrap_or(0);
            info!("  Recursion output (trace-only): {rv}");
        }
        RunConfig::TraceToFile => {
            info!("  Trace-only mode: Skipping proof generation and verification. Tracing to file: /tmp/{}.trace", guest.name());
            let (_, io_device) = recursion.trace_to_file(
                &input_bytes,
                &[],
                &[],
                &format!("/tmp/{}.trace", guest.name()).into(),
            );
            let rv = postcard::from_bytes::<u32>(&io_device.outputs).unwrap_or(0);
            info!("  Recursion output (trace-only): {rv}");
        }
    }
}

fn verify_proofs(
    guest: GuestProgram,
    use_embed: bool,
    workdir: &Path,
    output_dir: &Path,
    run_config: RunConfig,
) {
    info!("Verifying proofs for {} guest program...", guest.name());
    info!("Using embed mode: {use_embed}");

    generate_provable_macro(guest, use_embed, output_dir);

    let all_groups_data = load_proof_data(guest, workdir);

    check_data_integrity(&all_groups_data);

    if use_embed {
        info!("Running {} recursion with embedded bytes...", guest.name());

        generate_embedded_bytes(guest, &all_groups_data, output_dir);

        let memory_config = guest.get_memory_config(use_embed);

        let input_bytes = vec![];
        info!("Using empty input bytes (embedded mode)");

        run_recursion_proof(
            guest,
            run_config,
            input_bytes,
            memory_config,
            guest.get_max_trace_length(use_embed),
        );
    } else {
        info!("Running {} recursion with input data...", guest.name());

        info!("Testing basic serialization/deserialization...");
        let test_input_bytes = postcard::to_stdvec(&all_groups_data).unwrap();
        let test_deserialized: Vec<u8> = postcard::from_bytes(&test_input_bytes).unwrap();
        assert_eq!(all_groups_data, test_deserialized);
        info!("Basic serialization/deserialization test passed!");

        check_data_integrity(&all_groups_data);

        let mut input_bytes = vec![];
        input_bytes.append(&mut postcard::to_stdvec(&all_groups_data.as_slice()).unwrap());

        info!("Serialized input size: {} bytes", input_bytes.len());
        let memory_config = guest.get_memory_config(use_embed);

        assert!(
            input_bytes.len() < memory_config.max_input_size as usize,
            "Input size is too large"
        );

        run_recursion_proof(
            guest,
            run_config,
            input_bytes,
            memory_config,
            guest.get_max_trace_length(use_embed),
        );
    }
}

fn main() {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match &cli.command {
        Some(Commands::Generate { example, workdir }) => {
            let guest = match GuestProgram::from_str(example) {
                Some(guest) => guest,
                None => {
                    info!("Unknown example: {example}. Supported examples: fibonacci, muldiv");
                    return;
                }
            };
            generate_proofs(guest, workdir);
        }
        Some(Commands::Verify {
            example,
            workdir,
            embed,
        }) => {
            let guest = match GuestProgram::from_str(example) {
                Some(guest) => guest,
                None => {
                    info!("Unknown example: {example}. Supported examples: fibonacci, muldiv");
                    return;
                }
            };
            let output_dir = embed
                .as_ref()
                .and_then(|inner| inner.as_ref())
                .cloned()
                .unwrap_or_else(get_guest_src_dir);
            verify_proofs(
                guest,
                embed.is_some(),
                workdir,
                &output_dir,
                RunConfig::Prove,
            );
        }
        Some(Commands::Trace {
            example,
            workdir,
            embed,
            trace_to_file,
        }) => {
            let guest = match GuestProgram::from_str(example) {
                Some(guest) => guest,
                None => {
                    info!("Unknown example: {example}. Supported examples: fibonacci, muldiv");
                    return;
                }
            };
            let output_dir = embed
                .as_ref()
                .and_then(|inner| inner.as_ref())
                .cloned()
                .unwrap_or_else(get_guest_src_dir);
            let run_config = if *trace_to_file {
                RunConfig::TraceToFile
            } else {
                RunConfig::Trace
            };
            verify_proofs(guest, embed.is_some(), workdir, &output_dir, run_config);
        }
        None => {
            info!("No subcommand specified. Available commands:");
            info!("  generate --example <fibonacci|muldiv> [--workdir <DIR>]");
            info!("  verify --example <fibonacci|muldiv> [--workdir <DIR>] [--embed <DIR>]");
            info!("");
            info!("Examples:");
            info!("  cargo run --release -- generate --example fibonacci");
            info!("  cargo run --release -- generate --example fibonacci --workdir ./output");
            info!("  cargo run --release -- verify --example fibonacci");
            info!("  cargo run --release -- verify --example fibonacci --workdir ./output --embed");
            info!("  cargo run --release -- trace --example fibonacci --embed");
            info!("  cargo run --release -- trace --example fibonacci --embed --disk");
        }
    }
}
