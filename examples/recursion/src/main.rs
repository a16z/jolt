use clap::{Parser, Subcommand, ValueEnum};
#[cfg(feature = "akita")]
use jolt_akita::{AkitaField, AkitaScheme};
#[cfg(feature = "akita")]
use jolt_field::Ring;
use jolt_inlines_blake2 as _;
#[cfg(feature = "ntt-inline")]
use jolt_inlines_ntt as _;
use jolt_riscv::JoltInstructionRow;
#[cfg(feature = "akita")]
use jolt_sdk::host::JoltProgramSource;
use jolt_sdk::host::Program;
#[cfg(feature = "akita")]
use jolt_sdk::jolt_prover::akita::preprocessing::AkitaVc;
use jolt_sdk::jolt_verifier::preprocessing::ProgramPreprocessing as VerifierProgramPreprocessing;
#[cfg(feature = "akita")]
use jolt_sdk::jolt_verifier::proof::JoltProofClaims;
#[cfg(feature = "akita")]
use jolt_sdk::jolt_verifier::{JoltProof, JoltVerifierPreprocessing};
use jolt_sdk::{JoltDevice, MemoryConfig, MemoryLayout};
#[cfg(not(feature = "akita"))]
use jolt_sdk::{JoltProverPreprocessing, JoltVerifierPreprocessing, RV64IMACProof};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::cmp::PartialEq;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tracing::info;

/// The proof and verifier preprocessing the guest consumes, per commitment
/// build: Dory on the homomorphic build, Akita on `--features akita`.
#[cfg(not(feature = "akita"))]
type GuestProof = RV64IMACProof;
#[cfg(not(feature = "akita"))]
type GuestVerifierPreprocessing = JoltVerifierPreprocessing;
#[cfg(feature = "akita")]
type GuestProof = JoltProof<AkitaScheme, AkitaVc>;
#[cfg(feature = "akita")]
type GuestVerifierPreprocessing = JoltVerifierPreprocessing<AkitaScheme, AkitaVc>;

/// Guest records are `[u64 length][body][zero padding to 8 bytes]`, so every
/// body starts 8-byte aligned relative to the stream: raw payloads (the
/// Akita setup keys) are then used where they lie, and the Akita
/// public matrix is viewed in place without copying.
const RECORD_ALIGN: usize = 8;

fn push_record<T: Serialize>(buffer: &mut Vec<u8>, value: &T) {
    let bytes = bincode::serde::encode_to_vec(value, bincode::config::standard()).unwrap();
    push_raw(buffer, &bytes);
}

/// Raw bytes behind a length prefix: the guest consumes them in place instead
/// of decoding a `Vec<u8>` byte by byte.
fn push_raw(buffer: &mut Vec<u8>, bytes: &[u8]) {
    let len = u64::try_from(bytes.len()).unwrap();
    buffer.extend_from_slice(&len.to_le_bytes());
    buffer.extend_from_slice(bytes);
    buffer.resize(buffer.len().next_multiple_of(RECORD_ALIGN), 0);
}

/// The stream's leading record: `pad` zero bytes chosen so that, at the
/// address the guest sees the stream, every following body is 8-byte
/// aligned. Only this record is not padded to 8 bytes itself.
fn push_alignment_pad(buffer: &mut Vec<u8>, pad: usize) {
    let len = u64::try_from(pad).unwrap();
    buffer.extend_from_slice(&len.to_le_bytes());
    buffer.resize(buffer.len() + pad, 0);
}

fn read_raw<'a>(buffer: &'a [u8], offset: &mut usize) -> Result<&'a [u8], String> {
    if buffer.len().saturating_sub(*offset) < 8 {
        return Err("missing record length prefix".to_string());
    }
    let mut len_bytes = [0u8; 8];
    len_bytes.copy_from_slice(&buffer[*offset..*offset + 8]);
    *offset += 8;
    let len = usize::try_from(u64::from_le_bytes(len_bytes)).unwrap();
    if buffer.len().saturating_sub(*offset) < len {
        return Err("truncated raw record".to_string());
    }
    let end = *offset + len;
    let bytes = &buffer[*offset..end];
    *offset = end.next_multiple_of(RECORD_ALIGN).min(buffer.len());
    Ok(bytes)
}

fn read_record<T: DeserializeOwned>(buffer: &[u8], offset: &mut usize) -> Result<T, String> {
    if buffer.len().saturating_sub(*offset) < 8 {
        return Err("missing record length prefix".to_string());
    }
    let mut len_bytes = [0u8; 8];
    len_bytes.copy_from_slice(&buffer[*offset..*offset + 8]);
    *offset += 8;

    let len = usize::try_from(u64::from_le_bytes(len_bytes)).unwrap();
    if buffer.len().saturating_sub(*offset) < len {
        return Err("truncated serialized record".to_string());
    }
    let end = *offset + len;
    let (value, consumed) =
        bincode::serde::decode_from_slice(&buffer[*offset..end], bincode::config::standard())
            .map_err(|error| error.to_string())?;
    assert_eq!(consumed, len, "record decoder left trailing bytes");
    *offset = end.next_multiple_of(RECORD_ALIGN).min(buffer.len());
    Ok(value)
}

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
    /// Build and retain a non-embedded verifier ELF and its framed input.
    PrepareGuest {
        #[arg(long)]
        example: String,
        #[arg(long)]
        workdir: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Execute retained bytes without rebuilding; panic never counts as rejection.
    ExecutePrepared {
        #[arg(long)]
        directory: PathBuf,
        #[arg(long, value_enum)]
        expect: ExpectedVerification,
    },
    /// Write a parseable Akita proof with one deliberately invalid opening.
    #[cfg(feature = "akita")]
    TamperOpening {
        #[arg(long)]
        example: String,
        #[arg(long)]
        workdir: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Generate proofs for guest programs
    Generate {
        /// Example to run (fibonacci or muldiv)
        #[arg(long, value_name = "EXAMPLE")]
        example: String,
        /// Working directory for output files
        #[arg(long, value_name = "DIRECTORY", default_value = "output")]
        workdir: PathBuf,
        /// Use committed program mode for the inner guest proof
        #[arg(long, value_name = "CHUNKS")]
        committed_bytecode: Option<usize>,
        /// Number of inner proofs the recursion guest verifies in one run
        #[arg(long, value_name = "COUNT", default_value_t = 1)]
        proofs: usize,
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
        /// Store trace rows on disk instead of executing without row storage
        #[arg(short = 'd', long = "disk", default_value_t = false)]
        trace_to_file: bool,
    },
}

#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum GuestProgram {
    Fibonacci,
    Muldiv,
}

#[derive(Clone, PartialEq, Eq)]
enum RunConfig {
    Prove,
    Trace,
    TraceToFile,
    Prepare(PathBuf),
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

    /// `count` distinct inputs: one inner proof each, all verified by one run
    /// of the recursion guest (setup decode is paid once per run).
    fn inputs(&self, count: usize) -> Vec<Vec<u8>> {
        (0..count as u32)
            .map(|index| match self {
                GuestProgram::Fibonacci => postcard::to_stdvec(&(2u32 + index)).unwrap(),
                GuestProgram::Muldiv => {
                    postcard::to_stdvec(&(10u32 + index, 5u32, 2u32 + index)).unwrap()
                }
            })
            .collect()
    }

    fn get_memory_config(&self, use_embed: bool) -> MemoryConfig {
        match self {
            GuestProgram::Fibonacci => {
                if use_embed {
                    MemoryConfig {
                        max_input_size: 16_000_000,
                        max_output_size: 4096,
                        max_untrusted_advice_size: 0,
                        max_trusted_advice_size: 0,
                        heap_size: 134217728,
                        stack_size: 33554432,
                        program_size: None,
                    }
                } else {
                    MemoryConfig {
                        max_input_size: 40_000_000,
                        max_output_size: 4096,
                        max_untrusted_advice_size: 0,
                        max_trusted_advice_size: 0,
                        heap_size: 134217728,
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
                        heap_size: 134217728,
                        stack_size: 33554432,
                        program_size: None,
                    }
                } else {
                    MemoryConfig {
                        max_input_size: 2000000,
                        max_output_size: 4096,
                        max_untrusted_advice_size: 0,
                        max_trusted_advice_size: 0,
                        heap_size: 33554432,
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
            heap_size = {},
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
        memory_config.heap_size,
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
        memory_config.heap_size,
        memory_config.stack_size,
        max_trace_length
    );
}

/// Where the proof section starts in a saved stream: the setup section (the
/// verifier preprocessing and its detached payloads) precedes it, so an
/// embedded build can bake the setup into the guest image and feed only the
/// proofs as input.
struct StreamLayout {
    setup_len: usize,
    proof_count: u32,
}

fn check_data_integrity(all_groups_data: &[u8]) -> StreamLayout {
    info!("Checking data integrity...");

    let mut offset = 0;
    let verifier_preprocessing: GuestVerifierPreprocessing =
        read_record(all_groups_data, &mut offset).unwrap();
    let verifier_bytes =
        bincode::serde::encode_to_vec(&verifier_preprocessing, bincode::config::standard())
            .unwrap();
    info!(
        "✓ Verifier preprocessing deserialized successfully ({} bytes)",
        verifier_bytes.len()
    );
    let payload_count: u32 = read_record(all_groups_data, &mut offset).unwrap();
    for i in 0..payload_count {
        let payload = read_raw(all_groups_data, &mut offset).expect("decode setup payload");
        info!("Setup payload {i}: {} bytes", payload.len());
    }
    let setup_len = offset;

    let n: u32 = read_record(all_groups_data, &mut offset).unwrap();
    info!("✓ Number of proofs deserialized: {n}");

    for i in 0..n {
        let _: GuestProof = read_record(all_groups_data, &mut offset).expect("decode proof");
        let _: JoltDevice = read_record(all_groups_data, &mut offset).expect("decode device");
        info!("Decoded proof and device {i}");
    }

    let remaining = all_groups_data.len() - offset;
    info!("✓ Remaining data size: {remaining} bytes");
    assert_eq!(
        remaining, 0,
        "Not all data was consumed during deserialization"
    );

    StreamLayout {
        setup_len,
        proof_count: n,
    }
}

#[cfg(not(feature = "akita"))]
fn preprocess_guest_prover(
    guest_prog: &mut Program,
    memory_config: MemoryConfig,
    max_trace_length: usize,
    bytecode_chunk_count: Option<usize>,
) -> JoltProverPreprocessing {
    jolt_sdk::preprocess_program(
        guest_prog,
        memory_config,
        max_trace_length,
        bytecode_chunk_count,
    )
    .unwrap()
}

/// The packed (Akita) inner proofs use the modular prover over fp128 and the
/// packed verifier preprocessing from
/// the host's own verification of each proof.
#[cfg(feature = "akita")]
fn collect_guest_proofs(
    guest: GuestProgram,
    target_dir: &str,
    _use_embed: bool,
    bytecode_chunk_count: Option<usize>,
    proofs: usize,
) -> Vec<u8> {
    use jolt_akita::AkitaScheduleArtifacts;
    use jolt_program::execution::{ExecutionBackend, TraceInputs};
    use jolt_program::preprocess::JoltProgramPreprocessing;
    use jolt_sdk::jolt_prover::akita::preprocessing::{self, AkitaTranscript};
    use jolt_sdk::jolt_prover::akita::{self, JoltAkitaBackend};
    use jolt_sdk::jolt_prover::ProverConfig;
    use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};
    use tracer::execution_backend::TracerBackend;

    let max_trace_length = guest.get_max_trace_length(false);
    let mut memory_config = MemoryConfig { heap_size: 32768u64, ..Default::default() };
    let mut program = Program::new(guest.name());
    #[cfg(feature = "field-inline")]
    program.enable_field_inline();
    program.set_func(guest.func());
    program.set_std(false);
    program.set_memory_config(memory_config);
    program.build(target_dir);
    let jolt_program = Arc::new(program.build_jolt_program().expect("build inner program"));
    let inputs = guest.inputs(proofs);
    let (_, _, _, io_device) = program.trace(&inputs[0], &[], &[]);
    memory_config.program_size = Some(io_device.memory_layout.program_size);
    let program_data = Arc::new(JoltProgramPreprocessing::new(
        jolt_program.expanded_bytecode.clone(), jolt_program.memory_init.clone(),
        io_device.memory_layout.clone(), jolt_program.entry_address,
        max_trace_length, program.instruction_profile(),
    ).expect("inner program preprocessing"));
    let schedule_artifacts = AkitaScheduleArtifacts::shared_from_default_directory();
    let mut all_groups_data = Vec::new();
    let n = inputs.len() as u32;
    let mut verifier_preprocessing = None;
    let mut records = Vec::new();
    for input_bytes in inputs {
        let trace = TracerBackend::new().trace(
            &jolt_program, TraceInputs::new(input_bytes, Vec::new(), Vec::new(), memory_config),
        ).expect("trace inner program");
        let config = ProverConfig::derive::<AkitaField>(
            trace.trace.rows(), &program_data.memory_layout,
            program_data.ram.min_bytecode_address, program_data.ram.bytecode_words.len(),
            max_trace_length,
        ).expect("inner proof configuration");
        let prover_preprocessing = match bytecode_chunk_count {
            Some(chunks) => preprocessing::preprocess_committed(
                &schedule_artifacts, program_data.as_ref().clone(), &config, chunks,
            ),
            None => preprocessing::preprocess_full(
                &schedule_artifacts, program_data.as_ref().clone(), &config,
            ),
        }.expect("packed preprocessing");
        let public_io = trace.device.clone();
        let witness = TraceBackend::new(
            JoltVmWitnessConfig::new(
                config.trace_length.ilog2() as usize, config.ram_K, config.one_hot_config,
            ),
            JoltVmWitnessInputs::new(&jolt_program, &program_data, trace),
        );
        #[cfg(feature = "field-inline")]
        let witness = witness.with_field_inline().expect("field witness");
        let now = Instant::now();
        let proof = akita::prove::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript, _>(
            &JoltAkitaBackend::optimized(), &prover_preprocessing, &config, None, &witness, &public_io,
        ).expect("packed proof");
        info!("  Packed prove time: {:.3}s", now.elapsed().as_secs_f64());
        let mut preprocessing = prover_preprocessing.verifier;
        // The guest's setup is a trusted constant: carry the expanded backend
        // verifier keys instead of re-deriving them from the seed in-circuit.
        let embedded = preprocessing
            .pcs_setup
            .embed_prepared_backend_verifiers()
            .expect("embed prepared Akita backend verifiers");
        info!("  Embedded prepared Akita backend verifier keys: {embedded} bytes");
        let ntt_cache = preprocessing
            .pcs_setup
            .embed_prepared_terminal_ntt_cache(proof.joint_opening_proof.schedule_row_digest())
            .expect("embed prepared Akita terminal NTT cache");
        info!("  Embedded prepared Akita terminal NTT cache: {ntt_cache} bytes");
        records.push((proof, public_io));
        verifier_preprocessing = Some(preprocessing);
    }
    // The multi-megabyte setup payloads travel out of line, so the guest reads
    // them where they lie instead of copying them out of the bincode record.
    let mut verifier_preprocessing = verifier_preprocessing.unwrap();
    let selections: Vec<_> = records
        .iter()
        .map(|(proof, _)| proof.joint_opening_proof.schedule_row_digest())
        .collect();
    let catalog_bytes = verifier_preprocessing
        .pcs_setup
        .embed_prepared_schedule_catalog_views(&selections)
        .expect("prepare Akita verifier catalog views");
    info!("  Prepared Akita verifier catalog views: {catalog_bytes} bytes");
    // Rebuild skipped caches and verify every proof against the exact setup
    // transported to the guest, including the restricted catalog coverage.
    let (mut verifier_preprocessing, _): (GuestVerifierPreprocessing, _) =
        bincode::serde::decode_from_slice(
            &bincode::serde::encode_to_vec(&verifier_preprocessing, bincode::config::standard())
                .unwrap(),
            bincode::config::standard(),
        )
        .unwrap();
    for (proof, public_io) in &records {
        jolt_sdk::jolt_verifier::verify::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript>(
            &verifier_preprocessing,
            public_io,
            proof,
            None,
        )
        .expect("verify proof against prepared guest setup");
        info!("  Verification result: true");
    }
    let payloads = verifier_preprocessing
        .pcs_setup
        .detach_prepared_payloads()
        .expect("detach prepared Akita payloads");
    push_record(&mut all_groups_data, &verifier_preprocessing);
    push_record(&mut all_groups_data, &(payloads.len() as u32));
    for payload in &payloads {
        push_raw(&mut all_groups_data, payload);
    }
    push_record(&mut all_groups_data, &n);
    for (proof, public_io) in records {
        push_record(&mut all_groups_data, &proof);
        push_record(&mut all_groups_data, &public_io);
    }
    info!("Total data size: {} bytes", all_groups_data.len());
    all_groups_data
}

#[cfg(not(feature = "akita"))]
fn collect_guest_proofs(
    guest: GuestProgram,
    target_dir: &str,
    use_embed: bool,
    bytecode_chunk_count: Option<usize>,
    proofs: usize,
) -> Vec<u8> {
    info!("Starting collect_guest_proofs for {}", guest.name());
    let max_trace_length = guest.get_max_trace_length(use_embed);

    // This should match the example being run, it can cause layout issues if the guest's macro and our assumption here differ
    let memory_config = MemoryConfig {
        heap_size: 32768u64,
        ..Default::default()
    };

    info!("Creating program...");
    let mut program = Program::new(guest.name());
    program.set_func(guest.func());
    program.set_std(false);
    program.set_memory_config(memory_config);
    info!("Building program...");
    program.build(target_dir);
    info!("Getting ELF contents...");
    info!("Preprocessing guest prover...");
    let guest_prover_preprocessing = preprocess_guest_prover(
        &mut program,
        memory_config,
        max_trace_length,
        bytecode_chunk_count,
    );
    info!("Preprocessing guest verifier...");
    let guest_verifier_preprocessing =
        jolt_sdk::verifier_preprocessing_from_prover(&guest_prover_preprocessing);

    let inputs = guest.inputs(proofs);
    info!("Got inputs: {inputs:?}");

    let mut all_groups_data = Vec::new();
    let mut total_prove_time = 0.0;

    push_record(&mut all_groups_data, &guest_verifier_preprocessing);
    // No out-of-line setup payloads: the Dory setup travels inside the record.
    push_record(&mut all_groups_data, &0u32);

    let n = inputs.len() as u32;
    push_record(&mut all_groups_data, &n);

    info!("Starting {} recursion with {}", guest.name(), n);

    for (i, input_bytes) in inputs.into_iter().enumerate() {
        info!("Processing input {i}: {:#?}", &input_bytes);

        let now = Instant::now();

        // Running tracing allows things like JOLT_BACKTRACE=1 to work properly
        info!("  Tracing...");
        let (_, _, _, device_io) = program.trace(&input_bytes, &[], &[]);
        assert!(!device_io.panic, "Guest program panicked during tracing");

        info!("  Proving...");
        let (proof, io_device): (RV64IMACProof, _) = jolt_sdk::prove_program(
            &program,
            &guest_prover_preprocessing,
            &input_bytes,
            &[],
            &[],
            None,
            None,
            None,
        )
        .expect("prover should produce verifier-native proof");
        let prove_time = now.elapsed().as_secs_f64();
        total_prove_time += prove_time;
        info!(
            "  Input: {:?}, Prove time: {:.3}s",
            &input_bytes, prove_time
        );

        push_record(&mut all_groups_data, &proof);
        push_record(&mut all_groups_data, &io_device);

        info!("  Verifying...");
        let is_valid = jolt_sdk::jolt_verifier::verify::<
            jolt_sdk::VerifierField,
            jolt_sdk::VerifierPCS,
            jolt_sdk::VerifierVC,
            jolt_sdk::VerifierTranscript,
        >(&guest_verifier_preprocessing, &io_device, &proof, None)
        .is_ok();
        info!("  Verification result: {is_valid}");
    }
    info!("Total prove time: {total_prove_time:.3}s");
    info!("Total data size: {} bytes", all_groups_data.len());
    all_groups_data
}

/// Bake the setup section of the stream into the guest image: `embedded.bin`
/// next to the guest sources, included 8-byte aligned by the generated
/// `embedded_bytes.rs`. The guest then takes its verifier setup from its own
/// image and only the proofs from its input.
fn generate_embedded_bytes(guest: GuestProgram, setup_section: &[u8], output_dir: &Path) {
    let mut offset = 0;
    let mut preprocessing: GuestVerifierPreprocessing =
        read_record(setup_section, &mut offset).unwrap();
    let bytecode = match &mut preprocessing.program {
        VerifierProgramPreprocessing::Full(program) => {
            std::mem::take(&mut Arc::make_mut(program).bytecode.bytecode)
        }
        VerifierProgramPreprocessing::Committed(_) => Vec::new(),
    };
    let mut compiled_setup = Vec::new();
    push_record(&mut compiled_setup, &preprocessing);
    compiled_setup.extend_from_slice(&setup_section[offset..]);
    let setup_section = compiled_setup.as_slice();

    info!(
        "Generating embedded setup for {} guest program ({} bytes)...",
        guest.name(),
        setup_section.len()
    );
    let mut image = Vec::with_capacity(setup_section.len() + RECORD_ALIGN);
    // The static is 8-byte aligned, so no leading pad is needed.
    push_alignment_pad(&mut image, 0);
    image.extend_from_slice(setup_section);

    std::fs::create_dir_all(output_dir).unwrap();
    let bin_path = output_dir.join("embedded.bin");
    std::fs::write(&bin_path, &image).unwrap();
    let source = format!(
        "// Generated by the recursion host: the verifier setup baked into this image.\n\
         #[repr(C, align(8))]\n\
         struct Aligned<T: ?Sized>(T);\n\
         static ALIGNED: Aligned<[u8; {len}]> = Aligned(*include_bytes!(\"embedded.bin\"));\n\
         pub static EMBEDDED_BYTES: &[u8] = &ALIGNED.0;\n",
        len = image.len()
    );
    std::fs::write(
        output_dir.join("embedded_bytecode.rs"),
        render_embedded_bytecode(&bytecode),
    )
    .unwrap();
    let source_path = output_dir.join("embedded_bytes.rs");
    std::fs::write(&source_path, source).unwrap();
    info!("Embedded setup written to {}", bin_path.display());
}

/// Render owned program rows as typed constants for the already-embedded setup.
fn render_embedded_bytecode(rows: &[JoltInstructionRow]) -> String {
    use std::fmt::Write;

    if rows.is_empty() {
        return "use jolt_riscv::JoltInstructionRow;\npub static EMBEDDED_BYTECODE: &[JoltInstructionRow] = &[];\n".to_owned();
    }
    let mut source = String::from("use jolt_riscv::{JoltInstructionKind as Kind, JoltInstructionRow, JoltInstructionTag, NormalizedOperands};\npub static EMBEDDED_BYTECODE: &[JoltInstructionRow] = &[\n");
    for row in rows {
        writeln!(source,
            "JoltInstructionRow {{ instruction_kind: Kind::from_tag(JoltInstructionTag({})).unwrap(), address: {}, operands: NormalizedOperands {{ rs1: {:?}, rs2: {:?}, rd: {:?}, imm: {} }}, virtual_sequence_remaining: {:?}, is_first_in_sequence: {}, is_compressed: {} }},",
            row.instruction_kind.tag().0, row.address, row.operands.rs1, row.operands.rs2,
            row.operands.rd, row.operands.imm, row.virtual_sequence_remaining,
            row.is_first_in_sequence, row.is_compressed,
        ).unwrap();
    }
    source.push_str("];\n");
    source
}

/// Undo [`generate_embedded_bytes`]: an input-mode build must not carry a
/// stale baked setup.
fn clear_embedded_bytes(output_dir: &Path) {
    std::fs::create_dir_all(output_dir).unwrap();
    std::fs::write(
        output_dir.join("embedded_bytecode.rs"),
        render_embedded_bytecode(&[]),
    )
    .unwrap();
    std::fs::write(
        output_dir.join("embedded_bytes.rs"),
        "pub static EMBEDDED_BYTES: &[u8] = &[];\n",
    )
    .unwrap();
    let _ = std::fs::remove_file(output_dir.join("embedded.bin"));
}

/// Frame `section` as the guest's input: `postcard` prefixes the byte slice
/// with a varint length, so a leading pad record lands every body 8-byte
/// aligned at the address the guest reads it from.
fn frame_guest_input(section: &[u8], memory_config: &MemoryConfig) -> Vec<u8> {
    fn varint_len(value: usize) -> usize {
        postcard::to_stdvec(&value).unwrap().len()
    }
    // The I/O region does not depend on the program size, which the layout
    // insists on knowing (as the SDK's own macro does, pass a placeholder).
    let layout = MemoryLayout::new(&MemoryConfig {
        program_size: Some(0),
        ..*memory_config
    });
    let input_start = usize::try_from(layout.input_start).unwrap();
    let mut pad = 0;
    // The varint width depends on the total length, which depends on the pad;
    // iterate to the fixpoint (the width changes only at 2^(7k) boundaries).
    for _ in 0..4 {
        let total = 8 + pad + section.len();
        let body_start = input_start + varint_len(total) + 8;
        let next = body_start.next_multiple_of(RECORD_ALIGN) - body_start;
        if next == pad {
            break;
        }
        pad = next;
    }
    let mut stream = Vec::with_capacity(8 + pad + section.len());
    push_alignment_pad(&mut stream, pad);
    stream.extend_from_slice(section);
    assert_eq!(
        (input_start + varint_len(stream.len()) + 8 + pad) % RECORD_ALIGN,
        0,
        "guest input bodies must be 8-byte aligned"
    );
    postcard::to_stdvec(&stream.as_slice()).unwrap()
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

fn generate_proofs(
    guest: GuestProgram,
    workdir: &Path,
    bytecode_chunk_count: Option<usize>,
    proofs: usize,
) {
    info!(
        "Generating {proofs} proof(s) for {} guest program...",
        guest.name()
    );

    let target_dir = "/tmp/jolt-guest-targets";

    // Collect guest proofs
    let all_groups_data =
        collect_guest_proofs(guest, target_dir, false, bytecode_chunk_count, proofs);

    // Save proof data
    save_proof_data(guest, &all_groups_data, workdir);

    info!("Proof generation completed for {}", guest.name());
}

fn decode_verifier_output(bytes: &[u8]) -> u32 {
    let (output, remaining) =
        postcard::take_from_bytes::<u32>(bytes).expect("decode verifier output");
    assert!(remaining.is_empty(), "trailing verifier output bytes");
    output
}

fn configured_recursion_program(memory_config: MemoryConfig) -> Program {
    let mut program = Program::new("recursion-guest");
    program.set_func("verify");
    program.set_std(true);
    // The verifier guest computes its field arithmetic through the
    // field-inline instructions, so it decodes under the FR profile.
    #[cfg(feature = "field-inline")]
    program.enable_field_inline();
    #[cfg(feature = "akita")]
    program.add_guest_feature("akita");
    #[cfg(feature = "ntt-inline")]
    program.add_guest_feature("ntt-inline");
    program.add_guest_feature("fast-alloc");
    program.add_guest_feature("blake2-inline");
    // The verifier preprocessing is the recursion circuit's own trusted
    // constant: its group elements need no subgroup validation on decode.
    #[cfg(not(feature = "akita"))]
    program.add_guest_feature("trusted-preprocessing");
    program.set_memory_config(memory_config);
    program
}

#[derive(Clone, Copy, ValueEnum)]
enum ExpectedVerification {
    Accept,
    Reject,
}

#[derive(Serialize, Deserialize)]
struct PreparedGuest {
    guest: GuestProgram,
    akita: bool,
    field_inline: bool,
    ntt_inline: bool,
    input: Vec<u8>,
}

impl PreparedGuest {
    fn save(guest: GuestProgram, program: &Program, input: Vec<u8>, directory: &Path) {
        std::fs::create_dir(directory).expect("fresh prepared guest directory");
        let prepared = Self {
            guest,
            akita: cfg!(feature = "akita"),
            field_inline: cfg!(feature = "field-inline"),
            ntt_inline: cfg!(feature = "ntt-inline"),
            input,
        };
        std::fs::write(
            directory.join("guest.elf"),
            program.get_elf_contents().expect("built verifier ELF"),
        )
        .unwrap();
        std::fs::write(
            directory.join("execution.bin"),
            bincode::serde::encode_to_vec(&prepared, bincode::config::standard()).unwrap(),
        )
        .unwrap();
    }

    fn execute(directory: &Path, expected: ExpectedVerification) {
        let bytes = std::fs::read(directory.join("execution.bin")).unwrap();
        let (prepared, consumed): (Self, usize) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
        assert_eq!(consumed, bytes.len(), "trailing prepared execution bytes");
        assert_eq!(
            prepared.akita,
            cfg!(feature = "akita"),
            "Akita profile mismatch"
        );
        assert_eq!(
            prepared.field_inline,
            cfg!(feature = "field-inline"),
            "field profile mismatch"
        );
        assert_eq!(
            prepared.ntt_inline,
            cfg!(feature = "ntt-inline"),
            "NTT profile mismatch"
        );
        let memory = prepared.guest.get_memory_config(false);
        assert!(prepared.input.len() < memory.max_input_size as usize);
        let mut program = configured_recursion_program(memory);
        program.elf = Some(directory.join("guest.elf"));
        let elf = program.get_elf_contents().expect("retained verifier ELF");
        let (rows, device) = program.execute_with_output(&prepared.input, &[], &[]);
        assert!(!device.panic, "retained verifier guest panicked");
        let output = decode_verifier_output(&device.outputs);
        let expected = match expected {
            ExpectedVerification::Accept => 1,
            ExpectedVerification::Reject => 0,
        };
        assert_eq!(output, expected, "unexpected guest verification result");
        assert_eq!(
            program.get_elf_contents().unwrap(),
            elf,
            "retained ELF changed"
        );
        assert_eq!(
            std::fs::read(directory.join("execution.bin")).unwrap(),
            bytes
        );
        info!("Retained verifier output: {output}; trace length: {rows}");
    }
}

#[cfg(feature = "akita")]
fn tamper_opening(guest: GuestProgram, workdir: &Path, output: &Path) {
    let bytes = load_proof_data(guest, workdir);
    let mut offset = 0;
    let _: GuestVerifierPreprocessing = read_record(&bytes, &mut offset).unwrap();
    let payloads: u32 = read_record(&bytes, &mut offset).unwrap();
    for _ in 0..payloads {
        let _ = read_raw(&bytes, &mut offset).unwrap();
    }
    let count: u32 = read_record(&bytes, &mut offset).unwrap();
    assert!(count > 0, "tamper fixture requires a proof");
    let mut tampered = bytes[..offset].to_vec();
    for index in 0..count {
        let mut proof: GuestProof = read_record(&bytes, &mut offset).unwrap();
        let device: JoltDevice = read_record(&bytes, &mut offset).unwrap();
        if index == 0 {
            let JoltProofClaims::Clear(claims) = &mut proof.claims else {
                panic!("Akita fixture requires clear claims");
            };
            claims.stage1.outer.outer_remainder.left_instruction_input += AkitaField::from_u64(1);
        }
        push_record(&mut tampered, &proof);
        push_record(&mut tampered, &device);
    }
    assert_eq!(offset, bytes.len(), "trailing proof stream bytes");
    // A successful typed decode is distinct from the guest's verification result.
    assert_eq!(check_data_integrity(&tampered).proof_count, count);
    std::fs::create_dir(output).expect("fresh tampered proof directory");
    save_proof_data(guest, &tampered, output);
}

fn run_recursion_proof(
    guest: GuestProgram,
    run_config: RunConfig,
    input_bytes: Vec<u8>,
    memory_config: MemoryConfig,
    max_trace_length: usize,
) {
    let target_dir = "/tmp/jolt-guest-targets";

    let mut program = configured_recursion_program(memory_config);
    program.build(target_dir);
    match run_config {
        RunConfig::Prepare(directory) => {
            PreparedGuest::save(guest, &program, input_bytes, &directory);
        }
        RunConfig::Trace | RunConfig::TraceToFile => {
            let io_device = if run_config == RunConfig::Trace {
                let (rows, device) = program.execute_with_output(&input_bytes, &[], &[]);
                info!("  trace length: {rows}");
                device
            } else {
                let trace_path = PathBuf::from(format!("/tmp/{}-recursion.trace", guest.name()));
                program.trace_to_file(&input_bytes, &[], &[], &trace_path).1
            };
            assert!(!io_device.panic, "Recursion verifier guest panicked");
            let rv = decode_verifier_output(&io_device.outputs);
            assert_eq!(rv, 1, "Recursion verifier rejected the proof");
            info!("  Recursion output (trace-only): {rv}");
        }
        RunConfig::Prove => {
            #[cfg(feature = "akita")]
            {
                let _ = max_trace_length;
                panic!("packed recursion supports trace-only runs");
            }
            #[cfg(not(feature = "akita"))]
            {
                let preprocessing = preprocess_guest_prover(
                    &mut program, memory_config, max_trace_length, None,
                );
                let verifier = jolt_sdk::verifier_preprocessing_from_prover(&preprocessing);
                let (proof, io_device): (RV64IMACProof, _) = jolt_sdk::prove_program(
                    &program, &preprocessing, &input_bytes, &[], &[], None, None, None,
                ).expect("outer recursion proof");
                jolt_sdk::jolt_verifier::verify::<
                    jolt_sdk::VerifierField,
                    jolt_sdk::VerifierPCS,
                    jolt_sdk::VerifierVC,
                    jolt_sdk::VerifierTranscript,
                >(&verifier, &io_device, &proof, None)
                .expect("verify outer recursion proof");
                let rv = decode_verifier_output(&io_device.outputs);
                info!("  Recursion verification result: {rv}");
            }
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
    let layout = check_data_integrity(&all_groups_data);
    let memory_config = guest.get_memory_config(use_embed);

    let input_section = if use_embed {
        info!(
            "Running {} recursion with the setup baked into the guest image...",
            guest.name()
        );
        let (setup_section, proof_section) = all_groups_data.split_at(layout.setup_len);
        generate_embedded_bytes(guest, setup_section, output_dir);
        proof_section
    } else {
        info!("Running {} recursion with input data...", guest.name());
        clear_embedded_bytes(output_dir);
        all_groups_data.as_slice()
    };
    let input_bytes = frame_guest_input(input_section, &memory_config);
    info!(
        "Serialized input size: {} bytes ({} proofs)",
        input_bytes.len(),
        layout.proof_count
    );
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

fn main() {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match &cli.command {
        Some(Commands::PrepareGuest {
            example,
            workdir,
            output,
        }) => {
            let guest = GuestProgram::from_str(example).expect("supported guest example");
            verify_proofs(
                guest,
                false,
                workdir,
                &get_guest_src_dir(),
                RunConfig::Prepare(output.clone()),
            );
        }
        Some(Commands::ExecutePrepared { directory, expect }) => {
            PreparedGuest::execute(directory, *expect);
        }
        #[cfg(feature = "akita")]
        Some(Commands::TamperOpening {
            example,
            workdir,
            output,
        }) => {
            let guest = GuestProgram::from_str(example).expect("supported guest example");
            tamper_opening(guest, workdir, output);
        }
        Some(Commands::Generate {
            example,
            workdir,
            committed_bytecode,
            proofs,
        }) => {
            let guest = match GuestProgram::from_str(example) {
                Some(guest) => guest,
                None => {
                    info!("Unknown example: {example}. Supported examples: fibonacci, muldiv");
                    return;
                }
            };
            generate_proofs(guest, workdir, *committed_bytecode, *proofs);
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
            info!("  cargo run --release -- generate --example fibonacci --committed-bytecode 16");
            info!("  cargo run --release -- verify --example fibonacci");
            info!("  cargo run --release -- verify --example fibonacci --workdir ./output --embed");
            info!("  cargo run --release -- trace --example fibonacci --embed");
            info!("  cargo run --release -- trace --example fibonacci --embed --disk");
        }
    }
}
