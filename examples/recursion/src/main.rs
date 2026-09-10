use clap::{Parser, Subcommand};
// Linked for its inline registration: the guest transcripts hash with the
// Blake2b inline, which the tracer expands only for registered extensions.
use jolt_inlines_blake2 as _;
#[cfg(not(feature = "akita"))]
use jolt_sdk::guest::program::Program;
use jolt_sdk::{JoltDevice, MemoryConfig, MemoryLayout};
#[cfg(not(feature = "akita"))]
use jolt_sdk::{
    JoltProverPreprocessing, JoltSharedPreprocessing, JoltVerifierPreprocessing,
    ProgramPreprocessing, RV64IMACProof,
};
use serde::{de::DeserializeOwned, Serialize};
use std::cmp::PartialEq;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tracing::{error, info};

/// The proof and verifier preprocessing the guest consumes, per commitment
/// build: Dory on the homomorphic build, Akita on `--features akita`.
#[cfg(not(feature = "akita"))]
type GuestProof = RV64IMACProof;
#[cfg(not(feature = "akita"))]
type GuestVerifierPreprocessing = JoltVerifierPreprocessing;
#[cfg(feature = "akita")]
type GuestProof = jolt_sdk::jolt_verifier::JoltProof<
    jolt_sdk::jolt_prover_legacy::zkvm::packed::AkitaScheme,
    jolt_sdk::jolt_prover_legacy::zkvm::packed::AkitaVc,
>;
#[cfg(feature = "akita")]
type GuestVerifierPreprocessing = jolt_sdk::jolt_verifier::JoltVerifierPreprocessing<
    jolt_sdk::jolt_prover_legacy::zkvm::packed::AkitaScheme,
    jolt_sdk::jolt_prover_legacy::zkvm::packed::AkitaVc,
>;

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
        match read_raw(all_groups_data, &mut offset) {
            Ok(payload) => info!("✓ Setup payload {i} read ({} bytes)", payload.len()),
            Err(e) => error!("✗ Failed to read setup payload {i}: {e:?}"),
        }
    }
    let setup_len = offset;

    let n: u32 = read_record(all_groups_data, &mut offset).unwrap();
    info!("✓ Number of proofs deserialized: {n}");

    for i in 0..n {
        match read_record::<GuestProof>(all_groups_data, &mut offset) {
            Ok(_) => info!("✓ Proof {i} deserialized"),
            Err(e) => error!("✗ Failed to deserialize proof {i}: {e:?}"),
        }
        match read_record::<JoltDevice>(all_groups_data, &mut offset) {
            Ok(_) => info!("✓ Device {i} deserialized"),
            Err(e) => error!("✗ Failed to deserialize device {i}: {e:?}"),
        }
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
    max_trace_length: usize,
    bytecode_chunk_count: Option<usize>,
) -> JoltProverPreprocessing<jolt_sdk::F, jolt_sdk::Curve, jolt_sdk::PCS> {
    if let Some(chunk_count) = bytecode_chunk_count {
        let (bytecode, memory_init, program_size, e_entry) = guest_prog.decode();
        let mut memory_config = guest_prog.memory_config;
        memory_config.program_size = Some(program_size);
        let memory_layout = MemoryLayout::new(&memory_config);
        let program = ProgramPreprocessing::preprocess(bytecode, memory_init, e_entry).unwrap();
        let (shared, committed_program_prover_data, generators) =
            JoltSharedPreprocessing::new_committed(
                program,
                memory_layout,
                max_trace_length,
                chunk_count,
            );
        JoltProverPreprocessing::new_committed(shared, committed_program_prover_data, generators)
    } else {
        jolt_sdk::guest::prover::preprocess(guest_prog, max_trace_length).unwrap()
    }
}

/// The packed (Akita) inner proofs: legacy packed prover over fp128, the
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
    use jolt_openings::CommitmentScheme as VerifierCommitmentScheme;
    use jolt_sdk::jolt_prover_legacy::zkvm::packed::{
        akita_verifier_preprocessing, shared_preprocessing_with_direct_program, AkitaField,
        AkitaPackedProver, AkitaPackedScheme, AkitaScheduleArtifacts, AkitaScheme, AkitaTranscript,
        AkitaVc,
    };
    use jolt_sdk::jolt_prover_legacy::zkvm::preprocessing::JoltSharedPreprocessing;
    use jolt_sdk::jolt_prover_legacy::zkvm::program::ProgramPreprocessing;
    use jolt_sdk::jolt_prover_legacy::zkvm::prover::{JoltCpuProver, JoltProverPreprocessing};

    info!("Starting packed collect_guest_proofs for {}", guest.name());
    let max_trace_length = guest.get_max_trace_length(false);
    let memory_config = MemoryConfig {
        heap_size: 32768u64,
        ..Default::default()
    };
    let mut program = jolt_sdk::host::Program::new(guest.name());
    program.set_func(guest.func());
    program.set_std(false);
    program.set_memory_config(memory_config);
    program.build(target_dir);
    let (bytecode, init_memory_state, _, e_entry) = program.decode();
    let elf_contents = program.get_elf_contents().unwrap();
    let inputs = guest.inputs(proofs);
    let (_, _, _, io_device) = program.trace(&inputs[0], &[], &[]);
    let program_data =
        ProgramPreprocessing::preprocess(bytecode, init_memory_state, e_entry).unwrap();
    // With a chunk count the bytecode is committed: the guest verifier opens a
    // committed bytecode polynomial instead of walking every bytecode row, and
    // its preprocessing carries commitments rather than rows. The opening adds
    // two precommitted objects to the batch, so it pays only once the bytecode
    // pass outgrows that (not for a program this small).
    let schedule_artifacts = AkitaScheduleArtifacts::shared_from_default_directory();
    let (prover_preprocessing, direct_program) = match bytecode_chunk_count {
        Some(chunk_count) => {
            let (shared, prover_data, direct_program) = shared_preprocessing_with_direct_program(
                &schedule_artifacts,
                program_data,
                io_device.memory_layout.clone(),
                max_trace_length,
                chunk_count,
            )
            .expect("packed committed preprocessing");
            (
                JoltProverPreprocessing::new_committed(shared, prover_data, AkitaPackedScheme),
                Some(direct_program),
            )
        }
        None => {
            let shared: JoltSharedPreprocessing<AkitaPackedScheme> = JoltSharedPreprocessing::new(
                program_data,
                io_device.memory_layout.clone(),
                max_trace_length,
            );
            (JoltProverPreprocessing::new(shared), None)
        }
    };

    let mut all_groups_data = Vec::new();
    let n = inputs.len() as u32;
    let mut verifier_preprocessing = None;
    let mut records = Vec::new();
    for input_bytes in inputs {
        let prover: AkitaPackedProver<'_> = JoltCpuProver::gen_from_elf(
            &prover_preprocessing,
            &elf_contents,
            &input_bytes,
            &[],
            &[],
            None,
            None,
            None,
        )
        .unwrap();
        let public_io = prover.program_io.clone();
        let (object_setup, verifier_setup) = <AkitaScheme as VerifierCommitmentScheme>::setup(
            prover.one_hot_trace_setup_params(schedule_artifacts.clone()),
        )
        .unwrap();
        let now = Instant::now();
        let proof = prover
            .prove_packed(&object_setup, None, direct_program.as_ref())
            .unwrap();
        info!("  Packed prove time: {:.3}s", now.elapsed().as_secs_f64());
        let mut preprocessing = akita_verifier_preprocessing(
            &prover_preprocessing,
            verifier_setup,
            direct_program.as_ref(),
        );
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
        // Record against the preprocessing exactly as the guest receives it:
        // the setup's `#[serde(skip)]` backend cache is rebuilt inside the
        // guest's verify, so the host must rebuild it too for the operation
        // sequences to agree.
        let (preprocessing, _): (
            jolt_sdk::jolt_verifier::JoltVerifierPreprocessing<AkitaScheme, AkitaVc>,
            _,
        ) = bincode::serde::decode_from_slice(
            &bincode::serde::encode_to_vec(&preprocessing, bincode::config::standard()).unwrap(),
            bincode::config::standard(),
        )
        .unwrap();
        info!("  Verifying...");
        let is_valid = jolt_sdk::jolt_verifier::verify::<
            AkitaField,
            AkitaScheme,
            AkitaVc,
            AkitaTranscript,
        >(&preprocessing, &public_io, &proof, None)
        .inspect_err(|error| error!("  Verification failed: {error:?}"))
        .is_ok();
        info!("  Verification result: {is_valid}");
        records.push((proof, public_io));
        verifier_preprocessing = Some(preprocessing);
    }
    // The multi-megabyte setup payloads travel out of line, so the guest reads
    // them where they lie instead of copying them out of the bincode record.
    let mut verifier_preprocessing = verifier_preprocessing.unwrap();
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
        preprocess_guest_prover(&mut guest_prog, max_trace_length, bytecode_chunk_count);
    info!("Preprocessing guest verifier...");
    let guest_verifier_preprocessing =
        jolt_sdk::jolt_prover_legacy::zkvm::proof::verifier_preprocessing_from_prover::<
            jolt_sdk::F,
            jolt_sdk::Curve,
            jolt_sdk::PCS,
        >(&guest_prover_preprocessing);
    let guest_verifier_preprocessing: JoltVerifierPreprocessing =
        bincode::serde::decode_from_slice(
            &bincode::serde::encode_to_vec(
                &guest_verifier_preprocessing,
                bincode::config::standard(),
            )
            .unwrap(),
            bincode::config::standard(),
        )
        .unwrap()
        .0;

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

        let mut output_bytes = vec![0; 4096];

        // Running tracing allows things like JOLT_BACKTRACE=1 to work properly
        info!("  Tracing...");
        guest_prog.memory_config.program_size = Some(
            guest_verifier_preprocessing
                .program
                .memory_layout()
                .program_size,
        );
        let (_, _, _, device_io) = guest_prog.trace(&input_bytes, &[], &[]);
        assert!(!device_io.panic, "Guest program panicked during tracing");

        info!("  Proving...");
        let (proof, io_device, _debug): (RV64IMACProof, _, _) = jolt_sdk::guest::prover::prove::<
            jolt_sdk::F,
            jolt_sdk::Curve,
            jolt_sdk::PCS,
            jolt_sdk::ProofTranscript,
        >(
            &guest_prog,
            &input_bytes,
            &[],
            &[],
            None,
            None,
            &mut output_bytes,
            &guest_prover_preprocessing,
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
    let source_path = output_dir.join("embedded_bytes.rs");
    std::fs::write(&source_path, source).unwrap();
    info!("Embedded setup written to {}", bin_path.display());
}

/// Undo [`generate_embedded_bytes`]: an input-mode build must not carry a
/// stale baked setup.
fn clear_embedded_bytes(output_dir: &Path) {
    std::fs::create_dir_all(output_dir).unwrap();
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
    // The verifier guest computes its field arithmetic through the
    // field-inline instructions, so it decodes under the FR profile.
    #[cfg(feature = "field-inline")]
    program.enable_field_inline();
    #[cfg(feature = "akita")]
    program.add_guest_feature("akita");
    program.add_guest_feature("fast-alloc");
    program.add_guest_feature("blake2-inline");
    // The verifier preprocessing is the recursion circuit's own trusted
    // constant: its group elements need no subgroup validation on decode.
    #[cfg(not(feature = "akita"))]
    program.add_guest_feature("trusted-preprocessing");
    program.set_memory_config(memory_config);
    program.build(target_dir);
    let elf_contents = program.get_elf_contents().unwrap();
    if run_config == RunConfig::Trace {
        // Trace through the host program: it decodes under the FR profile the
        // verifier guest needs, and tracing needs no PCS setup.
        info!("  Trace-only mode: Skipping proof generation and verification.");
        // Streamed to disk: a multi-gigacycle verifier trace does not fit in
        // memory as rows.
        let trace_path = std::path::PathBuf::from(format!("/tmp/{}-recursion.trace", guest.name()));
        let (_, io_device) = program.trace_to_file(&input_bytes, &[], &[], &trace_path);
        let _ = std::fs::remove_file(&trace_path);
        let rv = postcard::from_bytes::<u32>(&io_device.outputs).unwrap_or(0);
        info!("  Recursion output (trace-only): {rv}");
        let _ = max_trace_length;
        return;
    }
    let mut recursion = jolt_sdk::guest::program::Program::new(&elf_contents, &memory_config);
    recursion.elf = program.elf;

    if run_config == RunConfig::Trace || run_config == RunConfig::TraceToFile {
        // shorten the max_trace_length for tracing only. Speeds up setup time for tracing purposes.
        max_trace_length = 0;
    }
    #[cfg(feature = "akita")]
    {
        // Packed mode traces only: the recursion guest itself is not proven here.
        let _ = (max_trace_length, recursion);
        info!("  Packed mode supports trace-only runs.");
    }
    #[cfg(not(feature = "akita"))]
    {
        let recursion_prover_preprocessing =
            jolt_sdk::guest::prover::preprocess(&recursion, max_trace_length).unwrap();
        let recursion_verifier_preprocessing =
            jolt_sdk::jolt_prover_legacy::zkvm::proof::verifier_preprocessing_from_prover::<
                jolt_sdk::F,
                jolt_sdk::Curve,
                jolt_sdk::PCS,
            >(&recursion_prover_preprocessing);

        // update program_size in memory_config now that we know it
        recursion.memory_config.program_size = Some(
            recursion_verifier_preprocessing
                .program
                .memory_layout()
                .program_size,
        );

        let mut output_bytes = vec![
            0;
            recursion_verifier_preprocessing
                .program
                .memory_layout()
                .max_output_size as usize
        ];
        match run_config {
            RunConfig::Prove => {
                let (proof, io_device, _debug): (RV64IMACProof, _, _) =
                    jolt_sdk::guest::prover::prove::<
                        jolt_sdk::F,
                        jolt_sdk::Curve,
                        jolt_sdk::PCS,
                        jolt_sdk::ProofTranscript,
                    >(
                        &recursion,
                        &input_bytes,
                        &[],
                        &[],
                        None,
                        None,
                        &mut output_bytes,
                        &recursion_prover_preprocessing,
                    )
                    .expect("prover should produce verifier-native proof");
                let is_valid = jolt_sdk::jolt_verifier::verify::<
                    jolt_sdk::VerifierField,
                    jolt_sdk::VerifierPCS,
                    jolt_sdk::VerifierVC,
                    jolt_sdk::VerifierTranscript,
                >(
                    &recursion_verifier_preprocessing, &io_device, &proof, None
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
