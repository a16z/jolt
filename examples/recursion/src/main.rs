#[path = "../guest/src/bounded.rs"]
mod bounded;
use bounded::RecordLimits;
#[cfg(any(feature = "akita", test))]
#[path = "../guest/src/compiled_rows.rs"]
mod compiled_rows;

use clap::{Parser, Subcommand, ValueEnum};
#[cfg(feature = "akita")]
use jolt_akita::{
    AkitaChunkProfile, AkitaField, AkitaProverTranscript, AkitaScheduleArtifacts, AkitaScheme,
    AkitaVerifierTranscript, TrustedBytes,
};
#[cfg(feature = "akita")]
use jolt_claims::protocols::jolt::JoltAdviceKind;
use jolt_field::Ring;
use jolt_inlines_blake2::Blake2b;
#[cfg(feature = "ntt-inline")]
use jolt_inlines_ntt as _;
use jolt_poly::{CompressedPoly, UnivariatePoly};
use jolt_riscv::JoltInstructionRow;
#[cfg(feature = "akita")]
use jolt_sdk::host::JoltProgramSource;
use jolt_sdk::host::Program;
#[cfg(feature = "akita")]
use jolt_sdk::jolt_prover::akita::preprocessing::AkitaVc;
#[cfg(feature = "akita")]
use jolt_sdk::jolt_prover::akita::witness::commit_advice;
use jolt_sdk::jolt_verifier::preprocessing::ProgramPreprocessing as VerifierProgramPreprocessing;
use jolt_sdk::jolt_verifier::proof::JoltProofClaims;
#[cfg(feature = "akita")]
use jolt_sdk::jolt_verifier::{JoltProof, JoltVerifierPreprocessing};
use jolt_sdk::{JoltDevice, MemoryConfig, MemoryLayout, ProgramPreprocessingMode};
#[cfg(not(feature = "akita"))]
use jolt_sdk::{
    JoltProverPreprocessing, JoltVerifierPreprocessing, RV64IMACProof, VerifierField,
    VerifierTranscript,
};
use jolt_sumcheck::{ClearProof, SumcheckProof};
use serde::{de::DeserializeOwned, Serialize};
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

#[cfg(feature = "akita")]
type GuestField = AkitaField;
#[cfg(not(feature = "akita"))]
type GuestField = VerifierField;

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
    let bytes = read_raw(buffer, offset)?;
    let (value, consumed) = bincode::serde::decode_from_slice(bytes, bincode::config::standard())
        .map_err(|error| error.to_string())?;
    assert_eq!(consumed, bytes.len(), "record decoder left trailing bytes");
    Ok(value)
}

impl RecordLimits {
    // Image construction already trusts these files. Runtime input never sets limits.
    fn read_record<T: DeserializeOwned>(
        &mut self,
        buffer: &[u8],
        offset: &mut usize,
    ) -> Result<T, String> {
        let bytes = read_raw(buffer, offset)?;
        let mut remaining = usize::MAX;
        let (value, consumed) =
            bounded::decode(bytes, &mut remaining).map_err(|error| error.to_string())?;
        assert_eq!(consumed, bytes.len(), "record decoder left trailing bytes");
        let charged = usize::MAX - remaining;
        // Reserve headroom for other honest proofs with the same image's setup.
        let allocation = charged
            .checked_mul(2)
            .and_then(usize::checked_next_power_of_two)
            .ok_or("record allocation budget overflow")?;
        self.allocation = self.allocation.max(allocation);
        self.bytes = self.bytes.max(
            bytes
                .len()
                .checked_next_power_of_two()
                .ok_or("record byte limit overflow")?,
        );
        info!(
            "Record: {} encoded bytes, {charged} charged bytes",
            bytes.len()
        );
        Ok(value)
    }
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
    /// Write a typed negative fixture from an existing proof stream.
    Tamper {
        #[arg(long)]
        example: String,
        #[arg(long)]
        workdir: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, value_enum)]
        target: TamperTarget,
        /// Zero-based proof index to modify; other proofs remain valid.
        #[arg(long, default_value_t = 0)]
        proof_index: u32,
    },
    /// Check malformed transport inputs against one compiled verifier image.
    CheckInputFraming {
        #[arg(long)]
        example: String,
        #[arg(long)]
        workdir: PathBuf,
        #[arg(long)]
        embed: bool,
        /// Supply the setup as trusted advice and the proofs as untrusted advice
        #[cfg(feature = "akita")]
        #[arg(long, conflicts_with = "embed")]
        advice: bool,
    },
    /// Compare compiled field kernels with native software results under the tracer.
    #[cfg(feature = "field-inline")]
    CheckFieldKernels,

    /// Write a parseable proof stream with one bit of its first Akita opening proof flipped.
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
        #[arg(long)]
        committed_bytecode: bool,
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
        /// Supply the setup as trusted advice and the proofs as untrusted
        /// advice; the input carries only the proof count and devices
        #[cfg(feature = "akita")]
        #[arg(long, conflicts_with = "embed")]
        advice: bool,
        /// Store trace rows on disk instead of executing without row storage
        #[arg(short = 'd', long = "disk", default_value_t = false)]
        trace_to_file: bool,
        /// Required guest verdict; a panic is neither verdict
        #[arg(long, value_enum, default_value = "accept")]
        expect: ExpectedVerification,
    },
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum GuestProgram {
    Fibonacci,
    Muldiv,
}

#[derive(Clone, PartialEq, Eq)]
enum RunConfig {
    Prove,
    Trace(ExpectedVerification),
    TraceToFile(ExpectedVerification),
    CheckInputFraming,
}

/// Where the recursion guest reads the inner verifier setup from.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SetupTransport {
    /// Baked into the guest image; the input carries the proofs and devices.
    Image,
    /// The input, ahead of the proofs and devices; the image pins its digest.
    Input,
    /// Trusted advice, with the proofs as untrusted advice; the input carries
    /// only the statements, the proof count and devices.
    #[cfg(feature = "akita")]
    Advice,
}

impl SetupTransport {
    fn guest_function(self) -> &'static str {
        match self {
            Self::Image | Self::Input => "verify",
            #[cfg(feature = "akita")]
            Self::Advice => "verify_advice",
        }
    }
}

/// Record streams for the guest's input and advice regions. The input region
/// always exists; an advice region exists only for a `Some` stream.
#[derive(Clone, Default)]
struct GuestStreams {
    input: Vec<u8>,
    untrusted_advice: Option<Vec<u8>>,
    trusted_advice: Option<Vec<u8>>,
}

/// [`GuestStreams`] as the tracer writes them into the guest's regions.
#[derive(Clone)]
struct GuestRegions {
    input: Vec<u8>,
    untrusted_advice: Vec<u8>,
    trusted_advice: Vec<u8>,
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

    fn get_memory_config(&self, transport: SetupTransport, streams: &GuestStreams) -> MemoryConfig {
        // Eight bytes for the pad prefix, at most seven pad bytes, ten for
        // postcard's length, and one byte for the strict size bound `frame`
        // asserts.
        let capacity = |stream: &[u8]| {
            stream
                .len()
                .checked_add(26)
                .and_then(usize::checked_next_power_of_two)
                .and_then(|size| u64::try_from(size).ok())
                .expect("recursion region size overflow")
        };
        MemoryConfig {
            max_input_size: capacity(&streams.input),
            max_output_size: 4096,
            max_untrusted_advice_size: streams.untrusted_advice.as_deref().map_or(0, capacity),
            max_trusted_advice_size: streams.trusted_advice.as_deref().map_or(0, capacity),
            heap_size: if *self == GuestProgram::Muldiv && transport == SetupTransport::Input {
                33554432
            } else {
                134217728
            },
            stack_size: 33554432,
            program_size: None,
        }
    }

    fn inner_max_trace_length(&self) -> usize {
        match self {
            GuestProgram::Fibonacci => 5000000,
            GuestProgram::Muldiv => 3000000,
        }
    }

    fn recursion_max_trace_length(&self, transport: SetupTransport) -> usize {
        if transport == SetupTransport::Input {
            return self.inner_max_trace_length();
        }
        match self {
            GuestProgram::Fibonacci => 67108864,
            GuestProgram::Muldiv => 800000,
        }
    }
}

fn generate_provable_macro(
    memory_config: &MemoryConfig,
    max_trace_length: usize,
    output_dir: &Path,
    limits: RecordLimits,
) {
    let macro_content = format!(
        r#"const RECORD_LIMITS: bounded::RecordLimits = bounded::RecordLimits {{
    bytes: {record_bytes},
    allocation: {record_allocation},
}};
macro_rules! provable_with_config {{
    ($item: item) => {{
        #[jolt::provable(
            input_error = 0,
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
        max_trace_length,
        record_bytes = limits.bytes,
        record_allocation = limits.allocation,
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
    limits: RecordLimits,
}

fn check_data_integrity(all_groups_data: &[u8]) -> StreamLayout {
    info!("Checking data integrity...");

    let mut offset = 0;
    let mut limits = RecordLimits::default();
    let verifier_preprocessing: GuestVerifierPreprocessing =
        limits.read_record(all_groups_data, &mut offset).unwrap();
    let verifier_bytes =
        bincode::serde::encode_to_vec(&verifier_preprocessing, bincode::config::standard())
            .unwrap();
    info!(
        "✓ Verifier preprocessing deserialized successfully ({} bytes)",
        verifier_bytes.len()
    );
    let payload_count: u32 = limits.read_record(all_groups_data, &mut offset).unwrap();
    for i in 0..payload_count {
        let payload = read_raw(all_groups_data, &mut offset).expect("decode setup payload");
        info!("Setup payload {i}: {} bytes", payload.len());
    }
    let setup_len = offset;

    let n: u32 = limits.read_record(all_groups_data, &mut offset).unwrap();
    info!("✓ Number of proofs deserialized: {n}");

    for i in 0..n {
        let _: GuestProof = limits
            .read_record(all_groups_data, &mut offset)
            .expect("decode proof");
        let _: JoltDevice = limits
            .read_record(all_groups_data, &mut offset)
            .expect("decode device");
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
        limits,
    }
}

#[cfg(not(feature = "akita"))]
fn preprocess_guest_prover(
    guest_prog: &mut Program,
    memory_config: MemoryConfig,
    max_trace_length: usize,
    mode: ProgramPreprocessingMode,
) -> JoltProverPreprocessing {
    jolt_sdk::preprocess_program(guest_prog, memory_config, max_trace_length, mode).unwrap()
}

/// The Akita schedule catalogs for the inner proofs and the setup commitment.
#[cfg(feature = "akita")]
fn schedule_artifacts() -> Arc<AkitaScheduleArtifacts> {
    AkitaScheduleArtifacts::shared_from_default_directory(AkitaChunkProfile::new(8, 2))
}

/// The packed (Akita) inner proofs use the modular prover over fp128 and the
/// packed verifier preprocessing from
/// the host's own verification of each proof.
#[cfg(feature = "akita")]
fn collect_guest_proofs(
    guest: GuestProgram,
    target_dir: &str,
    mode: ProgramPreprocessingMode,
    proofs: usize,
) -> Vec<u8> {
    use jolt_program::execution::{ExecutionBackend, TraceInputs};
    use jolt_program::preprocess::JoltProgramPreprocessing;
    use jolt_sdk::jolt_prover::akita;
    use jolt_sdk::jolt_prover::akita::preprocessing;
    use jolt_sdk::jolt_prover::{JoltBackend, ProverConfig};
    use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};
    use tracer::execution_backend::TracerBackend;

    let max_trace_length = guest.inner_max_trace_length();
    let mut memory_config = MemoryConfig {
        heap_size: 32768u64,
        ..Default::default()
    };
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
    let program_data = Arc::new(
        JoltProgramPreprocessing::new(
            jolt_program.expanded_bytecode.clone(),
            jolt_program.memory_init.clone(),
            io_device.memory_layout.clone(),
            jolt_program.entry_address,
            max_trace_length,
            program.instruction_profile(),
        )
        .expect("inner program preprocessing"),
    );
    let schedule_artifacts = schedule_artifacts();
    let mut all_groups_data = Vec::new();
    let n = inputs.len() as u32;
    let mut verifier_preprocessing = None;
    let mut records = Vec::new();
    for input_bytes in inputs {
        let trace = TracerBackend::new()
            .trace(
                &jolt_program,
                TraceInputs::new(input_bytes, Vec::new(), Vec::new(), memory_config),
            )
            .expect("trace inner program");
        let config = ProverConfig::derive_from_dimensions::<AkitaField>(
            trace.dimensions,
            &program_data.memory_layout,
            program_data.ram.min_bytecode_address,
            program_data.ram.bytecode_words.len(),
            max_trace_length,
        )
        .expect("inner proof configuration");
        let prover_preprocessing = match mode {
            ProgramPreprocessingMode::Committed => preprocessing::preprocess_committed(
                &schedule_artifacts,
                program_data.as_ref().clone(),
                &config,
            ),
            ProgramPreprocessingMode::Full => preprocessing::preprocess_full(
                &schedule_artifacts,
                program_data.as_ref().clone(),
                &config,
            ),
        }
        .expect("packed preprocessing");
        let public_io = trace.device.clone();
        let witness = TraceBackend::new(
            JoltVmWitnessConfig::new(
                config.trace_length.ilog2() as usize,
                config.ram_K,
                config.one_hot_config,
            ),
            JoltVmWitnessInputs::new(&jolt_program, &program_data, trace),
        );
        #[cfg(feature = "field-inline")]
        let witness = witness.with_field_inline().expect("field witness");
        let now = Instant::now();
        let proof = akita::prove::<AkitaField, AkitaScheme, AkitaVc, AkitaProverTranscript>(
            JoltBackend::optimized().with_witness(&witness),
            &prover_preprocessing,
            &config,
            None,
            &public_io,
        )
        .expect("packed proof");
        info!("  Packed prove time: {:.3}s", now.elapsed().as_secs_f64());
        let preprocessing = prover_preprocessing.verifier;
        records.push((proof, public_io));
        verifier_preprocessing = Some(preprocessing);
    }
    // The guest's setup is a trusted constant: carry the expanded backend key,
    // a catalog view, and the terminal matrix NTT for the row the proofs
    // selected instead of re-deriving them in the guest.
    let mut verifier_preprocessing = verifier_preprocessing.unwrap();
    let row_digest = records[0].0.joint_opening_proof.schedule_row_digest();
    assert!(
        records
            .iter()
            .all(|(proof, _)| proof.joint_opening_proof.schedule_row_digest() == row_digest),
        "a prepared guest verifier serves one schedule row"
    );
    verifier_preprocessing
        .pcs_setup
        .prepare_verifier(row_digest)
        .expect("prepare the Akita guest verifier");
    // Verify every proof against exactly what the guest will hold: the
    // transported record, with its multi-megabyte payloads detached out of
    // line and attached back in place, as the guest reads them.
    let (mut verifier_preprocessing, _): (GuestVerifierPreprocessing, _) =
        bincode::serde::decode_from_slice(
            &bincode::serde::encode_to_vec(&verifier_preprocessing, bincode::config::standard())
                .unwrap(),
            bincode::config::standard(),
        )
        .unwrap();
    let payloads = verifier_preprocessing
        .pcs_setup
        .detach_prepared_payloads()
        .expect("detach prepared Akita payloads");
    let mut attached = verifier_preprocessing.clone();
    let bodies: Vec<TrustedBytes> = payloads
        .iter()
        // SAFETY: the bodies were detached from this host's own prepared setup.
        .map(|payload| unsafe { TrustedBytes::new(Box::leak(payload.clone().into_boxed_slice())) })
        .collect();
    attached
        .pcs_setup
        .attach_prepared_payloads(&bodies)
        .expect("attach prepared Akita payloads");
    for (proof, public_io) in &records {
        jolt_sdk::jolt_verifier::verify::<
            AkitaField,
            AkitaScheme,
            AkitaVc,
            AkitaVerifierTranscript<'_>,
        >(&attached, public_io, proof, None)
        .expect("verify proof against prepared guest setup");
        info!("  Verification result: true");
    }
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
    mode: ProgramPreprocessingMode,
    proofs: usize,
) -> Vec<u8> {
    info!("Starting collect_guest_proofs for {}", guest.name());
    let max_trace_length = guest.inner_max_trace_length();

    // This should match the example being run, it can cause layout issues if the guest's macro and our assumption here differ
    let memory_config = MemoryConfig {
        heap_size: 32768u64,
        ..Default::default()
    };

    info!("Creating program...");
    let mut program = Program::new(guest.name());
    #[cfg(feature = "field-inline")]
    program.enable_field_inline();
    program.set_func(guest.func());
    program.set_std(false);
    program.set_memory_config(memory_config);
    info!("Building program...");
    program.build(target_dir);
    info!("Getting ELF contents...");
    info!("Preprocessing guest prover...");
    let guest_prover_preprocessing =
        preprocess_guest_prover(&mut program, memory_config, max_trace_length, mode);
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
            VerifierTranscript<'_>,
        >(&guest_verifier_preprocessing, &io_device, &proof, None)
        .is_ok();
        info!("  Verification result: {is_valid}");
    }
    info!("Total prove time: {total_prove_time:.3}s");
    info!("Total data size: {} bytes", all_groups_data.len());
    all_groups_data
}

/// A setup section with the full program's bytecode rows compiled out of its
/// preprocessing record.
struct CompiledSetup<'a> {
    /// The record without the rows, then the preprocessing digest computed
    /// over the complete program. The image or the committed trusted advice
    /// fixes both, so the guest assembles the preprocessing from them instead
    /// of re-encoding and hashing the program.
    preprocessing: Vec<u8>,
    rows: Vec<JoltInstructionRow>,
    /// The payload count and detached payloads, unchanged.
    payloads: &'a [u8],
}

fn compile_setup(setup_section: &[u8]) -> CompiledSetup<'_> {
    let mut offset = 0;
    let mut preprocessing: GuestVerifierPreprocessing =
        read_record(setup_section, &mut offset).unwrap();
    let rows = match &mut preprocessing.program {
        VerifierProgramPreprocessing::Full(program) => {
            std::mem::take(&mut Arc::make_mut(program).bytecode.bytecode)
        }
        VerifierProgramPreprocessing::Committed(_) => Vec::new(),
    };
    let digest = preprocessing.preprocessing_digest;
    let mut records = Vec::new();
    push_record(&mut records, &preprocessing);
    push_record(&mut records, &digest);
    CompiledSetup {
        preprocessing: records,
        rows,
        payloads: &setup_section[offset..],
    }
}

/// Bake the setup section of the stream into the guest image: `embedded.bin`
/// next to the guest sources, included 8-byte aligned by the generated
/// `embedded_bytes.rs`. The guest then takes its verifier setup from its own
/// image and only the proofs from its input.
fn generate_embedded_bytes(guest: GuestProgram, setup_section: &[u8], output_dir: &Path) {
    let CompiledSetup {
        preprocessing,
        rows: bytecode,
        payloads,
    } = compile_setup(setup_section);

    info!(
        "Generating embedded setup for {} guest program ({} bytes)...",
        guest.name(),
        preprocessing.len() + payloads.len()
    );
    let mut image = Vec::with_capacity(preprocessing.len() + payloads.len() + RECORD_ALIGN);
    // The static is 8-byte aligned, so no leading pad is needed.
    push_alignment_pad(&mut image, 0);
    image.extend_from_slice(&preprocessing);
    image.extend_from_slice(payloads);

    std::fs::create_dir_all(output_dir).unwrap();
    let bin_path = output_dir.join("embedded.bin");
    std::fs::write(&bin_path, &image).unwrap();
    let source = format!(
        "// Generated by the recursion host: the verifier setup baked into this image.\n\
         #[repr(C, align(8))]\n\
         struct Aligned<T: ?Sized>(T);\n\
         static ALIGNED: Aligned<[u8; {len}]> = Aligned(*include_bytes!(\"embedded.bin\"));\n\
         pub static EMBEDDED_BYTES: &[u8] = &ALIGNED.0;\n\
         pub const SETUP_DIGEST: Option<[u8; 64]> = None;\n",
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

/// Undo [`generate_embedded_bytes`]: an input-mode or advice-mode build must
/// not carry a stale baked setup. Input mode pins the digest of its setup
/// section; an advice-mode image bakes no setup data.
fn clear_embedded_bytes(output_dir: &Path, setup_digest: Option<[u8; 64]>) {
    std::fs::create_dir_all(output_dir).unwrap();
    std::fs::write(
        output_dir.join("embedded_bytecode.rs"),
        render_embedded_bytecode(&[]),
    )
    .unwrap();
    std::fs::write(
        output_dir.join("embedded_bytes.rs"),
        format!(
            "pub static EMBEDDED_BYTES: &[u8] = &[];\npub const SETUP_DIGEST: Option<[u8; 64]> = {setup_digest:?};\n",
        ),
    )
    .unwrap();
    let _ = std::fs::remove_file(output_dir.join("embedded.bin"));
}

/// The streams for advice mode. The trusted advice carries the compiled
/// setup with its rows in [`compiled_rows`]' fixed-width layout. The proofs
/// become untrusted advice, and the input keeps the statements: the proof
/// count, then each proof's device.
#[cfg(feature = "akita")]
fn advice_streams(setup_section: &[u8], proof_section: &[u8]) -> GuestStreams {
    let CompiledSetup {
        preprocessing,
        rows,
        payloads,
    } = compile_setup(setup_section);
    let words = compiled_rows::encode(&rows);
    assert_eq!(
        compiled_rows::decode(&words).as_ref(),
        Some(&rows),
        "compiled bytecode rows must decode to the setup's rows"
    );
    let mut trusted_advice = preprocessing;
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    push_raw(&mut trusted_advice, &bytes);
    trusted_advice.extend_from_slice(payloads);

    let mut offset = 0;
    let count: u32 = read_record(proof_section, &mut offset).unwrap();
    let mut input = Vec::new();
    push_record(&mut input, &count);
    let mut untrusted_advice = Vec::new();
    for _ in 0..count {
        push_raw(
            &mut untrusted_advice,
            read_raw(proof_section, &mut offset).unwrap(),
        );
        push_raw(&mut input, read_raw(proof_section, &mut offset).unwrap());
    }
    assert_eq!(offset, proof_section.len(), "trailing proof stream bytes");
    GuestStreams {
        input,
        untrusted_advice: Some(untrusted_advice),
        trusted_advice: Some(trusted_advice),
    }
}

/// Frame `stream` for a region starting at `region_start`: `postcard`
/// prefixes the byte slice with a varint length, so a leading pad record
/// lands every body 8-byte aligned at the address the guest reads it from.
fn frame_guest_region(stream: &[u8], region_start: u64) -> Vec<u8> {
    fn varint_len(value: usize) -> usize {
        postcard::to_stdvec(&value).unwrap().len()
    }
    let region_start = usize::try_from(region_start).unwrap();
    let mut pad = 0;
    // The varint width depends on the total length, which depends on the pad;
    // iterate to the fixpoint (the width changes only at 2^(7k) boundaries).
    for _ in 0..4 {
        let total = 8 + pad + stream.len();
        let body_start = region_start + varint_len(total) + 8;
        let next = body_start.next_multiple_of(RECORD_ALIGN) - body_start;
        if next == pad {
            break;
        }
        pad = next;
    }
    let mut framed = Vec::with_capacity(8 + pad + stream.len());
    push_alignment_pad(&mut framed, pad);
    framed.extend_from_slice(stream);
    assert_eq!(
        (region_start + varint_len(framed.len()) + 8 + pad) % RECORD_ALIGN,
        0,
        "guest region bodies must be 8-byte aligned"
    );
    postcard::to_stdvec(&framed.as_slice()).unwrap()
}

impl GuestStreams {
    fn frame(&self, memory_config: &MemoryConfig) -> GuestRegions {
        // The I/O region does not depend on the program size, which the layout
        // insists on knowing (as the SDK's own macro does, pass a placeholder).
        let layout = MemoryLayout::new(&MemoryConfig {
            program_size: Some(0),
            ..*memory_config
        });
        let advice = |stream: Option<&[u8]>, region_start| {
            stream.map_or_else(Vec::new, |stream| frame_guest_region(stream, region_start))
        };
        let regions = GuestRegions {
            input: frame_guest_region(&self.input, layout.input_start),
            untrusted_advice: advice(
                self.untrusted_advice.as_deref(),
                layout.untrusted_advice_start,
            ),
            trusted_advice: advice(self.trusted_advice.as_deref(), layout.trusted_advice_start),
        };
        for (name, bytes, capacity) in [
            ("input", &regions.input, layout.max_input_size),
            (
                "untrusted advice",
                &regions.untrusted_advice,
                layout.max_untrusted_advice_size,
            ),
            (
                "trusted advice",
                &regions.trusted_advice,
                layout.max_trusted_advice_size,
            ),
        ] {
            assert!(
                (bytes.len() as u64) < capacity || (capacity == 0 && bytes.is_empty()),
                "{name} does not fit its region"
            );
        }
        regions
    }
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
    mode: ProgramPreprocessingMode,
    proofs: usize,
) {
    info!(
        "Generating {proofs} proof(s) for {} guest program...",
        guest.name()
    );

    let target_dir = "/tmp/jolt-guest-targets";

    let all_groups_data = collect_guest_proofs(guest, target_dir, mode, proofs);

    save_proof_data(guest, &all_groups_data, workdir);

    info!("Proof generation completed for {}", guest.name());
}

fn decode_verifier_output(bytes: &[u8]) -> u32 {
    let (output, remaining) =
        postcard::take_from_bytes::<u32>(bytes).expect("decode verifier output");
    let canonical = postcard::to_stdvec(&output).expect("encode verifier output");
    assert!(
        bytes.starts_with(&canonical) && bytes.len() - remaining.len() == canonical.len(),
        "noncanonical verifier output"
    );
    // The tracer lowers narrow stores to aligned 8-byte read-modify-writes.
    let padded_len = canonical
        .len()
        .checked_next_multiple_of(8)
        .expect("verifier output length overflow");
    assert!(
        (bytes.len() == canonical.len() || bytes.len() == padded_len)
            && remaining.iter().all(|byte| *byte == 0),
        "invalid verifier output padding"
    );
    output
}

fn configured_recursion_program(memory_config: MemoryConfig, func: &str) -> Program {
    let mut program = Program::new("recursion-guest");
    program.set_func(func);
    program.set_std(true);
    // The verifier guest computes its field arithmetic through the
    // field-inline instructions, so it decodes under the field-inline profile, and
    // routes the one field the tracer's field-inline unit computes in.
    #[cfg(feature = "field-inline")]
    {
        program.enable_field_inline();
        program.add_guest_feature(if cfg!(feature = "akita") {
            "field-inline-fp128"
        } else {
            "field-inline-bn254"
        });
    }
    #[cfg(feature = "akita")]
    program.add_guest_feature("akita");
    #[cfg(feature = "ntt-inline")]
    program.add_guest_feature("ntt-inline");
    program.add_guest_feature("fast-alloc");
    program.add_guest_feature("hash-inlines");
    program.set_memory_config(memory_config);
    program
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ExpectedVerification {
    Accept,
    Reject,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum TamperTarget {
    Sumcheck,
    Claim,
    Output,
    Panic,
    Empty,
    #[cfg(feature = "akita")]
    Opening,
}

fn tamper_proof(
    guest: GuestProgram,
    workdir: &Path,
    output: &Path,
    target: TamperTarget,
    proof_index: u32,
) {
    let bytes = load_proof_data(guest, workdir);
    let mut offset = 0;
    let _: GuestVerifierPreprocessing = read_record(&bytes, &mut offset).unwrap();
    let payloads: u32 = read_record(&bytes, &mut offset).unwrap();
    for _ in 0..payloads {
        let _ = read_raw(&bytes, &mut offset).unwrap();
    }
    let count_offset = offset;
    let count: u32 = read_record(&bytes, &mut offset).unwrap();
    assert!(count > 0, "tamper fixture requires a proof");
    assert!(proof_index < count, "tamper proof index is out of range");
    let output_count = if target == TamperTarget::Empty {
        0
    } else {
        count
    };
    let mut tampered = bytes[..count_offset].to_vec();
    push_record(&mut tampered, &output_count);
    for index in 0..count {
        let mut proof: GuestProof = read_record(&bytes, &mut offset).unwrap();
        let mut device: JoltDevice = read_record(&bytes, &mut offset).unwrap();
        if target == TamperTarget::Empty {
            continue;
        }
        if index == proof_index {
            match target {
                TamperTarget::Sumcheck => match &mut proof.stages.stage1_sumcheck_proof {
                    SumcheckProof::Clear(ClearProof::Full(sumcheck)) => {
                        let round = sumcheck.round_polynomials.first_mut().expect("first round");
                        let mut coefficients = round.coefficients().to_vec();
                        *coefficients.first_mut().expect("round coefficient") +=
                            GuestField::from_u64(1);
                        *round = UnivariatePoly::new(coefficients);
                    }
                    SumcheckProof::Clear(ClearProof::Compressed(sumcheck)) => {
                        let round = sumcheck.round_polynomials.first_mut().expect("first round");
                        let mut coefficients = round.coeffs_except_linear_term().to_vec();
                        *coefficients.first_mut().expect("round coefficient") +=
                            GuestField::from_u64(1);
                        *round = CompressedPoly::new(coefficients);
                    }
                    SumcheckProof::Committed(_) => panic!("tamper requires a clear proof"),
                },
                TamperTarget::Claim => {
                    let JoltProofClaims::Clear(claims) = &mut proof.claims else {
                        panic!("tamper requires clear claims");
                    };
                    claims.stage1.outer.outer_remainder.left_instruction_input +=
                        GuestField::from_u64(1);
                }
                TamperTarget::Output => {
                    *device.outputs.first_mut().expect("device output byte") ^= 1;
                }
                TamperTarget::Panic => device.panic = !device.panic,
                TamperTarget::Empty => unreachable!(),
                #[cfg(feature = "akita")]
                TamperTarget::Opening => {
                    // Akita's opening is the argument string's suffix: nothing
                    // is written after it.
                    *proof.narg.last_mut().expect("Akita argument string") ^= 1;
                }
            }
        }
        push_record(&mut tampered, &proof);
        push_record(&mut tampered, &device);
    }
    assert_eq!(offset, bytes.len(), "trailing proof stream bytes");
    // A successful typed decode is distinct from the guest's verification result.
    assert_eq!(check_data_integrity(&tampered).proof_count, output_count);
    std::fs::create_dir(output).expect("fresh tampered proof directory");
    save_proof_data(guest, &tampered, output);
}

/// Transport inputs `check-input-framing` runs against one image, with the
/// verdict each must produce: malformed framing rejects without a panic.
fn framing_cases(
    streams: &GuestStreams,
    honest: &GuestRegions,
    memory_config: &MemoryConfig,
) -> Vec<(&'static str, GuestRegions, u32)> {
    let with_input = |input: Vec<u8>| GuestRegions {
        input,
        ..honest.clone()
    };
    let oversized = postcard::to_stdvec(&memory_config.max_input_size).unwrap();
    let mut suffix = honest.input.clone();
    suffix.push(1);
    let mut zero_padding = honest.input.clone();
    zero_padding.resize(memory_config.max_input_size as usize, 0);
    let mut last_byte = zero_padding.clone();
    *last_byte.last_mut().unwrap() = 1;
    let mut cases = vec![
        ("honest", honest.clone(), 1),
        ("overflowing length", with_input(vec![0xff; 10]), 0),
        ("oversize length", with_input(oversized), 0),
        ("nonzero suffix", with_input(suffix), 0),
        ("zero transport padding", with_input(zero_padding), 1),
        ("nonzero final capacity byte", with_input(last_byte), 0),
    ];
    if let Some(proofs) = &streams.untrusted_advice {
        let with_proofs = |untrusted_advice: Vec<u8>| GuestRegions {
            untrusted_advice,
            ..honest.clone()
        };
        let with_stream = |proofs: Vec<u8>| {
            GuestStreams {
                untrusted_advice: Some(proofs),
                ..streams.clone()
            }
            .frame(memory_config)
        };
        let oversized = postcard::to_stdvec(&memory_config.max_untrusted_advice_size).unwrap();
        let mut trailing = proofs.clone();
        push_raw(&mut trailing, &[]);
        cases.extend([
            ("overflowing advice length", with_proofs(vec![0xff; 10]), 0),
            ("oversize advice length", with_proofs(oversized), 0),
            ("missing proof record", with_stream(Vec::new()), 0),
            ("trailing proof record", with_stream(trailing), 0),
        ]);
    }
    cases
}

fn run_recursion_proof(
    guest: GuestProgram,
    run_config: RunConfig,
    transport: SetupTransport,
    streams: &GuestStreams,
    regions: GuestRegions,
    memory_config: MemoryConfig,
    max_trace_length: usize,
) {
    let target_dir = "/tmp/jolt-guest-targets";

    let mut program = configured_recursion_program(memory_config, transport.guest_function());
    program.build(target_dir);
    match run_config {
        RunConfig::CheckInputFraming => {
            // Build once: malformed inputs must not change the image or its capacity.
            for (name, regions, expected) in framing_cases(streams, &regions, &memory_config) {
                let (rows, device) = program.execute_with_output(
                    &regions.input,
                    &regions.untrusted_advice,
                    &regions.trusted_advice,
                );
                assert!(!device.panic, "{name}: recursion verifier guest panicked");
                assert_eq!(decode_verifier_output(&device.outputs), expected, "{name}");
                info!("  {name}: verdict {expected}, trace length {rows}");
            }
        }
        RunConfig::Trace(expected) | RunConfig::TraceToFile(expected) => {
            let io_device = if matches!(run_config, RunConfig::Trace(_)) {
                let (rows, device) = program.execute_with_output(
                    &regions.input,
                    &regions.untrusted_advice,
                    &regions.trusted_advice,
                );
                info!("  trace length: {rows}");
                device
            } else {
                let trace_path = PathBuf::from(format!("/tmp/{}-recursion.trace", guest.name()));
                program
                    .trace_to_file(
                        &regions.input,
                        &regions.untrusted_advice,
                        &regions.trusted_advice,
                        &trace_path,
                    )
                    .1
            };
            assert!(!io_device.panic, "Recursion verifier guest panicked");
            let rv = decode_verifier_output(&io_device.outputs);
            info!("  Recursion output (trace-only): {rv}");
            let expected = match expected {
                ExpectedVerification::Accept => 1,
                ExpectedVerification::Reject => 0,
            };
            assert_eq!(rv, expected, "unexpected recursion verifier verdict");
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
                    &mut program,
                    memory_config,
                    max_trace_length,
                    ProgramPreprocessingMode::Full,
                );
                let verifier = jolt_sdk::verifier_preprocessing_from_prover(&preprocessing);
                let (proof, io_device): (RV64IMACProof, _) = jolt_sdk::prove_program(
                    &program,
                    &preprocessing,
                    &regions.input,
                    &[],
                    &[],
                    None,
                    None,
                    None,
                )
                .expect("outer recursion proof");
                jolt_sdk::jolt_verifier::verify::<
                    jolt_sdk::VerifierField,
                    jolt_sdk::VerifierPCS,
                    jolt_sdk::VerifierVC,
                    VerifierTranscript<'_>,
                >(&verifier, &io_device, &proof, None)
                .expect("verify outer recursion proof");
                let rv = decode_verifier_output(&io_device.outputs);
                info!("  Recursion verification result: {rv}");
            }
        }
    }
}

#[cfg(feature = "akita")]
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn verify_proofs(
    guest: GuestProgram,
    transport: SetupTransport,
    workdir: &Path,
    output_dir: &Path,
    run_config: RunConfig,
) {
    info!("Verifying proofs for {} guest program...", guest.name());

    let all_groups_data = load_proof_data(guest, workdir);
    let layout = check_data_integrity(&all_groups_data);
    let (setup_section, proof_section) = all_groups_data.split_at(layout.setup_len);

    let streams = match transport {
        SetupTransport::Image => {
            info!(
                "Running {} recursion with the setup baked into the guest image...",
                guest.name()
            );
            generate_embedded_bytes(guest, setup_section, output_dir);
            GuestStreams {
                input: proof_section.to_vec(),
                ..GuestStreams::default()
            }
        }
        SetupTransport::Input => {
            info!("Running {} recursion with input data...", guest.name());
            clear_embedded_bytes(output_dir, Some(Blake2b::digest(setup_section)));
            GuestStreams {
                input: all_groups_data.clone(),
                ..GuestStreams::default()
            }
        }
        #[cfg(feature = "akita")]
        SetupTransport::Advice => {
            info!(
                "Running {} recursion with the setup as trusted advice...",
                guest.name()
            );
            clear_embedded_bytes(output_dir, None);
            advice_streams(setup_section, proof_section)
        }
    };
    let mut memory_config = guest.get_memory_config(transport, &streams);
    // Setup, proof, and device coexist; retain space for geometric growth and verification.
    let decode_heap = layout
        .limits
        .allocation
        .checked_mul(8)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .expect("recursion decode heap size overflow");
    memory_config.heap_size = memory_config.heap_size.max(decode_heap);
    let max_trace_length = guest.recursion_max_trace_length(transport);
    generate_provable_macro(&memory_config, max_trace_length, output_dir, layout.limits);
    let regions = streams.frame(&memory_config);
    info!(
        "Serialized input size: {} bytes, untrusted advice: {} bytes, trusted advice: {} bytes ({} proofs)",
        regions.input.len(),
        regions.untrusted_advice.len(),
        regions.trusted_advice.len(),
        layout.proof_count
    );
    // An outer proof of an advice-mode run binds the setup through this
    // commitment, which the consumer pins as `trusted_advice_commitment`, and
    // its public input carries the inner statements.
    #[cfg(feature = "akita")]
    if transport == SetupTransport::Advice {
        let commitment = commit_advice::<AkitaScheme>(
            &schedule_artifacts(),
            JoltAdviceKind::Trusted,
            &regions.trusted_advice,
            usize::try_from(memory_config.max_trusted_advice_size).unwrap(),
        )
        .expect("commit the setup as trusted advice")
        .commitment;
        let encoded =
            bincode::serde::encode_to_vec(&commitment, bincode::config::standard()).unwrap();
        info!(
            "Setup commitment over {} of {} trusted-advice bytes: {}",
            regions.trusted_advice.len(),
            memory_config.max_trusted_advice_size,
            hex(&encoded)
        );
        let mut offset = 0;
        let count: u32 = read_record(&streams.input, &mut offset).unwrap();
        for index in 0..count {
            let device: JoltDevice = read_record(&streams.input, &mut offset).unwrap();
            info!(
                "Inner statement {index}: inputs 0x{}, outputs 0x{}, panic {}",
                hex(&device.inputs),
                hex(&device.outputs),
                device.panic
            );
        }
    }

    run_recursion_proof(
        guest,
        run_config,
        transport,
        &streams,
        regions,
        memory_config,
        max_trace_length,
    );
}

fn main() {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match &cli.command {
        Some(Commands::CheckInputFraming {
            example,
            workdir,
            embed,
            #[cfg(feature = "akita")]
            advice,
        }) => {
            let guest = GuestProgram::from_str(example).expect("supported guest example");
            let transport = if *embed {
                SetupTransport::Image
            } else {
                SetupTransport::Input
            };
            #[cfg(feature = "akita")]
            let transport = if *advice {
                SetupTransport::Advice
            } else {
                transport
            };
            verify_proofs(
                guest,
                transport,
                workdir,
                &get_guest_src_dir(),
                RunConfig::CheckInputFraming,
            );
        }
        Some(Commands::Tamper {
            example,
            workdir,
            output,
            target,
            proof_index,
        }) => {
            let guest = GuestProgram::from_str(example).expect("supported guest example");
            tamper_proof(guest, workdir, output, *target, *proof_index);
        }
        #[cfg(feature = "field-inline")]
        Some(Commands::CheckFieldKernels) => kernel_tests::run(),
        #[cfg(feature = "akita")]
        Some(Commands::TamperOpening {
            example,
            workdir,
            output,
        }) => {
            let guest = GuestProgram::from_str(example).expect("supported guest example");
            tamper_proof(guest, workdir, output, TamperTarget::Opening, 0);
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
            generate_proofs(
                guest,
                workdir,
                if *committed_bytecode {
                    ProgramPreprocessingMode::Committed
                } else {
                    ProgramPreprocessingMode::Full
                },
                *proofs,
            );
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
            let transport = if embed.is_some() {
                SetupTransport::Image
            } else {
                SetupTransport::Input
            };
            verify_proofs(guest, transport, workdir, &output_dir, RunConfig::Prove);
        }
        Some(Commands::Trace {
            example,
            workdir,
            embed,
            #[cfg(feature = "akita")]
            advice,
            trace_to_file,
            expect,
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
                RunConfig::TraceToFile(*expect)
            } else {
                RunConfig::Trace(*expect)
            };
            let transport = if embed.is_some() {
                SetupTransport::Image
            } else {
                SetupTransport::Input
            };
            #[cfg(feature = "akita")]
            let transport = if *advice {
                SetupTransport::Advice
            } else {
                transport
            };
            verify_proofs(guest, transport, workdir, &output_dir, run_config);
        }
        None => {
            info!("No subcommand specified. Available commands:");
            info!("  generate --example <fibonacci|muldiv> [--workdir <DIR>]");
            info!("  verify --example <fibonacci|muldiv> [--workdir <DIR>] [--embed <DIR>]");
            info!("");
            info!("Examples:");
            info!("  cargo run --release -- generate --example fibonacci");
            info!("  cargo run --release -- generate --example fibonacci --workdir ./output");
            info!("  cargo run --release -- generate --example fibonacci --committed-bytecode");
            info!("  cargo run --release -- verify --example fibonacci");
            info!("  cargo run --release -- verify --example fibonacci --workdir ./output --embed");
            info!("  cargo run --release -- trace --example fibonacci --embed");
            info!("  cargo run --release -- trace --example fibonacci --embed --disk");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::decode_verifier_output;

    #[test]
    fn verifier_output_accepts_canonical_compact_and_word_padded_values() {
        for (value, encoded) in [
            (0, vec![0]),
            (1, vec![1]),
            (128, vec![0x80, 1]),
            (u32::MAX, vec![0xff, 0xff, 0xff, 0xff, 0x0f]),
        ] {
            assert_eq!(decode_verifier_output(&encoded), value);
            let mut padded = encoded;
            padded.resize(8, 0);
            assert_eq!(decode_verifier_output(&padded), value);
        }
    }

    #[test]
    fn verifier_output_rejects_malformed_encoding_and_padding() {
        for bytes in [
            vec![],
            vec![0x80],
            vec![0x81, 0],
            vec![1, 0],
            vec![1, 1, 0, 0, 0, 0, 0, 0],
            vec![0; 16],
        ] {
            assert!(std::panic::catch_unwind(|| decode_verifier_output(&bytes)).is_err());
        }
    }
}

#[cfg(test)]
#[path = "../guest/src/records.rs"]
mod guest_records;

#[cfg(test)]
mod input_size_tests {
    use super::*;

    #[test]
    fn framed_proof_sections_fit_every_example() {
        for guest in [GuestProgram::Fibonacci, GuestProgram::Muldiv] {
            for transport in [SetupTransport::Image, SetupTransport::Input] {
                for len in [0, 1, 127, 128, 1120, 16383, 16384, 32768] {
                    let streams = GuestStreams {
                        input: vec![0; len],
                        untrusted_advice: Some(vec![0; len]),
                        trusted_advice: Some(vec![0; len]),
                    };
                    let config = guest.get_memory_config(transport, &streams);
                    // `frame` asserts that every stream fits its region.
                    streams.frame(&config);
                }
            }
        }
    }
}

#[cfg(feature = "field-inline")]
mod kernel_cases;
#[cfg(feature = "field-inline")]
mod kernel_tests;

#[cfg(test)]
mod record_limit_tests {
    use super::guest_records::Records;
    use super::*;
    use jolt_sdk::jolt_verifier::proof::PROOF_ARGUMENT_NEWTYPE;
    use serde::de::{Deserialize, Deserializer, Error as DeError, Visitor};
    use std::fmt::{Formatter, Result as FmtResult};

    #[test]
    fn honest_field_record_exceeding_default_budget_uses_image_limits() {
        let fields = vec![GuestField::from_u64(7); (1 << 18) + 1];
        let mut record = Vec::new();
        push_record(&mut record, &fields);
        let mut limits = RecordLimits::default();
        let mut offset = 0;
        let decoded: Vec<GuestField> = limits.read_record(&record, &mut offset).unwrap();
        assert_eq!(decoded, fields);
        assert_eq!(offset, record.len());
        assert!(limits.allocation > RecordLimits::default().allocation);

        let mut framed = vec![0; record.len() + 15];
        let pad = (8 - (framed.as_ptr() as usize + 8) % 8) % 8;
        let base = 8 + pad;
        framed[..8].copy_from_slice(&(pad as u64).to_le_bytes());
        framed[base..base + record.len()].copy_from_slice(&record);
        framed.truncate(base + record.len());
        assert!(Records::new(&framed, RecordLimits::default())
            .unwrap()
            .record::<Vec<GuestField>>()
            .is_err());
        let mut records = Records::new(&framed, limits).unwrap();
        assert_eq!(records.record::<Vec<GuestField>>().unwrap(), fields);
        records.finish().unwrap();

        // Runtime lengths cannot increase the limits baked into the image.
        let mut huge_length = vec![253];
        huge_length.extend_from_slice(&u64::MAX.to_le_bytes());
        let mut budget = limits.allocation;
        assert!(bounded::decode::<Vec<GuestField>>(&huge_length, &mut budget).is_err());
    }

    /// Stands for the proof argument, which jolt-verifier's tests cover on the
    /// proof type: it decodes only from a byte string requested under the
    /// argument's name.
    #[derive(Debug)]
    struct Argument(Vec<u8>);

    impl<'de> Deserialize<'de> for Argument {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            struct BytesOnly;

            impl Visitor<'_> for BytesOnly {
                type Value = Argument;

                fn expecting(&self, formatter: &mut Formatter<'_>) -> FmtResult {
                    formatter.write_str("a byte string")
                }

                fn visit_bytes<E: DeError>(self, bytes: &[u8]) -> Result<Argument, E> {
                    Ok(Argument(bytes.to_vec()))
                }
            }

            deserializer.deserialize_newtype_struct(PROOF_ARGUMENT_NEWTYPE, BytesOnly)
        }
    }

    #[test]
    fn bounded_decoder_reads_the_proof_argument_as_bytes_at_the_sequence_charge() {
        let config = bincode::config::standard();
        let bytes: Vec<u8> = (0..=u8::MAX).chain(0..44).collect();
        let encoded = bincode::serde::encode_to_vec(&bytes, config).unwrap();

        // Charged as the `Vec<u8>` it stands for: twice its length.
        let mut budget = 2 * bytes.len();
        let (argument, consumed) = bounded::decode::<Argument>(&encoded, &mut budget).unwrap();
        assert_eq!(
            (argument.0, consumed, budget),
            (bytes.clone(), encoded.len(), 0)
        );
        let mut budget = 2 * bytes.len() - 1;
        assert!(bounded::decode::<Argument>(&encoded, &mut budget).is_err());
    }
}
