use std::fs::File;

use crate::{
    field::JoltField,
    poly::commitment::commitment_scheme::CommitmentScheme,
    poly::commitment::dory::DoryCommitmentScheme,
    poly::commitment::hyperkzg::HyperKZG,
    poly::opening_proof::ProverOpeningAccumulator,
    transcripts::Blake2bTranscript,
    transcripts::Transcript,
};
use ark_bn254::{Bn254, Fr};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::constants::{HYPERKZG_THRESHOLD_LOG_T, ONEHOT_CHUNK_THRESHOLD_LOG_T};
use eyre::Result;
use proof_serialization::JoltProof;
#[cfg(feature = "prover")]
use prover::JoltCpuProver;
use std::io::Cursor;
use std::path::PathBuf;
use tracer::JoltDevice;
use verifier::{JoltSharedPreprocessing, JoltVerifier, JoltVerifierPreprocessing};

pub mod bytecode;
pub mod claim_reductions;
pub mod config;
pub mod instruction;
pub mod instruction_lookups;
pub mod lookup_table;
pub mod proof_serialization;
#[cfg(feature = "prover")]
pub mod prover;
pub mod r1cs;
pub mod ram;
pub mod registers;
pub mod spartan;
pub mod verifier;
pub mod witness;

// Scoped CPU profiler for performance analysis. Feature-gated by "pprof".
// Usage: let _guard = pprof_scope!("label");
//
// Writes pprof/label.pb on scope exit
// View with: go tool pprof -http=:8080 pprof/label.pb

// Public type for the profiling guard
#[cfg(feature = "pprof")]
pub struct PprofGuard {
    guard: pprof::ProfilerGuard<'static>,
    label: &'static str,
}

#[cfg(not(feature = "pprof"))]
pub struct PprofGuard;

#[cfg(feature = "pprof")]
impl Drop for PprofGuard {
    fn drop(&mut self) {
        if let Ok(report) = self.guard.report().build() {
            let prefix = std::env::var("PPROF_PREFIX")
                .unwrap_or_else(|_| String::from("benchmark-runs/pprof/"));
            let filename = format!("{}{}.pb", prefix, self.label);
            // Extract directory from prefix for creation
            if let Some(dir) = std::path::Path::new(&filename).parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            if let Ok(mut f) = std::fs::File::create(&filename) {
                use pprof::protos::Message;
                if let Ok(p) = report.pprof() {
                    let mut buf = Vec::new();
                    if p.encode(&mut buf).is_ok() {
                        let _ = std::io::Write::write_all(&mut f, &buf);
                        tracing::info!("Wrote pprof profile to {}", filename);
                    }
                }
            }
        }
    }
}

#[macro_export]
macro_rules! pprof_scope {
    ($label:expr) => {{
        #[cfg(feature = "pprof")]
        {
            Some($crate::zkvm::PprofGuard {
                guard: pprof::ProfilerGuardBuilder::default()
                    .frequency(
                        std::env::var("PPROF_FREQ")
                            .unwrap_or("100".to_string())
                            .parse::<i32>()
                            .unwrap(),
                    )
                    .blocklist(&["libc", "libgcc", "pthread", "vdso"])
                    .build()
                    .expect("Failed to initialize profiler"),
                label: $label,
            })
        }
        #[cfg(not(feature = "pprof"))]
        None::<$crate::zkvm::PprofGuard>
    }};
    () => {
        pprof_scope!("default");
    };
}

#[allow(dead_code)]
pub struct ProverDebugInfo<F, ProofTranscript, PCS>
where
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    pub(crate) transcript: ProofTranscript,
    pub(crate) opening_accumulator: ProverOpeningAccumulator<F>,
    pub(crate) prover_setup: PCS::ProverSetup,
}

// TODO: Perhaps better we have something like a JoltClaim with this stuff in
// it and have a method on that to append that to the transcript.
pub fn fiat_shamir_preamble(
    program_io: &JoltDevice,
    ram_K: usize,
    trace_length: usize,
    transcript: &mut impl Transcript,
) {
    transcript.append_u64(program_io.memory_layout.max_input_size);
    transcript.append_u64(program_io.memory_layout.max_output_size);
    transcript.append_u64(program_io.memory_layout.memory_size);
    transcript.append_bytes(&program_io.inputs);
    transcript.append_bytes(&program_io.outputs);
    transcript.append_u64(program_io.panic as u64);
    transcript.append_u64(ram_K as u64);
    transcript.append_u64(trace_length as u64);
}

// Internal type aliases (not public API)
#[cfg(feature = "prover")]
type DoryProver<'a> = JoltCpuProver<'a, Fr, DoryCommitmentScheme, Blake2bTranscript>;
#[cfg(feature = "prover")]
type HyperKZGProver<'a> = JoltCpuProver<'a, Fr, HyperKZG<Bn254>, Blake2bTranscript>;

type DoryVerifier<'a> = JoltVerifier<'a, Fr, DoryCommitmentScheme, Blake2bTranscript>;
type HyperKZGVerifier<'a> = JoltVerifier<'a, Fr, HyperKZG<Bn254>, Blake2bTranscript>;

type DoryProof = JoltProof<Fr, DoryCommitmentScheme, Blake2bTranscript>;
type HyperKZGProof = JoltProof<Fr, HyperKZG<Bn254>, Blake2bTranscript>;

/// Unified proof type that can hold either Dory or HyperKZG proofs.
/// The PCS is chosen automatically based on max_trace_length at preprocessing time.
pub enum RV64IMACProof {
    Dory(DoryProof),
    HyperKZG(HyperKZGProof),
}

impl RV64IMACProof {
    pub fn trace_length(&self) -> usize {
        match self {
            RV64IMACProof::Dory(p) => p.trace_length,
            RV64IMACProof::HyperKZG(p) => p.trace_length,
        }
    }
}

impl CanonicalSerialize for RV64IMACProof {
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        mut writer: W,
        compress: ark_serialize::Compress,
    ) -> std::result::Result<(), ark_serialize::SerializationError> {
        match self {
            RV64IMACProof::Dory(p) => {
                0u8.serialize_with_mode(&mut writer, compress)?;
                p.serialize_with_mode(&mut writer, compress)
            }
            RV64IMACProof::HyperKZG(p) => {
                1u8.serialize_with_mode(&mut writer, compress)?;
                p.serialize_with_mode(&mut writer, compress)
            }
        }
    }

    fn serialized_size(&self, compress: ark_serialize::Compress) -> usize {
        1 + match self {
            RV64IMACProof::Dory(p) => p.serialized_size(compress),
            RV64IMACProof::HyperKZG(p) => p.serialized_size(compress),
        }
    }
}

impl ark_serialize::Valid for RV64IMACProof {
    fn check(&self) -> std::result::Result<(), ark_serialize::SerializationError> {
        match self {
            RV64IMACProof::Dory(p) => p.check(),
            RV64IMACProof::HyperKZG(p) => p.check(),
        }
    }
}

impl CanonicalDeserialize for RV64IMACProof {
    fn deserialize_with_mode<R: std::io::Read>(
        mut reader: R,
        compress: ark_serialize::Compress,
        validate: ark_serialize::Validate,
    ) -> std::result::Result<Self, ark_serialize::SerializationError> {
        let variant = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        match variant {
            0 => Ok(RV64IMACProof::Dory(DoryProof::deserialize_with_mode(
                reader, compress, validate,
            )?)),
            1 => Ok(RV64IMACProof::HyperKZG(
                HyperKZGProof::deserialize_with_mode(reader, compress, validate)?,
            )),
            _ => Err(ark_serialize::SerializationError::InvalidData),
        }
    }
}

/// Prover generators that hold either Dory or HyperKZG setup.
#[cfg(feature = "prover")]
pub enum PCSProverGenerators {
    Dory(<DoryCommitmentScheme as CommitmentScheme>::ProverSetup),
    HyperKZG(<HyperKZG<Bn254> as CommitmentScheme>::ProverSetup),
}

/// Verifier generators that hold either Dory or HyperKZG setup.
pub enum PCSVerifierGenerators {
    Dory(<DoryCommitmentScheme as CommitmentScheme>::VerifierSetup),
    HyperKZG(<HyperKZG<Bn254> as CommitmentScheme>::VerifierSetup),
}

pub trait Serializable: CanonicalSerialize + CanonicalDeserialize + Sized {
    /// Gets the byte size of the serialized data
    fn size(&self) -> Result<usize> {
        let mut buffer = Vec::new();
        self.serialize_compressed(&mut buffer)?;
        Ok(buffer.len())
    }

    /// Saves the data to a file
    fn save_to_file<P: Into<PathBuf>>(&self, path: P) -> Result<()> {
        let file = File::create(path.into())?;
        self.serialize_compressed(file)?;
        Ok(())
    }

    /// Reads data from a file
    fn from_file<P: Into<PathBuf>>(path: P) -> Result<Self> {
        let file = File::open(path.into())?;
        Ok(Self::deserialize_compressed(file)?)
    }

    /// Serializes the data to a byte vector
    fn serialize_to_bytes(&self) -> Result<Vec<u8>> {
        let mut buffer = Vec::new();
        self.serialize_compressed(&mut buffer)?;
        Ok(buffer)
    }

    /// Deserializes data from a byte vector
    fn deserialize_from_bytes(bytes: &[u8]) -> Result<Self> {
        let cursor = Cursor::new(bytes);
        Ok(Self::deserialize_compressed(cursor)?)
    }

    /// Deserializes data from bytes but skips checks for performance
    fn deserialize_from_bytes_unchecked(bytes: &[u8]) -> Result<Self> {
        let cursor = Cursor::new(bytes);
        Ok(Self::deserialize_with_mode(
            cursor,
            ark_serialize::Compress::Yes,
            ark_serialize::Validate::No,
        )?)
    }
}

impl Serializable for RV64IMACProof {}
impl Serializable for JoltDevice {}

/// Unified preprocessing for RV64IMAC proving.
/// Automatically selects PCS (Dory or HyperKZG) based on max_trace_length.
#[cfg(feature = "prover")]
pub struct RV64IMACPreprocessing {
    shared: JoltSharedPreprocessing,
    generators: PCSProverGenerators,
    max_trace_length: usize,
}

#[cfg(feature = "prover")]
impl RV64IMACPreprocessing {
    /// Creates preprocessing for the given max_trace_length.
    /// Uses HyperKZG for small traces (≤ 2^20), Dory for larger traces.
    #[tracing::instrument(skip_all, name = "RV64IMACPreprocessing::new")]
    pub fn new(shared: JoltSharedPreprocessing, max_trace_length: usize) -> Self {
        let max_T = max_trace_length.next_power_of_two();
        let max_log_T = max_T.ilog2() as usize;
        let max_log_k_chunk = if max_log_T < ONEHOT_CHUNK_THRESHOLD_LOG_T {
            4
        } else {
            8
        };
        let num_vars = max_log_k_chunk + max_log_T;

        let generators = if max_log_T <= HYPERKZG_THRESHOLD_LOG_T {
            PCSProverGenerators::HyperKZG(HyperKZG::<Bn254>::setup_prover(num_vars))
        } else {
            PCSProverGenerators::Dory(DoryCommitmentScheme::setup_prover(num_vars))
        };

        Self {
            shared,
            generators,
            max_trace_length,
        }
    }

    /// Proves a program execution, returning the proof and program IO.
    #[tracing::instrument(skip_all, name = "RV64IMACPreprocessing::prove")]
    pub fn prove(
        &self,
        elf_contents: &[u8],
        inputs: &[u8],
        untrusted_advice: &[u8],
        trusted_advice: &[u8],
    ) -> (RV64IMACProof, JoltDevice) {
        match &self.generators {
            PCSProverGenerators::HyperKZG(generators) => {
                let prover_preprocessing = prover::JoltProverPreprocessing {
                    generators: generators.clone(),
                    shared: self.shared.clone(),
                };
                let prover = HyperKZGProver::gen_from_elf(
                    &prover_preprocessing,
                    elf_contents,
                    inputs,
                    untrusted_advice,
                    trusted_advice,
                    None,
                );
                let program_io = prover.program_io.clone();
                let (proof, _) = prover.prove();
                (RV64IMACProof::HyperKZG(proof), program_io)
            }
            PCSProverGenerators::Dory(generators) => {
                let prover_preprocessing = prover::JoltProverPreprocessing {
                    generators: generators.clone(),
                    shared: self.shared.clone(),
                };
                let prover = DoryProver::gen_from_elf(
                    &prover_preprocessing,
                    elf_contents,
                    inputs,
                    untrusted_advice,
                    trusted_advice,
                    None,
                );
                let program_io = prover.program_io.clone();
                let (proof, _) = prover.prove();
                (RV64IMACProof::Dory(proof), program_io)
            }
        }
    }

    /// Returns the shared preprocessing data.
    pub fn shared(&self) -> &JoltSharedPreprocessing {
        &self.shared
    }

    /// Returns the max trace length this preprocessing was created for.
    pub fn max_trace_length(&self) -> usize {
        self.max_trace_length
    }

    /// Creates verifier preprocessing from this prover preprocessing.
    pub fn to_verifier(&self) -> RV64IMACVerifierPreprocessing {
        let generators = match &self.generators {
            PCSProverGenerators::HyperKZG(g) => {
                PCSVerifierGenerators::HyperKZG(HyperKZG::<Bn254>::setup_verifier(g))
            }
            PCSProverGenerators::Dory(g) => {
                PCSVerifierGenerators::Dory(DoryCommitmentScheme::setup_verifier(g))
            }
        };
        RV64IMACVerifierPreprocessing {
            shared: self.shared.clone(),
            generators,
            max_trace_length: self.max_trace_length,
        }
    }
}

/// Unified preprocessing for RV64IMAC verification.
pub struct RV64IMACVerifierPreprocessing {
    shared: JoltSharedPreprocessing,
    generators: PCSVerifierGenerators,
    max_trace_length: usize,
}

impl RV64IMACVerifierPreprocessing {
    /// Creates verifier preprocessing for the given max_trace_length.
    /// Uses HyperKZG for small traces (≤ 2^20), Dory for larger traces.
    #[tracing::instrument(skip_all, name = "RV64IMACVerifierPreprocessing::new")]
    pub fn new(shared: JoltSharedPreprocessing, max_trace_length: usize) -> Self {
        let max_T = max_trace_length.next_power_of_two();
        let max_log_T = max_T.ilog2() as usize;
        let max_log_k_chunk = if max_log_T < ONEHOT_CHUNK_THRESHOLD_LOG_T {
            4
        } else {
            8
        };
        let num_vars = max_log_k_chunk + max_log_T;

        let generators = if max_log_T <= HYPERKZG_THRESHOLD_LOG_T {
            let prover_setup = HyperKZG::<Bn254>::setup_prover(num_vars);
            PCSVerifierGenerators::HyperKZG(HyperKZG::<Bn254>::setup_verifier(&prover_setup))
        } else {
            let prover_setup = DoryCommitmentScheme::setup_prover(num_vars);
            PCSVerifierGenerators::Dory(DoryCommitmentScheme::setup_verifier(&prover_setup))
        };

        Self {
            shared,
            generators,
            max_trace_length,
        }
    }

    /// Verifies a proof, dispatching to the appropriate verifier based on proof type.
    #[tracing::instrument(skip_all, name = "RV64IMACVerifierPreprocessing::verify")]
    pub fn verify(
        &self,
        proof: RV64IMACProof,
        program_io: JoltDevice,
    ) -> std::result::Result<(), anyhow::Error> {
        match (&self.generators, proof) {
            (PCSVerifierGenerators::HyperKZG(generators), RV64IMACProof::HyperKZG(p)) => {
                let verifier_preprocessing = JoltVerifierPreprocessing {
                    generators: generators.clone(),
                    shared: self.shared.clone(),
                };
                let verifier =
                    HyperKZGVerifier::new(&verifier_preprocessing, p, program_io, None, None)?;
                verifier.verify()
            }
            (PCSVerifierGenerators::Dory(generators), RV64IMACProof::Dory(p)) => {
                let verifier_preprocessing = JoltVerifierPreprocessing {
                    generators: generators.clone(),
                    shared: self.shared.clone(),
                };
                let verifier =
                    DoryVerifier::new(&verifier_preprocessing, p, program_io, None, None)?;
                verifier.verify()
            }
            (PCSVerifierGenerators::HyperKZG(_), RV64IMACProof::Dory(_)) => {
                Err(anyhow::anyhow!(
                    "PCS mismatch: verifier configured for HyperKZG but received Dory proof"
                ))
            }
            (PCSVerifierGenerators::Dory(_), RV64IMACProof::HyperKZG(_)) => {
                Err(anyhow::anyhow!(
                    "PCS mismatch: verifier configured for Dory but received HyperKZG proof"
                ))
            }
        }
    }

    /// Returns the shared preprocessing data.
    pub fn shared(&self) -> &JoltSharedPreprocessing {
        &self.shared
    }

    /// Returns the max trace length this preprocessing was created for.
    pub fn max_trace_length(&self) -> usize {
        self.max_trace_length
    }
}
