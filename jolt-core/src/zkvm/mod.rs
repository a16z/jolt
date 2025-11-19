use std::fs::File;

use crate::{
    field::JoltField,
    poly::opening_proof::ProverOpeningAccumulator,
    poly::{
        commitment::commitment_scheme::CommitmentScheme, commitment::dory::DoryCommitmentScheme,
    },
    transcripts::Blake2bTranscript,
    transcripts::Transcript,
};
use ark_bn254::Fr;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use eyre::Result;
use proof_serialization::JoltProof;
use prover::JoltCpuProver;
use std::io::Cursor;
use std::path::PathBuf;
use tracer::JoltDevice;
use verifier::JoltVerifier;

pub mod bytecode;
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

pub type RV64IMACProver<'a> = JoltCpuProver<'a, Fr, DoryCommitmentScheme, Blake2bTranscript>;
pub type RV64IMACVerifier<'a> = JoltVerifier<'a, Fr, DoryCommitmentScheme, Blake2bTranscript>;
pub type RV64IMACProof = JoltProof<Fr, DoryCommitmentScheme, Blake2bTranscript>;

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
