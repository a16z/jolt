use std::fs::File;

use crate::zkvm::config::{OneHotConfig, OneHotParams, ProgramMode, ReadWriteConfig};
use crate::zkvm::witness::CommittedPolynomial;
use crate::{
    curve::Bn254Curve,
    field::JoltField,
    poly::commitment::commitment_scheme::CommitmentScheme,
    poly::commitment::dory::{DoryCommitmentScheme, DoryLayout},
    poly::opening_proof::{
        OpeningAccumulator, OpeningId, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
        BIG_ENDIAN,
    },
    utils::errors::ProofVerifyError,
    zkvm::claim_reductions::AdviceKind,
};

// Compile-time error if multiple transcript features are enabled
// When none of the transcript features are enabled, Jolt defaults to `Blake2bTranscript`
#[cfg(any(
    all(feature = "transcript-poseidon", feature = "transcript-keccak"),
    all(feature = "transcript-poseidon", feature = "transcript-blake2b"),
    all(feature = "transcript-keccak", feature = "transcript-blake2b"),
    all(
        feature = "transcript-poseidon",
        feature = "transcript-keccak",
        feature = "transcript-blake2b"
    )
))]
compile_error!("Cannot enable multiple transcript features simultaneously. Please choose exactly one of: 'transcript-poseidon', 'transcript-keccak', or 'transcript-blake2b'.");

/// The spongefish sponge RV64IMAC's prover/verifier/proof are instantiated over
/// (the phantom `H` of the proof). Cfg-selected, defaulting to Blake2b.
#[cfg(any(
    feature = "transcript-blake2b",
    not(any(feature = "transcript-poseidon", feature = "transcript-keccak"))
))]
pub type RV64IMACSponge = jolt_transcript::Blake2b512;
#[cfg(feature = "transcript-keccak")]
pub type RV64IMACSponge = jolt_transcript::Keccak;
#[cfg(feature = "transcript-poseidon")]
pub type RV64IMACSponge = jolt_transcript::PoseidonSponge;
use ark_bn254::Fr;
use jolt_transcript::DuplexSpongeInterface;
use crate::transcript_msgs::AbsorbFs;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use eyre::Result;
use proof_serialization::JoltProof;
#[cfg(feature = "prover")]
use prover::JoltCpuProver;
use std::io::Cursor;
use std::path::PathBuf;
use tracer::JoltDevice;
use verifier::JoltVerifier;

pub mod bytecode;
pub mod claim_reductions;
pub mod config;
pub mod instruction;
pub mod instruction_lookups;
pub mod lookup_table;
pub mod program;
pub mod proof_serialization;
#[cfg(feature = "prover")]
pub mod prover;
pub mod r1cs;
pub mod ram;
pub mod registers;
pub mod spartan;
/// Symbolic verifier used to transpile the on-chain verifier (gated behind the
/// `transpiler` feature). It relies on the removed `crate::transcripts::Transcript`
/// trait and cannot be expressed over spongefish's concrete `VerifierState`, so it
/// is disabled during the transcript→spongefish migration (spec Non-Goal #2 / DEV-21).
#[cfg(feature = "transpiler")]
pub mod transpilable_verifier;
pub mod verifier;
pub mod witness;

pub(crate) fn stage8_opening_ids(
    one_hot_params: &OneHotParams,
    include_trusted_advice: bool,
    include_untrusted_advice: bool,
    program_mode: ProgramMode,
    bytecode_chunk_count: usize,
) -> Vec<OpeningId> {
    let mut opening_ids = Vec::new();

    opening_ids.push(OpeningId::committed(
        CommittedPolynomial::RamInc,
        SumcheckId::IncClaimReduction,
    ));
    opening_ids.push(OpeningId::committed(
        CommittedPolynomial::RdInc,
        SumcheckId::IncClaimReduction,
    ));

    for i in 0..one_hot_params.instruction_d {
        opening_ids.push(OpeningId::committed(
            CommittedPolynomial::InstructionRa(i),
            SumcheckId::HammingWeightClaimReduction,
        ));
    }
    for i in 0..one_hot_params.bytecode_d {
        opening_ids.push(OpeningId::committed(
            CommittedPolynomial::BytecodeRa(i),
            SumcheckId::HammingWeightClaimReduction,
        ));
    }
    for i in 0..one_hot_params.ram_d {
        opening_ids.push(OpeningId::committed(
            CommittedPolynomial::RamRa(i),
            SumcheckId::HammingWeightClaimReduction,
        ));
    }

    if include_trusted_advice {
        opening_ids.push(OpeningId::TrustedAdvice(SumcheckId::AdviceClaimReduction));
    }
    if include_untrusted_advice {
        opening_ids.push(OpeningId::UntrustedAdvice(SumcheckId::AdviceClaimReduction));
    }
    if program_mode == ProgramMode::Committed {
        for i in 0..bytecode_chunk_count {
            opening_ids.push(OpeningId::committed(
                CommittedPolynomial::BytecodeChunk(i),
                SumcheckId::BytecodeClaimReduction,
            ));
        }
    }
    if program_mode == ProgramMode::Committed {
        opening_ids.push(OpeningId::committed(
            CommittedPolynomial::ProgramImageInit,
            SumcheckId::ProgramImageClaimReduction,
        ));
    }

    opening_ids
}

pub(crate) fn compute_final_opening_point<F: JoltField>(
    opening_accumulator: &impl OpeningAccumulator<F>,
    native_main_vars: usize,
    log_k_chunk: usize,
    layout: DoryLayout,
    program_mode: ProgramMode,
    bytecode_chunk_count: usize,
) -> Result<OpeningPoint<BIG_ENDIAN, F>, ProofVerifyError> {
    let mut opening_candidates: Vec<(String, OpeningPoint<BIG_ENDIAN, F>)> = Vec::new();
    if let Some((point, _)) = opening_accumulator
        .get_advice_opening(AdviceKind::Trusted, SumcheckId::AdviceClaimReduction)
    {
        opening_candidates.push(("trusted_advice".to_string(), point));
    }
    if let Some((point, _)) = opening_accumulator
        .get_advice_opening(AdviceKind::Untrusted, SumcheckId::AdviceClaimReduction)
    {
        opening_candidates.push(("untrusted_advice".to_string(), point));
    }
    if program_mode == ProgramMode::Committed {
        for chunk_idx in 0..bytecode_chunk_count {
            let (point, _) = opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::BytecodeChunk(chunk_idx),
                SumcheckId::BytecodeClaimReduction,
            );
            opening_candidates.push((format!("bytecode_chunk[{chunk_idx}]"), point));
        }
    }
    if program_mode == ProgramMode::Committed {
        let (program_image_point, _) = opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::ProgramImageInit,
            SumcheckId::ProgramImageClaimReduction,
        );
        opening_candidates.push(("program_image".to_string(), program_image_point));
    }

    let (hamming_point, _) = opening_accumulator.get_committed_polynomial_opening(
        CommittedPolynomial::InstructionRa(0),
        SumcheckId::HammingWeightClaimReduction,
    );
    let (r_cycle_stage6, _) = opening_accumulator.get_committed_polynomial_opening(
        CommittedPolynomial::RamInc,
        SumcheckId::IncClaimReduction,
    );

    let max_len = opening_candidates
        .iter()
        .map(|(_, point)| point.r.len())
        .max()
        .unwrap_or(0);
    if max_len > native_main_vars {
        let dominant = opening_candidates
            .iter()
            .find(|(_, point)| point.r.len() == max_len)
            .expect("at least one dominant precommitted candidate expected");
        for (name, point) in opening_candidates
            .iter()
            .filter(|(_, point)| point.r.len() == max_len)
        {
            if point.r != dominant.1.r {
                return Err(ProofVerifyError::DoryError(format!(
                    "incompatible dominant precommitted anchors: {} and {} have equal dimensionality {} but different opening points",
                    dominant.0, name, max_len
                )));
            }
        }
        Ok(OpeningPoint::<BIG_ENDIAN, F>::new(dominant.1.r.clone()))
    } else {
        let r_address_stage7 = hamming_point.r[..log_k_chunk].to_vec();

        match layout {
            DoryLayout::AddressMajor => Ok(OpeningPoint::<BIG_ENDIAN, F>::new(
                [r_cycle_stage6.r.as_slice(), r_address_stage7.as_slice()].concat(),
            )),
            DoryLayout::CycleMajor => {
                let native_cycle = &hamming_point.r[log_k_chunk..];
                if r_cycle_stage6.r.len() < native_cycle.len() {
                    return Err(ProofVerifyError::DoryError(
                        "stage6 cycle challenges shorter than native cycle vars".to_string(),
                    ));
                }
                if r_cycle_stage6.r[..native_cycle.len()] != *native_cycle {
                    return Err(ProofVerifyError::DoryError(format!(
                        "cycle-major Stage-8 expects stage6 cycle prefix to equal native cycle vars \
                         (cycle_full_len={}, native_len={})",
                        r_cycle_stage6.r.len(),
                        native_cycle.len()
                    )));
                }
                let cycle_extra = &r_cycle_stage6.r[native_cycle.len()..];
                let cycle_extra_and_anchor =
                    [cycle_extra, r_address_stage7.as_slice(), native_cycle].concat();
                Ok(OpeningPoint::<BIG_ENDIAN, F>::new(cycle_extra_and_anchor))
            }
        }
    }
}

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
pub struct ProverDebugInfo<F, H, PCS>
where
    F: JoltField,
    H: DuplexSpongeInterface,
    PCS: CommitmentScheme<Field = F>,
{
    pub(crate) opening_accumulator: ProverOpeningAccumulator<F>,
    pub(crate) prover_setup: PCS::ProverSetup,
    pub(crate) _marker: std::marker::PhantomData<fn() -> H>,
}

/// Absorb public instance data into the transcript for Fiat-Shamir.
///
/// The statement is `absorb`'d (public_message) on both prover and verifier — it is
/// recomputable by the verifier, so it is not shipped in the NARG. (Milestone: this
/// scattered absorb is the binding; folding it into `instance = Blake2b(statement)`
/// is a follow-up — see DEV-19.)
#[allow(clippy::too_many_arguments)]
pub fn fiat_shamir_preamble<F: JoltField>(
    program_io: &JoltDevice,
    ram_K: usize,
    trace_length: usize,
    entry_address: u64,
    rw_config: &ReadWriteConfig,
    one_hot_config: &OneHotConfig,
    dory_layout: DoryLayout,
    preprocessing_digest: &[u8; 32],
    transcript: &mut impl AbsorbFs<F>,
) {
    transcript.absorb(&preprocessing_digest.to_vec());
    transcript.absorb(&program_io.memory_layout.max_input_size);
    transcript.absorb(&program_io.memory_layout.max_output_size);
    transcript.absorb(&program_io.memory_layout.heap_size);
    transcript.absorb(&program_io.inputs);
    transcript.absorb(&program_io.outputs);
    transcript.absorb(&(program_io.panic as u64));
    transcript.absorb(&(ram_K as u64));
    transcript.absorb(&(trace_length as u64));
    transcript.absorb(&entry_address);
    transcript.absorb(&(rw_config.ram_rw_phase1_num_rounds as u64));
    transcript.absorb(&(rw_config.ram_rw_phase2_num_rounds as u64));
    transcript.absorb(&(rw_config.registers_rw_phase1_num_rounds as u64));
    transcript.absorb(&(rw_config.registers_rw_phase2_num_rounds as u64));
    transcript.absorb(&(one_hot_config.log_k_chunk as u64));
    transcript.absorb(&(one_hot_config.lookups_ra_virtual_log_k_chunk as u64));
    transcript.absorb(&(dory_layout as u64));
}

// The per-sponge variance lives entirely in `RV64IMACSponge` (cfg-gated above), so
// these aliases are sponge-agnostic. `RV64IMACProver` needs the `prover` feature; the
// verifier/proof are available in any build.
#[cfg(feature = "prover")]
pub type RV64IMACProver<'a> =
    JoltCpuProver<'a, Fr, Bn254Curve, DoryCommitmentScheme, RV64IMACSponge>;
pub type RV64IMACVerifier<'a> =
    JoltVerifier<'a, Fr, Bn254Curve, DoryCommitmentScheme, RV64IMACSponge>;
pub type RV64IMACProof = JoltProof<Fr, Bn254Curve, DoryCommitmentScheme, RV64IMACSponge>;

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
}

impl Serializable for RV64IMACProof {}
impl Serializable for JoltDevice {}
