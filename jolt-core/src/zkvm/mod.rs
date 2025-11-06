use std::{
    fs::File,
    io::{Read, Write},
    path::Path,
    time::{Duration, Instant},
};

use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::commitment::commitment_scheme::StreamingCommitmentScheme;
#[cfg(test)]
use crate::poly::commitment::dory::DoryGlobals;
use crate::{
    field::JoltField,
    poly::opening_proof::{OpeningPoint, ProverOpeningAccumulator, VerifierOpeningAccumulator},
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
    zkvm::{
        bytecode::BytecodePreprocessing,
        dag::{jolt_dag::DagVerifier, proof_serialization::JoltProof},
        ram::RAMPreprocessing,
        witness::DTH_ROOT_OF_K,
    },
};
use ark_bn254::Fr;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::jolt_device::MemoryLayout;
use tracer::{instruction::Instruction, JoltDevice};

pub mod bytecode;
pub mod dag;
pub mod instruction;
pub mod instruction_lookups;
pub mod lookup_table;
pub mod r1cs;
pub mod ram;
pub mod registers;
pub mod spartan;
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

#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltSharedPreprocessing {
    pub bytecode: BytecodePreprocessing,
    pub ram: RAMPreprocessing,
    pub memory_layout: MemoryLayout,
}

#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    pub generators: PCS::VerifierSetup,
    pub bytecode: BytecodePreprocessing,
    pub ram: RAMPreprocessing,
    pub memory_layout: MemoryLayout,
}

impl<F, PCS> Serializable for JoltVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
}

impl<F, PCS> JoltVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    pub fn save_to_target_dir(&self, target_dir: &str) -> std::io::Result<()> {
        let filename = Path::new(target_dir).join("jolt_verifier_preprocessing.dat");
        let mut file = File::create(filename.as_path())?;
        let mut data = Vec::new();
        self.serialize_compressed(&mut data).unwrap();
        file.write_all(&data)?;
        Ok(())
    }

    pub fn read_from_target_dir(target_dir: &str) -> std::io::Result<Self> {
        let filename = Path::new(target_dir).join("jolt_verifier_preprocessing.dat");
        let mut file = File::open(filename.as_path())?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        Ok(Self::deserialize_compressed(&*data).unwrap())
    }
}

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltProverPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    pub generators: PCS::ProverSetup,
    pub bytecode: BytecodePreprocessing,
    pub ram: RAMPreprocessing,
    pub memory_layout: MemoryLayout,
}

impl<F, PCS> Serializable for JoltProverPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
}

impl<F, PCS> JoltProverPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    pub fn save_to_target_dir(&self, target_dir: &str) -> std::io::Result<()> {
        let filename = Path::new(target_dir).join("jolt_prover_preprocessing.dat");
        let mut file = File::create(filename.as_path())?;
        let mut data = Vec::new();
        self.serialize_compressed(&mut data).unwrap();
        file.write_all(&data)?;
        Ok(())
    }

    pub fn read_from_target_dir(target_dir: &str) -> std::io::Result<Self> {
        let filename = Path::new(target_dir).join("jolt_prover_preprocessing.dat");
        let mut file = File::open(filename.as_path())?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        Ok(Self::deserialize_compressed(&*data).unwrap())
    }
}

impl<F, PCS> From<&JoltProverPreprocessing<F, PCS>> for JoltVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    fn from(preprocessing: &JoltProverPreprocessing<F, PCS>) -> Self {
        let generators = PCS::setup_verifier(&preprocessing.generators);
        JoltVerifierPreprocessing {
            generators,
            bytecode: preprocessing.bytecode.clone(),
            ram: preprocessing.ram.clone(),
            memory_layout: preprocessing.memory_layout.clone(),
        }
    }
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

pub trait Jolt<F: JoltField, PCS: StreamingCommitmentScheme<Field = F>, FS: Transcript> {
    #[tracing::instrument(skip_all, name = "Jolt::prover_preprocess")]
    fn prover_preprocess(
        bytecode: Vec<Instruction>,
        memory_layout: MemoryLayout,
        memory_init: Vec<(u64, u8)>,
        max_trace_length: usize,
    ) -> JoltProverPreprocessing<F, PCS> {
        let bytecode = BytecodePreprocessing::preprocess(bytecode);
        let ram = RAMPreprocessing::preprocess(memory_init);

        let max_T: usize = max_trace_length.next_power_of_two();
        let generators = PCS::setup_prover(DTH_ROOT_OF_K.log_2() + max_T.log_2());

        JoltProverPreprocessing {
            generators,
            bytecode,
            ram,
            memory_layout,
        }
    }

    #[allow(clippy::type_complexity)]
    #[cfg(feature = "prover")]
    fn prove(
        preprocessing: &JoltProverPreprocessing<F, PCS>,
        elf_contents: &[u8],
        inputs: &[u8],
        untrusted_advice: &[u8],
        trusted_advice: &[u8],
        trusted_advice_commitment: Option<<PCS as CommitmentScheme>::Commitment>,
    ) -> (
        JoltProof<F, PCS, FS>,
        JoltDevice,
        Option<ProverDebugInfo<F, FS, PCS>>,
        Duration,
    ) {
        use crate::{guest, zkvm::dag::state_manager::StateManager};
        use common::jolt_device::MemoryConfig;
        use rayon::prelude::*;
        use tracer::instruction::Cycle;

        let memory_config = MemoryConfig {
            max_untrusted_advice_size: preprocessing.memory_layout.max_untrusted_advice_size,
            max_trusted_advice_size: preprocessing.memory_layout.max_trusted_advice_size,
            max_input_size: preprocessing.memory_layout.max_input_size,
            max_output_size: preprocessing.memory_layout.max_output_size,
            stack_size: preprocessing.memory_layout.stack_size,
            memory_size: preprocessing.memory_layout.memory_size,
            program_size: Some(preprocessing.memory_layout.program_size),
        };

        let (lazy_trace, mut trace, final_memory_state, mut program_io) = {
            let _pprof_trace = pprof_scope!("trace");
            guest::program::trace(
                elf_contents,
                None,
                inputs,
                untrusted_advice,
                trusted_advice,
                &memory_config,
            )
        };
        let num_riscv_cycles: usize = trace
            .par_iter()
            .map(|cycle| {
                // Count the cycle if the instruction is not part of a inline sequence
                // (`virtual_sequence_remaining` is `None`) or if it's the first instruction
                // in a inline sequence (`virtual_sequence_remaining` is `Some(0)`)
                if let Some(virtual_sequence_remaining) =
                    cycle.instruction().normalize().virtual_sequence_remaining
                {
                    if virtual_sequence_remaining > 0 {
                        return 0;
                    }
                }
                1
            })
            .sum();
        tracing::info!(
            "{num_riscv_cycles} raw RISC-V instructions + {} virtual instructions = {} total cycles",
            trace.len() - num_riscv_cycles,
            trace.len(),
        );

        // Setup trace length and padding
        let trace_length = trace.len();
        let padded_trace_length = (trace.len() + 1).next_power_of_two();
        trace.resize(padded_trace_length, Cycle::NoOp);

        // truncate trailing zeros on device outputs
        program_io.outputs.truncate(
            program_io
                .outputs
                .iter()
                .rposition(|&b| b != 0)
                .map_or(0, |pos| pos + 1),
        );

        let (proof, debug_info, prove_duration) = {
            use crate::zkvm::dag::jolt_dag::prove_jolt_dag;

            let _pprof_prove = pprof_scope!("prove");
            let start = Instant::now();
            let opening_accumulator = ProverOpeningAccumulator::new(trace.len().log_2());
            let state_manager = StateManager::new_prover(
                preprocessing,
                lazy_trace,
                trace,
                program_io.clone(),
                trusted_advice_commitment,
                final_memory_state,
            );
            let transcript = &mut FS::new(b"Jolt");
            let (proof, debug_info) =
                prove_jolt_dag(state_manager, opening_accumulator, transcript)
                    .ok()
                    .unwrap();
            let prove_duration = start.elapsed();
            tracing::info!(
                "Proved in {:.1}s ({:.1} kHz / padded {:.1} kHz)",
                prove_duration.as_secs_f64(),
                trace_length as f64 / prove_duration.as_secs_f64() / 1000.0,
                padded_trace_length as f64 / prove_duration.as_secs_f64() / 1000.0,
            );
            (proof, debug_info, prove_duration)
        };

        (proof, program_io, debug_info, prove_duration)
    }

    #[tracing::instrument(skip_all, name = "Jolt::verify")]
    fn verify(
        preprocessing: &JoltVerifierPreprocessing<F, PCS>,
        proof: JoltProof<F, PCS, FS>,
        mut program_io: JoltDevice,
        trusted_advice_commitment: Option<<PCS as CommitmentScheme>::Commitment>,
        _debug_info: Option<ProverDebugInfo<F, FS, PCS>>,
    ) -> Result<(), ProofVerifyError> {
        let _pprof_verify = pprof_scope!("verify");

        #[cfg(test)]
        let T = proof.trace_length.next_power_of_two();
        // Need to initialize globals because the verifier computes commitments
        // in `VerifierOpeningAccumulator::append` inside of a `#[cfg(test)]` block
        #[cfg(test)]
        let _guard = DoryGlobals::initialize(DTH_ROOT_OF_K, T);

        // Memory layout checks
        if program_io.memory_layout != preprocessing.memory_layout {
            return Err(ProofVerifyError::MemoryLayoutMismatch);
        }
        if program_io.inputs.len() > preprocessing.memory_layout.max_input_size as usize {
            return Err(ProofVerifyError::InputTooLarge);
        }
        if program_io.outputs.len() > preprocessing.memory_layout.max_output_size as usize {
            return Err(ProofVerifyError::OutputTooLarge);
        }

        // truncate trailing zeros on device outputs
        program_io.outputs.truncate(
            program_io
                .outputs
                .iter()
                .rposition(|&b| b != 0)
                .map_or(0, |pos| pos + 1),
        );

        let mut opening_accumulator = VerifierOpeningAccumulator::new(proof.trace_length.log_2());
        // Populate claims in the verifier accumulator
        for (key, (_, claim)) in &proof.opening_claims.0 {
            opening_accumulator
                .openings
                .insert(*key, (OpeningPoint::default(), *claim));
        }

        let transcript = &mut FS::new(b"Jolt");

        #[cfg(test)]
        {
            if let Some(debug_info) = _debug_info {
                transcript.compare_to(debug_info.transcript);
                opening_accumulator.compare_to(debug_info.opening_accumulator);
            }
        }

        DagVerifier {
            trusted_advice_commitment,
            program_io,
            proof,
            opening_accumulator,
            transcript,
            preprocessing,
        }
        .verify()
        .expect("Verification failed");

        Ok(())
    }
}

pub struct JoltRV64IMAC;
impl Jolt<Fr, DoryCommitmentScheme, Blake2bTranscript> for JoltRV64IMAC {}
pub type RV64IMACJoltProof = JoltProof<Fr, DoryCommitmentScheme, Blake2bTranscript>;

use crate::poly::commitment::dory::DoryCommitmentScheme;
use crate::transcripts::Blake2bTranscript;
use eyre::Result;
use std::io::Cursor;
use std::path::PathBuf;

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

impl Serializable for RV64IMACJoltProof {}
impl Serializable for JoltDevice {}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;

    use crate::host;
    use crate::poly::commitment::mock::MockCommitScheme;
    use crate::zkvm::JoltVerifierPreprocessing;
    use crate::zkvm::{Jolt, JoltRV64IMAC};
    use serial_test::serial;

    use crate::transcripts::Blake2bTranscript;

    pub struct JoltRV64IMACMockPCS;
    impl Jolt<Fr, MockCommitScheme<Fr>, Blake2bTranscript> for JoltRV64IMACMockPCS {}

    #[test]
    #[serial]
    fn fib_e2e_mock() {
        let mut program = host::Program::new("fibonacci-guest");
        let inputs = postcard::to_stdvec(&9u32).unwrap();
        let (bytecode, init_memory_state, _) = program.decode();
        let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);

        let preprocessing = JoltRV64IMACMockPCS::prover_preprocess(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );
        let elf_contents_opt = program.get_elf_contents();
        let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
        let (jolt_proof, io_device, debug_info, _) =
            JoltRV64IMACMockPCS::prove(&preprocessing, elf_contents, &inputs, &[], &[], None);

        let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);
        let verification_result = JoltRV64IMACMockPCS::verify(
            &verifier_preprocessing,
            jolt_proof,
            io_device,
            None,
            debug_info,
        );
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }

    #[test]
    #[serial]
    fn fib_e2e_dory() {
        let mut program = host::Program::new("fibonacci-guest");
        let inputs = postcard::to_stdvec(&100u32).unwrap();
        let (bytecode, init_memory_state, _) = program.decode();
        let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);

        let preprocessing = JoltRV64IMAC::prover_preprocess(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );
        let elf_contents_opt = program.get_elf_contents();
        let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
        let (jolt_proof, io_device, debug_info, _) =
            JoltRV64IMAC::prove(&preprocessing, elf_contents, &inputs, &[], &[], None);

        let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);
        let verification_result = JoltRV64IMAC::verify(
            &verifier_preprocessing,
            jolt_proof,
            io_device,
            None,
            debug_info,
        );
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }

    #[test]
    #[serial]
    fn sha3_e2e_dory() {
        // Ensure SHA3 inline library is linked and auto-registered
        #[cfg(feature = "host")]
        use jolt_inlines_keccak256 as _;
        // SHA3 inlines are automatically registered via #[ctor::ctor]
        // when the jolt-inlines-keccak256 crate is linked (see lib.rs)

        let mut program = host::Program::new("sha3-guest");
        let (bytecode, init_memory_state, _) = program.decode();
        let inputs = postcard::to_stdvec(&[5u8; 32]).unwrap();
        let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);

        let preprocessing = JoltRV64IMAC::prover_preprocess(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );
        let elf_contents_opt = program.get_elf_contents();
        let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
        let (jolt_proof, io_device, debug_info, _) =
            JoltRV64IMAC::prove(&preprocessing, elf_contents, &inputs, &[], &[], None);

        let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);
        let verification_result = JoltRV64IMAC::verify(
            &verifier_preprocessing,
            jolt_proof,
            io_device.clone(),
            None,
            debug_info,
        );
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
        assert_eq!(
            io_device.inputs, inputs,
            "Inputs mismatch: expected {:?}, got {:?}",
            inputs, io_device.inputs
        );
        let expected_output = &[
            0xd0, 0x3, 0x5c, 0x96, 0x86, 0x6e, 0xe2, 0x2e, 0x81, 0xf5, 0xc4, 0xef, 0xbd, 0x88,
            0x33, 0xc1, 0x7e, 0xa1, 0x61, 0x10, 0x81, 0xfc, 0xd7, 0xa3, 0xdd, 0xce, 0xce, 0x7f,
            0x44, 0x72, 0x4, 0x66,
        ];
        assert_eq!(io_device.outputs, expected_output, "Outputs mismatch",);
    }

    #[test]
    #[serial]
    fn sha2_e2e_dory() {
        // Ensure SHA2 inline library is linked and auto-registered
        #[cfg(feature = "host")]
        use jolt_inlines_sha2 as _;
        // SHA2 inlines are automatically registered via #[ctor::ctor]
        // when the jolt-inlines-sha2 crate is linked (see lib.rs)
        let mut program = host::Program::new("sha2-guest");
        let (bytecode, init_memory_state, _) = program.decode();
        let inputs = postcard::to_stdvec(&[5u8; 32]).unwrap();
        let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);

        let preprocessing = JoltRV64IMAC::prover_preprocess(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );
        let elf_contents_opt = program.get_elf_contents();
        let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
        let (jolt_proof, io_device, debug_info, _) =
            JoltRV64IMAC::prove(&preprocessing, elf_contents, &inputs, &[], &[], None);

        let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);
        let verification_result = JoltRV64IMAC::verify(
            &verifier_preprocessing,
            jolt_proof,
            io_device.clone(),
            None,
            debug_info,
        );
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
        let expected_output = &[
            0x28, 0x9b, 0xdf, 0x82, 0x9b, 0x4a, 0x30, 0x26, 0x7, 0x9a, 0x3e, 0xa0, 0x89, 0x73,
            0xb1, 0x97, 0x2d, 0x12, 0x4e, 0x7e, 0xaf, 0x22, 0x33, 0xc6, 0x3, 0x14, 0x3d, 0xc6,
            0x3b, 0x50, 0xd2, 0x57,
        ];
        assert_eq!(
            io_device.outputs, expected_output,
            "Outputs mismatch: expected {:?}, got {:?}",
            expected_output, io_device.outputs
        );
    }

    #[test]
    #[serial]
    fn memory_ops_e2e_dory() {
        let mut program = host::Program::new("memory-ops-guest");
        let (bytecode, init_memory_state, _) = program.decode();
        let (_, _, _, io_device) = program.trace(&[], &[], &[]);

        let preprocessing = JoltRV64IMAC::prover_preprocess(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );
        let elf_contents_opt = program.get_elf_contents();
        let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
        let (jolt_proof, io_device, debug_info, _) =
            JoltRV64IMAC::prove(&preprocessing, elf_contents, &[], &[], &[], None);

        let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);
        let verification_result = JoltRV64IMAC::verify(
            &verifier_preprocessing,
            jolt_proof,
            io_device,
            None,
            debug_info,
        );
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }

    #[test]
    #[serial]
    fn btreemap_e2e_dory() {
        let mut program = host::Program::new("btreemap-guest");
        let (bytecode, init_memory_state, _) = program.decode();
        let inputs = postcard::to_stdvec(&50u32).unwrap();
        let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);

        let preprocessing = JoltRV64IMAC::prover_preprocess(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );
        let elf_contents_opt = program.get_elf_contents();
        let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
        let (jolt_proof, io_device, debug_info, _) =
            JoltRV64IMAC::prove(&preprocessing, elf_contents, &inputs, &[], &[], None);

        let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);
        let verification_result = JoltRV64IMAC::verify(
            &verifier_preprocessing,
            jolt_proof,
            io_device,
            None,
            debug_info,
        );
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }

    #[test]
    #[serial]
    fn muldiv_e2e_dory() {
        let mut program = host::Program::new("muldiv-guest");
        let (bytecode, init_memory_state, _) = program.decode();
        let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).unwrap();
        let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);

        let preprocessing = JoltRV64IMAC::prover_preprocess(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );
        let elf_contents_opt = program.get_elf_contents();
        let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
        let (jolt_proof, io_device, debug_info, _) =
            JoltRV64IMAC::prove(&preprocessing, elf_contents, &[50], &[], &[], None);

        let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);
        let verification_result = JoltRV64IMAC::verify(
            &verifier_preprocessing,
            jolt_proof,
            io_device,
            None,
            debug_info,
        );
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }
}
