use std::{
    fs::File,
    io::{Read, Write},
    path::Path,
};

use ark_bn254::Fr;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::jolt_device::MemoryLayout;
use tracer::JoltDevice;

use crate::{
    dag::proof_serialization::JoltProof,
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme, opening_proof::ProverOpeningAccumulator,
    },
    utils::transcript::Transcript,
    zkvm::{bytecode::BytecodePreprocessing, ram::RAMPreprocessing},
};

pub mod bytecode;
pub mod dag;
pub mod instruction;
pub mod instruction_lookups;
pub mod lookup_table;
pub mod r1cs;
pub mod ram;
pub mod registers;
pub mod witness;

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
    pub shared: JoltSharedPreprocessing,
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
    pub shared: JoltSharedPreprocessing,
    field: F::SmallValueLookupTables,
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
            shared: preprocessing.shared.clone(),
        }
    }
}

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

// fn fiat_shamir_preamble(
//     transcript: &mut ProofTranscript,
//     program_io: &JoltDevice,
//     memory_layout: &MemoryLayout,
//     trace_length: usize,
//     ram_K: usize,
// ) {
//     transcript.append_u64(trace_length as u64);
//     transcript.append_u64(ram_K as u64);
//     // transcript.append_u64(WORD_SIZE as u64);
//     transcript.append_u64(memory_layout.max_input_size);
//     transcript.append_u64(memory_layout.max_output_size);
//     transcript.append_bytes(&program_io.inputs);
//     transcript.append_bytes(&program_io.outputs);
//     transcript.append_u64(program_io.panic as u64);
// }

pub type RV32IMJoltProof<F, PCS, ProofTranscript> = JoltProof<F, PCS, ProofTranscript>;

use crate::poly::commitment::dory::DoryCommitmentScheme;
use crate::utils::transcript::KeccakTranscript;
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
}

pub type JoltTranscript = KeccakTranscript;
pub type PCS = DoryCommitmentScheme;

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltProofBundle {
    pub proof: RV32IMJoltProof<Fr, PCS, JoltTranscript>,
}

impl Serializable for JoltProofBundle {}

// ==================== TEST ====================

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;

    use crate::field::JoltField;
    use crate::host;
    use crate::poly::commitment::commitment_scheme::CommitmentScheme;
    use crate::poly::commitment::dory::DoryCommitmentScheme;
    use crate::poly::commitment::mock::MockCommitScheme;
    use crate::zkvm::JoltVerifierPreprocessing;
    use crate::zkvm::{Jolt, RV32IMJoltVM};
    use serial_test::serial;

    use crate::utils::transcript::KeccakTranscript;
    use std::sync::{LazyLock, Mutex};

    // If multiple tests try to read the same trace artifacts simultaneously, they will fail
    static FIB_FILE_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));
    static SHA3_FILE_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

    fn fib_e2e<F, PCS>()
    where
        F: JoltField,
        PCS: CommitmentScheme<Field = F>,
    {
        let artifact_guard = FIB_FILE_LOCK.lock().unwrap();
        let mut program = host::Program::new("fibonacci-guest");
        let inputs = postcard::to_stdvec(&9u32).unwrap();
        let (bytecode, init_memory_state) = program.decode();
        let (trace, final_memory_state, io_device) = program.trace(&inputs);
        drop(artifact_guard);

        let preprocessing = RV32IMJoltVM::prover_preprocess(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
            1 << 16,
            1 << 16,
        );
        let (proof, commitments, debug_info) = <RV32IMJoltVM as Jolt<32, F, PCS, _>>::prove(
            io_device,
            trace,
            final_memory_state,
            preprocessing.clone(),
        );

        let verifier_preprocessing = JoltVerifierPreprocessing::<F, PCS>::from(&preprocessing);
        let verification_result =
            RV32IMJoltVM::verify(verifier_preprocessing, proof, commitments, debug_info);
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }

    #[test]
    #[ignore]
    #[serial]
    fn fib_e2e_mock() {
        fib_e2e::<Fr, MockCommitScheme<Fr>>();
    }

    #[test]
    #[ignore]
    #[serial]
    fn fib_e2e_dory() {
        fib_e2e::<Fr, DoryCommitmentScheme>();
    }

    #[test]
    #[ignore]
    #[serial]
    fn sha3_e2e_dory() {
        let guard = SHA3_FILE_LOCK.lock().unwrap();
        let mut program = host::Program::new("sha3-guest");
        let (bytecode, init_memory_state) = program.decode();
        let inputs = postcard::to_stdvec(&[5u8; 32]).unwrap();
        let (trace, final_memory_state, io_device) = program.trace(&inputs);
        drop(guard);

        let preprocessing = RV32IMJoltVM::prover_preprocess(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
            1 << 16,
            1 << 16,
        );
        let (jolt_proof, jolt_commitments, debug_info) =
            <RV32IMJoltVM as Jolt<32, Fr, DoryCommitmentScheme, KeccakTranscript>>::prove(
                io_device,
                trace,
                final_memory_state,
                preprocessing.clone(),
            );

        let verifier_preprocessing =
            JoltVerifierPreprocessing::<Fr, DoryCommitmentScheme>::from(&preprocessing);
        let verification_result = RV32IMJoltVM::verify(
            verifier_preprocessing,
            jolt_proof,
            jolt_commitments,
            debug_info,
        );
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }

    #[test]
    #[ignore]
    #[serial]
    fn memory_ops_e2e_dory() {
        let mut program = host::Program::new("memory-ops-guest");
        let (bytecode, init_memory_state) = program.decode();
        let (trace, final_memory_state, io_device) = program.trace(&[]);

        let preprocessing = RV32IMJoltVM::prover_preprocess(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
            1 << 16,
            1 << 16,
        );
        let (jolt_proof, jolt_commitments, debug_info) =
            <RV32IMJoltVM as Jolt<32, Fr, DoryCommitmentScheme, KeccakTranscript>>::prove(
                io_device,
                trace,
                final_memory_state,
                preprocessing.clone(),
            );

        let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);
        let verification_result = RV32IMJoltVM::verify(
            verifier_preprocessing,
            jolt_proof,
            jolt_commitments,
            debug_info,
        );
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }

    #[test]
    #[ignore]
    #[serial]
    #[should_panic]
    fn truncated_trace() {
        let artifact_guard = FIB_FILE_LOCK.lock().unwrap();
        let mut program = host::Program::new("fibonacci-guest");
        let (bytecode, init_memory_state) = program.decode();
        let inputs = postcard::to_stdvec(&9u8).unwrap();
        let (mut trace, final_memory_state, mut io_device) = program.trace(&inputs);
        trace.truncate(100);
        io_device.outputs[0] = 0; // change the output to 0
        drop(artifact_guard);

        let preprocessing = RV32IMJoltVM::prover_preprocess(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
            1 << 16,
            1 << 16,
        );
        let (proof, commitments, debug_info) =
            <RV32IMJoltVM as Jolt<32, Fr, DoryCommitmentScheme, KeccakTranscript>>::prove(
                io_device,
                trace,
                final_memory_state,
                preprocessing.clone(),
            );
        let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);
        let _verification_result =
            RV32IMJoltVM::verify(verifier_preprocessing, proof, commitments, debug_info);
    }

    #[test]
    #[ignore]
    #[serial]
    #[should_panic]
    fn malicious_trace() {
        let artifact_guard = FIB_FILE_LOCK.lock().unwrap();
        let mut program = host::Program::new("fibonacci-guest");
        let inputs = postcard::to_stdvec(&1u8).unwrap();
        let (bytecode, init_memory_state) = program.decode();
        let (trace, final_memory_state, mut io_device) = program.trace(&inputs);
        let memory_layout = io_device.memory_layout.clone();
        drop(artifact_guard);

        // change memory address of output & termination bit to the same address as input
        // changes here should not be able to spoof the verifier result
        io_device.memory_layout.output_start = io_device.memory_layout.input_start;
        io_device.memory_layout.output_end = io_device.memory_layout.input_end;
        io_device.memory_layout.termination = io_device.memory_layout.input_start;

        // Since the preprocessing is done with the original memory layout, the verifier should fail
        let preprocessing = RV32IMJoltVM::prover_preprocess(
            bytecode.clone(),
            memory_layout,
            init_memory_state,
            1 << 16,
            1 << 16,
            1 << 16,
        );
        let (proof, commitments, debug_info) =
            <RV32IMJoltVM as Jolt<32, Fr, DoryCommitmentScheme, KeccakTranscript>>::prove(
                io_device,
                trace,
                final_memory_state,
                preprocessing.clone(),
            );
        let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);
        let _verification_result =
            RV32IMJoltVM::verify(verifier_preprocessing, proof, commitments, debug_info);
    }
}
