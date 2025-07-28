use crate::field::JoltField;
use crate::poly::commitment::hyperkzg::HyperKZG;
use crate::r1cs::constraints::JoltRV32IMConstraints;
use ark_bn254::{Bn254, Fr};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use super::{Jolt, JoltProof};
use crate::poly::commitment::commitment_scheme::CommitmentScheme;

const WORD_SIZE: usize = 32;

// ==================== JOLT ====================

pub enum RV32IJoltVM {}

impl<F, PCS> Jolt<WORD_SIZE, F, PCS, KeccakTranscript> for RV32IJoltVM
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    type Constraints = JoltRV32IMConstraints;
}

pub type RV32IJoltProof<F, PCS, ProofTranscript> = JoltProof<WORD_SIZE, F, PCS, ProofTranscript>;

use crate::utils::transcript::KeccakTranscript;
use eyre::Result;
use std::fs::File;
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

pub type ProofTranscript = KeccakTranscript;
pub type PCS = HyperKZG<Bn254>;
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone)]
pub struct JoltHyperKZGProof {
    pub proof: RV32IJoltProof<Fr, PCS, ProofTranscript>,
    // pub commitments: JoltCommitments<PCS, ProofTranscript>,
}

impl Serializable for JoltHyperKZGProof {}

// ==================== TEST ====================

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;

    use crate::field::JoltField;
    use crate::host;
    use crate::jolt::vm::rv32i_vm::{Jolt, RV32IJoltVM};
    use crate::jolt::vm::JoltVerifierPreprocessing;
    use crate::poly::commitment::commitment_scheme::CommitmentScheme;
    use crate::poly::commitment::dory::DoryCommitmentScheme;
    use crate::poly::commitment::mock::MockCommitScheme;
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

        let preprocessing = RV32IJoltVM::prover_preprocess(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
            1 << 16,
            1 << 16,
        );
        let (proof, commitments, debug_info) = <RV32IJoltVM as Jolt<32, F, PCS, _>>::prove(
            io_device,
            trace,
            final_memory_state,
            preprocessing.clone(),
        );

        let verifier_preprocessing = JoltVerifierPreprocessing::<F, PCS>::from(&preprocessing);
        let verification_result =
            RV32IJoltVM::verify(verifier_preprocessing, proof, commitments, debug_info);
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

        let preprocessing = RV32IJoltVM::prover_preprocess(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
            1 << 16,
            1 << 16,
        );
        let (jolt_proof, jolt_commitments, debug_info) =
            <RV32IJoltVM as Jolt<32, Fr, DoryCommitmentScheme, KeccakTranscript>>::prove(
                io_device,
                trace,
                final_memory_state,
                preprocessing.clone(),
            );

        let verifier_preprocessing =
            JoltVerifierPreprocessing::<Fr, DoryCommitmentScheme>::from(&preprocessing);
        let verification_result = RV32IJoltVM::verify(
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

        let preprocessing = RV32IJoltVM::prover_preprocess(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
            1 << 16,
            1 << 16,
        );
        let (jolt_proof, jolt_commitments, debug_info) =
            <RV32IJoltVM as Jolt<32, Fr, DoryCommitmentScheme, KeccakTranscript>>::prove(
                io_device,
                trace,
                final_memory_state,
                preprocessing.clone(),
            );

        let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);
        let verification_result = RV32IJoltVM::verify(
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

        let preprocessing = RV32IJoltVM::prover_preprocess(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
            1 << 16,
            1 << 16,
        );
        let (proof, commitments, debug_info) =
            <RV32IJoltVM as Jolt<32, Fr, DoryCommitmentScheme, KeccakTranscript>>::prove(
                io_device,
                trace,
                final_memory_state,
                preprocessing.clone(),
            );
        let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);
        let _verification_result =
            RV32IJoltVM::verify(verifier_preprocessing, proof, commitments, debug_info);
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
        let preprocessing = RV32IJoltVM::prover_preprocess(
            bytecode.clone(),
            memory_layout,
            init_memory_state,
            1 << 16,
            1 << 16,
            1 << 16,
        );
        let (proof, commitments, debug_info) =
            <RV32IJoltVM as Jolt<32, Fr, DoryCommitmentScheme, KeccakTranscript>>::prove(
                io_device,
                trace,
                final_memory_state,
                preprocessing.clone(),
            );
        let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);
        let _verification_result =
            RV32IJoltVM::verify(verifier_preprocessing, proof, commitments, debug_info);
    }
}
