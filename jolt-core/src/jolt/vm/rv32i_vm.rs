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

impl<F, PCS, ProofTranscript> Jolt<WORD_SIZE, F, PCS, ProofTranscript> for RV32IJoltVM
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    type Constraints = JoltRV32IMConstraints;
}

pub type RV32IJoltProof<F, PCS, ProofTranscript> = JoltProof<WORD_SIZE, F, PCS, ProofTranscript>;

use crate::utils::transcript::{KeccakTranscript, Transcript};
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
pub type PCS = HyperKZG<Bn254, ProofTranscript>;
#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltHyperKZGProof {
    pub proof: RV32IJoltProof<Fr, PCS, ProofTranscript>,
    // pub commitments: JoltCommitments<PCS, ProofTranscript>,
}

impl Serializable for JoltHyperKZGProof {}

// ==================== TEST ====================

#[cfg(test)]
mod tests {
    use ark_bn254::{Bn254, Fr};
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::{FmtSubscriber, Registry};

    use crate::field::JoltField;
    use crate::host;
    use crate::jolt::vm::rv32i_vm::{Jolt, RV32IJoltVM};
    use crate::jolt::vm::JoltProverPreprocessing;
    use crate::poly::commitment::commitment_scheme::CommitmentScheme;
    use crate::poly::commitment::hyperkzg::HyperKZG;
    use crate::poly::commitment::mock::MockCommitScheme;
    use crate::poly::commitment::zeromorph::Zeromorph;
    use crate::utils::transcript::{KeccakTranscript, Transcript};
    use std::sync::{LazyLock, Mutex};

    // If multiple tests try to read the same trace artifacts simultaneously, they will fail
    static FIB_FILE_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));
    static SHA3_FILE_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

    fn fib_e2e<F, PCS, ProofTranscript>()
    where
        F: JoltField,
        PCS: CommitmentScheme<ProofTranscript, Field = F>,
        ProofTranscript: Transcript,
    {
        let artifact_guard = FIB_FILE_LOCK.lock().unwrap();
        let mut program = host::Program::new("fibonacci-guest");
        let inputs = postcard::to_stdvec(&9u32).unwrap();
        let (bytecode, memory_init) = program.decode();
        let (io_device, trace) = program.trace(&inputs);
        drop(artifact_guard);

        let preprocessing = RV32IJoltVM::prover_preprocess(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            memory_init,
            1 << 20,
            1 << 20,
            1 << 20,
        );
        let (proof, commitments, debug_info) = <RV32IJoltVM as Jolt<
            32,
            Fr,
            HyperKZG<Bn254, KeccakTranscript>,
            KeccakTranscript,
        >>::prove(
            io_device, trace, preprocessing.clone()
        );
        let verification_result =
            RV32IJoltVM::verify(preprocessing.shared, proof, commitments, debug_info);
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }

    #[test]
    fn fib_e2e_mock() {
        fib_e2e::<Fr, MockCommitScheme<Fr, KeccakTranscript>, KeccakTranscript>();
    }

    #[test]
    fn fib_e2e_zeromorph() {
        fib_e2e::<Fr, Zeromorph<Bn254, KeccakTranscript>, KeccakTranscript>();
    }

    #[test]
    fn fib_e2e_hyperkzg() {
        let (chrome_layer, guard) = ChromeLayerBuilder::new()
            .file("trace.json")
            .include_args(true)
            .build();

        let subscriber = Registry::default().with(chrome_layer);
        tracing::subscriber::set_global_default(subscriber).unwrap();

        fib_e2e::<Fr, HyperKZG<Bn254, KeccakTranscript>, KeccakTranscript>();
        drop(guard);
    }

    #[test]
    fn sha3_e2e_zeromorph() {
        let guard = SHA3_FILE_LOCK.lock().unwrap();
        let mut program = host::Program::new("sha3-guest");
        let (bytecode, memory_init) = program.decode();
        let inputs = postcard::to_stdvec(&[5u8; 32]).unwrap();
        let (io_device, trace) = program.trace(&inputs);
        drop(guard);

        let preprocessing = RV32IJoltVM::prover_preprocess(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            memory_init,
            1 << 20,
            1 << 20,
            1 << 20,
        );
        let (jolt_proof, jolt_commitments, debug_info) = <RV32IJoltVM as Jolt<
            32,
            Fr,
            HyperKZG<Bn254, KeccakTranscript>,
            KeccakTranscript,
        >>::prove(
            io_device, trace, preprocessing.clone()
        );

        let verification_result = RV32IJoltVM::verify(
            preprocessing.shared,
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
    fn sha3_e2e_hyperkzg() {
        let guard = SHA3_FILE_LOCK.lock().unwrap();
        let mut program = host::Program::new("sha3-guest");
        let (bytecode, memory_init) = program.decode();
        let inputs = postcard::to_stdvec(&[5u8; 32]).unwrap();
        let (io_device, trace) = program.trace(&inputs);
        drop(guard);
        let preprocessing = RV32IJoltVM::prover_preprocess(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            memory_init,
            1 << 20,
            1 << 20,
            1 << 20,
        );
        let (jolt_proof, jolt_commitments, debug_info) = <RV32IJoltVM as Jolt<
            32,
            Fr,
            HyperKZG<Bn254, KeccakTranscript>,
            KeccakTranscript,
        >>::prove(
            io_device, trace, preprocessing.clone()
        );
        let verification_result = RV32IJoltVM::verify(
            preprocessing.shared,
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
    fn sha3_chain_e2e_hyperkzg() {
        let (chrome_layer, _guard) = ChromeLayerBuilder::new()
            .file("trace.json")
            .include_args(true)
            .build();

        let subscriber = Registry::default().with(chrome_layer);
        tracing::subscriber::set_global_default(subscriber).unwrap();

        let guard = SHA3_FILE_LOCK.lock().unwrap();
        let mut program = host::Program::new("sha3-chain-guest");
        let (bytecode, memory_init) = program.decode();
        let mut inputs = postcard::to_stdvec(&[5u8; 32]).unwrap();
        inputs.append(&mut postcard::to_stdvec(&512u32).unwrap());
        let (io_device, trace) = program.trace(&inputs);
        drop(guard);

        let preprocessing: crate::jolt::vm::JoltProverPreprocessing<
            Fr,
            HyperKZG<Bn254, KeccakTranscript>,
            KeccakTranscript,
        > = RV32IJoltVM::prover_preprocess(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            memory_init,
            1 << 20,
            1 << 20,
            1 << 20,
        );

        let (jolt_proof, jolt_commitments, debug_info) = <RV32IJoltVM as Jolt<
            32,
            Fr,
            HyperKZG<Bn254, KeccakTranscript>,
            KeccakTranscript,
        >>::prove(
            io_device, trace, preprocessing.clone()
        );

        let verification_result = RV32IJoltVM::verify(
            preprocessing.shared,
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
    fn memory_ops_e2e_hyperkzg() {
        let mut program = host::Program::new("memory-ops-guest");
        let (bytecode, memory_init) = program.decode();
        let (io_device, trace) = program.trace(&[]);

        let preprocessing = RV32IJoltVM::prover_preprocess(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            memory_init,
            1 << 20,
            1 << 20,
            1 << 20,
        );
        let (jolt_proof, jolt_commitments, debug_info) = <RV32IJoltVM as Jolt<
            32,
            Fr,
            HyperKZG<Bn254, KeccakTranscript>,
            KeccakTranscript,
        >>::prove(
            io_device, trace, preprocessing.clone()
        );

        let verification_result = RV32IJoltVM::verify(
            preprocessing.shared,
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
    #[should_panic]
    fn truncated_trace() {
        let artifact_guard = FIB_FILE_LOCK.lock().unwrap();
        let mut program = host::Program::new("fibonacci-guest");
        let (bytecode, memory_init) = program.decode();
        let inputs = postcard::to_stdvec(&9u8).unwrap();
        let (mut io_device, mut trace) = program.trace(&inputs);
        trace.truncate(100);
        io_device.outputs[0] = 0; // change the output to 0
        drop(artifact_guard);

        let preprocessing = RV32IJoltVM::prover_preprocess(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            memory_init,
            1 << 20,
            1 << 20,
            1 << 20,
        );
        let (proof, commitments, debug_info) = <RV32IJoltVM as Jolt<
            32,
            Fr,
            HyperKZG<Bn254, KeccakTranscript>,
            KeccakTranscript,
        >>::prove(
            io_device, trace, preprocessing.clone()
        );
        let _verification_result =
            RV32IJoltVM::verify(preprocessing.shared, proof, commitments, debug_info);
    }

    #[test]
    #[should_panic]
    fn malicious_trace() {
        let artifact_guard = FIB_FILE_LOCK.lock().unwrap();
        let mut program = host::Program::new("fibonacci-guest");
        let inputs = postcard::to_stdvec(&1u8).unwrap();
        let (bytecode, memory_init) = program.decode();
        let (mut io_device, trace) = program.trace(&inputs);
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
            memory_init,
            1 << 20,
            1 << 20,
            1 << 20,
        );
        let (proof, commitments, debug_info) = <RV32IJoltVM as Jolt<
            32,
            Fr,
            HyperKZG<Bn254, KeccakTranscript>,
            KeccakTranscript,
        >>::prove(
            io_device, trace, preprocessing.clone()
        );
        let _verification_result =
            RV32IJoltVM::verify(preprocessing.shared, proof, commitments, debug_info);
    }
}
