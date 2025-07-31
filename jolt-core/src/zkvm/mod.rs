use std::{
    fs::File,
    io::{Read, Write},
    path::Path,
};

use ark_bn254::Fr;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::jolt_device::MemoryLayout;
use tracer::{
    instruction::{RV32IMCycle, RV32IMInstruction},
    JoltDevice,
};

#[cfg(test)]
use crate::poly::commitment::dory::DoryGlobals;
use crate::{
    field::JoltField,
    host::Program,
    poly::{
        commitment::commitment_scheme::CommitmentScheme, opening_proof::ProverOpeningAccumulator,
    },
    utils::{errors::ProofVerifyError, math::Math, transcript::Transcript},
    zkvm::{
        bytecode::BytecodePreprocessing,
        dag::{jolt_dag::JoltDAG, proof_serialization::JoltProof, state_manager::StateManager},
        ram::{RAMPreprocessing, NUM_RA_I_VARS},
    },
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

pub trait Jolt<F, PCS, FS: Transcript>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    fn shared_preprocess(
        bytecode: Vec<RV32IMInstruction>,
        memory_layout: MemoryLayout,
        memory_init: Vec<(u64, u8)>,
    ) -> JoltSharedPreprocessing {
        let bytecode_preprocessing = BytecodePreprocessing::preprocess(bytecode);
        let ram_preprocessing = RAMPreprocessing::preprocess(memory_init);

        JoltSharedPreprocessing {
            memory_layout,
            bytecode: bytecode_preprocessing,
            ram: ram_preprocessing,
        }
    }

    #[tracing::instrument(skip_all, name = "Jolt::prover_preprocess")]
    fn prover_preprocess(
        bytecode: Vec<RV32IMInstruction>,
        memory_layout: MemoryLayout,
        memory_init: Vec<(u64, u8)>,
        max_trace_length: usize,
    ) -> JoltProverPreprocessing<F, PCS> {
        let small_value_lookup_tables = F::compute_lookup_tables();
        F::initialize_lookup_tables(small_value_lookup_tables.clone());

        let shared = Self::shared_preprocess(bytecode, memory_layout, memory_init);

        let max_K: usize = 1 << NUM_RA_I_VARS;
        let max_T: usize = max_trace_length.next_power_of_two();

        println!("setup...");
        // TODO(moodlezoup): Change setup parameter to # variables everywhere
        let generators = PCS::setup_prover(max_K.log_2() + max_T.log_2());
        println!("setup done");

        JoltProverPreprocessing {
            generators,
            shared,
            field: small_value_lookup_tables,
        }
    }

    #[allow(clippy::type_complexity)]
    fn prove(
        preprocessing: &JoltProverPreprocessing<F, PCS>,
        program: &mut Program,
        inputs: &[u8],
        write_proof_to_file: Option<&str>,
    ) -> (
        JoltProof<F, PCS, FS>,
        JoltDevice,
        Option<ProverDebugInfo<F, FS, PCS>>,
    ) {
        let (mut trace, final_memory_state, mut program_io) = program.trace(inputs);

        // Setup trace length and padding
        let padded_trace_length = (trace.len() + 1).next_power_of_two();
        trace.resize(padded_trace_length, RV32IMCycle::NoOp);

        // truncate trailing zeros on device outputs
        program_io.outputs.truncate(
            program_io
                .outputs
                .iter()
                .rposition(|&b| b != 0)
                .map_or(0, |pos| pos + 1),
        );

        let state_manager =
            StateManager::new_prover(preprocessing, trace, program_io.clone(), final_memory_state);
        let (proof, debug_info) = JoltDAG::prove(state_manager, write_proof_to_file)
            .ok()
            .unwrap();

        (proof, program_io, debug_info)
    }

    fn verify(
        preprocessing: &JoltVerifierPreprocessing<F, PCS>,
        proof: JoltProof<F, PCS, FS>,
        mut program_io: JoltDevice,
        _debug_info: Option<ProverDebugInfo<F, FS, PCS>>,
    ) -> Result<(), ProofVerifyError> {
        #[cfg(test)]
        let K = 1 << NUM_RA_I_VARS;
        #[cfg(test)]
        let T = proof.trace_length.next_power_of_two();
        // Need to initialize globals because the verifier computes commitments
        // in `VerifierOpeningAccumulator::append` inside of a `#[cfg(test)]` block
        #[cfg(test)]
        let _guard = DoryGlobals::initialize(K, T);

        // truncate trailing zeros on device outputs
        program_io.outputs.truncate(
            program_io
                .outputs
                .iter()
                .rposition(|&b| b != 0)
                .map_or(0, |pos| pos + 1),
        );

        let state_manager = proof.to_verifier_state_manager(preprocessing, program_io);

        #[cfg(test)]
        {
            if let Some(debug_info) = _debug_info {
                let mut transcript = state_manager.transcript.borrow_mut();
                transcript.compare_to(debug_info.transcript);
                let opening_accumulator = state_manager.get_verifier_accumulator();
                opening_accumulator
                    .borrow_mut()
                    .compare_to(debug_info.opening_accumulator);
            }
        }

        JoltDAG::verify(state_manager).expect("Verification failed");

        Ok(())
    }
}

pub struct JoltRV32IM;
impl Jolt<Fr, DoryCommitmentScheme, KeccakTranscript> for JoltRV32IM {}
pub type RV32IMJoltProof = JoltProof<Fr, DoryCommitmentScheme, KeccakTranscript>;

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

impl Serializable for RV32IMJoltProof {}

// ==================== TEST ====================

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;

    use crate::host;
    use crate::poly::commitment::mock::MockCommitScheme;
    use crate::zkvm::JoltVerifierPreprocessing;
    use crate::zkvm::{Jolt, JoltRV32IM};
    use serial_test::serial;

    use crate::utils::transcript::KeccakTranscript;

    pub struct JoltRV32IMMockPCS;
    impl Jolt<Fr, MockCommitScheme<Fr>, KeccakTranscript> for JoltRV32IMMockPCS {}

    #[test]
    #[serial]
    fn fib_e2e_mock() {
        let mut program = host::Program::new("fibonacci-guest");
        let inputs = postcard::to_stdvec(&9u32).unwrap();
        let (bytecode, init_memory_state, _) = program.decode();
        let (_, _, io_device) = program.trace(&inputs);

        let preprocessing = JoltRV32IMMockPCS::prover_preprocess(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );
        let (jolt_proof, io_device, debug_info) =
            JoltRV32IMMockPCS::prove(&preprocessing, &mut program, &inputs, None);

        let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);
        let verification_result =
            JoltRV32IMMockPCS::verify(&verifier_preprocessing, jolt_proof, io_device, debug_info);
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
        let (_, _, io_device) = program.trace(&inputs);

        let preprocessing = JoltRV32IM::prover_preprocess(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );
        let (jolt_proof, io_device, debug_info) =
            JoltRV32IM::prove(&preprocessing, &mut program, &inputs, None);

        let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);
        let verification_result =
            JoltRV32IM::verify(&verifier_preprocessing, jolt_proof, io_device, debug_info);
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }

    #[test]
    #[serial]
    fn sha3_e2e_dory() {
        let mut program = host::Program::new("sha3-guest");
        let (bytecode, init_memory_state, _) = program.decode();
        let inputs = postcard::to_stdvec(&[5u8; 32]).unwrap();
        let (_, _, io_device) = program.trace(&inputs);

        let preprocessing = JoltRV32IM::prover_preprocess(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );
        let (jolt_proof, io_device, debug_info) =
            JoltRV32IM::prove(&preprocessing, &mut program, &inputs, None);

        let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);
        let verification_result =
            JoltRV32IM::verify(&verifier_preprocessing, jolt_proof, io_device, debug_info);
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }

    #[test]
    #[serial]
    fn sha2_e2e_dory() {
        let mut program = host::Program::new("sha2-guest");
        let (bytecode, init_memory_state, _) = program.decode();
        let inputs = postcard::to_stdvec(&[5u8; 32]).unwrap();
        let (_, _, io_device) = program.trace(&inputs);

        let preprocessing = JoltRV32IM::prover_preprocess(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );
        let (jolt_proof, io_device, debug_info) =
            JoltRV32IM::prove(&preprocessing, &mut program, &inputs, None);

        let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);
        let verification_result =
            JoltRV32IM::verify(&verifier_preprocessing, jolt_proof, io_device, debug_info);
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }

    #[test]
    #[serial]
    fn memory_ops_e2e_dory() {
        let mut program = host::Program::new("memory-ops-guest");
        let (bytecode, init_memory_state, _) = program.decode();
        let (_, _, io_device) = program.trace(&[]);

        let preprocessing = JoltRV32IM::prover_preprocess(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );
        let (jolt_proof, io_device, debug_info) =
            JoltRV32IM::prove(&preprocessing, &mut program, &[], None);

        let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);
        let verification_result =
            JoltRV32IM::verify(&verifier_preprocessing, jolt_proof, io_device, debug_info);
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }
}
