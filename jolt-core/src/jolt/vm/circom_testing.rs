use super::rv32i_vm::{RV32ISubtables, RV32I};
use super::{Jolt, JoltProof};
use crate::field::JoltField;
use crate::host;
use crate::jolt::vm::rv32i_vm::{RV32IJoltVM, C, M};
use crate::jolt::vm::JoltStuff;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::r1cs::inputs::JoltR1CSInputs;
use crate::utils::transcript::Transcript;
use std::sync::{LazyLock, Mutex};

// If multiple tests try to read the same trace artifacts simultaneously, they will fail
static FIB_FILE_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));
static SHA3_FILE_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

pub fn fib_e2e_circom<F, PCS, ProofTranscript>() -> (
    JoltProof<C, M, JoltR1CSInputs, F, PCS, RV32I, RV32ISubtables<F>, ProofTranscript>,
    JoltStuff<<PCS as CommitmentScheme<ProofTranscript>>::Commitment>,
)
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    let artifact_guard = FIB_FILE_LOCK.lock().unwrap();
    let mut program = host::Program::new("fibonacci-guest");
    program.set_input(&9u32);
    let (bytecode, memory_init) = program.decode();
    let (io_device, trace) = program.trace();
    drop(artifact_guard);

    let preprocessing = RV32IJoltVM::preprocess(
        bytecode.clone(),
        io_device.memory_layout.clone(),
        memory_init,
        1 << 20,
        1 << 20,
        1 << 20,
    );
    let (proof, commitments, debug_info) =
        <RV32IJoltVM as Jolt<F, PCS, C, M, ProofTranscript>>::prove(
            io_device,
            trace,
            preprocessing.clone(),
        );
    (proof, commitments)
    // let verification_result =
    //     RV32IJoltVM::verify(preprocessing, proof, commitments, debug_info);
    // assert!(
    //     verification_result.is_ok(),
    //     "Verification failed with error: {:?}",
    //     verification_result.err()
    // );
}
