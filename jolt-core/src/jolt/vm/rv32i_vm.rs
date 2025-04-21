use crate::field::JoltField;
use crate::poly::commitment::hyperkzg::HyperKZG;
use crate::r1cs::constraints::JoltRV32IMConstraints;
use crate::r1cs::inputs::JoltR1CSInputs;
use ark_bn254::{Bn254, Fr};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use enum_dispatch::enum_dispatch;
use rand::{prelude::StdRng, RngCore};
use serde::{Deserialize, Serialize};
use strum::{EnumCount, IntoEnumIterator};
use strum_macros::{EnumCount as EnumCountMacro, EnumIter};

use super::{Jolt, JoltCommitments, JoltProof};
use crate::poly::commitment::commitment_scheme::CommitmentScheme;

// /// Generates an enum out of a list of JoltInstruction types. All JoltInstruction methods
// /// are callable on the enum type via enum_dispatch.
// macro_rules! instruction_set {
//     ($enum_name:ident, $($alias:ident: $struct:ty),+) => {
//         #[allow(non_camel_case_types)]
//         #[repr(u8)]
//         #[derive(Copy, Clone, Debug, PartialEq, EnumIter, EnumCountMacro, Serialize, Deserialize)]
//         #[enum_dispatch(JoltInstruction)]
//         pub enum $enum_name {
//             $($alias($struct)),+
//         }
//         impl JoltInstructionSet for $enum_name {}
//         impl $enum_name {
//             pub fn random_instruction(rng: &mut StdRng) -> Self {
//                 let index = rng.next_u64() as usize % $enum_name::COUNT;
//                 let instruction = $enum_name::iter()
//                     .enumerate()
//                     .filter(|(i, _)| *i == index)
//                     .map(|(_, x)| x)
//                     .next()
//                     .unwrap();
//                 instruction.random(rng)
//             }
//         }
//         // Need a default so that we can derive EnumIter on `JoltR1CSInputs`
//         impl Default for $enum_name {
//             fn default() -> Self {
//                 $enum_name::iter().collect::<Vec<_>>()[0]
//             }
//         }
//     };
// }

const WORD_SIZE: usize = 32;

// instruction_set!(
//   RV32I,
//   ADD: ADDInstruction<WORD_SIZE>,
//   SUB: SubTable<WORD_SIZE>,
//   AND: AndTable<WORD_SIZE>,
//   OR: OrTable<WORD_SIZE>,
//   XOR: XorTable<WORD_SIZE>,
//   BEQ: BEQInstruction<WORD_SIZE>,
//   BGE: BGEInstruction<WORD_SIZE>,
//   BGEU: BGEUInstruction<WORD_SIZE>,
//   BNE: BNEInstruction<WORD_SIZE>,
//   SLT: SLTInstruction<WORD_SIZE>,
//   SLTU: SLTUInstruction<WORD_SIZE>,
//   MOVSIGN: MOVSIGNInstruction<WORD_SIZE>,
//   MUL: MULInstruction<WORD_SIZE>,
//   MULU: MULUInstruction<WORD_SIZE>,
//   MULHU: MULHUInstruction<WORD_SIZE>,
//   VIRTUAL_ADVICE: ADVICEInstruction<WORD_SIZE>,
//   VIRTUAL_MOVE: MOVEInstruction<WORD_SIZE>,
//   VIRTUAL_ASSERT_LTE: ASSERTLTEInstruction<WORD_SIZE>,
//   VIRTUAL_ASSERT_VALID_SIGNED_REMAINDER: AssertValidSignedRemainderInstruction<WORD_SIZE>,
//   VIRTUAL_ASSERT_VALID_UNSIGNED_REMAINDER: AssertValidUnsignedRemainderInstruction<WORD_SIZE>,
//   VIRTUAL_ASSERT_VALID_DIV0: AssertValidDiv0Instruction<WORD_SIZE>,
//   VIRTUAL_ASSERT_HALFWORD_ALIGNMENT: AssertHalfwordAlignmentInstruction<WORD_SIZE>,
//   VIRTUAL_POW2: POW2Instruction<WORD_SIZE>,
//   VIRTUAL_SHIFT_RIGHT_BITMASK: ShiftRightBitmaskInstruction<WORD_SIZE>,
//   VIRTUAL_SRL: VirtualSRLTable<WORD_SIZE>,
//   VIRTUAL_SRA: VirtualSRATable<WORD_SIZE>
// );

// ==================== JOLT ====================

pub enum RV32IJoltVM {}

pub const C: usize = 4;
pub const M: usize = 1 << 16;

impl<F, PCS, ProofTranscript> Jolt<C, M, WORD_SIZE, F, PCS, ProofTranscript> for RV32IJoltVM
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    // type InstructionSet = RV32I;
    type Constraints = JoltRV32IMConstraints;
}

pub type RV32IJoltProof<F, PCS, ProofTranscript> =
    JoltProof<WORD_SIZE, C, M, JoltR1CSInputs, F, PCS, ProofTranscript>;

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
    pub commitments: JoltCommitments<PCS, ProofTranscript>,
}

impl Serializable for JoltHyperKZGProof {}

// ==================== TEST ====================

#[cfg(test)]
mod tests {
    use ark_bn254::{Bn254, Fr};

    use crate::field::JoltField;
    use crate::host;
    use crate::jolt::vm::rv32i_vm::{Jolt, RV32IJoltVM, C, M};
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
            <RV32IJoltVM as Jolt<
                C,
                M,
                32,
                Fr,
                HyperKZG<Bn254, KeccakTranscript>,
                KeccakTranscript,
            >>::prove(io_device, trace, preprocessing.clone());
        let verification_result =
            RV32IJoltVM::verify(preprocessing, proof, commitments, debug_info);
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
        fib_e2e::<Fr, HyperKZG<Bn254, KeccakTranscript>, KeccakTranscript>();
    }

    // TODO(sragss): Finish Binius.
    // #[test]
    // fn fib_e2e_binius() {
    //     type Field = crate::field::binius::BiniusField<binius_field::BinaryField128b>;
    //     fib_e2e::<Field, MockCommitScheme<Field>>();
    // }

    #[test]
    fn sha3_e2e_zeromorph() {
        let guard = SHA3_FILE_LOCK.lock().unwrap();

        let mut program = host::Program::new("sha3-guest");
        program.set_input(&[5u8; 32]);
        let (bytecode, memory_init) = program.decode();
        let (io_device, trace) = program.trace();
        drop(guard);

        let preprocessing = RV32IJoltVM::preprocess(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            memory_init,
            1 << 20,
            1 << 20,
            1 << 20,
        );
        let (jolt_proof, jolt_commitments, debug_info) =
            <RV32IJoltVM as Jolt<
                C,
                M,
                32,
                Fr,
                HyperKZG<Bn254, KeccakTranscript>,
                KeccakTranscript,
            >>::prove(io_device, trace, preprocessing.clone());

        let verification_result =
            RV32IJoltVM::verify(preprocessing, jolt_proof, jolt_commitments, debug_info);
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
        program.set_input(&[5u8; 32]);
        let (bytecode, memory_init) = program.decode();
        let (io_device, trace) = program.trace();
        drop(guard);

        let preprocessing = RV32IJoltVM::preprocess(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            memory_init,
            1 << 20,
            1 << 20,
            1 << 20,
        );
        let (jolt_proof, jolt_commitments, debug_info) =
            <RV32IJoltVM as Jolt<
                C,
                M,
                32,
                Fr,
                HyperKZG<Bn254, KeccakTranscript>,
                KeccakTranscript,
            >>::prove(io_device, trace, preprocessing.clone());

        let verification_result =
            RV32IJoltVM::verify(preprocessing, jolt_proof, jolt_commitments, debug_info);
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
        let (io_device, trace) = program.trace();

        let preprocessing = RV32IJoltVM::preprocess(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            memory_init,
            1 << 20,
            1 << 20,
            1 << 20,
        );
        let (jolt_proof, jolt_commitments, debug_info) =
            <RV32IJoltVM as Jolt<
                C,
                M,
                32,
                Fr,
                HyperKZG<Bn254, KeccakTranscript>,
                KeccakTranscript,
            >>::prove(io_device, trace, preprocessing.clone());

        let verification_result =
            RV32IJoltVM::verify(preprocessing, jolt_proof, jolt_commitments, debug_info);
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
        program.set_input(&9u32);
        let (bytecode, memory_init) = program.decode();
        let (mut io_device, mut trace) = program.trace();
        trace.truncate(100);
        io_device.outputs[0] = 0; // change the output to 0
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
            <RV32IJoltVM as Jolt<
                C,
                M,
                32,
                Fr,
                HyperKZG<Bn254, KeccakTranscript>,
                KeccakTranscript,
            >>::prove(io_device, trace, preprocessing.clone());
        let _verification_result =
            RV32IJoltVM::verify(preprocessing, proof, commitments, debug_info);
    }

    #[test]
    #[should_panic]
    fn malicious_trace() {
        let artifact_guard = FIB_FILE_LOCK.lock().unwrap();
        let mut program = host::Program::new("fibonacci-guest");
        program.set_input(&1u8); // change input to 1 so that termination bit equal true
        let (bytecode, memory_init) = program.decode();
        let (mut io_device, trace) = program.trace();
        let memory_layout = io_device.memory_layout.clone();
        drop(artifact_guard);

        // change memory address of output & termination bit to the same address as input
        // changes here should not be able to spoof the verifier result
        io_device.memory_layout.output_start = io_device.memory_layout.input_start;
        io_device.memory_layout.output_end = io_device.memory_layout.input_end;
        io_device.memory_layout.termination = io_device.memory_layout.input_start;

        // Since the preprocessing is done with the original memory layout, the verifier should fail
        let preprocessing = RV32IJoltVM::preprocess(
            bytecode.clone(),
            memory_layout,
            memory_init,
            1 << 20,
            1 << 20,
            1 << 20,
        );
        let (proof, commitments, debug_info) =
            <RV32IJoltVM as Jolt<
                C,
                M,
                32,
                Fr,
                HyperKZG<Bn254, KeccakTranscript>,
                KeccakTranscript,
            >>::prove(io_device, trace, preprocessing.clone());
        let _verification_result =
            RV32IJoltVM::verify(preprocessing, proof, commitments, debug_info);
    }
}
