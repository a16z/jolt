//! This module defines the [`ONNXJoltVM`] type and its instruction set, which is a Jolt VM & ISA for ONNX models.

use super::JoltProof;
use crate::field::JoltField;
use crate::jolt::instruction::add::ADDInstruction;
use crate::jolt::instruction::{JoltInstruction, JoltInstructionSet, SubtableIndices};
use crate::jolt::subtable::{
    identity::IdentitySubtable, JoltSubtableSet, LassoSubtable, SubtableId,
};
use crate::jolt_onnx::{instruction::relu::ReLUInstruction, subtable::is_pos::IsPosSubtable};
use enum_dispatch::enum_dispatch;
use rand::{prelude::StdRng, RngCore};
use serde::{Deserialize, Serialize};
use std::any::TypeId;
use strum::{EnumCount, IntoEnumIterator};
use strum_macros::{EnumCount as EnumCountMacro, EnumIter};

// TODO: Remove these duplicated macros. Original definitions are in jolt-core/src/jolt/vm/rv32i_vm.rs

/// Generates an enum out of a list of JoltInstruction types. All JoltInstruction methods
/// are callable on the enum type via enum_dispatch.
macro_rules! instruction_set {
    ($enum_name:ident, $($alias:ident: $struct:ty),+) => {
        #[allow(non_camel_case_types, missing_docs)]
        #[repr(u8)]
        #[derive(Copy, Clone, Debug, PartialEq, EnumIter, EnumCountMacro, Serialize, Deserialize)]
        #[enum_dispatch(JoltInstruction)]
        pub enum $enum_name {
            $($alias($struct)),+
        }
        impl JoltInstructionSet for $enum_name {}
        impl $enum_name {
            /// Create a random instruction from the enum.
            pub fn random_instruction(rng: &mut StdRng) -> Self {
                let index = rng.next_u64() as usize % $enum_name::COUNT;
                let instruction = $enum_name::iter()
                    .enumerate()
                    .filter(|(i, _)| *i == index)
                    .map(|(_, x)| x)
                    .next()
                    .unwrap();
                instruction.random(rng)
            }
        }
        // Need a default so that we can derive EnumIter on `JoltR1CSInputs`
        impl Default for $enum_name {
            fn default() -> Self {
                $enum_name::iter().collect::<Vec<_>>()[0]
            }
        }
    };
}

/// Generates an enum out of a list of LassoSubtable types. All LassoSubtable methods
/// are callable on the enum type via enum_dispatch.
macro_rules! subtable_enum {
    ($enum_name:ident, $($alias:ident: $struct:ty),+) => {
        #[allow(non_camel_case_types, missing_docs)]
        #[repr(u8)]
        #[enum_dispatch(LassoSubtable<F>)]
        #[derive(EnumCountMacro, EnumIter)]
        pub enum $enum_name<F: JoltField> { $($alias($struct)),+ }
        impl<F: JoltField> From<SubtableId> for $enum_name<F> {
          fn from(subtable_id: SubtableId) -> Self {
            $(
              if subtable_id == TypeId::of::<$struct>() {
                $enum_name::from(<$struct>::new())
              } else
            )+
            { panic!("Unexpected subtable id {:?}", subtable_id) }
          }
        }

        impl<F: JoltField> From<$enum_name<F>> for usize {
            fn from(subtable: $enum_name<F>) -> usize {
                // Discriminant: https://doc.rust-lang.org/reference/items/enumerations.html#pointer-casting
                let byte = unsafe { *(&subtable as *const $enum_name<F> as *const u8) };
                byte as usize
            }
        }
        impl<F: JoltField> JoltSubtableSet<F> for $enum_name<F> {}
    };
}

/// C constant in Jolt paper
pub const C_ONNX: usize = 4;
/// Size of subtable entries
pub const M_ONNX: usize = 1 << 16;
const WORD_SIZE: usize = 32;

instruction_set!(
  ONNXInstructionSet,
  ReLU: ReLUInstruction,
  ADD: ADDInstruction<WORD_SIZE>
);

subtable_enum!(
  ONNXSubtables,
  IDENTITY: IdentitySubtable<F>,
  IS_POS: IsPosSubtable<F>
);

/// The ONNX Jolt VM type, which is a Jolt VM for ONNX models.
pub type ONNXJoltVM<F, PCS, ProofTranscript> =
    JoltProof<C_ONNX, M_ONNX, F, PCS, ONNXInstructionSet, ONNXSubtables<F>, ProofTranscript>;

#[cfg(test)]
mod tests {
    use super::ONNXJoltVM;
    use crate::jolt_onnx::onnx_host::ONNXProgram;
    use crate::jolt_onnx::utils::random_floatvec;
    use crate::poly::commitment::hyperkzg::HyperKZG;
    use crate::utils::transcript::{KeccakTranscript, Transcript};
    use crate::{field::JoltField, poly::commitment::commitment_scheme::CommitmentScheme};
    use ark_bn254::{Bn254, Fr};
    use ark_std::test_rng;

    fn test_e2e_with<F, PCS, ProofTranscript>(onnx_program: &ONNXProgram)
    where
        F: JoltField,
        PCS: CommitmentScheme<ProofTranscript, Field = F>,
        ProofTranscript: Transcript,
    {
        // Setup model and get trace (input for proving)
        let model = onnx_program.decode();

        // Generate preprocessing
        let pp = ONNXJoltVM::<F, PCS, ProofTranscript>::prover_preprocess(&model, 1 << 20);

        // Prove
        let (io, trace) = onnx_program.trace();
        let (snark, commitments, verifier_io, _) =
            ONNXJoltVM::<F, PCS, ProofTranscript>::prove(io, trace, pp.clone());

        // Verify
        snark
            .verify(pp.shared, commitments, verifier_io, None)
            .unwrap();
    }

    #[test]
    fn test_perceptron() {
        test_e2e_with::<Fr, HyperKZG<Bn254, KeccakTranscript>, KeccakTranscript>(&ONNXProgram::new(
            "onnx/mlp/perceptron.onnx",
            Some(random_floatvec(&mut test_rng(), 10)),
        ))
    }

    #[test]
    fn test_perceptron_2() {
        test_e2e_with::<Fr, HyperKZG<Bn254, KeccakTranscript>, KeccakTranscript>(&ONNXProgram::new(
            "onnx/mlp/perceptron_2.onnx",
            Some(random_floatvec(&mut test_rng(), 4)),
        ))
    }

    #[test]
    fn test_accuracy() {
        test_e2e_with::<Fr, HyperKZG<Bn254, KeccakTranscript>, KeccakTranscript>(&ONNXProgram::new(
            "onnx/mlp/accuracy.onnx",
            Some(random_floatvec(&mut test_rng(), 41)),
        ))
    }
}
