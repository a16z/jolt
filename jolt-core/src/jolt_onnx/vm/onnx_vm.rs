use crate::field::JoltField;
use enum_dispatch::enum_dispatch;
use rand::{prelude::StdRng, RngCore};
use serde::{Deserialize, Serialize};
use std::any::TypeId;
use strum::{EnumCount, IntoEnumIterator};
use strum_macros::{EnumCount as EnumCountMacro, EnumIter};

use super::JoltProof;
use crate::jolt::instruction::{
    and::ANDInstruction, or::ORInstruction, xor::XORInstruction, JoltInstruction,
    JoltInstructionSet, SubtableIndices,
};
use crate::jolt::subtable::{
    and::AndSubtable, identity::IdentitySubtable, or::OrSubtable, xor::XorSubtable,
    JoltSubtableSet, LassoSubtable, SubtableId,
};
use crate::jolt_onnx::{instruction::relu::ReLUInstruction, subtable::is_pos::IsPosSubtable};

// TODO: Remove these duplicated macros. Original definitions are in jolt-core/src/jolt/vm/rv32i_vm.rs

/// Generates an enum out of a list of JoltInstruction types. All JoltInstruction methods
/// are callable on the enum type via enum_dispatch.
macro_rules! instruction_set {
    ($enum_name:ident, $($alias:ident: $struct:ty),+) => {
        #[allow(non_camel_case_types)]
        #[repr(u8)]
        #[derive(Copy, Clone, Debug, PartialEq, EnumIter, EnumCountMacro, Serialize, Deserialize)]
        #[enum_dispatch(JoltInstruction)]
        pub enum $enum_name {
            $($alias($struct)),+
        }
        impl JoltInstructionSet for $enum_name {}
        impl $enum_name {
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
        #[allow(non_camel_case_types)]
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
            { panic!("Unexpected subtable id {:?}", subtable_id) } // TODO(moodlezoup): better error handling
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

pub const C_ONNX: usize = 4;
pub const M_ONNX: usize = 1 << 16;
const WORD_SIZE: usize = 32;

instruction_set!(
  ONNX,
  AND: ANDInstruction<WORD_SIZE>,
  OR: ORInstruction<WORD_SIZE>,
  XOR: XORInstruction<WORD_SIZE>,
  ReLU: ReLUInstruction
);

subtable_enum!(
  ONNXSubtables,
  AND: AndSubtable<F>,
  OR: OrSubtable<F>,
  XOR: XorSubtable<F>,
  IDENTITY: IdentitySubtable<F>,
  IS_POS: IsPosSubtable<F>
);

pub type ONNXJoltVM<F, PCS, ProofTranscript> =
    JoltProof<C_ONNX, M_ONNX, F, PCS, ONNX, ONNXSubtables<F>, ProofTranscript>;

#[cfg(test)]
mod tests {
    use crate::poly::commitment::hyperkzg::HyperKZG;
    use crate::utils::transcript::{KeccakTranscript, Transcript};
    use crate::{
        field::JoltField,
        jolt_onnx::trace::onnx::{JoltONNXDevice, ONNXParser},
        poly::commitment::commitment_scheme::CommitmentScheme,
    };
    use ark_bn254::{Bn254, Fr};

    use super::ONNXJoltVM;

    fn test_e2e_with<F, PCS, ProofTranscript>(model_path: &str)
    where
        F: JoltField,
        PCS: CommitmentScheme<ProofTranscript, Field = F>,
        ProofTranscript: Transcript,
    {
        // Setup model and get trace (input for proving)
        let graph = ONNXParser::load_model(model_path).unwrap();
        let trace = graph.trace(); // TODO: make this more opaque to the user

        // Generate preprocessing
        let pp = ONNXJoltVM::<F, PCS, ProofTranscript>::prover_preprocess(1 << 20);

        // Prove
        let io = JoltONNXDevice::new(graph.input_count as u64, graph.output_count as u64);
        let (snark, commitments, verifier_io, _) =
            ONNXJoltVM::<F, PCS, ProofTranscript>::prove(io, trace, pp.clone());

        // Verify
        snark
            .verify(pp.shared, commitments, verifier_io, None)
            .unwrap();
    }

    #[test]
    fn test_bitwise_e2e_hkzg() {
        test_e2e_with::<Fr, HyperKZG<Bn254, KeccakTranscript>, KeccakTranscript>(
            "./onnx/bitwise_test.onnx",
        )
    }

    #[test]
    fn test_bitwise_with_relu_e2e_hkzg() {
        test_e2e_with::<Fr, HyperKZG<Bn254, KeccakTranscript>, KeccakTranscript>(
            "./onnx/bitwise_with_relu.onnx",
        )
    }

    #[test]
    fn test_add_mul_e2e_hkzg() {
        test_e2e_with::<Fr, HyperKZG<Bn254, KeccakTranscript>, KeccakTranscript>(
            "./onnx/add_mul.onnx",
        )
    }

    #[test]
    fn test_add_mul_sub_shift_e2e_hkzg() {
        test_e2e_with::<Fr, HyperKZG<Bn254, KeccakTranscript>, KeccakTranscript>(
            "./onnx/add_mul_sub_shift.onnx",
        )
    }
}
