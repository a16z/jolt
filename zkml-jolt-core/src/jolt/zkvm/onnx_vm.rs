use jolt_core::jolt::{
    instruction::{InstructionLookup, LookupQuery},
    lookup_table::LookupTables,
};

use crate::jolt::instruction::{add::ADD, mul::MUL, sub::SUB};

pub const WORD_SIZE: usize = 32;

macro_rules! define_lookup_enum {
    (
        enum $enum_name:ident,
        const $word_size:ident,
        trait $trait_name:ident,
        $($variant:ident : $inner:ty),+ $(,)?
    ) => {
        #[derive(Debug)]
        pub enum $enum_name {
            $(
                $variant($inner),
            )+
        }

        impl $trait_name<$word_size> for $enum_name {
            fn to_instruction_inputs(&self) -> (u64, i64) {
                match self {
                    $(
                        $enum_name::$variant(inner) => inner.to_instruction_inputs(),
                    )+
                }
            }

            fn to_lookup_index(&self) -> u64 {
                match self {
                    $(
                        $enum_name::$variant(inner) => inner.to_lookup_index(),
                    )+
                }
            }

            fn to_lookup_operands(&self) -> (u64, u64) {
                match self {
                    $(
                        $enum_name::$variant(inner) => inner.to_lookup_operands(),
                    )+
                }
            }

            fn to_lookup_output(&self) -> u64 {
                match self {
                    $(
                        $enum_name::$variant(inner) => inner.to_lookup_output(),
                    )+
                }
            }
        }
    };
}

define_lookup_enum!(
    enum ONNXLookup,
    const WORD_SIZE,
    trait LookupQuery,
    Add: ADD<WORD_SIZE>,
    Sub: SUB<WORD_SIZE>,
    Mul: MUL<WORD_SIZE>,
);

impl InstructionLookup<WORD_SIZE> for ONNXLookup {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        match self {
            ONNXLookup::Add(add) => add.lookup_table(),
            ONNXLookup::Sub(sub) => sub.lookup_table(),
            ONNXLookup::Mul(mul) => mul.lookup_table(),
        }
    }
}

#[cfg(test)]
mod e2e_tests {

    use crate::{
        jolt::zkvm::{JoltProverPreprocessing, JoltSNARK},
        program::ONNXProgram,
    };
    use ark_bn254::Fr;
    use jolt_core::poly::commitment::dory::DoryCommitmentScheme;
    use jolt_core::utils::transcript::KeccakTranscript;
    use onnx_tracer::{custom_addsubmul_model, logger::init_logger, tensor::Tensor};

    type PCS = DoryCommitmentScheme<KeccakTranscript>;

    #[test]
    fn test_custom_addsubmul() {
        // --- Preprocessing ---
        let custom_addsubmul_model = custom_addsubmul_model();
        let program_bytecode = onnx_tracer::decode_model(custom_addsubmul_model.clone());
        println!("Program code: {program_bytecode:#?}");
        let pp: JoltProverPreprocessing<Fr, PCS, KeccakTranscript> =
            JoltSNARK::prover_preprocess(program_bytecode);

        // --- Proving ---
        let input = Tensor::new(Some(&[60]), &[1]).unwrap();
        let execution_trace = onnx_tracer::execution_trace(custom_addsubmul_model, &input);
        println!("Execution trace: {execution_trace:#?}");
        let snark: JoltSNARK<Fr, PCS, KeccakTranscript> =
            JoltSNARK::prove(pp.clone(), execution_trace);

        // --- Verification ---
        snark.verify((&pp).into()).unwrap();
    }

    #[test]
    fn test_subgraph() {
        init_logger();
        let subgraph_program = ONNXProgram {
            model_path: "../onnx-tracer/models/subgraph/network.onnx".into(),
            inputs: Tensor::new(Some(&[1, 2, 3, 4]), &[1, 4]).unwrap(), // Example input
        };
        let program_bytecode = subgraph_program.decode();

        println!("Program decoded");
        println!("Program code: {program_bytecode:#?}",);

        // Test that the addresses of a subgraph are monotonically increasing
        let mut i = 0;
        for instr in program_bytecode {
            assert!(instr.address > i);
            i = instr.address;
        }

        subgraph_program.trace();
    }
}
