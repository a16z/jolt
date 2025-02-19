use core::fmt;

use ark_bn254::{Bn254, Fr as Scalar};
use crate::jolt::vm::{bytecode::BytecodePreprocessing, JoltPreprocessing};
use crate::jolt::vm::read_write_memory::ReadWriteMemoryPreprocessing;
use crate::poly::commitment::hyperkzg::HyperKZG;
use crate::utils::poseidon_transcript::PoseidonTranscript;
use super::struct_fq::{FqCircom, ReadWriteMemoryPreprocessingCircom};
use crate::jolt::vm::rv32i_vm::C;




#[derive(Clone,  Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct PIProofCircom{
    pub bytecode: BytecodePreprocessingCircom,
    pub read_write_memory: ReadWriteMemoryPreprocessingCircom
}


impl fmt::Debug for PIProofCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
        r#"{{
                "bytecode": {:?},
                "read_write_memory": {:?}
            }}"#,
            // self.generators,
            self.bytecode,
            self.read_write_memory
        )
    }
}




#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct BytecodePreprocessingCircom{
    pub v_init_final: Vec<Vec<FqCircom>>,
}

impl fmt::Debug for BytecodePreprocessingCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
                r#"{{
                    "v_init_final": {:?}
                }}"#,
            self.v_init_final
        )
    }
}

pub fn convert_byte_code_preprocessing_to_circom(bytecode_preprocess: BytecodePreprocessing<Scalar>) -> BytecodePreprocessingCircom{
    let mut v_init_final = Vec::new();
    for i in 0..bytecode_preprocess.v_init_final.len(){
        let mut temp = Vec::new();
        for j in 0..bytecode_preprocess.v_init_final[i].len(){
            temp.push(
                FqCircom(bytecode_preprocess.v_init_final[i].Z[j]),
        );
        }
        v_init_final.push(temp);
    }
    BytecodePreprocessingCircom{
        v_init_final: v_init_final,
    }
}


pub fn convert_rw_mem_processing(rw_mem_processing: ReadWriteMemoryPreprocessing) -> ReadWriteMemoryPreprocessingCircom {
    let mut bytecode_words = Vec::new();
    for i in 0..rw_mem_processing.bytecode_words.len(){
        bytecode_words.push(FqCircom(Scalar::from(rw_mem_processing.bytecode_words[i])));
    }
    ReadWriteMemoryPreprocessingCircom {
        bytecode_words: bytecode_words,
    }
}

pub fn convert_piproof_to_circom(jolt_preprocessing: JoltPreprocessing<C, Scalar, HyperKZG<Bn254, PoseidonTranscript<Scalar, Scalar>>,PoseidonTranscript<Scalar, Scalar>>) -> PIProofCircom{
    PIProofCircom{
        // generators: convert_hyperkzg_verifier_key_to_hyperkzg_verifier_key_circom(jolt_preprocessing.generators.1),
        bytecode: convert_byte_code_preprocessing_to_circom(jolt_preprocessing.bytecode),
        read_write_memory: convert_rw_mem_processing(jolt_preprocessing.read_write_memory),
    }
}
