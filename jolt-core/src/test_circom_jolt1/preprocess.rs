use core::fmt;

use ark_bn254::{Bn254, Fr as Scalar};
use crate::utils::poseidon_transcript::PoseidonTranscript;
use super::struct_fq::FqCircom;
#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]


pub struct JoltPreprocessingCircom{
    // pub generators : HyperKZGVerifierKeyCircom,
    pub v_init_final_hash: FqCircom,
    pub bytecode_words_hash : FqCircom
}

impl fmt::Debug for JoltPreprocessingCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
        r#"{{
                "v_init_final_hash": {:?},
                "bytecode_words_hash": {:?}
            }}"#,
            // self.generators,
            self.v_init_final_hash,
            self.bytecode_words_hash
        )
    }
}



use crate::jolt::vm::rv32i_vm::{C, M};
use crate::jolt::vm::JoltPreprocessing;
use crate::poly::commitment::hyperkzg::HyperKZG;



pub fn convert_joltpreprocessing_to_circom(jolt_preprocessing: &JoltPreprocessing<C, Scalar, HyperKZG<Bn254, PoseidonTranscript<Scalar, Scalar>>,PoseidonTranscript<Scalar, Scalar>>) -> JoltPreprocessingCircom{
    JoltPreprocessingCircom{
        v_init_final_hash: FqCircom(jolt_preprocessing.bytecode.v_init_final_hash),
        bytecode_words_hash: FqCircom(jolt_preprocessing.read_write_memory.hash),
        // generators: convert_hyperkzg_verifier_key_to_hyperkzg_verifier_key_circom(jolt_preprocessing.generators.1),
        
    }
}

