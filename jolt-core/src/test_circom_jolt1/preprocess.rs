use core::fmt;

use super::struct_fq::FqCircom;
use crate::utils::poseidon_transcript::PoseidonTranscript;
use ark_bn254::{Bn254, Fr as Scalar};
#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]

pub struct JoltPreprocessingCircom {
    pub v_init_final_hash: FqCircom,
    pub bytecode_words_hash: FqCircom,
}

impl fmt::Debug for JoltPreprocessingCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"{{
                "v_init_final_hash": {:?},
                "bytecode_words_hash": {:?}
            }}"#,
            self.v_init_final_hash, self.bytecode_words_hash
        )
    }
}

use crate::jolt::vm::rv32i_vm::{C, M};
use crate::jolt::vm::JoltPreprocessing;
use crate::poly::commitment::hyperkzg::HyperKZG;

pub fn convert_joltpreprocessing_to_circom(
    jolt_preprocessing: &JoltPreprocessing<
        C,
        Scalar,
        HyperKZG<Bn254, PoseidonTranscript<Scalar, Scalar>>,
        PoseidonTranscript<Scalar, Scalar>,
    >,
) -> JoltPreprocessingCircom {
    JoltPreprocessingCircom {
        v_init_final_hash: FqCircom(jolt_preprocessing.bytecode.v_init_final_hash),
        bytecode_words_hash: FqCircom(jolt_preprocessing.read_write_memory.hash),
    }
}
