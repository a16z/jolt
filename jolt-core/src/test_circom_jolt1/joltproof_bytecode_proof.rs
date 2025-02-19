use super::{
    struct_fq::FqCircom,
    sum_check_gkr::{convert_from_batched_GKRProof_to_circom, BatchedGrandProductProofCircom},
};
use crate::{
    jolt::vm::bytecode::BytecodeProof,
    lasso::memory_checking::MultisetHashes,
    poly::{commitment::hyperkzg::HyperKZG, unipoly::UniPoly},
    subprotocols::grand_product::BatchedGrandProductProof,
    utils::poseidon_transcript::PoseidonTranscript,
};
use ark_bn254::{Bn254, Fr as Scalar};
use core::fmt;

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct MultiSethashesCircom {
    pub read_hashes: Vec<FqCircom>,
    pub write_hashes: Vec<FqCircom>,
    pub init_hashes: Vec<FqCircom>,
    pub final_hashes: Vec<FqCircom>,
}

impl fmt::Debug for MultiSethashesCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"{{
                "read_hashes": {:?},
                "write_hashes": {:?},
                "init_hashes": {:?},
                "final_hashes": {:?}
            }}"#,
            self.read_hashes, self.write_hashes, self.init_hashes, self.final_hashes,
        )
    }
}

pub fn convert_multiset_hashes_to_circom(
    multiset_hash: &MultisetHashes<Scalar>,
) -> MultiSethashesCircom {
    let mut read_hashes = Vec::new();

    for i in 0..multiset_hash.read_hashes.len() {
        read_hashes.push(FqCircom(multiset_hash.read_hashes[i].clone()));
    }
    let mut write_hashes = Vec::new();
    for i in 0..multiset_hash.write_hashes.len() {
        write_hashes.push(FqCircom(multiset_hash.write_hashes[i].clone()))
    }
    let mut init_hashes = Vec::new();
    for i in 0..multiset_hash.init_hashes.len() {
        init_hashes.push(FqCircom(multiset_hash.init_hashes[i].clone()));
    }
    let mut final_hashes = Vec::new();
    for i in 0..multiset_hash.final_hashes.len() {
        final_hashes.push(FqCircom(multiset_hash.final_hashes[i].clone()));
    }
    MultiSethashesCircom {
        read_hashes,
        write_hashes,
        init_hashes,
        final_hashes,
    }
}

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct BytecodeProofCircom {
    pub multiset_hashes: MultiSethashesCircom,
    pub read_write_grand_product: BatchedGrandProductProofCircom,
    pub init_final_grand_product: BatchedGrandProductProofCircom,
    pub openings: Vec<FqCircom>,
}

impl fmt::Debug for BytecodeProofCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"{{
                "multiset_hashes": {:?},
                "read_write_grand_product": {:?},
                "init_final_grand_product": {:?},
                "openings": {:?}
            }}"#,
            self.multiset_hashes,
            self.read_write_grand_product,
            self.init_final_grand_product,
            self.openings,
        )
    }
}

use crate::lasso::memory_checking::StructuredPolynomialData;

pub fn convert_from_bytecode_proof_to_circom(
    bytecode_proof: BytecodeProof<
        Scalar,
        HyperKZG<Bn254, PoseidonTranscript<Scalar, Scalar>>,
        PoseidonTranscript<Scalar, Scalar>,
    >,
) -> BytecodeProofCircom {
    let mut openings = Vec::new();
    let previous_openings = bytecode_proof.openings;
    for opening in previous_openings.read_write_values() {
        openings.push(FqCircom(opening.clone()))
    }
    for opening in previous_openings.init_final_values() {
        openings.push(FqCircom(opening.clone()));
    }

    // Last 7 init_final values will be update inside verifier
    for i in 0..7 {
        openings.push(FqCircom(Scalar::from(0u8)));
    }

    return BytecodeProofCircom {
        multiset_hashes: convert_multiset_hashes_to_circom(&bytecode_proof.multiset_hashes),
        read_write_grand_product: convert_from_batched_GKRProof_to_circom(
            &bytecode_proof.read_write_grand_product,
        ),
        init_final_grand_product: convert_from_batched_GKRProof_to_circom(
            &bytecode_proof.init_final_grand_product,
        ),
        openings: openings,
    };
}
