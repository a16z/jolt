use super::{
    grand_product::BatchedGrandProductProofCircom,
    non_native::{convert_to_3_limbs, Fqq},
};
use crate::lasso::memory_checking::MultisetHashes;
use ark_grumpkin::Fr as Scalar;
use std::fmt;

pub struct SparkMemoryCheckingProofCircom {
    pub multiset_hashes: MultiSethashesCircom,
    pub read_write_grand_product: BatchedGrandProductProofCircom,
    pub init_final_grand_product: BatchedGrandProductProofCircom,
    pub openings: Vec<Fqq>,
}
impl fmt::Debug for SparkMemoryCheckingProofCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"{{
            "multiset_hashes": {:?},
            "read_write_grand_product": {:?},
            "init_final_grand_product": {:?},
            "openings": {:?},
            }}"#,
            self.multiset_hashes,
            self.read_write_grand_product,
            self.init_final_grand_product,
            self.openings
        )
    }
}

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct MultiSethashesCircom {
    pub read_hashes: Vec<Fqq>,
    pub write_hashes: Vec<Fqq>,
    pub init_hashes: Vec<Fqq>,
    pub final_hashes: Vec<Fqq>,
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
        read_hashes.push(Fqq {
            element: multiset_hash.read_hashes[i].clone(),
            limbs: convert_to_3_limbs(multiset_hash.read_hashes[i].clone()),
        });
    }
    let mut write_hashes = Vec::new();
    for i in 0..multiset_hash.write_hashes.len() {
        write_hashes.push(Fqq {
            element: multiset_hash.write_hashes[i].clone(),
            limbs: convert_to_3_limbs(multiset_hash.write_hashes[i].clone()),
        });
    }
    let mut init_hashes = Vec::new();
    for i in 0..multiset_hash.init_hashes.len() {
        init_hashes.push(Fqq {
            element: multiset_hash.init_hashes[i].clone(),
            limbs: convert_to_3_limbs(multiset_hash.init_hashes[i].clone()),
        });
    }
    let mut final_hashes = Vec::new();
    for i in 0..multiset_hash.final_hashes.len() {
        final_hashes.push(Fqq {
            element: multiset_hash.final_hashes[i].clone(),
            limbs: convert_to_3_limbs(multiset_hash.final_hashes[i].clone()),
        });
    }
    MultiSethashesCircom {
        read_hashes,
        write_hashes,
        init_hashes,
        final_hashes,
    }
}
