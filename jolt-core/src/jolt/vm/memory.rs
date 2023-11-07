use ark_ff::PrimeField;

use crate::{poly::dense_mlpoly::DensePolynomial, subprotocols::grand_product::BGPCInterpretable, lasso::fingerprint_strategy::MemBatchInfo};



pub struct Memory<F: PrimeField> {
    a: DensePolynomial<F>,
    v: DensePolynomial<F>,

    read_t: DensePolynomial<F>
}

impl<F: PrimeField> Memory<F> {
    fn new(read_set: Vec<(F, F, F)>, write_set: Vec<(F, F, F)>, final_set: Vec<(F, F, F)>) -> Self {
        todo!("construct")
    }
}

impl<F: PrimeField> MemBatchInfo for Memory<F> {
    fn ops_size(&self) -> usize {
        todo!()
    }

    fn mem_size(&self) -> usize {
        todo!()
    }

    fn num_memories(&self) -> usize {
        todo!()
    }
}

impl<F: PrimeField> BGPCInterpretable<F> for Memory<F> {
    fn a_ops(&self, memory_index: usize, leaf_index: usize) -> F {
        todo!()
    }

    fn v_mem(&self, memory_index: usize, leaf_index: usize) -> F {
        todo!()
    }

    fn v_ops(&self, memory_index: usize, leaf_index: usize) -> F {
        todo!()
    }

    fn t_final(&self, memory_index: usize, leaf_index: usize) -> F {
        todo!()
    }

    fn t_read(&self, memory_index: usize, leaf_index: usize) -> F {
        todo!()
    }
}

// TODO(sragss): FingerprintStrategy

#[cfg(test)]
mod tests {
    #[test]
    fn prod_layer_proof() {
        todo!()
    }

    #[test]
    fn e2e_mem_checking() {
        todo!()

    }
}