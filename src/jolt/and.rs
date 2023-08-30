use ark_ff::PrimeField;
use ark_std::log2;

use crate::utils::split_bits;

use crate::jolt::jolt_strategy::{JoltStrategy, SubtableStrategy, InstructionStrategy};


pub enum AndVMInstruction {
    And(u64, u64),
}

pub struct AndVM {}

impl<F: PrimeField> JoltStrategy<F> for AndVM {
    type Instruction = AndVMInstruction;

    fn instructions() -> Vec<Box<dyn InstructionStrategy<F>>> {
        vec![Box::new(AndInstruction::new(4, 1 << 16))]
    }
}

pub struct AndInstruction {
    C: usize,
    M: usize,
}
impl AndInstruction {
    pub fn new(C: usize, M: usize) -> Self {
        Self { C, M }
    }
}

impl<F: PrimeField> InstructionStrategy<F> for AndInstruction {
    fn subtables(&self) -> Vec<Box<dyn SubtableStrategy<F>>> {
        vec![Box::new(AndSubtable::new(self.C, self.M))]
    }

    fn combine_lookups(&self, vals: &[F]) -> F {
        assert_eq!(vals.len(), self.C);

        let increment = log2(self.M) as usize;
        let mut sum = F::zero();
        for i in 0..self.C {
          let weight: u64 = 1u64 << (i * increment);
          sum += F::from(weight) * vals[i];
        }
        sum
    }

    fn g_poly_degree(&self) -> usize {
        1
    }
}

pub struct AndSubtable {
    C: usize,
    M: usize
}
impl AndSubtable {
    fn new(C: usize, M: usize) -> Self {
        Self { C, M }
    }
}

impl<F: PrimeField> SubtableStrategy<F> for AndSubtable {
    fn dimensions(&self) -> usize {
        self.C
    }

    fn memory_size(&self) -> usize {
        self.M
    }

    fn materialize(&self) -> Vec<F> {
        let M: usize = <AndSubtable as SubtableStrategy<F>>::memory_size(self);

        let mut materialized: Vec<F> = Vec::with_capacity(M);
        let bits_per_operand = (log2(M) / 2) as usize;
    
        // Materialize table in counting order where lhs | rhs counts 0->m
        for idx in 0..M {
          let (lhs, rhs) = split_bits(idx, bits_per_operand);
          let row = F::from((lhs & rhs) as u64);
          materialized.push(row);
        }
    
        materialized
    }

    fn evaluate_mle(&self, point: &[F]) -> F {
        debug_assert!(point.len() % 2 == 0);
        let b = point.len() / 2;
        let (x, y) = point.split_at(b);
    
        let mut result = F::zero();
        for i in 0..b {
          result += F::from(1u64 << (i)) * x[b - i - 1] * y[b - i - 1];
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::PrimeField;
    use ark_std::{test_rng, log2};
    use merlin::Transcript;
    use rand_chacha::rand_core::RngCore;
    use ark_curve25519::{Fr, EdwardsProjective};

    use crate::{lasso::{densified::DensifiedRepresentation, surge::{SparsePolyCommitmentGens, SparsePolynomialEvaluationProof}}, jolt::and::AndVM, utils::random::RandomTape};

    pub fn gen_indices<const C: usize>(sparsity: usize, memory_size: usize) -> Vec<Vec<usize>> {
        let mut rng = test_rng();
        let mut all_indices: Vec<Vec<usize>> = Vec::new();
        for _ in 0..sparsity {
          let indices = vec![rng.next_u64() as usize % memory_size; C];
          all_indices.push(indices);
        }
        all_indices
    }

    pub fn gen_random_point<F: PrimeField>(memory_bits: usize) -> Vec<F> {
        let mut rng = test_rng();
        let mut r_i: Vec<F> = Vec::with_capacity(memory_bits);
        for _ in 0..memory_bits {
            r_i.push(F::rand(&mut rng));
        }
        r_i
    }

    #[test]
    fn e2e() {
        const C: usize = 4;
        const S: usize = 1 << 8;
        const M: usize = 1 << 16;

        let log_m = log2(M) as usize;
        let log_s: usize = log2(S) as usize;

        let nz: Vec<Vec<usize>> = gen_indices::<C>(S, M);
        let r: Vec<Fr> = gen_random_point::<Fr>(log_s);

        let mut dense: DensifiedRepresentation<Fr, AndVM> = DensifiedRepresentation::from_lookup_indices(&nz, log_m);
        let gens = SparsePolyCommitmentGens::<EdwardsProjective>::new(b"gens_sparse_poly", C, S, C, log_m);
        let commitment = dense.commit::<EdwardsProjective>(&gens);
        let mut random_tape = RandomTape::new(b"proof");
        let mut prover_transcript = Transcript::new(b"example");
        let proof = SparsePolynomialEvaluationProof::<EdwardsProjective, AndVM>::prove(
            &mut dense,
            &r,
            &gens,
            &mut prover_transcript,
            &mut random_tape
        );

        let mut verify_transcript = Transcript::new(b"example");
        proof.verify(&commitment, &r, &gens, &mut verify_transcript).expect("should verify");
    }
}