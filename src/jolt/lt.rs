use std::marker::PhantomData;

use ark_ff::PrimeField;
use ark_std::log2;

use crate::utils::split_bits;

use super::jolt_strategy::{InstructionStrategy, JoltStrategy, SubtableStrategy};

pub enum LTVMInstruction {
  LT(u64, u64),
}

pub struct LTVM {}

impl<F: PrimeField> JoltStrategy<F> for LTVM {
  type Instruction = LTVMInstruction;

  fn instructions() -> Vec<Box<dyn InstructionStrategy<F>>> {
    vec![Box::new(LTInstruction {
      _marker: PhantomData::<F>,
    })]
  }
}

pub struct LTInstruction<F: PrimeField> {
  _marker: PhantomData<F>,
}

impl<F: PrimeField> InstructionStrategy<F> for LTInstruction<F> {
  fn subtables(&self) -> Vec<Box<dyn SubtableStrategy<F>>> {
    vec![
      Box::new(LTSubtable {
        _marker: PhantomData::<F>,
      }),
      Box::new(EQSubtable {
        _marker: PhantomData::<F>,
      }),
    ]
  }

  fn combine_lookups(&self, vals: &[F]) -> F {
    assert_eq!(vals.len(), self.num_memories());
    let mut sum = F::zero();
    let mut eq_prod = F::one();

    let C: usize = self.subtables()[0].dimensions();

    for i in 0..C {
      sum += vals[2 * i] * eq_prod;
      eq_prod *= vals[2 * i + 1];
    }
    sum
  }

  fn g_poly_degree(&self) -> usize {
    4
  }
}

pub struct LTSubtable<F: PrimeField> {
  _marker: PhantomData<F>,
}
impl<F: PrimeField> SubtableStrategy<F> for LTSubtable<F> {
  fn dimensions(&self) -> usize {
    4
  }

  fn memory_size(&self) -> usize {
    1 << 16
  }

  fn materialize(&self) -> Vec<F> {
    let M: usize = self.memory_size();
    let bits_per_operand = (log2(M) / 2) as usize;

    let mut materialized_lt: Vec<F> = Vec::with_capacity(M);

    // Materialize table in counting order where lhs | rhs counts 0->m
    for idx in 0..M {
      let (lhs, rhs) = split_bits(idx, bits_per_operand);
      materialized_lt.push(F::from((lhs < rhs) as u64));
    }

    materialized_lt
  }

  fn evaluate_mle(&self, point: &[F]) -> F {
    debug_assert!(point.len() % 2 == 0);
    let b = point.len() / 2;
    let (x, y) = point.split_at(b);

    let mut result = F::zero();
    let mut eq_term = F::one();
    for i in 0..b {
      result += (F::one() - x[i]) * y[i] * eq_term;
      eq_term *= F::one() - x[i] - y[i] + F::from(2u64) * x[i] * y[i];
    }
    result
  }
}

pub struct EQSubtable<F: PrimeField> {
  _marker: PhantomData<F>,
}
impl<F: PrimeField> SubtableStrategy<F> for EQSubtable<F> {
  fn dimensions(&self) -> usize {
    4
  }

  fn memory_size(&self) -> usize {
    1 << 16
  }

  fn materialize(&self) -> Vec<F> {
    let M: usize = self.memory_size();
    let bits_per_operand = (log2(M) / 2) as usize;

    let mut materialized_eq: Vec<F> = Vec::with_capacity(M);

    // Materialize table in counting order where lhs | rhs counts 0->m
    for idx in 0..M {
      let (lhs, rhs) = split_bits(idx, bits_per_operand);
      materialized_eq.push(F::from((lhs == rhs) as u64));
    }

    materialized_eq
  }

  fn evaluate_mle(&self, point: &[F]) -> F {
    debug_assert!(point.len() % 2 == 0);
    let b = point.len() / 2;
    let (x, y) = point.split_at(b);

    let mut eq_term = F::one();
    for i in 0..b {
      eq_term *= F::one() - x[i] - y[i] + F::from(2u64) * x[i] * y[i];
    }
    eq_term
  }
}

#[cfg(test)]
mod tests {
  use ark_curve25519::{EdwardsProjective, Fr};
  use ark_ff::PrimeField;
  use ark_std::{log2, test_rng, One, Zero};
  use merlin::Transcript;
  use rand_chacha::rand_core::RngCore;

  use crate::{
    jolt::{jolt_strategy::JoltStrategy, lt::LTVM},
    lasso::{
      densified::DensifiedRepresentation,
      surge::{SparsePolyCommitmentGens, SparsePolynomialEvaluationProof},
    },
    poly::{dense_mlpoly::DensePolynomial, eq_poly::EqPolynomial},
    subtables::Subtables,
    utils::{ff_bitvector_dbg, index_to_field_bitvector, random::RandomTape, split_bits},
  };

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

    let mut dense: DensifiedRepresentation<Fr, LTVM> =
      DensifiedRepresentation::from_lookup_indices(&nz, log_m);
    let gens =
      SparsePolyCommitmentGens::<EdwardsProjective>::new(b"gens_sparse_poly", C, S, 2 * C, log_m);
    let commitment = dense.commit::<EdwardsProjective>(&gens);
    let mut random_tape = RandomTape::new(b"proof");
    let mut prover_transcript = Transcript::new(b"example");
    let proof = SparsePolynomialEvaluationProof::<EdwardsProjective, LTVM>::prove(
      &mut dense,
      &r,
      &gens,
      &mut prover_transcript,
      &mut random_tape,
    );

    let mut verify_transcript = Transcript::new(b"example");
    proof
      .verify(&commitment, &r, &gens, &mut verify_transcript)
      .expect("should verify");
  }

  #[test]
  fn to_lookup_polys() {
    const C: usize = 4;
    const S: usize = 1 << 3;
    const M: usize = 1 << 16;

    let nz: Vec<Vec<usize>> = vec![
      vec![
        1,            // LT: True, EQ: False
        1 << 8,       // LT: False, EQ: False
        1 + (1 << 8), // LT: False, EQ: True
        5,
        4,
        3,
        2,
        1
      ];
      C
    ];
    let (lhs, rhs) = split_bits(1 + (1 << 8), 8);
    assert_eq!(lhs, rhs);

    let subtable_entries: Vec<Vec<Fr>> = LTVM::materialize_subtables();
    assert_eq!(subtable_entries[0][0], Fr::zero()); // LT 0 > 0 = 0
    assert_eq!(subtable_entries[0][1], Fr::one()); // LT 1 > 0 = 1
    assert_eq!(subtable_entries[0][(1 << 15) - 1], Fr::one()); // LT MAX > 0 = 1
    assert_eq!(subtable_entries[0][1 << 15], Fr::zero()); // LT 0 > 1 = 0

    let lookup_polys: Vec<DensePolynomial<Fr>> = LTVM::to_lookup_polys(&subtable_entries, &nz, S);
    // LT on even indices
    assert_eq!(lookup_polys[0][0], Fr::one()); // True
    assert_eq!(lookup_polys[0][1], Fr::zero()); // False
    assert_eq!(lookup_polys[0][2], Fr::zero()); // False
    assert_eq!(lookup_polys[2][0], Fr::one()); // True
    assert_eq!(lookup_polys[4][1], Fr::zero()); // False
    assert_eq!(lookup_polys[6][2], Fr::zero()); // False

    // EQ on odd indices
    assert_eq!(lookup_polys[1][0], Fr::zero()); // False
    assert_eq!(lookup_polys[1][1], Fr::zero()); // False
    assert_eq!(lookup_polys[1][2], Fr::one()); // True
    assert_eq!(lookup_polys[3][0], Fr::zero()); // False
    assert_eq!(lookup_polys[5][1], Fr::zero()); // False
    assert_eq!(lookup_polys[7][2], Fr::one()); // True
  }

  #[test]
  fn subtable_construction() {
    const C: usize = 4;
    const S: usize = 1 << 2;
    const M: usize = 1 << 16;

    let log_m = log2(M) as usize;
    let log_s: usize = log2(S) as usize;

    // Densified takes indices - 'sparsity' x 'C' sized and rotates to make dense.dim_usize
    let nz: Vec<Vec<usize>> = vec![
      vec![1; C],
      vec![1 << 8; C],
      vec![1 + (1 << 8); C],
      vec![5; C],
    ];

    let dense: DensifiedRepresentation<Fr, LTVM> =
      DensifiedRepresentation::from_lookup_indices(&nz, log_m);
    let subtables = Subtables::<Fr, LTVM>::new(&dense.dim_usize, dense.s);

    let demo_poly = DensePolynomial::new(vec![Fr::from(5), Fr::from(6), Fr::from(7), Fr::from(8)]);
    assert_eq!(
      demo_poly.evaluate(&index_to_field_bitvector(0, 2)),
      demo_poly[0]
    );
    assert_eq!(
      demo_poly.evaluate(&index_to_field_bitvector(1, 2)),
      demo_poly[1]
    );
    assert_eq!(
      demo_poly.evaluate(&index_to_field_bitvector(2, 2)),
      demo_poly[2]
    );
    assert_eq!(
      demo_poly.evaluate(&index_to_field_bitvector(3, 2)),
      demo_poly[3]
    );

    // LT
    let eval_point = index_to_field_bitvector(0, log_s);
    let eval = subtables.lookup_polys[0].evaluate(&eval_point);
    assert_eq!(eval, Fr::one());

    let eval_point = index_to_field_bitvector(1, log_s);
    let eval = subtables.lookup_polys[0].evaluate(&eval_point);
    assert_eq!(eval, Fr::zero());

    let eval_point = index_to_field_bitvector(2, log_s);
    let eval = subtables.lookup_polys[0].evaluate(&eval_point);
    assert_eq!(eval, Fr::zero());

    // EQ
    let eval_point = index_to_field_bitvector(0, log_s);
    let eval = subtables.lookup_polys[1].evaluate(&eval_point);
    assert_eq!(eval, Fr::zero());

    let eval_point = index_to_field_bitvector(1, log_s);
    let eval = subtables.lookup_polys[1].evaluate(&eval_point);
    assert_eq!(eval, Fr::zero());

    let eval_point = index_to_field_bitvector(2, log_s);
    let eval = subtables.lookup_polys[1].evaluate(&eval_point);
    assert_eq!(eval, Fr::one());
  }

  #[test]
  fn combine_lookups() {
    const C: usize = 4;
    const S: usize = 1 << 3;
    const M: usize = 1 << 16;

    // let nz: Vec<Vec<usize>> = gen_indices::<C>(S, M);
    let nz: Vec<Vec<usize>> = vec![
      vec![
        1,            // LT: True, EQ: False
        1 << 8,       // LT: False, EQ: False
        1 + (1 << 8), // LT: False, EQ: True
        5,
        4,
        3,
        2,
        1
      ];
      C
    ];

    let subtable_entries: Vec<Vec<Fr>> = LTVM::materialize_subtables();
    assert_eq!(subtable_entries[0][0], Fr::zero()); // LT 0 > 0 = 0
    assert_eq!(subtable_entries[0][1], Fr::one()); // LT 1 > 0 = 1
    assert_eq!(subtable_entries[0][(1 << 15) - 1], Fr::one()); // LT MAX > 0 = 1
    assert_eq!(subtable_entries[0][1 << 15], Fr::zero()); // LT 0 > 1 = 0

    // LT[0], EQ[0], LT[1], EQ[1], ...
    // LT[0] + LT[1]EQ[0] + LT[2]EQ[0]EQ[1]
    let lt = vec![Fr::from(2), Fr::from(3), Fr::from(4), Fr::from(5)];
    let eq = vec![Fr::from(6), Fr::from(7), Fr::from(8), Fr::from(9)];

    let vals = vec![lt[0], eq[0], lt[1], eq[1], lt[2], eq[2], lt[3], eq[3]];

    let combined = LTVM::combine_lookups(&vals);
    let expected = lt[0] + lt[1] * eq[0] + lt[2] * eq[0] * eq[1] + lt[3] * eq[0] * eq[1] * eq[2];
    assert_eq!(combined, expected);
  }
}
