use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use merlin::Transcript;
use std::any::TypeId;
use strum::{EnumCount, IntoEnumIterator};

use crate::jolt::instruction::{JoltInstruction, Opcode};
use crate::jolt::subtable::LassoSubtable;

use crate::{
  lasso::memory_checking::MemoryCheckingProof,
  poly::{
    dense_mlpoly::{DensePolynomial, PolyCommitment, PolyCommitmentGens},
    eq_poly::EqPolynomial,
  },
  subprotocols::sumcheck::SumcheckInstanceProof,
  utils::{math::Math, random::RandomTape},
};

pub struct PolynomialRepresentation<F: PrimeField> {
  pub dim: Vec<DensePolynomial<F>>,
  pub read_cts: Vec<DensePolynomial<F>>,
  pub final_cts: Vec<DensePolynomial<F>>,
  pub E_polys: Vec<DensePolynomial<F>>,
  pub flag: DensePolynomial<F>,
  // TODO(moodlezoup): Consider pulling out combined polys into separate struct
  pub combined_dim_read_poly: DensePolynomial<F>,
  pub combined_final_poly: DensePolynomial<F>,
  pub combined_E_poly: DensePolynomial<F>,
}

impl<F: PrimeField> PolynomialRepresentation<F> {
  fn commit<G: CurveGroup<ScalarField = F>>(
    &self,
    generators: &SurgeCommitmentGenerators<G>,
  ) -> SurgeCommitment<G> {
    let (dim_read_commitment, _) = self
      .combined_dim_read_poly
      .commit(&generators.dim_read_commitment_gens, None);
    let (final_commitment, _) = self
      .combined_final_poly
      .commit(&generators.final_commitment_gens, None);
    let (flag_commitment, _) = self.flag.commit(&generators.flag_commitment_gens, None);
    let (E_commitment, _) = self
      .combined_E_poly
      .commit(&generators.E_commitment_gens, None);

    SurgeCommitment {
      dim_read_commitment,
      final_commitment,
      flag_commitment,
      E_commitment,
    }
  }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct SurgeCommitment<G: CurveGroup> {
  pub dim_read_commitment: PolyCommitment<G>,
  pub final_commitment: PolyCommitment<G>,
  pub flag_commitment: PolyCommitment<G>,
  pub E_commitment: PolyCommitment<G>,
}

pub struct SurgeCommitmentGenerators<G: CurveGroup> {
  pub dim_read_commitment_gens: PolyCommitmentGens<G>,
  pub final_commitment_gens: PolyCommitmentGens<G>,
  pub flag_commitment_gens: PolyCommitmentGens<G>,
  pub E_commitment_gens: PolyCommitmentGens<G>,
}

pub trait Jolt<F: PrimeField, G: CurveGroup<ScalarField = F>> {
  type InstructionSet: JoltInstruction + Opcode + IntoEnumIterator + EnumCount;
  type Subtables: LassoSubtable<F> + IntoEnumIterator + EnumCount + From<TypeId> + Into<usize>;

  const C: usize;
  const M: usize;
  const NUM_SUBTABLES: usize = Self::Subtables::COUNT;
  const NUM_INSTRUCTIONS: usize = Self::InstructionSet::COUNT;
  const NUM_MEMORIES: usize = Self::C * Self::Subtables::COUNT;

  fn prove(ops: Vec<Self::InstructionSet>, r: Vec<F>, transcript: &mut Transcript) {
    let m = ops.len().next_power_of_two();
    // TODO(moodlezoup): transcript stuff

    let materialized_subtables = Self::materialize_subtables();
    let subtable_lookup_indices = Self::subtable_lookup_indices(&ops);

    let polynomials = Self::polynomialize(&ops, &subtable_lookup_indices, &materialized_subtables);

    // TODO(moodlezoup): commit to polynomials

    let eq = EqPolynomial::new(r.to_vec());
    let sumcheck_claim = Self::compute_sumcheck_claim(&ops, &polynomials.E_polys, &eq);

    // TODO(moodlezoup): jolt-specific sumcheck implementation?
    let mut sumcheck_polys = polynomials.E_polys.clone();
    sumcheck_polys.push(DensePolynomial::new(eq.evals()));
    sumcheck_polys.push(polynomials.flag);
    let (primary_sumcheck_proof, r_sumcheck, _) =
      SumcheckInstanceProof::prove_arbitrary::<_, G, Transcript>(
        &sumcheck_claim,
        m.log_2(),
        &mut sumcheck_polys,
        Self::combine_lookups,
        Self::sumcheck_poly_degree(),
        transcript,
      );

    // let gamma = F::zero();
    // let tau = F::zero();
    // let mut random_tape = RandomTape::new(b"proof");
    // let commitment_generators = Self::commitment_generators(m);
    // MemoryCheckingProof::prove(
    //   polynomials,
    //   gamma,
    //   tau,
    //   &materialized_subtables,
    //   &commitment_generators,
    //   &mut transcript,
    //   &mut random_tape,
    // );
  }

  fn commitment_generators(m: usize) -> SurgeCommitmentGenerators<G> {
    // dim_1, ... dim_C, read_1, ..., read_C
    // log_2(C * m + C * m)
    let num_vars_dim_read = (2 * Self::C * m).next_power_of_two().log_2();
    // final_1, ..., final_C
    // log_2(C * M)
    let num_vars_final = (Self::C * Self::M).next_power_of_two().log_2();
    // E_1, ..., E_alpha
    // log_2(alpha * m)
    let num_vars_E = (Self::NUM_MEMORIES * m).next_power_of_two().log_2();
    let num_vars_flag =
      m.next_power_of_two().log_2() + Self::NUM_INSTRUCTIONS.next_power_of_two().log_2();

    let dim_read_commitment_gens = PolyCommitmentGens::new(num_vars_dim_read, b"asdf");
    let final_commitment_gens = PolyCommitmentGens::new(num_vars_final, b"asd");
    let E_commitment_gens = PolyCommitmentGens::new(num_vars_E, b"asdf");
    let flag_commitment_gens = PolyCommitmentGens::new(num_vars_flag, b"1234");

    SurgeCommitmentGenerators {
      dim_read_commitment_gens,
      final_commitment_gens,
      flag_commitment_gens,
      E_commitment_gens,
    }
  }

  fn polynomialize(
    ops: &Vec<Self::InstructionSet>,
    subtable_lookup_indices: &Vec<Vec<usize>>,
    materialized_subtables: &Vec<Vec<F>>,
  ) -> PolynomialRepresentation<F> {
    let m: usize = ops.len().next_power_of_two();

    let mut opcodes: Vec<u8> = ops.iter().map(|op| op.to_opcode()).collect();
    opcodes.resize(m, 0);

    let mut dim: Vec<DensePolynomial<_>> = Vec::with_capacity(Self::C);
    let mut read_cts: Vec<DensePolynomial<_>> = Vec::with_capacity(Self::C);
    let mut final_cts: Vec<DensePolynomial<_>> = Vec::with_capacity(Self::C);
    let mut E_polys: Vec<DensePolynomial<_>> = Vec::with_capacity(Self::NUM_MEMORIES);

    for i in 0..Self::C {
      let access_sequence: &Vec<usize> = &subtable_lookup_indices[i];

      let mut final_cts_i = vec![0usize; Self::M];
      let mut read_cts_i = vec![0usize; m];

      for j in 0..m {
        let memory_address = access_sequence[j];
        debug_assert!(memory_address < Self::M);
        let counter = final_cts_i[memory_address];
        read_cts_i[j] = counter;
        final_cts_i[memory_address] = counter + 1;
      }

      dim.push(DensePolynomial::from_usize(access_sequence));
      read_cts.push(DensePolynomial::from_usize(&read_cts_i));
      final_cts.push(DensePolynomial::from_usize(&final_cts_i));
    }

    for subtable_index in 0..Self::NUM_SUBTABLES {
      for i in 0..Self::C {
        let subtable_lookups = subtable_lookup_indices[i]
          .iter()
          .map(|&lookup_index| materialized_subtables[subtable_index][lookup_index])
          .collect();
        E_polys.push(DensePolynomial::new(subtable_lookups))
      }
    }

    let mut flag_bitvectors: Vec<Vec<usize>> = Vec::with_capacity(m);
    for j in 0..m {
      let mut opcode_bitvector = vec![0usize, Self::NUM_INSTRUCTIONS.next_power_of_two()];
      opcode_bitvector[opcodes[j] as usize] = 1;
      flag_bitvectors.push(opcode_bitvector);
    }
    let flag_bitvectors: Vec<usize> = flag_bitvectors.into_iter().flatten().collect();
    let flag = DensePolynomial::from_usize(&flag_bitvectors);

    let dim_read_polys = [dim.as_slice(), read_cts.as_slice()].concat();
    let combined_dim_read_poly = DensePolynomial::merge(&dim_read_polys);
    let combined_final_poly = DensePolynomial::merge(&final_cts);
    let combined_E_poly = DensePolynomial::merge(&E_polys);

    PolynomialRepresentation {
      dim,
      read_cts,
      final_cts,
      flag,
      E_polys,
      combined_dim_read_poly,
      combined_final_poly,
      combined_E_poly,
    }
  }

  fn compute_sumcheck_claim(
    ops: &Vec<Self::InstructionSet>,
    E_polys: &Vec<DensePolynomial<F>>,
    eq: &EqPolynomial<F>,
  ) -> F {
    let m = ops.len().next_power_of_two();
    E_polys.iter().for_each(|E_i| assert_eq!(E_i.len(), m));

    let eq_evals = eq.evals();

    let mut claim = F::zero();
    for (k, op) in ops.iter().enumerate() {
      let memory_indices = Self::instruction_to_memory_indices(&op);
      let mut filtered_operands = Vec::with_capacity(memory_indices.len());
      for index in memory_indices {
        filtered_operands.push(E_polys[index][k]);
      }
      claim += eq_evals[k] * op.combine_lookups(&filtered_operands, Self::C, Self::M);
    }

    claim
  }

  fn instruction_to_memory_indices(op: &Self::InstructionSet) -> Vec<usize> {
    let instruction_subtables: Vec<Self::Subtables> = op
      .subtables::<F>()
      .iter()
      .map(|subtable| Self::Subtables::from(subtable.subtable_id()))
      .collect();

    let mut memory_indices = Vec::with_capacity(Self::C * instruction_subtables.len());
    for subtable in instruction_subtables {
      let index: usize = subtable.into();
      memory_indices.extend((Self::C * index)..(Self::C * (index + 1)));
    }

    memory_indices
  }

  fn combine_lookups(vals: &[F]) -> F {
    assert_eq!(vals.len(), Self::NUM_MEMORIES + 2);

    let mut sum = F::zero();
    for instruction in Self::InstructionSet::iter() {
      let memory_indices = Self::instruction_to_memory_indices(&instruction);
      let mut filtered_operands = Vec::with_capacity(memory_indices.len());
      for index in memory_indices {
        filtered_operands.push(vals[index]);
      }
      sum += instruction.combine_lookups(&filtered_operands, Self::C, Self::M);
    }
    // eq(...) * flag(...) * g(...)
    vals[vals.len() - 2] * vals[vals.len() - 1] * sum
  }

  fn sumcheck_poly_degree() -> usize {
    Self::InstructionSet::iter()
      .map(|instruction| instruction.g_poly_degree(Self::C))
      .max()
      .unwrap()
      + 2 // eq and flag
  }

  fn materialize_subtables() -> Vec<Vec<F>> {
    let mut subtables: Vec<Vec<_>> = Vec::with_capacity(Self::Subtables::COUNT);
    for subtable in Self::Subtables::iter() {
      subtables.push(subtable.materialize(Self::M));
    }
    subtables
  }

  fn subtable_lookup_indices(ops: &Vec<Self::InstructionSet>) -> Vec<Vec<usize>> {
    let m = ops.len().next_power_of_two();
    let log_m = Self::M.log_2();
    let chunked_indices: Vec<Vec<usize>> =
      ops.iter().map(|op| op.to_indices(Self::C, log_m)).collect();

    let mut subtable_lookup_indices: Vec<Vec<usize>> = Vec::with_capacity(Self::C);
    for i in 0..Self::C {
      let mut access_sequence: Vec<usize> =
        chunked_indices.iter().map(|chunks| chunks[i]).collect();
      access_sequence.resize(m, 0);
      subtable_lookup_indices.push(access_sequence);
    }
    subtable_lookup_indices
  }
}
