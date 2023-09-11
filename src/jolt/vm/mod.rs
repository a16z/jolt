use ark_ff::PrimeField;
use std::any::TypeId;
use strum::{EnumCount, IntoEnumIterator};

use crate::{
  jolt::{
    instruction::{JoltInstruction, Opcode},
    subtable::LassoSubtable,
  },
  poly::{dense_mlpoly::DensePolynomial, eq_poly::EqPolynomial},
  utils::math::Math,
};

pub struct PolynomialRepresentation<F: PrimeField> {
  pub dim: Vec<DensePolynomial<F>>,
  pub read_cts: Vec<DensePolynomial<F>>,
  pub final_cts: Vec<DensePolynomial<F>>,
  pub flags: Vec<DensePolynomial<F>>,
  pub E_polys: Vec<DensePolynomial<F>>,
  pub combined_dim_read_poly: DensePolynomial<F>,
  pub combined_final_poly: DensePolynomial<F>,
  pub combined_flags_poly: DensePolynomial<F>,
  pub combined_E_poly: DensePolynomial<F>,
}

pub trait Jolt<F: PrimeField> {
  type InstructionSet: JoltInstruction + Opcode + IntoEnumIterator + EnumCount;
  type Subtables: LassoSubtable<F> + IntoEnumIterator + EnumCount + From<TypeId> + Into<usize>;

  const C: usize;
  const M: usize;
  const NUM_SUBTABLES: usize = Self::Subtables::COUNT;
  const NUM_INSTRUCTIONS: usize = Self::InstructionSet::COUNT;
  const NUM_MEMORIES: usize = Self::C * Self::Subtables::COUNT;

  fn prove(ops: Vec<Self::InstructionSet>) {
    let materialized_subtables = Self::materialize_subtables();
    let subtable_lookup_indices = Self::subtable_lookup_indices(&ops);

    let polynomials = Self::polynomialize(ops, &subtable_lookup_indices, &materialized_subtables);

    // TODO(moodlezoup): commit to polynomials

    // let eq = EqPolynomial::new(r.to_vec());
    // let sumcheck_claim = Self::compute_sumcheck_claim(ops, &polynomials.E_polys, &eq);

    // TODO(moodlezoup): sumcheck (jolt-specific sumcheck implementation?)

    // TODO(moodlezoup): memory checking
    // MemoryCheckingProof::prove(
    //   polynomials,
    //   gamma, tau,
    //   &materialized_subtables,
    //   gens,
    //   transcript,
    //   random_tape
    // );
  }

  fn polynomialize(
    ops: Vec<Self::InstructionSet>,
    subtable_lookup_indices: &Vec<Vec<usize>>,
    materialized_subtables: &Vec<Vec<F>>,
  ) -> PolynomialRepresentation<F> {
    let m: usize = ops.len().next_power_of_two();

    let mut opcodes: Vec<u8> = ops.iter().map(|op| op.to_opcode()).collect();
    opcodes.resize(m, 0);

    let mut dim: Vec<DensePolynomial<F>> = Vec::with_capacity(Self::C);
    let mut read_cts: Vec<DensePolynomial<F>> = Vec::with_capacity(Self::C);
    let mut final_cts: Vec<DensePolynomial<F>> = Vec::with_capacity(Self::C);
    let mut flags: Vec<DensePolynomial<F>> = Vec::with_capacity(Self::C);
    let mut E_polys: Vec<DensePolynomial<F>> = Vec::with_capacity(Self::NUM_MEMORIES);

    for i in 0..Self::C {
      let access_sequence: &Vec<usize> = &subtable_lookup_indices[i];

      let mut final_cts_i = vec![0usize; Self::M];
      let mut read_cts_i = vec![0usize; m];
      let mut flags_i: Vec<Vec<usize>> = Vec::with_capacity(m);

      for j in 0..m {
        let memory_address = access_sequence[j];
        debug_assert!(memory_address < Self::M);
        let counter = final_cts_i[memory_address];
        read_cts_i[j] = counter;
        final_cts_i[memory_address] = counter + 1;

        let mut opcode_bitvector = vec![0usize, Self::NUM_INSTRUCTIONS.next_power_of_two()];
        opcode_bitvector[opcodes[j] as usize] = 1;
        flags_i.push(opcode_bitvector);
      }

      let flags_i: Vec<usize> = flags_i.into_iter().flatten().collect();

      dim.push(DensePolynomial::from_usize(access_sequence));
      read_cts.push(DensePolynomial::from_usize(&read_cts_i));
      final_cts.push(DensePolynomial::from_usize(&final_cts_i));
      flags.push(DensePolynomial::from_usize(&flags_i));
    }

    for subtable_index in 0..Self::NUM_SUBTABLES {
      for i in 0..Self::C {
        let subtable_lookups: Vec<F> = subtable_lookup_indices[i]
          .iter()
          .map(|&lookup_index| materialized_subtables[subtable_index][lookup_index])
          .collect();
        E_polys.push(DensePolynomial::new(subtable_lookups))
      }
    }

    let dim_read_polys = [dim.as_slice(), read_cts.as_slice()].concat();
    let combined_dim_read_poly = DensePolynomial::merge(&dim_read_polys);
    let combined_final_poly = DensePolynomial::merge(&final_cts);
    let combined_flags_poly = DensePolynomial::merge(&flags);
    let combined_E_poly = DensePolynomial::merge(&E_polys);

    PolynomialRepresentation {
      dim,
      read_cts,
      final_cts,
      flags,
      E_polys,
      combined_dim_read_poly,
      combined_final_poly,
      combined_flags_poly,
      combined_E_poly,
    }
  }

  fn compute_sumcheck_claim(
    ops: Vec<Self::InstructionSet>,
    E_polys: &Vec<DensePolynomial<F>>,
    eq: &EqPolynomial<F>,
  ) -> F {
    let m = ops.len().next_power_of_two();
    E_polys.iter().for_each(|E_i| assert_eq!(E_i.len(), m));

    let eq_evals = eq.evals();

    let mut claim = F::zero();
    for (k, op) in ops.iter().enumerate() {
      let memory_indices = Self::instruction_to_memory_indices(&op);
      let mut filtered_operands: Vec<F> = Vec::with_capacity(memory_indices.len());
      for index in memory_indices {
        filtered_operands.push(E_polys[index][k]);
      }
      claim += eq_evals[k] * op.combine_lookups::<F>(&filtered_operands, Self::C, Self::M);
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
    unimplemented!("TODO");
  }

  fn materialize_subtables() -> Vec<Vec<F>> {
    let mut subtables: Vec<Vec<F>> = Vec::with_capacity(Self::Subtables::COUNT);
    for subtable in Self::Subtables::iter() {
      subtables.push(subtable.materialize(Self::M));
    }
    subtables
  }

  fn subtable_lookup_indices(ops: &Vec<Self::InstructionSet>) -> Vec<Vec<usize>> {
    let m = ops.len().next_power_of_two();
    let chunked_indices: Vec<Vec<usize>> = ops
      .iter()
      .map(|op| op.to_indices(Self::C, Self::M.log_2()))
      .collect();

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

pub mod test_vm;
