use ark_ff::PrimeField;
use std::any::TypeId;
use std::fmt::Debug;
use strum::{EnumCount, IntoEnumIterator};

use crate::{
  jolt::{
    instruction::{ChunkIndices, Opcode, SubtableDecomposition},
    subtable::LassoSubtable,
  },
  poly::dense_mlpoly::DensePolynomial,
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
  type InstructionSet: ChunkIndices + SubtableDecomposition + Opcode + IntoEnumIterator + EnumCount;
  type Subtables: LassoSubtable<F> + IntoEnumIterator + EnumCount + Debug + From<TypeId>;

  const C: usize;
  const M: usize;
  const NUM_SUBTABLES: usize = Self::Subtables::COUNT;
  const NUM_INSTRUCTIONS: usize = Self::InstructionSet::COUNT;
  const NUM_MEMORIES: usize = Self::C * Self::Subtables::COUNT;

  fn prove(ops: Vec<Self::InstructionSet>) {
    let materialized_subtables = Self::materialize_subtables();
    let subtable_lookup_indices = Self::subtable_lookup_indices(&ops);

    let polynomials = Self::polynomialize(ops, &subtable_lookup_indices, &materialized_subtables);

    // commit to polynomials

    // - Compute primary sumcheck claim
    // - sumcheck
    // - memory checking
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

      for subtable_index in 0..Self::NUM_SUBTABLES {
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

  fn combine_lookups(vals: &[F]) -> F {
    for instruction in Self::InstructionSet::iter() {
      for subtable in instruction.subtables::<F>().iter() {
        println!(
          "{:?}",
          <Self as Jolt<F>>::Subtables::from(subtable.subtable_id())
        );
      }
    }

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
