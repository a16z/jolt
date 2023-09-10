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
  pub dim_usize: Vec<Vec<usize>>,
  pub dim: Vec<DensePolynomial<F>>,
  pub read_cts: Vec<DensePolynomial<F>>,
  pub final_cts: Vec<DensePolynomial<F>>,
  pub flags: Vec<DensePolynomial<F>>,
  pub combined_dim_read_poly: DensePolynomial<F>,
  pub combined_final_poly: DensePolynomial<F>,
  pub combined_flags_poly: DensePolynomial<F>,
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
    for instruction in Self::InstructionSet::iter() {
      for subtable in instruction.subtables::<F>().iter() {
        println!(
          "{:?}",
          <Self as Jolt<F>>::Subtables::from(subtable.subtable_id())
        );
      }
    }

    // let mut dim_i, read_i, final_i =
    Self::polynomialize(ops);

    // - commit dim_i, read_i, final_i
    // - materialize subtables
    // - Create NUM_MEMORIES E_i polys (to_lookup_polys + commit)
    //   - subtable -> C E_i polys?
    // - Compute primary sumcheck claim
    // - sumcheck
    // - memory checking
  }

  fn polynomialize(ops: Vec<Self::InstructionSet>) -> PolynomialRepresentation<F> {
    let log_M: usize = Self::M.log_2();
    let m: usize = ops.len().next_power_of_two();

    let chunked_indices: Vec<Vec<usize>> =
      ops.iter().map(|op| op.to_indices(Self::C, log_M)).collect();

    let mut opcodes: Vec<u8> = ops.iter().map(|op| op.to_opcode()).collect();
    opcodes.resize(m, 0);

    let mut dim_usize: Vec<Vec<usize>> = Vec::with_capacity(Self::C);
    let mut dim: Vec<DensePolynomial<F>> = Vec::with_capacity(Self::C);
    let mut read_cts: Vec<DensePolynomial<F>> = Vec::with_capacity(Self::C);
    let mut final_cts: Vec<DensePolynomial<F>> = Vec::with_capacity(Self::C);
    let mut flags: Vec<DensePolynomial<F>> = Vec::with_capacity(Self::C);

    for i in 0..Self::C {
      let mut access_sequence: Vec<usize> =
        chunked_indices.iter().map(|chunks| chunks[i]).collect();
      access_sequence.resize(m, 0);

      let mut final_cts_i = vec![0usize, Self::M];
      let mut read_cts_i = vec![0usize, m];
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

      dim.push(DensePolynomial::from_usize(&access_sequence));
      dim_usize.push(access_sequence);
      read_cts.push(DensePolynomial::from_usize(&read_cts_i));
      final_cts.push(DensePolynomial::from_usize(&final_cts_i));
      flags.push(DensePolynomial::from_usize(&flags_i));
    }

    let dim_read_polys = [dim.as_slice(), read_cts.as_slice()].concat();
    let combined_dim_read_poly = DensePolynomial::merge(&dim_read_polys);
    let combined_final_poly = DensePolynomial::merge(&final_cts);
    let combined_flags_poly = DensePolynomial::merge(&flags);

    PolynomialRepresentation {
      dim_usize,
      dim,
      read_cts,
      final_cts,
      flags,
      combined_dim_read_poly,
      combined_final_poly,
      combined_flags_poly,
    }
  }

  fn combine_lookups(vals: &[F]) -> F {
    for instruction in Self::InstructionSet::iter() {}

    unimplemented!("TODO");
  }

  fn materialize_subtables() -> Vec<Vec<F>> {
    let mut subtables: Vec<Vec<F>> = Vec::with_capacity(Self::Subtables::COUNT);
    for subtable in Self::Subtables::iter() {
      subtables.push(subtable.materialize(Self::M));
    }
    subtables
  }
}

pub mod test_vm;
