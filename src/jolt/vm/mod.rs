use ark_ff::PrimeField;
use strum::{EnumCount, IntoEnumIterator};

use crate::jolt::{
  instruction::{ChunkIndices, Opcode, SubtableDecomposition},
  subtable::LassoSubtable,
};

pub trait Jolt<F: PrimeField> {
  type InstructionSet: ChunkIndices + SubtableDecomposition + Opcode + IntoEnumIterator + EnumCount;
  type Subtables: LassoSubtable<F> + IntoEnumIterator + EnumCount;

  const C: usize;
  const LOG_M: usize;
  const NUM_SUBTABLES: usize = Self::Subtables::COUNT;
  const NUM_MEMORIES: usize = Self::C * Self::Subtables::COUNT;

  fn prove(ops: Vec<Self::InstructionSet>) {
    for instruction in Self::InstructionSet::iter() {
      for subtable in instruction.subtables::<F>().iter() {
        // println!(
        //   "{:?}",
        //   TestSubtables::<F>::from_subtable_id(subtable.subtable_id())
        // );
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

  fn polynomialize(ops: Vec<Self::InstructionSet>) {
    // let chunked_indices: Vec<[usize; C]> =
    //   ops.iter().map(|op| op.to_indices::<C, LOG_M>()).collect();
    let ops_u8: Vec<u8> = ops.iter().map(|op| op.to_opcode()).collect();
    println!("{:?}", ops_u8);

    // let ops_subtables: Vec<_> = ops
    //   .iter()
    //   .map(|op| {
    //     for subtable in op.subtables::<F>() {
    //       let t: Self::Subtables = Self::coerce_subtable(subtable);
    //       println!("{:?}", t);
    //     }
    //   })
    //   .collect();
  }

  fn combine_lookups(vals: &[F]) -> F {
    for instruction in Self::InstructionSet::iter() {}

    unimplemented!("TODO");
  }

  fn materialize_subtables() -> Vec<Vec<F>> {
    let mut subtables: Vec<Vec<F>> = Vec::with_capacity(Self::Subtables::COUNT);
    for subtable in Self::Subtables::iter() {
      subtables.push(subtable.materialize());
    }
    subtables
  }
}

pub mod test_vm;
