use std::fmt::Debug;
use std::marker::PhantomData;

use ark_ff::PrimeField;
use ark_std::log2;

use enum_dispatch::enum_dispatch;
use itertools::Chunk;
use strum::{EnumCount, IntoEnumIterator};
use strum_macros::{EnumCount as EnumCountMacro, EnumIter};

use crate::poly::dense_mlpoly::DensePolynomial;
use crate::utils::math::Math;
use crate::utils::split_bits;

// ==================== INSTRUCTIONS ====================

trait JoltInstruction<F: PrimeField> {
  // TODO: C, M

  fn combine_lookups(vals: &[F]) -> F;
  fn g_poly_degree() -> usize;
}

#[enum_dispatch]
pub trait SubtableDecomposition {
  fn subtables<F: PrimeField>(&self) -> Vec<Box<dyn LassoSubtable<F>>>;
}

#[enum_dispatch]
pub trait ChunkIndices {
  fn to_indices<const C: usize, const LOG_M: usize>(&self) -> [usize; C];
}

pub trait Opcode {
  fn to_opcode(&self) -> u8;
}

#[repr(u8)]
#[derive(Copy, Clone)]
#[enum_dispatch(ChunkIndices, SubtableDecomposition)]
pub enum TestInstructionSet {
  XOR(XORInstruction),
  EQ(EQInstruction),
}

impl Opcode for TestInstructionSet {
  fn to_opcode(&self) -> u8 {
    unsafe { *<*const _>::from(self).cast::<u8>() }
  }
}

#[derive(Copy, Clone)]
pub struct XORInstruction(u64, u64);
#[derive(Copy, Clone)]
pub struct EQInstruction(u64, u64);

impl<F: PrimeField> JoltInstruction<F> for XORInstruction {
  fn combine_lookups(vals: &[F]) -> F {
    unimplemented!("TODO");
  }

  fn g_poly_degree() -> usize {
    1
  }
}

impl SubtableDecomposition for XORInstruction {
  fn subtables<F: PrimeField>(&self) -> Vec<Box<dyn LassoSubtable<F>>> {
    vec![Box::new(XORSubtable {
      _field: PhantomData,
    })]
  }
}

impl ChunkIndices for XORInstruction {
  fn to_indices<const C: usize, const LOG_M: usize>(&self) -> [usize; C] {
    unimplemented!("TODO");
  }
}

impl<F: PrimeField> JoltInstruction<F> for EQInstruction {
  fn combine_lookups(vals: &[F]) -> F {
    unimplemented!("TODO");
  }

  fn g_poly_degree() -> usize {
    unimplemented!("TODO");
  }
}

impl SubtableDecomposition for EQInstruction {
  fn subtables<F: PrimeField>(&self) -> Vec<Box<dyn LassoSubtable<F>>> {
    vec![Box::new(EQSubtable {
      _field: PhantomData,
    })]
  }
}

impl ChunkIndices for EQInstruction {
  fn to_indices<const C: usize, const LOG_M: usize>(&self) -> [usize; C] {
    unimplemented!("TODO");
  }
}

// ==================== SUBTABLES ====================

#[enum_dispatch]
pub trait LassoSubtable<F: PrimeField> {
  // TODO: M

  fn materialize(&self) -> Vec<F>;
  fn evaluate_mle(&self, point: &[F]) -> F;
}

#[enum_dispatch(LassoSubtable<F>, ToEnum<F>)]
#[derive(EnumCountMacro, EnumIter)]
pub enum TestSubtables<F: PrimeField> {
  XOR(XORSubtable<F>),
  EQ(EQSubtable<F>),
}

#[derive(Default)]
pub struct XORSubtable<F: PrimeField> {
  _field: PhantomData<F>,
}
#[derive(Default)]
pub struct EQSubtable<F: PrimeField> {
  _field: PhantomData<F>,
}

impl<F: PrimeField> LassoSubtable<F> for XORSubtable<F> {
  fn materialize(&self) -> Vec<F> {
    unimplemented!("TODO");
  }

  fn evaluate_mle(&self, point: &[F]) -> F {
    unimplemented!("TODO");
  }
}

impl<F: PrimeField> LassoSubtable<F> for EQSubtable<F> {
  fn materialize(&self) -> Vec<F> {
    unimplemented!("TODO");
  }

  fn evaluate_mle(&self, point: &[F]) -> F {
    unimplemented!("TODO");
  }
}

// ==================== JOLT ====================
pub trait Jolt<F: PrimeField, const C: usize, const LOG_M: usize> {
  type InstructionSet: ChunkIndices + SubtableDecomposition + Opcode;
  type Subtables: LassoSubtable<F> + IntoEnumIterator + EnumCount;

  const NUM_SUBTABLES: usize = Self::Subtables::COUNT;
  const NUM_MEMORIES: usize = C * Self::Subtables::COUNT;

  fn prove(ops: Vec<Self::InstructionSet>) {
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
    // E_1(x) ... E_{NUM_MEMORIES}(x)
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

pub struct TestJoltVM<F: PrimeField, const C: usize, const LOG_M: usize> {
  _field: PhantomData<F>,
}

impl<F: PrimeField, const C: usize, const LOG_M: usize> Jolt<F, C, LOG_M>
  for TestJoltVM<F, C, LOG_M>
{
  type InstructionSet = TestInstructionSet;
  type Subtables = TestSubtables<F>;
}

// ==================== TEST ====================

#[cfg(test)]
mod tests {
  use ark_curve25519::{EdwardsProjective, Fr};
  use ark_ff::PrimeField;
  use ark_std::{log2, test_rng, One, Zero};
  use merlin::Transcript;
  use rand_chacha::rand_core::RngCore;

  use crate::{
    jolt::test_vm::{EQInstruction, Jolt, TestInstructionSet, TestJoltVM, XORInstruction},
    utils::{index_to_field_bitvector, random::RandomTape, split_bits},
  };

  #[test]
  fn e2e() {
    const C: usize = 4;
    const M: usize = 1 << 16;

    TestJoltVM::<Fr, C, 16>::prove(vec![
      TestInstructionSet::XOR(XORInstruction(420, 69)),
      TestInstructionSet::EQ(EQInstruction(420, 69)),
      TestInstructionSet::EQ(EQInstruction(420, 420)),
    ]);
  }
}
