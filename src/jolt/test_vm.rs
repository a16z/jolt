use std::marker::PhantomData;

use ark_ff::PrimeField;
use ark_std::log2;

use enum_dispatch::enum_dispatch;
use strum::{EnumCount, IntoEnumIterator};
use strum_macros::{EnumCount as EnumCountMacro, EnumIter};

use crate::utils::split_bits;

// use super::jolt_strategy::{InstructionStrategy, JoltStrategy, SubtableStrategy};

trait JoltInstruction<F: PrimeField> {
  // TODO: C, M

  type Subtables;

  fn combine_lookups(&self, vals: &[F]) -> F;
  fn g_poly_degree(&self) -> usize;
}

#[repr(u8)]
enum TestInstructionSet {
  XOR(XORInstruction) = 0,
  EQ(EQInstruction) = 1,
  LT(LTInstruction) = 2,
  NOT(NOTInstruction) = 3,
}

struct XORInstruction(u64, u64);
struct EQInstruction(u64, u64);
struct LTInstruction(u64, u64);
struct NOTInstruction(u64);

impl<F: PrimeField> JoltInstruction<F> for XORInstruction {
  // TODO: assocsiated tuple vs associated enum vs method
  type Subtables = (XORSubtable<F>, EQSubtable<F>);

  fn combine_lookups(&self, vals: &[F]) -> F {
    unimplemented!("TODO");
  }

  fn g_poly_degree(&self) -> usize {
    unimplemented!("TODO");
  }
}

#[enum_dispatch]
trait LassoSubtable<F: PrimeField> {
  // TODO: M

  fn materialize(&self) -> Vec<F>;
  fn evaluate_mle(&self, point: &[F]) -> F;
}

#[enum_dispatch(LassoSubtable<F>)]
#[derive(EnumCountMacro, EnumIter)]

enum TestSubtables<F: PrimeField> {
  XOR(XORSubtable<F>),
  EQ(EQSubtable<F>),
}

#[derive(Default)]
struct XORSubtable<F: PrimeField> {
  _field: PhantomData<F>,
}
#[derive(Default)]
struct EQSubtable<F: PrimeField> {
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

pub trait Jolt<F: PrimeField> {
  type InstructionSet;
  type Subtables: LassoSubtable<F> + IntoEnumIterator + EnumCount;

  fn prove(ops: Vec<Self::InstructionSet>) {
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

struct TestJoltVM;

impl<F: PrimeField> Jolt<F> for TestJoltVM {
  type InstructionSet = TestInstructionSet;
  type Subtables = TestSubtables<F>;
}
