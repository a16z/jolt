use ark_ff::PrimeField;
use std::marker::PhantomData;

use super::LassoSubtable;

#[derive(Default, Debug)]
pub struct XORSubtable<F: PrimeField> {
  _field: PhantomData<F>,
}

impl<F: PrimeField> XORSubtable<F> {
  pub fn new() -> Self {
    Self {
      _field: PhantomData,
    }
  }
}

impl<F: PrimeField> LassoSubtable<F> for XORSubtable<F> {
  fn materialize(&self) -> Vec<F> {
    unimplemented!("TODO");
  }

  fn evaluate_mle(&self, point: &[F]) -> F {
    unimplemented!("TODO");
  }
}
