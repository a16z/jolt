use crate::Field;

/// Unified fused-multiply-add trait for accumulators.
/// Perform: acc += left * right.
pub trait FMAdd<Left, Right>: Sized {
    fn fmadd(&mut self, left: &Left, right: &Right);
}

/// Trait for accumulators that finish with Barrett reduction to a field element
pub trait BarrettReduce<F: Field> {
    fn barrett_reduce(&self) -> F;
}

/// Trait for accumulators that finish with Montgomery reduction to a field element
pub trait MontgomeryReduce<F: Field> {
    fn montgomery_reduce(&self) -> F;
}
