//! Deferred-reduction accumulator traits for fused multiply-add.
//!
//! In sumcheck inner loops, many products are summed before reduction to a
//! field element. These traits let accumulators collect unreduced limbs via
//! [`FMAdd::fmadd`], then reduce once at the end via [`BarrettReduce`] or
//! [`MontgomeryReduce`]. This amortizes the expensive modular reduction
//! across hundreds of multiply-add operations.

use crate::Field;

/// Fused multiply-add: `self += left * right` without intermediate reduction.
pub trait FMAdd<Left, Right>: Sized {
    fn fmadd(&mut self, left: &Left, right: &Right);
}

/// Finalizes an unreduced accumulator via Barrett reduction.
///
/// Barrett reduction uses a precomputed approximate inverse of the modulus
/// and is faster than Montgomery REDC when the accumulator exceeds `2N` limbs.
pub trait BarrettReduce<F: Field> {
    fn barrett_reduce(&self) -> F;
}

/// Finalizes an unreduced accumulator via Montgomery REDC.
pub trait MontgomeryReduce<F: Field> {
    fn montgomery_reduce(&self) -> F;
}
