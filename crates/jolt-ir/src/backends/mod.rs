//! Backend implementations that derive from the expression IR.
//!
//! Each backend implements [`ExprVisitor`](crate::ExprVisitor) or operates on
//! the normalized [`SumOfProducts`](crate::SumOfProducts) form.

pub mod evaluate;
pub mod r1cs;
