//! Backend implementations that derive from the expression IR.
//!
//! Each backend implements [`ExprVisitor`](crate::ExprVisitor) or operates on
//! the normalized [`SumOfProducts`](crate::SumOfProducts) form.

pub mod circuit;
pub mod evaluate;
pub mod lean;
pub mod r1cs;

#[cfg(feature = "z3")]
pub mod z3;
