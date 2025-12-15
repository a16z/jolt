//! Stage 1: Constraint Sumchecks
//!
//! This module contains the three parallel sumcheck protocols for proving
//! different types of constraints in the recursion SNARK:
//!
//! - `square_and_multiply`: GT exponentiation constraints
//! - `gt_mul`: GT multiplication constraints
//! - `g1_scalar_mul`: G1 scalar multiplication constraints

pub mod g1_scalar_mul;
pub mod gt_mul;
pub mod square_and_multiply;