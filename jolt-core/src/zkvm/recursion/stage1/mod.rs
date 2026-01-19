//! Stage 1: Constraint Sumchecks
//!
//! This module contains the sumcheck protocols for proving
//! different types of constraints in the recursion SNARK:
//!
//! - `square_and_multiply`: GT exponentiation constraints (legacy per-step)
//! - `packed_gt_exp`: GT exponentiation constraints (optimized packed 12-var)
//! - `gt_mul`: GT multiplication constraints
//! - `g1_scalar_mul`: G1 scalar multiplication constraints
//! - `shift_rho`: Shift sumcheck for verifying rho_next claims (Stage 1b)

pub mod g1_scalar_mul;
pub mod gt_mul;
pub mod packed_gt_exp;
pub mod shift_rho;
pub mod square_and_multiply;
