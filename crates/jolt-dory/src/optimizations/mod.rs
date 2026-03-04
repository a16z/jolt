// Ported from a16z/arkworks-algebra fork (jolt-optimizations crate).
// BN254-specific optimized EC operations: GLV scalar multiplication,
// batch affine addition, Frobenius endomorphism.
//
// Original code dual-licensed under Apache-2.0 and MIT.

pub mod batch_addition;
mod constants;
mod decomp_2d;
mod decomp_4d;
mod frobenius;
mod glv_four;
mod glv_two;

mod dory_g1;
mod dory_g2;

pub use batch_addition::batch_g1_additions_multi;
pub use dory_g1::{vector_add_scalar_mul_g1_online, vector_scalar_mul_add_gamma_g1_online};
pub use dory_g2::{vector_add_scalar_mul_g2_online, vector_scalar_mul_add_gamma_g2_online};
pub use glv_four::glv_four_scalar_mul_online;
pub use glv_two::fixed_base_vector_msm_g1;
