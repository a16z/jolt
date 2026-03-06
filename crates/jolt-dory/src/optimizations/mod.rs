//! BN254-specific optimized EC operations for Dory.
//!
//! Provides GLV scalar multiplication (2D for G1, 4D for G2 via Frobenius endomorphism),
//! batch affine addition via Montgomery's trick, and vector-scalar operations used in
//! Dory's inner-product argument rounds.
//!
//! Ported from the `a16z/arkworks-algebra` fork (jolt-optimizations crate).
//! Original code dual-licensed under Apache-2.0 and MIT.
//!
//! References:
//! - GLV: Gallant, Lambert, Vanstone. [Faster Point Multiplication on Elliptic Curves](https://link.springer.com/chapter/10.1007/3-540-44647-8_11) (CRYPTO 2001)
//! - Dory: Lee. [Dory: Efficient, Transparent arguments for Generalised Inner Products](https://eprint.iacr.org/2020/1274)

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
