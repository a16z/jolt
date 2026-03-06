//! Re-exports of GLV and batch addition optimizations from `jolt-crypto`.
//!
//! The implementations live in `jolt_crypto::arkworks::bn254::{glv, batch_addition}`.

pub use jolt_crypto::arkworks::bn254::batch_addition::batch_g1_additions_multi_affine
    as batch_g1_additions_multi;
pub use jolt_crypto::arkworks::bn254::glv::dory_g1::{
    vector_add_scalar_mul_g1_online, vector_scalar_mul_add_gamma_g1_online,
};
pub use jolt_crypto::arkworks::bn254::glv::dory_g2::{
    vector_add_scalar_mul_g2_online, vector_scalar_mul_add_gamma_g2_online,
};
pub use jolt_crypto::arkworks::bn254::glv::glv_four::glv_four_scalar_mul_online;
pub use jolt_crypto::arkworks::bn254::glv::glv_two::fixed_base_vector_msm_g1;
