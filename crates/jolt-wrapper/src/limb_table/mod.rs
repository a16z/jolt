//! T2: non-native BN254 Fq limb arithmetic (96-bit limbs, 16-bit chunks, the
//! limb-polynomial identity at a challenge, grouped-inverse LogUp range
//! checks) for the Dory deferred check, wired by a signed-digit Straus
//! schedule whose operands are committed columns copied by public kernels
//! and looked up by digit.

pub mod adapter;
pub mod columns;
pub mod digit_link;
pub mod digits;
pub mod dory;
pub mod export;
pub mod layout;
pub mod lookup;
pub mod ops;
pub mod program;
pub mod relation;
pub mod row_sumcheck;
pub mod schedule;
pub mod stream;
pub mod template;
pub mod terms;
pub mod tower;
pub mod verifier;
pub mod wiring;
