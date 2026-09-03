//! T2: non-native BN254 Fq limb arithmetic (96-bit limbs, 16-bit chunks, CRT
//! identity, grouped-inverse LogUp range checks) for the Dory deferred check,
//! wired by a signed-digit Straus schedule.

pub mod columns;
pub mod dory;
pub mod glv;
pub mod ops;
pub mod program;
pub mod relation;
pub mod schedule;
pub mod tower;
pub mod wiring;
