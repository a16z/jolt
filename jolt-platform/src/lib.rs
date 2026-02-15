#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "random")]
pub mod random;
#[cfg(feature = "random")]
pub use random::*;

pub mod print;
pub use print::*;

pub mod exit;
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
pub use exit::*;

pub mod cycle_tracking;
pub use cycle_tracking::*;

pub mod advice;
pub use advice::*;

#[cfg(all(
    feature = "malloc-shim",
    any(target_arch = "riscv32", target_arch = "riscv64")
))]
mod malloc_shim;
