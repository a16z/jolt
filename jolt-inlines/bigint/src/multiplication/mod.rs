const INPUT_LIMBS: usize = 4;
const OUTPUT_LIMBS: usize = 2 * INPUT_LIMBS;

pub mod sdk;
pub use sdk::*;

#[cfg(feature = "host")]
pub mod exec;
#[cfg(feature = "host")]
pub mod trace_generator;

#[cfg(all(test, feature = "host"))]
pub mod test_utils;
#[cfg(all(test, feature = "host"))]
pub mod tests;
