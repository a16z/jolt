#[cfg(feature = "host")]
pub use tracer::utils::inline_helpers::INLINE_OPCODE;

pub const BIGINT256_MUL_FUNCT3: u32 = 0x00;
pub const BIGINT256_MUL_FUNCT7: u32 = 0x04;
pub const BIGINT256_MUL_NAME: &str = "BIGINT256_MUL_INLINE";

const INPUT_LIMBS: usize = 4;
const OUTPUT_LIMBS: usize = 2 * INPUT_LIMBS;

pub mod sdk;
pub use sdk::*;

#[cfg(feature = "host")]
pub mod exec;
#[cfg(feature = "host")]
pub mod sequence_builder;

#[cfg(all(test, feature = "host"))]
pub mod test_utils;
#[cfg(all(test, feature = "host"))]
pub mod tests;
