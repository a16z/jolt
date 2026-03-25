use crate::spec::Blake2bCompressionSpec;
use jolt_inlines_sdk::spec::{InlineSpec, InlineTestHarness, INLINE};

pub fn create_blake2_harness() -> InlineTestHarness {
    Blake2bCompressionSpec::create_harness()
}

pub fn load_blake2_data(
    harness: &mut InlineTestHarness,
    state: &[u64; crate::STATE_VECTOR_LEN],
    message: &[u64; crate::MSG_BLOCK_LEN],
    counter: u64,
    is_final: bool,
) {
    let mut combined_input = [0u64; 18];
    combined_input[..16].copy_from_slice(message);
    combined_input[16] = counter;
    combined_input[17] = is_final as u64;

    let input = (*state, combined_input);
    Blake2bCompressionSpec::load(harness, &input);
}

pub fn read_state(harness: &mut InlineTestHarness) -> [u64; crate::STATE_VECTOR_LEN] {
    Blake2bCompressionSpec::read(harness)
}

pub fn instruction() -> INLINE {
    Blake2bCompressionSpec::instruction()
}
