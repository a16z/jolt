use jolt_inlines_sdk::{InlineReference, InlineSpec};
use rand::RngCore;
use jolt_tracer::utils::inline_test_harness::{InlineMemoryLayout, InlineTestHarness};

use crate::exec::execute_keccak_f;
use crate::sequence_builder::Keccak256Permutation;
use crate::test_constants::TestVectors;
use crate::{Keccak256State, NUM_LANES};

impl InlineReference for Keccak256Permutation {
    type Input = Keccak256State;
    type Output = Keccak256State;

    fn reference(input: &Self::Input) -> Self::Output {
        let mut state = *input;
        execute_keccak_f(&mut state);
        state
    }
}

impl InlineSpec for Keccak256Permutation {
    fn edge_cases() -> impl IntoIterator<Item = Self::Input> {
        TestVectors::get_standard_test_vectors()
            .into_iter()
            .map(|(_, state)| state)
    }

    fn random(rng: &mut impl RngCore) -> Self::Input {
        core::array::from_fn(|_| rng.next_u64())
    }

    fn harness() -> InlineTestHarness {
        InlineTestHarness::new(InlineMemoryLayout::single_input(136, 200))
    }

    fn load(harness: &mut InlineTestHarness, input: &Self::Input) {
        harness.load_state64(input);
    }

    fn read(harness: &mut InlineTestHarness) -> Self::Output {
        let result = harness.read_output64(NUM_LANES);
        result.try_into().unwrap()
    }
}
