use jolt_inlines_sdk::{InlineReference, InlineSpec};
use rand::RngCore;
use tracer::utils::inline_test_harness::{InlineMemoryLayout, InlineTestHarness};

use crate::exec::execute_keccak_f;
use crate::sequence_builder::{Keccak256AbsorbPermutation, Keccak256Permutation};
use crate::test_constants::TestVectors;
use crate::{Keccak256State, NUM_LANES, RATE_IN_BYTES, RATE_IN_U64};

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

impl InlineReference for Keccak256AbsorbPermutation {
    type Input = (Keccak256State, [u64; RATE_IN_U64]);
    type Output = Keccak256State;

    fn reference((state, block): &Self::Input) -> Self::Output {
        let mut state = *state;
        for (lane, block_lane) in state.iter_mut().zip(block) {
            *lane ^= block_lane;
        }
        execute_keccak_f(&mut state);
        state
    }
}

impl InlineSpec for Keccak256AbsorbPermutation {
    fn edge_cases() -> impl IntoIterator<Item = Self::Input> {
        [
            ([0; NUM_LANES], [0; RATE_IN_U64]),
            ([u64::MAX; NUM_LANES], [u64::MAX; RATE_IN_U64]),
        ]
    }

    fn random(rng: &mut impl RngCore) -> Self::Input {
        (
            core::array::from_fn(|_| rng.next_u64()),
            core::array::from_fn(|_| rng.next_u64()),
        )
    }

    fn harness() -> InlineTestHarness {
        InlineTestHarness::new(InlineMemoryLayout::single_input(
            RATE_IN_BYTES,
            NUM_LANES * size_of::<u64>(),
        ))
    }

    fn load(harness: &mut InlineTestHarness, (state, block): &Self::Input) {
        harness.load_state64(state);
        harness.load_input64(block);
    }

    fn read(harness: &mut InlineTestHarness) -> Self::Output {
        harness.read_output64(NUM_LANES).try_into().unwrap()
    }
}
