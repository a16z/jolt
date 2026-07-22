use jolt_inlines_sdk::{InlineReference, InlineSpec};
use rand::RngCore;
use jolt_tracer::utils::inline_test_harness::{InlineMemoryLayout, InlineTestHarness};

use crate::exec::{execute_sha256_compression, execute_sha256_compression_initial};
use crate::sequence_builder::{Sha256Compression, Sha256CompressionInitial};
use crate::test_constants::{Sha256Block, Sha256State, TestVectors};

impl InlineReference for Sha256Compression {
    type Input = (Sha256State, Sha256Block);
    type Output = Sha256State;

    fn reference((state, block): &Self::Input) -> Self::Output {
        execute_sha256_compression(*state, *block)
    }
}

impl InlineSpec for Sha256Compression {
    fn edge_cases() -> impl IntoIterator<Item = Self::Input> {
        TestVectors::get_standard_test_vectors()
            .into_iter()
            .map(|(_, block, state, _)| (state, block))
    }

    fn random(rng: &mut impl RngCore) -> Self::Input {
        (
            core::array::from_fn(|_| rng.next_u32()),
            core::array::from_fn(|_| rng.next_u32()),
        )
    }

    fn harness() -> InlineTestHarness {
        InlineTestHarness::new(InlineMemoryLayout::single_input(64, 32))
    }

    fn load(harness: &mut InlineTestHarness, (state, block): &Self::Input) {
        harness.load_input32(block);
        harness.load_state32(state);
    }

    fn read(harness: &mut InlineTestHarness) -> Self::Output {
        harness.read_output32(8).try_into().unwrap()
    }
}

impl InlineReference for Sha256CompressionInitial {
    type Input = Sha256Block;
    type Output = Sha256State;

    fn reference(block: &Self::Input) -> Self::Output {
        execute_sha256_compression_initial(*block)
    }
}

impl InlineSpec for Sha256CompressionInitial {
    fn edge_cases() -> impl IntoIterator<Item = Self::Input> {
        TestVectors::get_standard_test_vectors()
            .into_iter()
            .map(|(_, block, _, _)| block)
    }

    fn random(rng: &mut impl RngCore) -> Self::Input {
        core::array::from_fn(|_| rng.next_u32())
    }

    fn harness() -> InlineTestHarness {
        InlineTestHarness::new(InlineMemoryLayout::single_input(64, 32))
    }

    fn load(harness: &mut InlineTestHarness, block: &Self::Input) {
        harness.load_input32(block);
    }

    fn read(harness: &mut InlineTestHarness) -> Self::Output {
        harness.read_output32(8).try_into().unwrap()
    }
}
