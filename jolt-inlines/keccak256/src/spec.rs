use jolt_inlines_sdk::{InlineReference, InlineSpec};
use rand::RngCore;
use tracer::utils::inline_test_harness::{InlineMemoryLayout, InlineTestHarness, INLINE_RS2};

use crate::exec::{execute_absorb_permute, execute_init_absorb_permute, execute_keccak_f};
use crate::sequence_builder::{
    Keccak256AbsorbPermutation, Keccak256AbsorbPermutationUnaligned,
    Keccak256InitAbsorbPermutation, Keccak256InitAbsorbPermutationUnaligned, Keccak256Permutation,
};
use crate::test_constants::TestVectors;
use crate::{Keccak256State, NUM_LANES, RATE_IN_BYTES, RATE_IN_U64};

type Block = [u64; RATE_IN_U64];

const ZERO_STATE: Keccak256State = [0; NUM_LANES];
const ONES_STATE: Keccak256State = [u64::MAX; NUM_LANES];
const ZERO_BLOCK: Block = [0; RATE_IN_U64];
const ONES_BLOCK: Block = [u64::MAX; RATE_IN_U64];

fn random_state(rng: &mut impl RngCore) -> Keccak256State {
    core::array::from_fn(|_| rng.next_u64())
}

fn random_block(rng: &mut impl RngCore) -> Block {
    core::array::from_fn(|_| rng.next_u64())
}

fn aligned_harness() -> InlineTestHarness {
    InlineTestHarness::new(InlineMemoryLayout::single_input(
        RATE_IN_BYTES,
        NUM_LANES * size_of::<u64>(),
    ))
}

/// One doubleword more than the block, so the 18th containing doubleword of
/// a misaligned block lies inside the input region.
fn unaligned_harness() -> InlineTestHarness {
    InlineTestHarness::new(InlineMemoryLayout::single_input(
        RATE_IN_BYTES + size_of::<u64>(),
        NUM_LANES * size_of::<u64>(),
    ))
}

/// The input region for the unaligned ops: `block` starts `offset` bytes in,
/// surrounded by junk the funnel shift must discard.
fn containing_words(block: &Block, offset: usize) -> [u64; RATE_IN_U64 + 1] {
    let mut bytes = [0xA5u8; RATE_IN_BYTES + 8];
    for (bytes, word) in bytes[offset..offset + RATE_IN_BYTES]
        .chunks_exact_mut(8)
        .zip(block)
    {
        bytes.copy_from_slice(&word.to_le_bytes());
    }
    core::array::from_fn(|i| u64::from_le_bytes(bytes[8 * i..8 * i + 8].try_into().unwrap()))
}

/// Places `block` at `offset` and points `rs2` at it; `setup_registers` has
/// already run when `load` is called.
fn load_unaligned_block(harness: &mut InlineTestHarness, block: &Block, offset: usize) {
    harness.load_input64(&containing_words(block, offset));
    harness.cpu.x[INLINE_RS2 as usize] += offset as i64;
}

fn read_state(harness: &mut InlineTestHarness) -> Keccak256State {
    harness.read_output64(NUM_LANES).try_into().unwrap()
}

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
        random_state(rng)
    }

    fn harness() -> InlineTestHarness {
        aligned_harness()
    }

    fn load(harness: &mut InlineTestHarness, input: &Self::Input) {
        harness.load_state64(input);
    }

    fn read(harness: &mut InlineTestHarness) -> Self::Output {
        read_state(harness)
    }
}

impl InlineReference for Keccak256AbsorbPermutation {
    type Input = (Keccak256State, Block);
    type Output = Keccak256State;

    fn reference((state, block): &Self::Input) -> Self::Output {
        let mut state = *state;
        execute_absorb_permute(&mut state, block);
        state
    }
}

impl InlineSpec for Keccak256AbsorbPermutation {
    fn edge_cases() -> impl IntoIterator<Item = Self::Input> {
        [(ZERO_STATE, ZERO_BLOCK), (ONES_STATE, ONES_BLOCK)]
    }

    fn random(rng: &mut impl RngCore) -> Self::Input {
        (random_state(rng), random_block(rng))
    }

    fn harness() -> InlineTestHarness {
        aligned_harness()
    }

    fn load(harness: &mut InlineTestHarness, (state, block): &Self::Input) {
        harness.load_state64(state);
        harness.load_input64(block);
    }

    fn read(harness: &mut InlineTestHarness) -> Self::Output {
        read_state(harness)
    }
}

/// The state operand is the junk found at `rs1`, which the op must ignore.
impl InlineReference for Keccak256InitAbsorbPermutation {
    type Input = (Keccak256State, Block);
    type Output = Keccak256State;

    fn reference((_, block): &Self::Input) -> Self::Output {
        execute_init_absorb_permute(block)
    }
}

impl InlineSpec for Keccak256InitAbsorbPermutation {
    fn edge_cases() -> impl IntoIterator<Item = Self::Input> {
        [
            (ONES_STATE, ZERO_BLOCK),
            (ZERO_STATE, ONES_BLOCK),
            (ONES_STATE, ONES_BLOCK),
        ]
    }

    fn random(rng: &mut impl RngCore) -> Self::Input {
        (random_state(rng), random_block(rng))
    }

    fn harness() -> InlineTestHarness {
        aligned_harness()
    }

    fn load(harness: &mut InlineTestHarness, (state, block): &Self::Input) {
        harness.load_state64(state);
        harness.load_input64(block);
    }

    fn read(harness: &mut InlineTestHarness) -> Self::Output {
        read_state(harness)
    }
}

/// `(state, block, offset)`: the block starts `offset` bytes past an 8-byte
/// boundary. Offset 0 pins the documented aligned-`rs2` behaviour.
impl InlineReference for Keccak256AbsorbPermutationUnaligned {
    type Input = (Keccak256State, Block, usize);
    type Output = Keccak256State;

    fn reference((state, block, _): &Self::Input) -> Self::Output {
        let mut state = *state;
        execute_absorb_permute(&mut state, block);
        state
    }
}

impl InlineSpec for Keccak256AbsorbPermutationUnaligned {
    fn edge_cases() -> impl IntoIterator<Item = Self::Input> {
        [0, 1, 7].into_iter().flat_map(|offset| {
            [
                (ZERO_STATE, ZERO_BLOCK, offset),
                (ONES_STATE, ONES_BLOCK, offset),
            ]
        })
    }

    fn random(rng: &mut impl RngCore) -> Self::Input {
        let offset = 1 + rng.next_u32() as usize % 7;
        (random_state(rng), random_block(rng), offset)
    }

    fn harness() -> InlineTestHarness {
        unaligned_harness()
    }

    fn load(harness: &mut InlineTestHarness, (state, block, offset): &Self::Input) {
        harness.load_state64(state);
        load_unaligned_block(harness, block, *offset);
    }

    fn read(harness: &mut InlineTestHarness) -> Self::Output {
        read_state(harness)
    }
}

/// As [`Keccak256AbsorbPermutationUnaligned`], with the state operand being
/// junk at `rs1` the op must ignore.
impl InlineReference for Keccak256InitAbsorbPermutationUnaligned {
    type Input = (Keccak256State, Block, usize);
    type Output = Keccak256State;

    fn reference((_, block, _): &Self::Input) -> Self::Output {
        execute_init_absorb_permute(block)
    }
}

impl InlineSpec for Keccak256InitAbsorbPermutationUnaligned {
    fn edge_cases() -> impl IntoIterator<Item = Self::Input> {
        [0, 1, 7].into_iter().flat_map(|offset| {
            [
                (ONES_STATE, ZERO_BLOCK, offset),
                (ZERO_STATE, ONES_BLOCK, offset),
                (ONES_STATE, ONES_BLOCK, offset),
            ]
        })
    }

    fn random(rng: &mut impl RngCore) -> Self::Input {
        let offset = 1 + rng.next_u32() as usize % 7;
        (random_state(rng), random_block(rng), offset)
    }

    fn harness() -> InlineTestHarness {
        unaligned_harness()
    }

    fn load(harness: &mut InlineTestHarness, (state, block, offset): &Self::Input) {
        harness.load_state64(state);
        load_unaligned_block(harness, block, *offset);
    }

    fn read(harness: &mut InlineTestHarness) -> Self::Output {
        read_state(harness)
    }
}
