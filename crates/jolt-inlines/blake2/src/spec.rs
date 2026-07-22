use jolt_inlines_sdk::{InlineReference, InlineSpec};
use jolt_tracer::utils::inline_test_harness::{InlineMemoryLayout, InlineTestHarness};
use rand::RngCore;

use crate::exec::execute_blake2b_compression;
use crate::sequence_builder::Blake2bCompression;
use crate::{IV, MSG_BLOCK_LEN, STATE_VECTOR_LEN};

pub type Blake2CompressionInput = ([u64; STATE_VECTOR_LEN], [u64; MSG_BLOCK_LEN], u64, bool);

impl InlineReference for Blake2bCompression {
    type Input = Blake2CompressionInput;
    type Output = [u64; STATE_VECTOR_LEN];

    fn reference((state, message, counter, is_final): &Self::Input) -> Self::Output {
        let mut state = *state;
        let mut message_words = [0u64; MSG_BLOCK_LEN + 2];
        message_words[..MSG_BLOCK_LEN].copy_from_slice(message);
        message_words[MSG_BLOCK_LEN] = *counter;
        message_words[MSG_BLOCK_LEN + 1] = u64::from(*is_final);
        execute_blake2b_compression(&mut state, &message_words);
        state
    }
}

impl InlineSpec for Blake2bCompression {
    fn edge_cases() -> impl IntoIterator<Item = Self::Input> {
        let initial_state = blake2b_initial_state();
        let mut abc_message = [0u64; MSG_BLOCK_LEN];
        abc_message[0] = 0x0000000000636261;

        [
            (initial_state, [0u64; MSG_BLOCK_LEN], 0, true),
            (initial_state, abc_message, 3, true),
            (
                [u64::MAX; STATE_VECTOR_LEN],
                [u64::MAX; MSG_BLOCK_LEN],
                128,
                false,
            ),
        ]
    }

    fn random(rng: &mut impl RngCore) -> Self::Input {
        (
            core::array::from_fn(|_| rng.next_u64()),
            core::array::from_fn(|_| rng.next_u64()),
            rng.next_u64(),
            rng.next_u32() & 1 == 1,
        )
    }

    fn harness() -> InlineTestHarness {
        InlineTestHarness::new(InlineMemoryLayout::single_input(144, 64))
    }

    fn load(harness: &mut InlineTestHarness, (state, message, counter, is_final): &Self::Input) {
        harness.load_state64(state);

        let mut input = [0u64; MSG_BLOCK_LEN + 2];
        input[..MSG_BLOCK_LEN].copy_from_slice(message);
        input[MSG_BLOCK_LEN] = *counter;
        input[MSG_BLOCK_LEN + 1] = u64::from(*is_final);
        harness.load_input64(&input);
    }

    fn read(harness: &mut InlineTestHarness) -> Self::Output {
        let output = harness.read_output64(STATE_VECTOR_LEN);
        output.try_into().unwrap()
    }
}

fn blake2b_initial_state() -> [u64; STATE_VECTOR_LEN] {
    let mut state = IV;
    state[0] ^= 0x01010000 ^ 64u64;
    state
}
