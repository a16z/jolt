use jolt_inlines_sdk::{InlineReference, InlineSpec};
use rand::RngCore;
use tracer::utils::inline_test_harness::{InlineMemoryLayout, InlineTestHarness};

use crate::exec::execute_blake3_compression;
use crate::sequence_builder::{Blake3Compression, Blake3Keyed64Compression};
use crate::{
    CHAINING_VALUE_LEN, FLAG_CHUNK_END, FLAG_CHUNK_START, FLAG_KEYED_HASH, FLAG_ROOT, IV,
    MSG_BLOCK_LEN,
};

pub type ChainingValue = [u32; CHAINING_VALUE_LEN];
pub type MessageBlock = [u32; MSG_BLOCK_LEN];
pub type Blake3CompressionInput = (ChainingValue, MessageBlock, [u32; 2], u32, u32);
pub type Blake3Keyed64Input = (ChainingValue, ChainingValue, ChainingValue);

impl InlineReference for Blake3Compression {
    type Input = Blake3CompressionInput;
    type Output = ChainingValue;

    fn reference(
        (chaining_value, message, counter, block_len, flags): &Self::Input,
    ) -> Self::Output {
        let mut output = *chaining_value;
        execute_blake3_compression(&mut output, message, counter, *block_len, *flags);
        output
    }
}

impl InlineSpec for Blake3Compression {
    fn edge_cases() -> impl IntoIterator<Item = Self::Input> {
        let mut abc = [0u32; MSG_BLOCK_LEN];
        abc[0] = u32::from_le_bytes(*b"abc\0");

        [
            (
                IV,
                [0u32; MSG_BLOCK_LEN],
                [0, 0],
                0,
                FLAG_CHUNK_START | FLAG_CHUNK_END | FLAG_ROOT,
            ),
            (
                IV,
                abc,
                [0, 0],
                3,
                FLAG_CHUNK_START | FLAG_CHUNK_END | FLAG_ROOT,
            ),
            (
                [u32::MAX; CHAINING_VALUE_LEN],
                [u32::MAX; MSG_BLOCK_LEN],
                [u32::MAX, u32::MAX],
                64,
                FLAG_CHUNK_START | FLAG_CHUNK_END | FLAG_ROOT | FLAG_KEYED_HASH,
            ),
        ]
    }

    fn random(rng: &mut impl RngCore) -> Self::Input {
        (
            core::array::from_fn(|_| rng.next_u32()),
            core::array::from_fn(|_| rng.next_u32()),
            [rng.next_u32(), rng.next_u32()],
            rng.next_u32() % 65,
            rng.next_u32() & 0x1F,
        )
    }

    fn harness() -> InlineTestHarness {
        InlineTestHarness::new(InlineMemoryLayout::single_input(80, 32))
    }

    fn load(
        harness: &mut InlineTestHarness,
        (chaining_value, message, counter, block_len, flags): &Self::Input,
    ) {
        harness.load_state32(chaining_value);

        let mut input = [0u32; MSG_BLOCK_LEN + 4];
        input[..MSG_BLOCK_LEN].copy_from_slice(message);
        input[MSG_BLOCK_LEN..MSG_BLOCK_LEN + 2].copy_from_slice(counter);
        input[MSG_BLOCK_LEN + 2] = *block_len;
        input[MSG_BLOCK_LEN + 3] = *flags;
        harness.load_input32(&input);
    }

    fn read(harness: &mut InlineTestHarness) -> Self::Output {
        let output = harness.read_output32(CHAINING_VALUE_LEN);
        output.try_into().unwrap()
    }
}

impl InlineReference for Blake3Keyed64Compression {
    type Input = Blake3Keyed64Input;
    type Output = ChainingValue;

    fn reference((left, right, key): &Self::Input) -> Self::Output {
        let mut output = *key;
        let mut message = [0u32; MSG_BLOCK_LEN];
        message[..CHAINING_VALUE_LEN].copy_from_slice(left);
        message[CHAINING_VALUE_LEN..].copy_from_slice(right);
        execute_blake3_compression(
            &mut output,
            &message,
            &[0, 0],
            64,
            FLAG_CHUNK_START | FLAG_CHUNK_END | FLAG_ROOT | FLAG_KEYED_HASH,
        );
        output
    }
}

impl InlineSpec for Blake3Keyed64Compression {
    fn edge_cases() -> impl IntoIterator<Item = Self::Input> {
        [
            ([0u32; CHAINING_VALUE_LEN], [0u32; CHAINING_VALUE_LEN], IV),
            (
                [0xAAAAAAAA; CHAINING_VALUE_LEN],
                [0x55555555; CHAINING_VALUE_LEN],
                IV,
            ),
            (
                [u32::MAX; CHAINING_VALUE_LEN],
                [u32::MAX; CHAINING_VALUE_LEN],
                [u32::MAX; CHAINING_VALUE_LEN],
            ),
        ]
    }

    fn random(rng: &mut impl RngCore) -> Self::Input {
        (
            core::array::from_fn(|_| rng.next_u32()),
            core::array::from_fn(|_| rng.next_u32()),
            core::array::from_fn(|_| rng.next_u32()),
        )
    }

    fn harness() -> InlineTestHarness {
        InlineTestHarness::new(InlineMemoryLayout::two_inputs(32, 32, 32))
    }

    fn load(harness: &mut InlineTestHarness, (left, right, key): &Self::Input) {
        harness.load_input32(left);
        harness.load_input2_32(right);
        harness.load_state32(key);
    }

    fn read(harness: &mut InlineTestHarness) -> Self::Output {
        let output = harness.read_output32(CHAINING_VALUE_LEN);
        output.try_into().unwrap()
    }
}
