use crate::sequence_builder::{Blake3Compression, Blake3Keyed64Compression};
use crate::{CHAINING_VALUE_LEN, FLAG_CHUNK_END, FLAG_CHUNK_START, FLAG_KEYED_HASH, FLAG_ROOT, IV};
use jolt_inlines_sdk::host::Xlen;
use jolt_inlines_sdk::spec::rand::rngs::StdRng;
use jolt_inlines_sdk::spec::rand::Rng;
use jolt_inlines_sdk::spec::{InlineMemoryLayout, InlineSpec, InlineTestHarness, INLINE};

const MSG_PERMUTATION: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

fn g(state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, mx: u32, my: u32) {
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(mx);
    state[d] = (state[d] ^ state[a]).rotate_right(16);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_right(12);
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(my);
    state[d] = (state[d] ^ state[a]).rotate_right(8);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_right(7);
}

fn round(state: &mut [u32; 16], m: &[u32; 16]) {
    g(state, 0, 4, 8, 12, m[0], m[1]);
    g(state, 1, 5, 9, 13, m[2], m[3]);
    g(state, 2, 6, 10, 14, m[4], m[5]);
    g(state, 3, 7, 11, 15, m[6], m[7]);
    g(state, 0, 5, 10, 15, m[8], m[9]);
    g(state, 1, 6, 11, 12, m[10], m[11]);
    g(state, 2, 7, 8, 13, m[12], m[13]);
    g(state, 3, 4, 9, 14, m[14], m[15]);
}

fn permute(m: &mut [u32; 16]) {
    let mut permuted = [0; 16];
    for i in 0..16 {
        permuted[i] = m[MSG_PERMUTATION[i]];
    }
    *m = permuted;
}

/// Reference BLAKE3 compression function.
/// See https://github.com/BLAKE3-team/BLAKE3/blob/master/reference_impl/reference_impl.rs
fn compress(
    chaining_value: &[u32; 8],
    block_words: &[u32; 16],
    counter: &[u32; 2],
    block_len: u32,
    flags: u32,
) -> [u32; 8] {
    #[rustfmt::skip]
    let mut state = [
        chaining_value[0], chaining_value[1], chaining_value[2], chaining_value[3],
        chaining_value[4], chaining_value[5], chaining_value[6], chaining_value[7],
        IV[0],             IV[1],             IV[2],             IV[3],
        counter[0],        counter[1],        block_len,         flags,
    ];
    let mut block = *block_words;

    round(&mut state, &block); // round 1
    permute(&mut block);
    round(&mut state, &block); // round 2
    permute(&mut block);
    round(&mut state, &block); // round 3
    permute(&mut block);
    round(&mut state, &block); // round 4
    permute(&mut block);
    round(&mut state, &block); // round 5
    permute(&mut block);
    round(&mut state, &block); // round 6
    permute(&mut block);
    round(&mut state, &block); // round 7

    let mut result = [0u32; 8];
    for i in 0..8 {
        result[i] = state[i] ^ state[i + 8];
    }
    result
}

impl InlineSpec for Blake3Compression {
    type Input = ([u32; 8], [u32; 16], [u32; 2], u32, u32);
    type Output = [u32; CHAINING_VALUE_LEN];

    fn random_input(rng: &mut StdRng) -> Self::Input {
        (
            core::array::from_fn(|_| rng.gen()),
            core::array::from_fn(|_| rng.gen()),
            core::array::from_fn(|_| rng.gen()),
            rng.gen(),
            rng.gen(),
        )
    }

    fn reference(input: &Self::Input) -> Self::Output {
        compress(&input.0, &input.1, &input.2, input.3, input.4)
    }

    fn create_harness() -> InlineTestHarness {
        let layout = InlineMemoryLayout::single_input(80, 32);
        InlineTestHarness::new(layout, Xlen::Bit64)
    }

    fn instruction() -> INLINE {
        InlineTestHarness::create_default_instruction(
            crate::INLINE_OPCODE,
            crate::BLAKE3_FUNCT3,
            crate::BLAKE3_FUNCT7,
        )
    }

    fn load(harness: &mut InlineTestHarness, input: &Self::Input) {
        harness.setup_registers();
        harness.load_state32(&input.0);

        let mut combined = Vec::with_capacity(20);
        combined.extend_from_slice(&input.1);
        combined.extend_from_slice(&input.2);
        combined.push(input.3);
        combined.push(input.4);
        harness.load_input32(&combined);
    }

    fn read(harness: &mut InlineTestHarness) -> Self::Output {
        harness
            .read_output32(CHAINING_VALUE_LEN)
            .try_into()
            .unwrap()
    }
}

impl InlineSpec for Blake3Keyed64Compression {
    type Input = ([u32; 8], [u32; 8], [u32; 8]);
    type Output = [u32; CHAINING_VALUE_LEN];

    fn random_input(rng: &mut StdRng) -> Self::Input {
        (
            core::array::from_fn(|_| rng.gen()),
            core::array::from_fn(|_| rng.gen()),
            core::array::from_fn(|_| rng.gen()),
        )
    }

    fn reference(input: &Self::Input) -> Self::Output {
        let mut message = [0u32; 16];
        message[..8].copy_from_slice(&input.0);
        message[8..].copy_from_slice(&input.1);

        compress(
            &input.2,
            &message,
            &[0, 0],
            64,
            FLAG_CHUNK_START | FLAG_CHUNK_END | FLAG_ROOT | FLAG_KEYED_HASH,
        )
    }

    fn create_harness() -> InlineTestHarness {
        let layout = InlineMemoryLayout::two_inputs(32, 32, 32);
        InlineTestHarness::new(layout, Xlen::Bit64)
    }

    fn instruction() -> INLINE {
        InlineTestHarness::create_default_instruction(
            crate::INLINE_OPCODE,
            crate::BLAKE3_KEYED64_FUNCT3,
            crate::BLAKE3_FUNCT7,
        )
    }

    fn load(harness: &mut InlineTestHarness, input: &Self::Input) {
        harness.setup_registers();
        harness.load_input32(&input.0);
        harness.load_input2_32(&input.1);
        harness.load_state32(&input.2);
    }

    fn read(harness: &mut InlineTestHarness) -> Self::Output {
        harness
            .read_output32(CHAINING_VALUE_LEN)
            .try_into()
            .unwrap()
    }
}
