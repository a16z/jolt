use crate::{IV, SIGMA};
use jolt_inlines_sdk::host::Xlen;
use jolt_inlines_sdk::spec::{InlineMemoryLayout, InlineSpec, InlineTestHarness, INLINE};

pub fn execute_blake2b_compression(state: &mut [u64; 8], message_words: &[u64; 18]) {
    let mut v = [0u64; 16];
    v[0..8].copy_from_slice(state);
    v[8..16].copy_from_slice(&IV);

    v[12] ^= message_words[16];

    if message_words[17] != 0 {
        v[14] = !v[14];
    }

    for s in SIGMA {
        g(
            &mut v,
            0,
            4,
            8,
            12,
            message_words[s[0]],
            message_words[s[1]],
        );
        g(
            &mut v,
            1,
            5,
            9,
            13,
            message_words[s[2]],
            message_words[s[3]],
        );
        g(
            &mut v,
            2,
            6,
            10,
            14,
            message_words[s[4]],
            message_words[s[5]],
        );
        g(
            &mut v,
            3,
            7,
            11,
            15,
            message_words[s[6]],
            message_words[s[7]],
        );

        g(
            &mut v,
            0,
            5,
            10,
            15,
            message_words[s[8]],
            message_words[s[9]],
        );
        g(
            &mut v,
            1,
            6,
            11,
            12,
            message_words[s[10]],
            message_words[s[11]],
        );
        g(
            &mut v,
            2,
            7,
            8,
            13,
            message_words[s[12]],
            message_words[s[13]],
        );
        g(
            &mut v,
            3,
            4,
            9,
            14,
            message_words[s[14]],
            message_words[s[15]],
        );
    }

    for i in 0..8 {
        state[i] ^= v[i] ^ v[i + 8];
    }
}

fn g(v: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize, x: u64, y: u64) {
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
    v[d] = (v[d] ^ v[a]).rotate_right(32);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(24);
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
    v[d] = (v[d] ^ v[a]).rotate_right(16);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(63);
}

pub struct Blake2bCompressionSpec;

impl InlineSpec for Blake2bCompressionSpec {
    type Input = ([u64; 8], [u64; 18]);
    type Output = [u64; 8];

    fn reference(input: &Self::Input) -> Self::Output {
        let mut state = input.0;
        execute_blake2b_compression(&mut state, &input.1);
        state
    }

    fn create_harness() -> InlineTestHarness {
        let layout = InlineMemoryLayout::single_input(144, 64);
        InlineTestHarness::new(layout, Xlen::Bit64)
    }

    fn instruction() -> INLINE {
        InlineTestHarness::create_default_instruction(
            crate::INLINE_OPCODE,
            crate::BLAKE2_FUNCT3,
            crate::BLAKE2_FUNCT7,
        )
    }

    fn load(harness: &mut InlineTestHarness, input: &Self::Input) {
        harness.setup_registers();
        harness.load_state64(&input.0);
        harness.load_input64(&input.1);
    }

    fn read(harness: &mut InlineTestHarness) -> Self::Output {
        let vec = harness.read_output64(8);
        vec.try_into().unwrap()
    }
}
