use crate::sequence_builder::{BLOCK, K};
use jolt_inlines_sdk::host::Xlen;
use jolt_inlines_sdk::spec::{InlineMemoryLayout, InlineSpec, InlineTestHarness, INLINE};

pub fn execute_sha256_compression(initial_state: [u32; 8], input: [u32; 16]) -> [u32; 8] {
    let mut a = initial_state[0];
    let mut b = initial_state[1];
    let mut c = initial_state[2];
    let mut d = initial_state[3];
    let mut e = initial_state[4];
    let mut f = initial_state[5];
    let mut g = initial_state[6];
    let mut h = initial_state[7];

    let mut w = [0u32; 64];

    w[..16].copy_from_slice(&input);

    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }

    for i in 0..64 {
        let ch = (e & f) ^ ((!e) & g);
        let maj = (a & b) ^ (a & c) ^ (b & c);

        let sigma0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let sigma1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);

        let t1 = h
            .wrapping_add(sigma1)
            .wrapping_add(ch)
            .wrapping_add(K[i] as u32)
            .wrapping_add(w[i]);
        let t2 = sigma0.wrapping_add(maj);

        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }

    [
        initial_state[0].wrapping_add(a),
        initial_state[1].wrapping_add(b),
        initial_state[2].wrapping_add(c),
        initial_state[3].wrapping_add(d),
        initial_state[4].wrapping_add(e),
        initial_state[5].wrapping_add(f),
        initial_state[6].wrapping_add(g),
        initial_state[7].wrapping_add(h),
    ]
}

pub fn execute_sha256_compression_initial(input: [u32; 16]) -> [u32; 8] {
    execute_sha256_compression(BLOCK.map(|x| x as u32), input)
}

fn create_harness() -> InlineTestHarness {
    let layout = InlineMemoryLayout::single_input(64, 32);
    InlineTestHarness::new(layout, Xlen::Bit64)
}

pub struct Sha256CompressionSpec;

impl InlineSpec for Sha256CompressionSpec {
    type Input = ([u32; 8], [u32; 16]);
    type Output = [u32; 8];

    fn reference(input: &Self::Input) -> Self::Output {
        execute_sha256_compression(input.0, input.1)
    }

    fn create_harness() -> InlineTestHarness {
        create_harness()
    }

    fn instruction() -> INLINE {
        InlineTestHarness::create_default_instruction(
            crate::INLINE_OPCODE,
            crate::SHA256_FUNCT3,
            crate::SHA256_FUNCT7,
        )
    }

    fn load(harness: &mut InlineTestHarness, input: &Self::Input) {
        harness.setup_registers();
        harness.load_input32(&input.1);
        harness.load_state32(&input.0);
    }

    fn read(harness: &mut InlineTestHarness) -> Self::Output {
        harness.read_output32(8).try_into().unwrap()
    }
}

pub struct Sha256CompressionInitialSpec;

impl InlineSpec for Sha256CompressionInitialSpec {
    type Input = [u32; 16];
    type Output = [u32; 8];

    fn reference(input: &Self::Input) -> Self::Output {
        execute_sha256_compression_initial(*input)
    }

    fn create_harness() -> InlineTestHarness {
        create_harness()
    }

    fn instruction() -> INLINE {
        InlineTestHarness::create_default_instruction(
            crate::INLINE_OPCODE,
            crate::SHA256_INIT_FUNCT3,
            crate::SHA256_INIT_FUNCT7,
        )
    }

    fn load(harness: &mut InlineTestHarness, input: &Self::Input) {
        harness.setup_registers();
        harness.load_input32(input);
    }

    fn read(harness: &mut InlineTestHarness) -> Self::Output {
        harness.read_output32(8).try_into().unwrap()
    }
}
