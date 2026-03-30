use crate::sequence_builder::{Sha256Compression, Sha256CompressionInitial, BLOCK, K};
use jolt_inlines_sdk::host::Xlen;
use jolt_inlines_sdk::spec::rand::rngs::StdRng;
use jolt_inlines_sdk::spec::rand::Rng;
use jolt_inlines_sdk::spec::{InlineMemoryLayout, InlineSpec, InlineTestHarness};

fn create_harness() -> InlineTestHarness {
    let layout = InlineMemoryLayout::single_input(64, 32);
    InlineTestHarness::new(layout, Xlen::Bit64)
}

impl InlineSpec for Sha256Compression {
    type Input = ([u32; 8], [u32; 16]);
    type Output = [u32; 8];

    fn random_input(rng: &mut StdRng) -> Self::Input {
        (
            core::array::from_fn(|_| rng.gen()),
            core::array::from_fn(|_| rng.gen()),
        )
    }

    fn reference(input: &Self::Input) -> Self::Output {
        let (initial_state, input) = input;

        let mut a = initial_state[0];
        let mut b = initial_state[1];
        let mut c = initial_state[2];
        let mut d = initial_state[3];
        let mut e = initial_state[4];
        let mut f = initial_state[5];
        let mut g = initial_state[6];
        let mut h = initial_state[7];

        let mut w = [0u32; 64];

        w[..16].copy_from_slice(input);

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

    fn create_harness() -> InlineTestHarness {
        create_harness()
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

impl InlineSpec for Sha256CompressionInitial {
    type Input = [u32; 16];
    type Output = [u32; 8];

    fn random_input(rng: &mut StdRng) -> Self::Input {
        core::array::from_fn(|_| rng.gen())
    }

    fn reference(input: &Self::Input) -> Self::Output {
        Sha256Compression::reference(&(BLOCK.map(|x| x as u32), *input))
    }

    fn create_harness() -> InlineTestHarness {
        create_harness()
    }

    fn load(harness: &mut InlineTestHarness, input: &Self::Input) {
        harness.setup_registers();
        harness.load_input32(input);
    }

    fn read(harness: &mut InlineTestHarness) -> Self::Output {
        harness.read_output32(8).try_into().unwrap()
    }
}
