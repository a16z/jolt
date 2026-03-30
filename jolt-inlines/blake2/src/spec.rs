use crate::sequence_builder::Blake2bCompression;
use crate::{IV, SIGMA};
use jolt_inlines_sdk::host::Xlen;
use jolt_inlines_sdk::spec::rand::rngs::StdRng;
use jolt_inlines_sdk::spec::rand::Rng;
use jolt_inlines_sdk::spec::{InlineMemoryLayout, InlineSpec, InlineTestHarness};

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

impl InlineSpec for Blake2bCompression {
    type Input = ([u64; 8], [u64; 18]);
    type Output = [u64; 8];

    fn random_input(rng: &mut StdRng) -> Self::Input {
        let state: [u64; 8] = core::array::from_fn(|_| rng.gen());
        let mut message: [u64; 18] = core::array::from_fn(|_| rng.gen());
        message[17] = rng.gen_range(0..=1);
        (state, message)
    }

    fn reference(input: &Self::Input) -> Self::Output {
        let (state, w) = input;
        let mut v = [0u64; 16];
        v[0..8].copy_from_slice(state);
        v[8..16].copy_from_slice(&IV);

        v[12] ^= w[16];

        if w[17] != 0 {
            v[14] = !v[14];
        }

        for s in SIGMA {
            g(&mut v, 0, 4, 8, 12, w[s[0]], w[s[1]]);
            g(&mut v, 1, 5, 9, 13, w[s[2]], w[s[3]]);
            g(&mut v, 2, 6, 10, 14, w[s[4]], w[s[5]]);
            g(&mut v, 3, 7, 11, 15, w[s[6]], w[s[7]]);

            g(&mut v, 0, 5, 10, 15, w[s[8]], w[s[9]]);
            g(&mut v, 1, 6, 11, 12, w[s[10]], w[s[11]]);
            g(&mut v, 2, 7, 8, 13, w[s[12]], w[s[13]]);
            g(&mut v, 3, 4, 9, 14, w[s[14]], w[s[15]]);
        }

        let mut result = *state;
        for i in 0..8 {
            result[i] ^= v[i] ^ v[i + 8];
        }
        result
    }

    fn create_harness() -> InlineTestHarness {
        let layout = InlineMemoryLayout::single_input(144, 64);
        InlineTestHarness::new(layout, Xlen::Bit64)
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
