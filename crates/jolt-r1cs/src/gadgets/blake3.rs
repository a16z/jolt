//! BLAKE3 compression over boolean R1CS variables.

use jolt_field::JoltField;

use super::{bit::xor_word, Bit};
use crate::{LinearCombination, R1csBuilder};

const WORD_BITS: usize = 32;
const IV: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];
const MESSAGE_PERMUTATION: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

/// A little-endian 32-bit BLAKE3 word.
pub type Word = [Bit; WORD_BITS];

#[cfg(test)]
fn allocate_word<F: JoltField>(builder: &mut R1csBuilder<F>, value: u32) -> Word {
    std::array::from_fn(|bit| Bit::allocate(builder, value & (1 << bit) != 0))
}

fn constant_word(value: u32) -> Word {
    std::array::from_fn(|bit| Bit::constant(value & (1 << bit) != 0))
}

fn word_value(word: &Word) -> u32 {
    word.iter().enumerate().fold(0, |value, (bit, assigned)| {
        value | (u32::from(assigned.value()) << bit)
    })
}

fn rotate_right(mut word: Word, amount: usize) -> Word {
    word.rotate_left(amount);
    word
}

fn assert_sum<F: JoltField>(
    builder: &mut R1csBuilder<F>,
    inputs: &[&Word],
    result: &Word,
    overflow: &[Bit],
) {
    let mut input_sum = LinearCombination::zero();
    for input in inputs {
        let mut scale = F::one();
        for bit in *input {
            bit.add_scaled(&mut input_sum, scale);
            scale += scale;
        }
    }

    let mut output_sum = LinearCombination::zero();
    let mut scale = F::one();
    for bit in result {
        bit.add_scaled(&mut output_sum, scale);
        scale += scale;
    }
    for bit in overflow {
        bit.add_scaled(&mut output_sum, scale);
        scale += scale;
    }
    builder.assert_equal(input_sum, output_sum);
}

fn add2<F: JoltField>(builder: &mut R1csBuilder<F>, lhs: &Word, rhs: &Word) -> Word {
    let sum = u64::from(word_value(lhs)) + u64::from(word_value(rhs));
    let result = std::array::from_fn(|bit| Bit::allocate(builder, sum & (1 << bit) != 0));
    let overflow = Bit::allocate(builder, sum & (1 << WORD_BITS) != 0);
    assert_sum(builder, &[lhs, rhs], &result, &[overflow]);
    result
}

fn add3<F: JoltField>(builder: &mut R1csBuilder<F>, a: &Word, b: &Word, c: &Word) -> Word {
    let sum = u64::from(word_value(a)) + u64::from(word_value(b)) + u64::from(word_value(c));
    let result = std::array::from_fn(|bit| Bit::allocate(builder, sum & (1 << bit) != 0));
    let overflow = [
        Bit::allocate(builder, sum & (1 << WORD_BITS) != 0),
        Bit::allocate(builder, sum & (1 << (WORD_BITS + 1)) != 0),
    ];
    assert_sum(builder, &[a, b, c], &result, &overflow);
    result
}

#[expect(
    clippy::indexing_slicing,
    reason = "BLAKE3's fixed permutation indices are all within 16 words"
)]
fn g<F: JoltField>(
    builder: &mut R1csBuilder<F>,
    state: &mut [Word; 16],
    indices: [usize; 4],
    x: &Word,
    y: &Word,
) {
    let [a, b, c, d] = indices;
    let mut va = state[a];
    let mut vb = state[b];
    let mut vc = state[c];
    let mut vd = state[d];

    va = add3(builder, &va, &vb, x);
    vd = rotate_right(xor_word(builder, &vd, &va), 16);
    vc = add2(builder, &vc, &vd);
    vb = rotate_right(xor_word(builder, &vb, &vc), 12);
    va = add3(builder, &va, &vb, y);
    vd = rotate_right(xor_word(builder, &vd, &va), 8);
    vc = add2(builder, &vc, &vd);
    vb = rotate_right(xor_word(builder, &vb, &vc), 7);

    state[a] = va;
    state[b] = vb;
    state[c] = vc;
    state[d] = vd;
}

fn round<F: JoltField>(builder: &mut R1csBuilder<F>, state: &mut [Word; 16], message: &[Word; 16]) {
    g(builder, state, [0, 4, 8, 12], &message[0], &message[1]);
    g(builder, state, [1, 5, 9, 13], &message[2], &message[3]);
    g(builder, state, [2, 6, 10, 14], &message[4], &message[5]);
    g(builder, state, [3, 7, 11, 15], &message[6], &message[7]);
    g(builder, state, [0, 5, 10, 15], &message[8], &message[9]);
    g(builder, state, [1, 6, 11, 12], &message[10], &message[11]);
    g(builder, state, [2, 7, 8, 13], &message[12], &message[13]);
    g(builder, state, [3, 4, 9, 14], &message[14], &message[15]);
}

#[expect(
    clippy::indexing_slicing,
    reason = "BLAKE3's fixed message permutation indices are in range"
)]
fn permute(message: &[Word; 16]) -> [Word; 16] {
    MESSAGE_PERMUTATION.map(|index| message[index])
}

/// Constrains one BLAKE3 compression invocation and returns its final 16-word state.
pub fn compress<F: JoltField>(
    builder: &mut R1csBuilder<F>,
    cv: &[Word; 8],
    block: &[Word; 16],
    counter: u64,
    block_len: u32,
    flags: u32,
) -> [Word; 16] {
    let mut state = [constant_word(0); 16];
    state[..8].copy_from_slice(cv);
    state[8..12].copy_from_slice(&IV.map(constant_word)[..4]);
    state[12] = constant_word(counter as u32);
    state[13] = constant_word((counter >> 32) as u32);
    state[14] = constant_word(block_len);
    state[15] = constant_word(flags);

    let mut message = *block;
    for round_index in 0..7 {
        round(builder, &mut state, &message);
        if round_index != 6 {
            message = permute(&message);
        }
    }
    state
}

/// Applies BLAKE3's feed-forward XOR to a compression state.
#[expect(
    clippy::indexing_slicing,
    reason = "the two compression-state halves each contain exactly eight words"
)]
pub fn chaining_value<F: JoltField>(
    builder: &mut R1csBuilder<F>,
    output: &[Word; 16],
) -> [Word; 8] {
    std::array::from_fn(|index| xor_word(builder, &output[index], &output[index + 8]))
}

#[cfg(test)]
#[expect(
    clippy::indexing_slicing,
    reason = "tests index fixed-size hash blocks"
)]
#[expect(clippy::expect_used, reason = "tests may panic on assertion failures")]
mod tests {
    use jolt_field::Fr;
    use rand_chacha::ChaCha20Rng;
    use rand_core::{RngCore, SeedableRng};

    use super::*;
    use crate::{ConstraintMatrices, Variable};

    const CHUNK_START: u32 = 1;
    const CHUNK_END: u32 = 2;
    const ROOT: u32 = 8;

    fn output_bytes(words: &[Word; 8]) -> [u8; 32] {
        let mut output = [0; 32];
        for (word_index, word) in words.iter().enumerate() {
            output[word_index * 4..word_index * 4 + 4]
                .copy_from_slice(&word_value(word).to_le_bytes());
        }
        output
    }

    fn hash_gadget(input: &[u8]) -> (R1csBuilder<Fr>, [u8; 32]) {
        let mut padded = [0u8; 64];
        padded[..input.len()].copy_from_slice(input);
        let mut builder = R1csBuilder::new();
        let cv = IV.map(constant_word);
        let block = std::array::from_fn(|word| {
            let start = word * 4;
            allocate_word(
                &mut builder,
                u32::from_le_bytes(padded[start..start + 4].try_into().expect("four bytes")),
            )
        });
        let output = compress(
            &mut builder,
            &cv,
            &block,
            0,
            input.len() as u32,
            CHUNK_START | CHUNK_END | ROOT,
        );
        let cv = chaining_value(&mut builder, &output);
        (builder, output_bytes(&cv))
    }

    fn nonzeros(matrices: &ConstraintMatrices<Fr>) -> usize {
        [&matrices.a, &matrices.b, &matrices.c]
            .into_iter()
            .flat_map(|matrix| matrix.iter())
            .map(Vec::len)
            .sum()
    }

    fn shape_by_class(
        matrices: &ConstraintMatrices<Fr>,
        cv_bits: usize,
        block_bits: usize,
        output_xors: usize,
    ) -> [(usize, usize); 6] {
        let mut classes = [(0, 0); 6];
        for (index, ((a, b), c)) in matrices
            .a
            .iter()
            .zip(&matrices.b)
            .zip(&matrices.c)
            .enumerate()
        {
            let class = if index < cv_bits {
                0
            } else if index < cv_bits + block_bits {
                1
            } else if index >= matrices.num_constraints - output_xors {
                5
            } else if !c.is_empty() {
                4
            } else if b.len() == 1 && b[0].0 == Variable::ONE.index() {
                3
            } else {
                2
            };
            classes[class].0 += 1;
            classes[class].1 += a.len() + b.len() + c.len();
        }
        classes
    }

    #[test]
    fn compression_shape_is_stable() {
        let mut builder = R1csBuilder::<Fr>::new();
        let cv = IV.map(|word| allocate_word(&mut builder, word));
        let block = std::array::from_fn(|word| allocate_word(&mut builder, word as u32));
        let output = compress(&mut builder, &cv, &block, 0, 64, CHUNK_START | CHUNK_END);
        let _cv = chaining_value(&mut builder, &output);
        let witness = builder.witness().expect("complete witness");
        let matrices = builder.into_matrices();

        assert!(matrices.check_witness(&witness).is_ok());
        assert_eq!(matrices.num_constraints, 15_792);
        assert_eq!(matrices.num_vars - 1, 15_568);
        assert_eq!(
            shape_by_class(&matrices, 256, 512, 256),
            [
                (256, 768),
                (512, 1_536),
                (7_504, 22_512),
                (224, 25_524),
                (7_040, 35_206),
                (256, 1_280),
            ]
        );
        assert_eq!(nonzeros(&matrices), 86_826);
    }

    #[test]
    fn fixed_iv_compression_shape_is_stable() {
        let mut builder = R1csBuilder::<Fr>::new();
        let cv = IV.map(constant_word);
        let block = std::array::from_fn(|word| allocate_word(&mut builder, word as u32));
        let output = compress(&mut builder, &cv, &block, 0, 64, CHUNK_START | CHUNK_END);
        let _cv = chaining_value(&mut builder, &output);
        let witness = builder.witness().expect("complete witness");
        let matrices = builder.into_matrices();

        assert!(matrices.check_witness(&witness).is_ok());
        assert_eq!(matrices.num_constraints, 15_408);
        assert_eq!(matrices.num_vars - 1, 15_184);
        assert_eq!(
            shape_by_class(&matrices, 0, 512, 256),
            [
                (0, 0),
                (512, 1_536),
                (7_504, 22_512),
                (224, 25_276),
                (6_912, 34_692),
                (256, 1_280),
            ]
        );
        assert_eq!(nonzeros(&matrices), 85_296);
    }

    #[test]
    fn matches_blake3_on_random_single_block_inputs() {
        let mut rng = ChaCha20Rng::seed_from_u64(0x5eed_b1a3_0000_0064);
        for len in 1..=64 {
            let mut input = vec![0; len];
            rng.fill_bytes(&mut input);
            let (builder, actual) = hash_gadget(&input);
            assert_eq!(actual, *blake3::hash(&input).as_bytes());
            let witness = builder.witness().expect("complete witness");
            assert!(builder.into_matrices().check_witness(&witness).is_ok());
        }
    }
}
