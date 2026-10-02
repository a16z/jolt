//! BLAKE2b compression over boolean R1CS variables.

use jolt_field::JoltField;

use super::{bit::xor_word, Bit};
use crate::{LinearCombination, R1csBuilder};

const WORD_BITS: usize = 64;
const IV: [u64; 8] = [
    0x6a09_e667_f3bc_c908,
    0xbb67_ae85_84ca_a73b,
    0x3c6e_f372_fe94_f82b,
    0xa54f_f53a_5f1d_36f1,
    0x510e_527f_ade6_82d1,
    0x9b05_688c_2b3e_6c1f,
    0x1f83_d9ab_fb41_bd6b,
    0x5be0_cd19_137e_2179,
];
const SIGMA: [[usize; 16]; 12] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
];

/// A boolean value backed by either a constant or an R1CS variable.
pub type Blake2bBit = Bit;

/// A little-endian 64-bit BLAKE2b word.
pub type Blake2bWord = [Blake2bBit; WORD_BITS];

/// Allocates one 64-bit word as 64 boolean witness variables.
pub fn allocate_word<F: JoltField>(builder: &mut R1csBuilder<F>, value: u64) -> Blake2bWord {
    std::array::from_fn(|bit| Blake2bBit::allocate(builder, value & (1 << bit) != 0))
}

/// Creates one constant 64-bit word without allocating variables.
pub fn constant_word(value: u64) -> Blake2bWord {
    std::array::from_fn(|bit| Blake2bBit::constant(value & (1 << bit) != 0))
}

/// Returns the assigned value of a word.
pub fn word_value(word: &Blake2bWord) -> u64 {
    word.iter().enumerate().fold(0, |value, (bit, assigned)| {
        value | (u64::from(assigned.value()) << bit)
    })
}

fn rotate_right(mut word: Blake2bWord, amount: usize) -> Blake2bWord {
    word.rotate_left(amount);
    word
}

fn assert_sum<F: JoltField>(
    builder: &mut R1csBuilder<F>,
    inputs: &[&Blake2bWord],
    result: &Blake2bWord,
    carries: &[Blake2bBit],
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
    for carry in carries {
        carry.add_scaled(&mut output_sum, scale);
        scale += scale;
    }
    builder.assert_equal(input_sum, output_sum);
}

fn add2<F: JoltField>(
    builder: &mut R1csBuilder<F>,
    lhs: &Blake2bWord,
    rhs: &Blake2bWord,
) -> Blake2bWord {
    let sum = u128::from(word_value(lhs)) + u128::from(word_value(rhs));
    let result = std::array::from_fn(|bit| Blake2bBit::allocate(builder, sum & (1 << bit) != 0));
    let carry = Blake2bBit::allocate(builder, sum & (1 << WORD_BITS) != 0);
    assert_sum(builder, &[lhs, rhs], &result, &[carry]);
    result
}

fn add3<F: JoltField>(
    builder: &mut R1csBuilder<F>,
    a: &Blake2bWord,
    b: &Blake2bWord,
    c: &Blake2bWord,
) -> Blake2bWord {
    let sum = u128::from(word_value(a)) + u128::from(word_value(b)) + u128::from(word_value(c));
    let result = std::array::from_fn(|bit| Blake2bBit::allocate(builder, sum & (1 << bit) != 0));
    let carries = [
        Blake2bBit::allocate(builder, sum & (1 << WORD_BITS) != 0),
        Blake2bBit::allocate(builder, sum & (1 << (WORD_BITS + 1)) != 0),
    ];
    assert_sum(builder, &[a, b, c], &result, &carries);
    result
}

#[expect(
    clippy::indexing_slicing,
    reason = "BLAKE2b's fixed permutation indices are all within 16 words"
)]
fn g<F: JoltField>(
    builder: &mut R1csBuilder<F>,
    state: &mut [Blake2bWord],
    indices: [usize; 4],
    x: &Blake2bWord,
    y: &Blake2bWord,
) {
    let [a, b, c, d] = indices;
    let mut va = state[a];
    let mut vb = state[b];
    let mut vc = state[c];
    let mut vd = state[d];

    va = add3(builder, &va, &vb, x);
    vd = rotate_right(xor_word(builder, &vd, &va), 32);
    vc = add2(builder, &vc, &vd);
    vb = rotate_right(xor_word(builder, &vb, &vc), 24);
    va = add3(builder, &va, &vb, y);
    vd = rotate_right(xor_word(builder, &vd, &va), 16);
    vc = add2(builder, &vc, &vd);
    vb = rotate_right(xor_word(builder, &vb, &vc), 63);

    state[a] = va;
    state[b] = vb;
    state[c] = vc;
    state[d] = vd;
}

/// Constrains one RFC 7693 BLAKE2b compression invocation.
///
/// `h` and `message` must already be represented by boolean bits. `t` and
/// `last_block` are constants for each block in the digest schedule.
#[expect(
    clippy::indexing_slicing,
    reason = "BLAKE2b's fixed state and message permutation indices are in range"
)]
pub fn compress<F: JoltField>(
    builder: &mut R1csBuilder<F>,
    h: &[Blake2bWord; 8],
    message: &[Blake2bWord; 16],
    t: u128,
    last_block: bool,
) -> [Blake2bWord; 8] {
    let original_h = *h;
    let mut state = Vec::with_capacity(16);
    state.extend_from_slice(h);
    state.extend([
        constant_word(IV[0]),
        constant_word(IV[1]),
        constant_word(IV[2]),
        constant_word(IV[3]),
        constant_word(IV[4] ^ t as u64),
        constant_word(IV[5] ^ (t >> WORD_BITS) as u64),
        constant_word(IV[6] ^ if last_block { u64::MAX } else { 0 }),
        constant_word(IV[7]),
    ]);

    for sigma in SIGMA {
        g(
            builder,
            &mut state,
            [0, 4, 8, 12],
            &message[sigma[0]],
            &message[sigma[1]],
        );
        g(
            builder,
            &mut state,
            [1, 5, 9, 13],
            &message[sigma[2]],
            &message[sigma[3]],
        );
        g(
            builder,
            &mut state,
            [2, 6, 10, 14],
            &message[sigma[4]],
            &message[sigma[5]],
        );
        g(
            builder,
            &mut state,
            [3, 7, 11, 15],
            &message[sigma[6]],
            &message[sigma[7]],
        );
        g(
            builder,
            &mut state,
            [0, 5, 10, 15],
            &message[sigma[8]],
            &message[sigma[9]],
        );
        g(
            builder,
            &mut state,
            [1, 6, 11, 12],
            &message[sigma[10]],
            &message[sigma[11]],
        );
        g(
            builder,
            &mut state,
            [2, 7, 8, 13],
            &message[sigma[12]],
            &message[sigma[13]],
        );
        g(
            builder,
            &mut state,
            [3, 4, 9, 14],
            &message[sigma[14]],
            &message[sigma[15]],
        );
    }

    std::array::from_fn(|word| {
        let mixed = xor_word(builder, &original_h[word], &state[word]);
        xor_word(builder, &mixed, &state[word + 8])
    })
}

#[cfg(test)]
#[expect(
    clippy::indexing_slicing,
    reason = "tests index fixed-size hash blocks"
)]
#[expect(clippy::expect_used, reason = "tests may panic on assertion failures")]
mod tests {
    use blake2::digest::consts::{U32, U64};
    use blake2::{Blake2b, Digest};
    use jolt_field::Fr;
    use rand_chacha::ChaCha20Rng;
    use rand_core::{RngCore, SeedableRng};

    use super::*;
    use crate::{ConstraintMatrices, Variable};

    fn digest_gadget(
        builder: &mut R1csBuilder<Fr>,
        input: &[u8],
        output_len: usize,
    ) -> Vec<Blake2bBit> {
        let mut h = IV.map(constant_word);
        h[0] = constant_word(IV[0] ^ 0x0101_0000 ^ output_len as u64);
        let block_count = input.len().max(1).div_ceil(128);

        for block_index in 0..block_count {
            let start = block_index * 128;
            let end = (start + 128).min(input.len());
            let mut block = [0u8; 128];
            if start < input.len() {
                block[..end - start].copy_from_slice(&input[start..end]);
            }
            let message = std::array::from_fn(|word| {
                let start = word * 8;
                allocate_word(
                    builder,
                    u64::from_le_bytes(block[start..start + 8].try_into().expect("eight bytes")),
                )
            });
            h = compress(
                builder,
                &h,
                &message,
                end as u128,
                block_index + 1 == block_count,
            );
        }

        h.into_iter()
            .flat_map(|word| word.into_iter())
            .take(output_len * 8)
            .collect()
    }

    fn output_bytes(bits: &[Blake2bBit]) -> Vec<u8> {
        bits.chunks_exact(8)
            .map(|byte| {
                byte.iter().enumerate().fold(0, |value, (bit, assigned)| {
                    value | (u8::from(assigned.value()) << bit)
                })
            })
            .collect()
    }

    fn assert_digest<const OUTPUT_BYTES: usize>(input: &[u8], expected: &[u8]) {
        let mut builder = R1csBuilder::<Fr>::new();
        let output = digest_gadget(&mut builder, input, OUTPUT_BYTES);
        assert_eq!(output_bytes(&output), expected);
        let witness = builder.witness().expect("complete witness");
        assert!(builder.into_matrices().check_witness(&witness).is_ok());
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
        state_bits: usize,
        message_bits: usize,
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
            let class = if index < state_bits {
                0
            } else if index < state_bits + message_bits {
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
        let h = IV.map(|word| allocate_word(&mut builder, word));
        let message = std::array::from_fn(|word| allocate_word(&mut builder, word as u64));
        let _output = compress(&mut builder, &h, &message, 128, true);
        let witness = builder.witness().expect("complete witness");
        let matrices = builder.into_matrices();

        assert!(matrices.check_witness(&witness).is_ok());
        assert_eq!(matrices.num_constraints, 52_416);
        assert_eq!(matrices.num_vars - 1, 52_032);
        assert_eq!(
            shape_by_class(&matrices, 512, 1_024, 1_024),
            [
                (512, 1_536),
                (1_024, 3_072),
                (25_152, 75_456),
                (384, 86_724),
                (24_320, 121_840),
                (1_024, 5_120),
            ]
        );
        assert_eq!(nonzeros(&matrices), 293_748);
    }

    #[test]
    fn fixed_iv_compression_shape_is_stable() {
        let mut builder = R1csBuilder::<Fr>::new();
        let h = IV.map(constant_word);
        let message = std::array::from_fn(|word| allocate_word(&mut builder, word as u64));
        let _output = compress(&mut builder, &h, &message, 128, true);
        let witness = builder.witness().expect("complete witness");
        let matrices = builder.into_matrices();

        assert!(matrices.check_witness(&witness).is_ok());
        assert_eq!(matrices.num_constraints, 51_136);
        assert_eq!(matrices.num_vars - 1, 50_752);
        assert_eq!(
            shape_by_class(&matrices, 0, 1_024, 512),
            [
                (0, 0),
                (1_024, 3_072),
                (25_152, 75_456),
                (384, 86_220),
                (24_064, 120_826),
                (512, 3_112),
            ]
        );
        assert_eq!(nonzeros(&matrices), 288_686);
    }

    #[test]
    fn matches_rfc_7693_abc_vector() {
        let expected = hex::decode(
            "ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d1\
             7d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923",
        )
        .expect("valid vector");
        assert_digest::<64>(b"abc", &expected);
    }

    #[test]
    fn matches_blake2b_256_on_random_single_and_multi_block_inputs() {
        let mut rng = ChaCha20Rng::seed_from_u64(0x5eed_b1a2_e2b2_0256);
        for case in 0..50 {
            let len = if case % 2 == 0 {
                (rng.next_u32() as usize % 128) + 1
            } else {
                (rng.next_u32() as usize % 256) + 129
            };
            let mut input = vec![0; len];
            rng.fill_bytes(&mut input);
            let expected = Blake2b::<U32>::digest(&input);
            assert_digest::<32>(&input, &expected);
        }

        let empty_expected = Blake2b::<U32>::digest([]);
        assert_digest::<32>(&[], &empty_expected);
        let rfc_expected = Blake2b::<U64>::digest(b"abc");
        assert_digest::<64>(b"abc", &rfc_expected);
    }
}
