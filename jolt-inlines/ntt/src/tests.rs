#![expect(clippy::unwrap_used, reason = "fixed-size test harness outputs")]

use super::{
    forward_ntt64, pointwise_builder::PointwiseDot64, pointwise_dot64,
    sequence_builder::ForwardNtt64, DEGREE, DOT_PRODUCTS,
};
use jolt_inlines_sdk::{
    assert_edge_cases_match_reference, assert_random_cases_match_reference, InlineReference,
    InlineSpec,
};
use rand::RngCore;
use tracer::emulator::mmu::DRAM_BASE;
use tracer::utils::inline_test_harness::{InlineMemoryLayout, InlineTestHarness};

const P: i64 = 998_244_353;
const R: i64 = (1i64 << 32) % P;

fn pow(mut a: i64, mut n: usize) -> i64 {
    let mut result = 1;
    while n != 0 {
        if n & 1 != 0 {
            result = result * a % P;
        }
        a = a * a % P;
        n >>= 1;
    }
    result
}

fn parameters() -> ([i32; DEGREE], [i32; DEGREE], i32) {
    let root = pow(3, (P as usize - 1) / (2 * DEGREE));
    let center = |a: i64| if a > P / 2 { (a - P) as i32 } else { a as i32 };
    let psi = core::array::from_fn(|i| center(pow(root, i) * R % P));
    let mut twiddles = [0; DEGREE];
    for stage in 0..6 {
        let len = 1 << stage;
        for j in 0..len {
            twiddles[len - 1 + j] = center(pow(root, DEGREE / len * j) * R % P);
        }
    }
    let mut inverse = 1i32;
    for _ in 0..5 {
        inverse = inverse.wrapping_mul(2i32.wrapping_sub((P as i32).wrapping_mul(inverse)));
    }
    (psi, twiddles, inverse)
}

impl InlineReference for ForwardNtt64 {
    type Input = [i32; DEGREE];
    type Output = [i32; DEGREE];

    fn reference(state: &Self::Input) -> Self::Output {
        let root = pow(3, (P as usize - 1) / (2 * DEGREE));
        core::array::from_fn(|i| {
            let frequency = i.reverse_bits() >> (usize::BITS - 6);
            let point = pow(root, 2 * frequency + 1);
            state.iter().enumerate().fold(0i64, |sum, (j, a)| {
                (sum + i64::from(*a) * pow(point, j)).rem_euclid(P)
            }) as i32
        })
    }
}

impl InlineSpec for ForwardNtt64 {
    fn edge_cases() -> impl IntoIterator<Item = Self::Input> {
        let mut impulse = [0; DEGREE];
        impulse[0] = R as i32;
        [
            [0; DEGREE],
            impulse,
            [P as i32 - 1; DEGREE],
            [1 - P as i32; DEGREE],
            core::array::from_fn(|i| {
                if i & 1 == 0 {
                    P as i32 - 1
                } else {
                    1 - P as i32
                }
            }),
        ]
    }
    fn random(rng: &mut impl RngCore) -> Self::Input {
        core::array::from_fn(|_| (rng.next_u32() as i64 % (2 * P - 1) - P + 1) as i32)
    }
    fn harness() -> InlineTestHarness {
        InlineTestHarness::new(InlineMemoryLayout::single_input(
            24 + 2 * DEGREE * 4,
            DEGREE * 4,
        ))
    }
    fn load(h: &mut InlineTestHarness, state: &Self::Input) {
        let (psi, twiddles, pinv) = parameters();
        let mut words = vec![
            DRAM_BASE + 24,
            DRAM_BASE + 24 + DEGREE as u64 * 4,
            P as u64 | (u64::from(pinv as u32) << 32),
        ];
        for table in [psi, twiddles] {
            for pair in table.chunks_exact(2) {
                words.push(u64::from(pair[0] as u32) | (u64::from(pair[1] as u32) << 32));
            }
        }
        h.load_input64(&words);
        h.load_state32(&state.map(|a| a as u32));
    }
    fn read(h: &mut InlineTestHarness) -> Self::Output {
        h.read_output32(DEGREE)
            .into_iter()
            .map(|a| i64::from(a as i32).rem_euclid(P) as i32)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }
}

#[test]
fn expanded_ntt_matches_direct_dft() {
    assert_edge_cases_match_reference::<ForwardNtt64>();
    assert_random_cases_match_reference::<ForwardNtt64>(0x004e_5454, 16);
}

#[test]
fn portable_ntt_matches_direct_dft() {
    let (psi, twiddles, pinv) = parameters();
    for state in ForwardNtt64::edge_cases() {
        let mut actual = state;
        forward_ntt64(&mut actual, &psi, &twiddles, P as i32, pinv);
        assert_eq!(
            actual.map(|a| i64::from(a).rem_euclid(P) as i32),
            ForwardNtt64::reference(&state)
        );
    }
}

#[test]
fn expanded_ntt_preserves_wrapping_word_semantics() {
    use tracer::instruction::RISCVTrace;
    let (psi, twiddles, pinv) = parameters();
    for state in [
        [i32::MIN; DEGREE],
        [i32::MAX; DEGREE],
        core::array::from_fn(|i| if i & 1 == 0 { i32::MIN } else { i32::MAX }),
    ] {
        let mut expected = state;
        forward_ntt64(&mut expected, &psi, &twiddles, P as i32, pinv);
        let mut harness = ForwardNtt64::harness();
        harness.setup_registers();
        ForwardNtt64::load(&mut harness, &state);
        let mut rows = Vec::new();
        ForwardNtt64::instruction().trace(&mut harness.cpu, Some(&mut rows));
        let actual = harness
            .read_output32(DEGREE)
            .into_iter()
            .map(|x| x as i32)
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }
}

#[derive(Debug, Clone)]
pub struct DotInput {
    acc: [i32; DEGREE],
    lhs: [[i32; DEGREE]; DOT_PRODUCTS],
    rhs: [[i32; DEGREE]; DOT_PRODUCTS],
}

impl InlineReference for PointwiseDot64 {
    type Input = DotInput;
    type Output = [i32; DEGREE];

    fn reference(input: &Self::Input) -> Self::Output {
        let inverse = pow(R, P as usize - 2);
        core::array::from_fn(|lane| {
            input
                .lhs
                .iter()
                .zip(&input.rhs)
                .fold(i64::from(input.acc[lane]), |sum, (a, b)| {
                    (sum + (i64::from(a[lane]) * i64::from(b[lane]) % P) * inverse % P)
                        .rem_euclid(P)
                }) as i32
        })
    }
}

impl InlineSpec for PointwiseDot64 {
    fn edge_cases() -> impl IntoIterator<Item = Self::Input> {
        [0, 1, P as i32 - 1].map(|value| DotInput {
            acc: [P as i32 - 1; DEGREE],
            lhs: [[value; DEGREE]; DOT_PRODUCTS],
            rhs: [[value; DEGREE]; DOT_PRODUCTS],
        })
    }
    fn random(rng: &mut impl RngCore) -> Self::Input {
        DotInput {
            acc: core::array::from_fn(|_| (u64::from(rng.next_u32()) % P as u64) as i32),
            lhs: core::array::from_fn(|_| {
                core::array::from_fn(|_| (u64::from(rng.next_u32()) % P as u64) as i32)
            }),
            rhs: core::array::from_fn(|_| {
                core::array::from_fn(|_| (u64::from(rng.next_u32()) % P as u64) as i32)
            }),
        }
    }
    fn harness() -> InlineTestHarness {
        InlineTestHarness::new(InlineMemoryLayout::single_input(
            (2 * DOT_PRODUCTS + 1) * 8 + 2 * DOT_PRODUCTS * DEGREE * 4,
            DEGREE * 4,
        ))
    }
    fn load(h: &mut InlineTestHarness, input: &Self::Input) {
        let (_, _, pinv) = parameters();
        let data = DRAM_BASE + (2 * DOT_PRODUCTS + 1) as u64 * 8;
        let mut words = (0..2 * DOT_PRODUCTS)
            .map(|i| data + (i * DEGREE * 4) as u64)
            .collect::<Vec<_>>();
        words.push(P as u64 | (u64::from(pinv as u32) << 32));
        for row in input.lhs.iter().chain(&input.rhs) {
            for pair in row.chunks_exact(2) {
                words.push(u64::from(pair[0] as u32) | (u64::from(pair[1] as u32) << 32));
            }
        }
        h.load_input64(&words);
        h.load_state32(&input.acc.map(|a| a as u32));
    }
    fn read(h: &mut InlineTestHarness) -> Self::Output {
        h.read_output32(DEGREE)
            .into_iter()
            .map(|a| a as i32)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }
}

#[test]
fn expanded_pointwise_dot_matches_modular_relation() {
    assert_edge_cases_match_reference::<PointwiseDot64>();
    assert_random_cases_match_reference::<PointwiseDot64>(0x0044_4f54, 16);
}

#[test]
fn portable_pointwise_dot_matches_modular_relation() {
    let (_, _, pinv) = parameters();
    for input in PointwiseDot64::edge_cases() {
        let mut actual = input.acc;
        pointwise_dot64(
            &mut actual,
            input.lhs.each_ref(),
            input.rhs.each_ref(),
            P as i32,
            pinv,
        );
        assert_eq!(actual, PointwiseDot64::reference(&input));
    }
}

#[test]
fn expanded_pointwise_dot_preserves_wrapping_word_semantics() {
    use tracer::instruction::RISCVTrace;
    let (_, _, pinv) = parameters();
    for value in [i32::MIN, i32::MAX] {
        let input = DotInput {
            acc: [value; DEGREE],
            lhs: [[value; DEGREE]; DOT_PRODUCTS],
            rhs: [[value; DEGREE]; DOT_PRODUCTS],
        };
        let mut expected = input.acc;
        pointwise_dot64(
            &mut expected,
            input.lhs.each_ref(),
            input.rhs.each_ref(),
            P as i32,
            pinv,
        );
        let mut harness = PointwiseDot64::harness();
        harness.setup_registers();
        PointwiseDot64::load(&mut harness, &input);
        let mut rows = Vec::new();
        PointwiseDot64::instruction().trace(&mut harness.cpu, Some(&mut rows));
        assert_eq!(PointwiseDot64::read(&mut harness), expected);
    }
}
