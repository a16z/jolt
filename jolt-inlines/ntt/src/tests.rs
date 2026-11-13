#![expect(clippy::unwrap_used, reason = "fixed-size test harness outputs")]

use super::{
    forward_ntt64, pointwise_builder::PointwiseDot64, pointwise_dot64,
    sequence_builder::ForwardNtt64, DEGREE, DOT_PRODUCTS,
};
use crate::pointwise::DOT_NONNEGATIVE_FUNCT7;
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
        load_ntt(h, state, &psi, &twiddles, pinv);
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

fn load_ntt(
    h: &mut InlineTestHarness,
    state: &[i32; DEGREE],
    psi: &[i32; DEGREE],
    twiddles: &[i32; DEGREE],
    pinv: i32,
) {
    let mut words = vec![
        DRAM_BASE + 24,
        DRAM_BASE + 24 + DEGREE as u64 * 4,
        P as u64 | (u64::from(pinv as u32) << 32),
    ];
    for table in [psi, twiddles] {
        for pair in table.as_chunks::<2>().0 {
            words.push(u64::from(pair[0] as u32) | (u64::from(pair[1] as u32) << 32));
        }
    }
    h.load_input64(&words);
    h.load_state32(&state.map(|a| a as u32));
}

#[test]
fn expanded_ntt_preserves_signed_twiddle_words() {
    use tracer::instruction::RISCVTrace;
    let (psi, _, pinv) = parameters();
    let edges = [i32::MIN, i32::MAX, -1, 0, 1, -123_456_789];
    let state = core::array::from_fn(|i| edges[i % edges.len()]);
    for shift in 0..edges.len() {
        let twiddles = core::array::from_fn(|i| edges[(i + shift) % edges.len()]);
        let mut expected = state;
        forward_ntt64(&mut expected, &psi, &twiddles, P as i32, pinv);
        let mut harness = ForwardNtt64::harness();
        harness.setup_registers();
        load_ntt(&mut harness, &state, &psi, &twiddles, pinv);
        let mut rows = Vec::new();
        ForwardNtt64::instruction().trace(&mut harness.cpu, Some(&mut rows));
        let actual = harness
            .read_output32(DEGREE)
            .into_iter()
            .map(|word| word as i32)
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
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
        load_dot(h, input, P as i32, pinv);
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

fn load_dot(h: &mut InlineTestHarness, input: &DotInput, p: i32, pinv: i32) {
    let data = DRAM_BASE + (2 * DOT_PRODUCTS + 1) as u64 * 8;
    let mut words = (0..2 * DOT_PRODUCTS)
        .map(|i| data + (i * DEGREE * 4) as u64)
        .collect::<Vec<_>>();
    words.push(u64::from(p as u32) | (u64::from(pinv as u32) << 32));
    for row in input.lhs.iter().chain(&input.rhs) {
        for pair in row.as_chunks::<2>().0 {
            words.push(u64::from(pair[0] as u32) | (u64::from(pair[1] as u32) << 32));
        }
    }
    h.load_input64(&words);
    h.load_state32(&input.acc.map(|a| a as u32));
}

fn trace_dot(input: &DotInput, p: i32, pinv: i32, selector: u32) -> [i32; DEGREE] {
    use tracer::instruction::RISCVTrace;
    let mut harness = PointwiseDot64::harness();
    harness.setup_registers();
    load_dot(&mut harness, input, p, pinv);
    let mut instruction = PointwiseDot64::instruction();
    instruction.funct7 = selector;
    instruction.trace(&mut harness.cpu, None);
    PointwiseDot64::read(&mut harness)
}

#[test]
fn pointwise_nonnegative_correction_preserves_every_word_domain() {
    let edges = [i32::MIN, i32::MAX, -1, 0, 1, 1 << 30];
    for p in [
        0,
        1,
        2,
        1 << 30,
        i32::MAX,
        1_073_707_009,
        1_073_698_817,
        1_073_692_673,
        1_073_682_433,
        1_073_668_097,
        1_073_655_809,
    ] {
        let mut inverse = 1i32;
        for _ in 0..5 {
            inverse = inverse.wrapping_mul(2i32.wrapping_sub(p.wrapping_mul(inverse)));
        }
        for pinv in [0, -1, i32::MIN, i32::MAX, inverse] {
            let input = DotInput {
                acc: core::array::from_fn(|i| edges[i % edges.len()]),
                lhs: core::array::from_fn(|j| {
                    core::array::from_fn(|i| edges[(i + j) % edges.len()])
                }),
                rhs: core::array::from_fn(|j| {
                    core::array::from_fn(|i| edges[(i + 2 * j + 1) % edges.len()])
                }),
            };
            let mut expected = input.acc;
            pointwise_dot64(
                &mut expected,
                input.lhs.each_ref(),
                input.rhs.each_ref(),
                p,
                pinv,
            );
            let short = trace_dot(&input, p, pinv, DOT_NONNEGATIVE_FUNCT7);
            let signed = trace_dot(&input, p, pinv, crate::pointwise::DOT_FUNCT7);
            assert_eq!(short, expected, "p={p}, pinv={pinv}");
            assert_eq!(signed, short, "p={p}, pinv={pinv}");
        }
    }
}

// Fixed signed-word vectors for the portable operation and selector 9.
#[test]
fn negative_moduli_match_frozen_base_words() {
    use tracer::instruction::RISCVTrace;
    // Reducing the zero product gives 5; adding -6 presents x=-1, p=-5
    // to the final correction, where the two instruction domains differ.
    let input = DotInput {
        acc: [-6; DEGREE],
        lhs: [[0; DEGREE]; DOT_PRODUCTS],
        rhs: [[0; DEGREE]; DOT_PRODUCTS],
    };
    assert_eq!(
        trace_dot(&input, -5, 0, crate::pointwise::DOT_FUNCT7),
        [4; DEGREE]
    );
    assert_eq!(
        trace_dot(&input, -5, 0, DOT_NONNEGATIVE_FUNCT7),
        [-1; DEGREE]
    );
    let edges = [i32::MIN, i32::MAX, -1, 0, 1, 1 << 30];
    let cases: [(i32, i32, [i32; 6]); 30] = [
        (
            -1,
            0,
            [
                1_342_177_280,
                -1_073_741_826,
                536_870_912,
                -1_073_741_827,
                1_073_741_826,
                536_870_913,
            ],
        ),
        (
            -1,
            -1,
            [
                1_342_177_280,
                -1_073_741_825,
                536_870_912,
                -1_073_741_827,
                1_073_741_826,
                536_870_913,
            ],
        ),
        (
            -1,
            -2_147_483_648,
            [
                1_342_177_280,
                -1_073_741_826,
                536_870_912,
                -1_073_741_828,
                1_073_741_825,
                536_870_913,
            ],
        ),
        (
            -1,
            2_147_483_647,
            [
                1_342_177_280,
                -1_073_741_826,
                536_870_912,
                -1_073_741_827,
                1_073_741_826,
                536_870_913,
            ],
        ),
        (
            -1,
            -1,
            [
                1_342_177_280,
                -1_073_741_825,
                536_870_912,
                -1_073_741_827,
                1_073_741_826,
                536_870_913,
            ],
        ),
        (
            -2,
            0,
            [
                1_342_177_280,
                -1_073_741_826,
                536_870_914,
                -1_073_741_829,
                1_073_741_828,
                536_870_913,
            ],
        ),
        (
            -2,
            -1,
            [
                1_342_177_279,
                -1_073_741_825,
                536_870_913,
                -1_073_741_830,
                1_073_741_827,
                536_870_912,
            ],
        ),
        (
            -2,
            -2_147_483_648,
            [
                1_342_177_280,
                -1_073_741_827,
                536_870_914,
                -1_073_741_830,
                1_073_741_827,
                536_870_913,
            ],
        ),
        (
            -2,
            2_147_483_647,
            [
                1_342_177_279,
                -1_073_741_826,
                536_870_913,
                -1_073_741_829,
                1_073_741_828,
                536_870_912,
            ],
        ),
        (
            -2,
            -250_667_200,
            [
                1_342_177_280,
                -1_073_741_826,
                536_870_914,
                -1_073_741_829,
                1_073_741_828,
                536_870_913,
            ],
        ),
        (
            -1_073_741_824,
            0,
            [
                1_342_177_280,
                1_073_741_822,
                -1_610_612_738,
                -1_073_741_825,
                1_073_741_824,
                -1_610_612_735,
            ],
        ),
        (
            -1_073_741_824,
            -1,
            [
                1_073_741_824,
                1_342_177_278,
                2_147_483_646,
                -1_342_177_281,
                -1_342_177_280,
                -2_147_483_647,
            ],
        ),
        (
            -1_073_741_824,
            -2_147_483_648,
            [
                1_342_177_280,
                536_870_910,
                -1_610_612_738,
                -1_610_612_737,
                -1_610_612_736,
                -1_610_612_735,
            ],
        ),
        (
            -1_073_741_824,
            2_147_483_647,
            [
                1_073_741_824,
                805_306_366,
                2_147_483_646,
                1_342_177_279,
                1_342_177_280,
                -2_147_483_647,
            ],
        ),
        (
            -1_073_741_824,
            32,
            [
                1_342_177_280,
                1_073_741_814,
                -1_610_612_738,
                1_073_741_831,
                1_073_741_832,
                -1_610_612_735,
            ],
        ),
        (
            -1_073_707_009,
            0,
            [
                1_342_177_280,
                1_073_672_192,
                -1_610_682_368,
                1_073_811_453,
                -1_073_811_454,
                -1_610_682_365,
            ],
        ),
        (
            -1_073_707_009,
            -1,
            [
                1_073_750_528,
                1_342_098_945,
                2_147_431_424,
                -1_342_168_578,
                -1_342_238_207,
                17_409,
            ],
        ),
        (
            -1_073_707_009,
            -2_147_483_648,
            [
                1_342_177_280,
                536_818_688,
                -1_610_682_368,
                -1_610_595_330,
                -1_610_664_959,
                -1_610_682_365,
            ],
        ),
        (
            -1_073_707_009,
            2_147_483_647,
            [
                1_073_750_528,
                805_245_440,
                2_147_431_424,
                1_342_098_945,
                1_342_168_576,
                17_409,
            ],
        ),
        (
            -1_073_707_009,
            -138_446_849,
            [
                1_073_750_528,
                1_376_709_535,
                2_147_431_424,
                -1_376_779_167,
                -1_376_848_796,
                17_409,
            ],
        ),
        (
            -2_147_483_647,
            0,
            [
                1_342_177_278,
                -1_073_741_828,
                536_870_908,
                -1_073_741_827,
                1_073_741_822,
                536_870_911,
            ],
        ),
        (
            -2_147_483_647,
            -1,
            [
                805_306_366,
                -536_870_916,
                -536_870_915,
                -1_610_612_739,
                536_870_910,
                -536_870_912,
            ],
        ),
        (
            -2_147_483_647,
            -2_147_483_648,
            [
                1_342_177_278,
                2_147_483_645,
                536_870_908,
                2_147_483_645,
                -2,
                536_870_911,
            ],
        ),
        (
            -2_147_483_647,
            2_147_483_647,
            [
                805_306_366,
                -1_610_612_739,
                -536_870_915,
                -536_870_916,
                1_610_612_733,
                -536_870_912,
            ],
        ),
        (
            -2_147_483_647,
            -2_147_483_647,
            [
                1_879_048_190,
                -536_870_916,
                -536_870_915,
                -1_610_612_738,
                536_870_911,
                -536_870_912,
            ],
        ),
        (
            -2_147_483_648,
            0,
            [
                1_342_177_280,
                -1_073_741_826,
                536_870_910,
                -1_073_741_825,
                1_073_741_824,
                536_870_913,
            ],
        ),
        (
            -2_147_483_648,
            -1,
            [
                805_306_368,
                -536_870_913,
                -536_870_914,
                -1_610_612_738,
                536_870_911,
                -536_870_911,
            ],
        ),
        (
            -2_147_483_648,
            -2_147_483_648,
            [
                1_342_177_280,
                2_147_483_646,
                536_870_910,
                2_147_483_647,
                0,
                536_870_913,
            ],
        ),
        (
            -2_147_483_648,
            2_147_483_647,
            [
                805_306_368,
                -1_610_612_737,
                -536_870_914,
                -536_870_914,
                1_610_612_735,
                -536_870_911,
            ],
        ),
        (
            -2_147_483_648,
            32,
            [
                1_342_177_280,
                -1_073_741_842,
                536_870_910,
                -1_073_741_809,
                1_073_741_840,
                536_870_913,
            ],
        ),
    ];
    for (p, pinv, expected) in cases {
        let input = DotInput {
            acc: core::array::from_fn(|i| edges[i % edges.len()]),
            lhs: core::array::from_fn(|j| core::array::from_fn(|i| edges[(i + j) % edges.len()])),
            rhs: core::array::from_fn(|j| {
                core::array::from_fn(|i| edges[(i + 2 * j + 1) % edges.len()])
            }),
        };
        let mut actual = input.acc;
        pointwise_dot64(
            &mut actual,
            input.lhs.each_ref(),
            input.rhs.each_ref(),
            p,
            pinv,
        );
        assert_eq!(
            actual,
            core::array::from_fn(|i| expected[i % 6]),
            "p={p}, pinv={pinv}"
        );
        let mut harness = PointwiseDot64::harness();
        harness.setup_registers();
        load_dot(&mut harness, &input, p, pinv);
        PointwiseDot64::instruction().trace(&mut harness.cpu, None);
        assert_eq!(
            PointwiseDot64::read(&mut harness),
            actual,
            "p={p}, pinv={pinv}"
        );
    }
}
