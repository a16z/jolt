#![expect(clippy::unwrap_used, reason = "fixed valid NTT test parameters")]

use super::{CenteredI16NttInput, PreparedForwardNtt64};
use crate::{forward_ntt64, sequence_builder::ForwardNtt64, DEGREE, MERGED_FUNCT7};
use jolt_inlines_sdk::InlineSpec;
use rand::{rngs::StdRng, RngCore, SeedableRng};
use tracer::emulator::mmu::DRAM_BASE;
use tracer::instruction::RISCVTrace;
use tracer::utils::inline_test_harness::{InlineMemoryLayout, InlineTestHarness};
use tracer::utils::virtual_registers::VirtualRegisterAllocator;

const PRIMES: [i32; 6] = [
    1_073_707_009,
    1_073_698_817,
    1_073_692_673,
    1_073_682_433,
    1_073_668_097,
    1_073_655_809,
];

fn pow(mut a: i64, mut n: usize, p: i64) -> i64 {
    let mut value = 1;
    while n != 0 {
        if n & 1 != 0 {
            value = value * a % p;
        }
        a = a * a % p;
        n >>= 1;
    }
    value
}

struct Parameters {
    p: i32,
    pinv: i32,
    root: i64,
    psi: [i32; DEGREE],
    twiddles: [i32; DEGREE],
}

impl Parameters {
    fn new(p: i32) -> Self {
        let modulus = i64::from(p);
        let root = (2..modulus)
            .find(|&a| pow(a, (p as usize - 1) / 2, modulus) == modulus - 1)
            .map(|a| pow(a, (p as usize - 1) / (2 * DEGREE), modulus))
            .unwrap();
        let radix = (1i64 << 32) % modulus;
        let psi = core::array::from_fn(|i| {
            (pow(root, i, modulus) * radix % modulus * radix % modulus) as i32
        });
        let mut twiddles = [0; DEGREE];
        for length in [1, 2, 4, 8, 16, 32] {
            for j in 0..length {
                twiddles[length - 1 + j] =
                    (pow(root, DEGREE / length * j, modulus) * radix % modulus) as i32;
            }
        }
        let mut pinv = 1i32;
        for _ in 0..5 {
            pinv = pinv.wrapping_mul(2i32.wrapping_sub(p.wrapping_mul(pinv)));
        }
        Self {
            p,
            pinv,
            root,
            psi,
            twiddles,
        }
    }

    fn evaluate(&self, state: &[i32; DEGREE]) -> [i32; DEGREE] {
        let p = i64::from(self.p);
        core::array::from_fn(|i: usize| {
            let frequency = i.reverse_bits() >> (usize::BITS - 6);
            let point = pow(self.root, 2 * frequency + 1, p);
            let value = state.iter().enumerate().fold(0i64, |sum, (j, &value)| {
                (sum + i64::from(value) * pow(point, j, p)).rem_euclid(p)
            });
            (value * ((1i64 << 32) % p) % p) as i32
        })
    }
}

fn expanded(prepared: &PreparedForwardNtt64<'_>, state: &[i32; DEGREE]) -> Vec<i32> {
    let mut harness = InlineTestHarness::new(InlineMemoryLayout::single_input(
        24 + 2 * DEGREE * 4,
        DEGREE * 4,
    ));
    harness.setup_registers();
    let mut words = vec![
        // An invalid twist pointer proves the merged instruction never reads it.
        0,
        DRAM_BASE + 24 + DEGREE as u64 * 4,
        u64::from(prepared.p as u32) | (u64::from(prepared.pinv as u32) << 32),
    ];
    let twiddles = &prepared.merged.0;
    for table in [prepared.psi, twiddles] {
        for pair in table.as_chunks::<2>().0 {
            words.push(u64::from(pair[0] as u32) | (u64::from(pair[1] as u32) << 32));
        }
    }
    harness.load_input64(&words);
    harness.load_state32(&state.map(|x| x as u32));
    let mut instruction = ForwardNtt64::instruction();
    instruction.funct7 = MERGED_FUNCT7;
    instruction.trace(&mut harness.cpu, None);
    harness
        .read_output32(DEGREE)
        .into_iter()
        .map(|x| x as i32)
        .collect()
}

#[test]
fn raw_ntt_preserves_direct_evaluation_for_six_primes() {
    for p in PRIMES {
        let params = Parameters::new(p);
        let mut states = [0, 1, -1, i32::MIN, i32::MAX, p - 1, 1 - p]
            .map(|x| [x; DEGREE])
            .to_vec();
        states.push(core::array::from_fn(|i| {
            if i & 1 == 0 {
                i32::MIN
            } else {
                i32::MAX
            }
        }));
        states.push(core::array::from_fn(|i| {
            (i as i32).wrapping_mul(1_234_567_891)
        }));
        for state in states {
            let mut base = state;
            forward_ntt64(&mut base, &params.psi, &params.twiddles, p, params.pinv);
            assert_eq!(base, params.evaluate(&state), "p={p}");
        }
    }
}

#[test]
fn prepared_ntt_rejects_each_invalid_admission_condition() {
    for p in PRIMES {
        let params = Parameters::new(p);
        for p in [0, -1, 1 << 30, i32::MAX, 2] {
            assert!(
                PreparedForwardNtt64::new(&params.psi, &params.twiddles, p, params.pinv).is_none()
            );
        }
        assert!(PreparedForwardNtt64::new(
            &params.psi,
            &params.twiddles,
            params.p,
            params.pinv ^ 1
        )
        .is_none());
        for value in [-params.p, params.p, i32::MIN, i32::MAX] {
            let mut psi = params.psi;
            psi[63] = value;
            assert!(
                PreparedForwardNtt64::new(&psi, &params.twiddles, params.p, params.pinv).is_none()
            );
            let mut twiddles = params.twiddles;
            twiddles[63] = value;
            assert!(
                PreparedForwardNtt64::new(&params.psi, &twiddles, params.p, params.pinv).is_none()
            );
        }
        let mut twiddles = params.twiddles;
        twiddles[0] = 0;
        assert!(PreparedForwardNtt64::new(&params.psi, &twiddles, params.p, params.pinv).is_none());
    }
}

#[test]
fn forward_ntt_body_and_materialized_row_counts() {
    let instruction = ForwardNtt64::instruction();
    let rows = instruction.inline_sequence(&VirtualRegisterAllocator::new());
    assert_eq!(rows.len(), 3315);
    let mut merged = ForwardNtt64::instruction();
    merged.funct7 = MERGED_FUNCT7;
    let rows = merged.inline_sequence(&VirtualRegisterAllocator::new());
    assert_eq!(rows.len(), 2255);
    use crate::pointwise_builder::PointwiseDot64;
    let rows = PointwiseDot64::instruction().inline_sequence(&VirtualRegisterAllocator::new());
    assert_eq!(rows.len(), 3363);
    let mut nonnegative = PointwiseDot64::instruction();
    nonnegative.funct7 = crate::pointwise::DOT_NONNEGATIVE_FUNCT7;
    let rows = nonnegative.inline_sequence(&VirtualRegisterAllocator::new());
    assert_eq!(rows.len(), 3235);
}

#[test]
fn merged_ntt_preserves_direct_evaluation_and_exact_words_for_six_primes() {
    let mut rng = StdRng::seed_from_u64(0x4d45_5247_4544);
    for p in PRIMES {
        for negative_representatives in [false, true] {
            let mut params = Parameters::new(p);
            if negative_representatives {
                for value in params.psi.iter_mut().chain(&mut params.twiddles) {
                    if *value > 0 {
                        *value -= p;
                    }
                }
            }
            let prepared =
                PreparedForwardNtt64::new(&params.psi, &params.twiddles, p, params.pinv).unwrap();
            assert_eq!(prepared.merged.0.as_ptr() as usize & 7, 0);
            let mut states = [0, 1, -1, i16::MIN, i16::MAX].map(|x| [x; DEGREE]).to_vec();
            states.push(core::array::from_fn(|i| {
                if i & 1 == 0 {
                    i16::MIN
                } else {
                    i16::MAX
                }
            }));
            states.push(core::array::from_fn(|i| (i as i16).wrapping_mul(31723)));
            for lane in 0..DEGREE {
                let mut impulse = [0i16; DEGREE];
                impulse[lane] = i16::MIN;
                states.push(impulse);
            }
            for _ in 0..64 {
                states.push(core::array::from_fn(|_| rng.next_u32() as i16));
            }
            for digits in states {
                let state = digits.map(i32::from);
                let expected = params.evaluate(&state);
                let input = CenteredI16NttInput::new(&digits);
                let mut actual = [i32::MAX; DEGREE];
                prepared.transform_i16(&input, &mut actual);
                assert_eq!(actual, expected, "p={p}");
                assert_eq!(expanded(&prepared, &state), expected, "p={p}");
                let mut raw = state;
                forward_ntt64(&mut raw, &params.psi, &params.twiddles, p, params.pinv);
                assert_eq!(raw, expected, "p={p}");
            }
        }
    }
}

#[test]
fn merged_admission_rejects_each_invalid_root_condition() {
    for p in PRIMES {
        let params = Parameters::new(p);
        let radix = ((1u64 << 32) % params.p as u64) as i32;
        for case in 0..5 {
            let mut psi = params.psi;
            let mut twiddles = params.twiddles;
            match case {
                0 => psi[0] = 0,
                1 => psi[1] = 0,
                2 => psi[63] = 0,
                3 => twiddles[62] = 0,
                _ => {
                    let scale = psi[0];
                    psi.fill(scale);
                    twiddles.fill(radix);
                }
            }
            assert!(
                PreparedForwardNtt64::new(&psi, &twiddles, params.p, params.pinv).is_none(),
                "case {case}"
            );
        }
    }
}

#[test]
fn merged_zero_modulus_ring_and_unaligned_output_preserve_words() {
    let zeros = [0; DEGREE];
    let prepared = PreparedForwardNtt64::new(&zeros, &zeros, 1, 1).unwrap();
    let digits = core::array::from_fn(|i| if i & 1 == 0 { i16::MIN } else { i16::MAX });
    let input = CenteredI16NttInput::new(&digits);
    #[repr(align(8))]
    struct Unaligned([i32; DEGREE + 1]);
    let mut storage = Unaligned([i32::MAX; DEGREE + 1]);
    let state: &mut [i32; DEGREE] = (&mut storage.0[1..]).try_into().unwrap();
    assert_eq!(state.as_ptr() as usize & 7, 4);
    input.write_to(state);
    assert_eq!(*state, digits.map(i32::from));
    prepared.transform_i16(&input, state);
    assert_eq!(*state, zeros);
    assert_eq!(expanded(&prepared, &digits.map(i32::from)), zeros);
}

#[test]
fn merged_range_admission_rejects_an_oversized_final_scale() {
    let p = 1_073_728_897;
    let params = Parameters::new(p);
    assert!(PreparedForwardNtt64::new(&params.psi, &params.twiddles, p, params.pinv).is_none());
    let mut actual = [i32::from(i16::MIN); DEGREE];
    let expected = params.evaluate(&actual);
    forward_ntt64(&mut actual, &params.psi, &params.twiddles, p, params.pinv);
    assert_eq!(actual, expected);
}
