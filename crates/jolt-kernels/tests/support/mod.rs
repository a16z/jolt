use jolt_kernels::metal::solinas::{Fp128, Probe, OFFSET_275};
use num_bigint::BigUint;

pub const PRODUCT5_FACTORS: usize = 5;

pub fn expected_field_for_offset(
    probe: Probe,
    lhs: Fp128,
    rhs: Fp128,
    iterations: u32,
    offset: u32,
) -> Result<Fp128, &'static str> {
    let modulus = BigUint::from(modulus(offset));
    let lhs = BigUint::from(lhs.to_u128());
    let rhs = BigUint::from(rhs.to_u128());
    let result = match probe {
        Probe::Copy => lhs,
        Probe::Add => (lhs + rhs) % &modulus,
        Probe::Sub if lhs >= rhs => lhs - rhs,
        Probe::Sub => lhs + &modulus - rhs,
        Probe::MulWide => (lhs * rhs) % &modulus,
        Probe::ChainWide1 | Probe::ChainWide2 | Probe::ChainWide4 | Probe::ChainWide8 => {
            let mut accumulator = lhs;
            for _ in 0..iterations {
                accumulator = (accumulator * &rhs) % &modulus;
            }
            accumulator
        }
        Probe::Noop | Probe::U32MadIlp8 => return Err("not a field probe"),
    };
    Ok(biguint_to_fp128(&result))
}

pub fn expected_u32_mad(lhs: Fp128, rhs: Fp128, iterations: u32) -> Fp128 {
    let mut x = lhs.limbs();
    let mut y = rhs.limbs();
    let rhs = rhs.limbs();
    let lhs = lhs.limbs();
    let x_add = [0x9e37_79b9, 0x7f4a_7c15, 0xf39c_c060, 0x106a_a070];
    let y_add = [0x94d0_49bb, 0x369d_ea0f, 0xd2b7_4407, 0xb7e1_5163];
    for _ in 0..iterations {
        for limb in 0..4 {
            x[limb] = x[limb]
                .wrapping_mul(rhs[limb] | 1)
                .wrapping_add(x_add[limb]);
            y[limb] = y[limb]
                .wrapping_mul(lhs[limb] | 1)
                .wrapping_add(y_add[limb]);
        }
    }
    Fp128::from_limbs(std::array::from_fn(|limb| x[limb] ^ y[limb]))
}

pub fn product5_message(
    tables: &[Fp128],
    elements_per_table: usize,
    e_in: &[Fp128],
    e_out: &[Fp128],
    offset: u32,
) -> [Fp128; PRODUCT5_FACTORS] {
    assert_eq!(tables.len(), PRODUCT5_FACTORS * elements_per_table);
    assert_eq!(e_in.len() * e_out.len(), elements_per_table / 2);
    let modulus = BigUint::from(modulus(offset));
    let mut total: [BigUint; PRODUCT5_FACTORS] = Default::default();

    for (x_out, outer_weight) in e_out.iter().enumerate() {
        let mut block: [BigUint; PRODUCT5_FACTORS] = Default::default();
        for (x_in, inner_weight) in e_in.iter().enumerate() {
            let pair = x_out * e_in.len() + x_in;
            let endpoints: [(BigUint, BigUint); PRODUCT5_FACTORS] = std::array::from_fn(|factor| {
                let base = factor * elements_per_table + 2 * pair;
                (
                    BigUint::from(tables[base].to_u128()),
                    BigUint::from(tables[base + 1].to_u128()),
                )
            });
            let inner_weight = BigUint::from(inner_weight.to_u128());
            for t in 1..PRODUCT5_FACTORS {
                let mut product = inner_weight.clone();
                for (lo, hi) in &endpoints {
                    let step = sub_mod(hi, lo, &modulus);
                    let eval = (lo + BigUint::from(t) * step) % &modulus;
                    product = product * eval % &modulus;
                }
                block[t - 1] = (&block[t - 1] + product) % &modulus;
            }
            let mut leading = inner_weight;
            for (lo, hi) in &endpoints {
                leading = leading * sub_mod(hi, lo, &modulus) % &modulus;
            }
            block[PRODUCT5_FACTORS - 1] = (&block[PRODUCT5_FACTORS - 1] + leading) % &modulus;
        }
        let outer_weight = BigUint::from(outer_weight.to_u128());
        for (sum, block) in total.iter_mut().zip(block) {
            *sum = (&*sum + &outer_weight * block) % &modulus;
        }
    }

    total.map(|value| biguint_to_fp128(&value))
}

pub fn product5_fused_transition(
    tables: &[Fp128],
    elements_per_table: usize,
    challenge: Fp128,
    e_in: &[Fp128],
    e_out: &[Fp128],
    offset: u32,
) -> (Vec<Fp128>, [Fp128; PRODUCT5_FACTORS]) {
    assert_eq!(tables.len(), PRODUCT5_FACTORS * elements_per_table);
    let bound_elements = elements_per_table / 2;
    assert_eq!(e_in.len() * e_out.len(), bound_elements / 2);
    let modulus = BigUint::from(modulus(offset));
    let challenge = BigUint::from(challenge.to_u128());
    let mut bound = vec![Fp128::ZERO; PRODUCT5_FACTORS * bound_elements];

    for factor in 0..PRODUCT5_FACTORS {
        for output in 0..bound_elements {
            let input = factor * elements_per_table + 2 * output;
            let lo = BigUint::from(tables[input].to_u128());
            let hi = BigUint::from(tables[input + 1].to_u128());
            let value = (&lo + &challenge * sub_mod(&hi, &lo, &modulus)) % &modulus;
            bound[factor * bound_elements + output] = biguint_to_fp128(&value);
        }
    }
    let message = product5_message(&bound, bound_elements, e_in, e_out, offset);
    (bound, message)
}

pub fn inputs(count: usize) -> (Vec<Fp128>, Vec<Fp128>) {
    inputs_for_offset(count, OFFSET_275)
}

pub fn values(count: usize) -> Vec<Fp128> {
    let modulus = modulus(OFFSET_275);
    let mut state = 0x243f_6a88_85a3_08d3_1319_8a2e_0370_7344u128;
    (0..count)
        .map(|_| Fp128::from_u128(next_value(&mut state, modulus)))
        .collect()
}

pub fn inputs_for_offset(count: usize, offset: u32) -> (Vec<Fp128>, Vec<Fp128>) {
    let modulus = modulus(offset);
    let boundary_pairs = [
        (0, 0),
        (0, modulus - 1),
        (1, modulus - 1),
        (modulus - 1, modulus - 1),
        (modulus - 1, modulus - 2),
        (modulus - 2, modulus - 1),
        (offset as u128 - 1, offset as u128),
        (u32::MAX as u128, u32::MAX as u128),
        (1u128 << 32, u64::MAX as u128),
        (u64::MAX as u128, u64::MAX as u128),
        ((1u128 << 96) - 1, (1u128 << 96) - 1),
        (1u128 << 127, modulus - 1),
    ];
    let mut state = 0x243f_6a88_85a3_08d3_1319_8a2e_0370_7344u128;
    let mut lhs = Vec::with_capacity(count);
    let mut rhs = Vec::with_capacity(count);
    for index in 0..count {
        let (lhs_value, rhs_value) = if index < boundary_pairs.len() {
            boundary_pairs[index]
        } else {
            (
                next_value(&mut state, modulus),
                next_value(&mut state, modulus),
            )
        };
        lhs.push(Fp128::from_u128(lhs_value));
        rhs.push(Fp128::from_u128(rhs_value));
    }
    (lhs, rhs)
}

fn biguint_to_fp128(value: &BigUint) -> Fp128 {
    let encoded = value.to_bytes_le();
    assert!(encoded.len() <= 16);
    let mut bytes = [0u8; 16];
    bytes[..encoded.len()].copy_from_slice(&encoded);
    Fp128::from_u128(u128::from_le_bytes(bytes))
}

fn sub_mod(lhs: &BigUint, rhs: &BigUint, modulus: &BigUint) -> BigUint {
    if lhs >= rhs {
        lhs - rhs
    } else {
        lhs + modulus - rhs
    }
}

const fn reduce_u128(value: u128, modulus: u128) -> u128 {
    if value >= modulus {
        value - modulus
    } else {
        value
    }
}

fn next_value(state: &mut u128, modulus: u128) -> u128 {
    *state ^= *state << 23;
    *state ^= *state >> 17;
    *state ^= *state << 26;
    reduce_u128(*state, modulus)
}

const fn modulus(offset: u32) -> u128 {
    u128::MAX - offset as u128 + 1
}
