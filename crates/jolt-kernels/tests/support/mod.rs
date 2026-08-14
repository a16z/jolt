use jolt_kernels::metal::solinas::{Fp128, AKITA_OFFSET_FFFFA7F7};
use num_bigint::BigUint;

pub const PRODUCT5_FACTORS: usize = 5;

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

pub fn values(count: usize) -> Vec<Fp128> {
    let modulus = modulus(AKITA_OFFSET_FFFFA7F7);
    let mut state = 0x243f_6a88_85a3_08d3_1319_8a2e_0370_7344u128;
    (0..count)
        .map(|_| Fp128::from_u128(next_value(&mut state, modulus)))
        .collect()
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
