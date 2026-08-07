use jolt_field::{AkitaField, CanonicalBytes, Field, ReducingBytes};

pub const OUTER_COLUMNS: usize = 35;
pub const SHIFT_COLUMNS: [usize; 4] = [5, 4, 28, 33];

#[derive(Clone, Copy, Debug, Default)]
pub struct OuterAFlags {
    pub load: bool,
    pub store: bool,
    pub add: bool,
    pub sub: bool,
    pub mul: bool,
    pub jump: bool,
    pub should_branch: bool,
    pub assert_flag: bool,
    pub should_jump: bool,
    pub virtual_instruction: bool,
    pub is_last: bool,
    pub next_is_virtual: bool,
    pub next_is_first: bool,
    pub advice: bool,
    pub write_lookup: bool,
}

fn scaled<F: Field>(value: F, scalar: i64) -> F {
    let magnitude = F::from_u64(scalar.unsigned_abs());
    if scalar < 0 {
        -(value * magnitude)
    } else {
        value * magnitude
    }
}

pub fn direct_a<F: Field>(lagrange: &[F; 10], flags: OuterAFlags, second: bool) -> F {
    let bit = |value: bool| i64::from(value);
    let load = bit(flags.load);
    let store = bit(flags.store);
    let add = bit(flags.add);
    let sub = bit(flags.sub);
    let mul = bit(flags.mul);
    let jump = bit(flags.jump);
    let should_branch = bit(flags.should_branch);
    let rows = if second {
        [
            load + store,
            add,
            sub,
            mul,
            1 - add - sub - mul - bit(flags.advice),
            bit(flags.write_lookup),
            jump,
            should_branch,
            1 - should_branch - jump,
            0,
        ]
    } else {
        [
            1 - load - store,
            load,
            load,
            store,
            add + sub + mul,
            1 - add - sub - mul,
            bit(flags.assert_flag),
            bit(flags.should_jump),
            bit(flags.virtual_instruction) - bit(flags.is_last),
            bit(flags.next_is_virtual) - bit(flags.next_is_first),
        ]
    };
    rows.into_iter()
        .zip(lagrange)
        .fold(F::zero(), |sum, (row, weight)| sum + scaled(*weight, row))
}

fn add_if<F: Field>(sum: &mut F, set: bool, value: F) {
    if set {
        *sum += value;
    }
}

pub fn affine_a<F: Field>(lagrange: &[F; 10], flags: OuterAFlags, second: bool) -> F {
    if second {
        let mut sum = lagrange[4] + lagrange[8];
        add_if(&mut sum, flags.load, lagrange[0]);
        add_if(&mut sum, flags.store, lagrange[0]);
        add_if(&mut sum, flags.add, lagrange[1] - lagrange[4]);
        add_if(&mut sum, flags.sub, lagrange[2] - lagrange[4]);
        add_if(&mut sum, flags.mul, lagrange[3] - lagrange[4]);
        add_if(&mut sum, flags.advice, -lagrange[4]);
        add_if(&mut sum, flags.write_lookup, lagrange[5]);
        add_if(&mut sum, flags.jump, lagrange[6] - lagrange[8]);
        add_if(&mut sum, flags.should_branch, lagrange[7] - lagrange[8]);
        sum
    } else {
        let mut sum = lagrange[0] + lagrange[5];
        add_if(
            &mut sum,
            flags.load,
            -lagrange[0] + lagrange[1] + lagrange[2],
        );
        add_if(&mut sum, flags.store, -lagrange[0] + lagrange[3]);
        let operation = lagrange[4] - lagrange[5];
        add_if(&mut sum, flags.add, operation);
        add_if(&mut sum, flags.sub, operation);
        add_if(&mut sum, flags.mul, operation);
        add_if(&mut sum, flags.assert_flag, lagrange[6]);
        add_if(&mut sum, flags.should_jump, lagrange[7]);
        add_if(&mut sum, flags.virtual_instruction, lagrange[8]);
        add_if(&mut sum, flags.is_last, -lagrange[8]);
        add_if(&mut sum, flags.next_is_virtual, lagrange[9]);
        add_if(&mut sum, flags.next_is_first, -lagrange[9]);
        sum
    }
}

pub fn challenge_collapsed_a<F: Field>(lagrange: &[F; 10], flags: OuterAFlags, challenge: F) -> F {
    let first = affine_a(lagrange, flags, false);
    let second = affine_a(lagrange, flags, true);
    first + challenge * (second - first)
}

pub const AKITA_OFFSET: u32 = 0xffff_a7f7;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SignedMagnitude160 {
    limbs: [u32; 5],
    negative: bool,
}

impl SignedMagnitude160 {
    pub fn new(limbs: [u32; 5], negative: bool) -> Result<Self, DeferredDotError> {
        if limbs[4] > 3 {
            return Err(DeferredDotError::MagnitudeOutOfRange);
        }
        Ok(Self { limbs, negative })
    }

    pub const fn limbs(&self) -> &[u32; 5] {
        &self.limbs
    }

    pub const fn is_negative(&self) -> bool {
        self.negative
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeferredDotError {
    LengthMismatch,
    MagnitudeOutOfRange,
    AccumulatorOverflow,
    FoldOverflow,
}

pub fn deferred_signed_dot(
    weights: &[AkitaField],
    terms: &[SignedMagnitude160],
) -> Result<AkitaField, DeferredDotError> {
    if weights.len() != terms.len() {
        return Err(DeferredDotError::LengthMismatch);
    }
    let mut accumulator = [0_u32; 10];
    for (&weight, term) in weights.iter().zip(terms) {
        let mut product = multiply_160_by_field(term.limbs, weight);
        if term.negative {
            negate_limbs(&mut product);
        }
        add_signed(&mut accumulator, product)?;
    }
    reduce_signed_320(accumulator)
}

fn multiply_160_by_field(magnitude: [u32; 5], weight: AkitaField) -> [u32; 10] {
    let mut weight_bytes = [0_u8; 16];
    weight.to_bytes_le(&mut weight_bytes);
    let weight: [u32; 4] = std::array::from_fn(|index| {
        let offset = 4 * index;
        u32::from_le_bytes([
            weight_bytes[offset],
            weight_bytes[offset + 1],
            weight_bytes[offset + 2],
            weight_bytes[offset + 3],
        ])
    });
    let mut product = [0_u32; 10];
    for (i, &magnitude_limb) in magnitude.iter().enumerate() {
        let mut carry = 0_u64;
        for (j, &weight_limb) in weight.iter().enumerate() {
            let k = i + j;
            let word =
                u64::from(magnitude_limb) * u64::from(weight_limb) + u64::from(product[k]) + carry;
            product[k] = word as u32;
            carry = word >> 32;
        }
        product[i + 4] = carry as u32;
    }
    product
}

fn negate_limbs<const N: usize>(value: &mut [u32; N]) {
    let mut carry = 1_u64;
    for limb in value {
        let word = u64::from(!*limb) + carry;
        *limb = word as u32;
        carry = word >> 32;
    }
}

fn add_signed(accumulator: &mut [u32; 10], value: [u32; 10]) -> Result<(), DeferredDotError> {
    let accumulator_negative = accumulator[9] >> 31 != 0;
    let value_negative = value[9] >> 31 != 0;
    let mut carry = 0_u64;
    for (accumulator_limb, value_limb) in accumulator.iter_mut().zip(value) {
        let word = u64::from(*accumulator_limb) + u64::from(value_limb) + carry;
        *accumulator_limb = word as u32;
        carry = word >> 32;
    }
    let result_negative = accumulator[9] >> 31 != 0;
    if accumulator_negative == value_negative && result_negative != accumulator_negative {
        return Err(DeferredDotError::AccumulatorOverflow);
    }
    Ok(())
}

fn add_carry(value: &mut [u32; 8], index: usize, mut carry: u64) -> bool {
    for limb in &mut value[index..] {
        if carry == 0 {
            return true;
        }
        let word = u64::from(*limb) + carry;
        *limb = word as u32;
        carry = word >> 32;
    }
    carry == 0
}

fn reduce_signed_320(mut value: [u32; 10]) -> Result<AkitaField, DeferredDotError> {
    let negative = value[9] >> 31 != 0;
    if negative {
        negate_limbs(&mut value);
    }

    let mut folded = [0_u32; 8];
    folded[..4].copy_from_slice(&value[..4]);
    let mut carry = 0_u64;
    for i in 0..4 {
        let word = u64::from(value[i + 4]) * u64::from(AKITA_OFFSET) + u64::from(folded[i]) + carry;
        folded[i] = word as u32;
        carry = word >> 32;
    }
    if !add_carry(&mut folded, 4, carry) {
        return Err(DeferredDotError::FoldOverflow);
    }

    let offset_squared = u64::from(AKITA_OFFSET) * u64::from(AKITA_OFFSET);
    let factor = [offset_squared as u32, (offset_squared >> 32) as u32];
    for i in 0..2 {
        carry = 0;
        for (j, &factor_limb) in factor.iter().enumerate() {
            let k = i + j;
            let word =
                u64::from(value[i + 8]) * u64::from(factor_limb) + u64::from(folded[k]) + carry;
            folded[k] = word as u32;
            carry = word >> 32;
        }
        if !add_carry(&mut folded, i + 2, carry) {
            return Err(DeferredDotError::FoldOverflow);
        }
    }

    let mut bytes = [0_u8; 32];
    for (index, limb) in folded.into_iter().enumerate() {
        bytes[4 * index..4 * index + 4].copy_from_slice(&limb.to_le_bytes());
    }
    let reduced = AkitaField::from_le_bytes_mod_order(&bytes);
    Ok(if negative { -reduced } else { reduced })
}

#[derive(Clone, Debug, PartialEq)]
pub struct OpeningCarrier<F: Field> {
    pub current: Vec<Vec<F>>,
    pub successor: [Vec<F>; SHIFT_COLUMNS.len()],
}

pub fn opening_carrier<F: Field>(
    rows: &[[F; OUTER_COLUMNS]],
    high_weights: &[F],
    low: usize,
) -> OpeningCarrier<F> {
    assert_eq!(rows.len(), high_weights.len() * low);
    let mut current = vec![vec![F::zero(); low]; OUTER_COLUMNS];
    let mut successor = std::array::from_fn(|_| vec![F::zero(); low]);
    for x_low in 0..low {
        for (x_high, &weight) in high_weights.iter().enumerate() {
            let row = &rows[x_high * low + x_low];
            for (column, partials) in current.iter_mut().enumerate() {
                partials[x_low] += weight * row[column];
            }
            if x_high != 0 {
                let predecessor_weight = high_weights[x_high - 1];
                for (slot, &column) in SHIFT_COLUMNS.iter().enumerate() {
                    successor[slot][x_low] += predecessor_weight * row[column];
                }
            }
        }
    }
    OpeningCarrier { current, successor }
}

pub fn reduce_openings<F: Field>(carrier: &OpeningCarrier<F>, low_weights: &[F]) -> Vec<F> {
    carrier
        .current
        .iter()
        .map(|partials| {
            partials
                .iter()
                .zip(low_weights)
                .fold(F::zero(), |sum, (&value, &weight)| sum + value * weight)
        })
        .collect()
}

pub fn direct_openings<F: Field>(
    rows: &[[F; OUTER_COLUMNS]],
    high_weights: &[F],
    low_weights: &[F],
) -> Vec<F> {
    let low = low_weights.len();
    assert_eq!(rows.len(), high_weights.len() * low);
    let mut result = vec![F::zero(); OUTER_COLUMNS];
    for (x_high, &high_weight) in high_weights.iter().enumerate() {
        for (x_low, &low_weight) in low_weights.iter().enumerate() {
            let weight = high_weight * low_weight;
            let row = &rows[x_high * low + x_low];
            for (output, &value) in result.iter_mut().zip(row) {
                *output += weight * value;
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use jolt_field::{AkitaField, FromPrimitiveInt, ReducingBytes};

    use super::*;

    fn field(value: u64) -> AkitaField {
        AkitaField::from_u64(value)
    }

    fn direct_signed_dot(weights: &[AkitaField], terms: &[SignedMagnitude160]) -> AkitaField {
        weights
            .iter()
            .zip(terms)
            .fold(AkitaField::zero(), |sum, (&weight, term)| {
                let mut bytes = [0_u8; 20];
                for (index, limb) in term.limbs().iter().enumerate() {
                    bytes[4 * index..4 * index + 4].copy_from_slice(&limb.to_le_bytes());
                }
                let magnitude = AkitaField::from_le_bytes_mod_order(&bytes);
                let product = weight * magnitude;
                sum + if term.is_negative() {
                    -product
                } else {
                    product
                }
            })
    }

    #[test]
    fn affine_a_matches_the_row_definition() {
        let lagrange = std::array::from_fn(|index| field(3 + 7 * index as u64));
        for mask in 0..256u64 {
            let flag = |bit: u32| mask & (1u64 << bit) != 0;
            let flags = OuterAFlags {
                load: flag(0),
                store: flag(1),
                add: flag(2),
                sub: flag(3),
                mul: flag(4),
                jump: flag(5),
                should_branch: flag(6),
                advice: flag(7),
                assert_flag: flag(1),
                should_jump: flag(2),
                virtual_instruction: flag(3),
                is_last: flag(4),
                next_is_virtual: flag(5),
                next_is_first: flag(7),
                write_lookup: flag(0),
            };
            assert_eq!(
                direct_a(&lagrange, flags, false),
                affine_a(&lagrange, flags, false)
            );
            assert_eq!(
                direct_a(&lagrange, flags, true),
                affine_a(&lagrange, flags, true)
            );
            let challenge = field(41);
            let first = direct_a(&lagrange, flags, false);
            let second = direct_a(&lagrange, flags, true);
            assert_eq!(
                challenge_collapsed_a(&lagrange, flags, challenge),
                first + challenge * (second - first)
            );
        }
    }

    #[test]
    fn partial_carrier_matches_direct_openings_and_successor_definition() {
        let high = 4;
        let low = 8;
        let rows = (0..high * low)
            .map(|row| std::array::from_fn(|column| field((11 * row + 5 * column + 1) as u64)))
            .collect::<Vec<_>>();
        let high_weights = (0..high)
            .map(|index| field(13 + index as u64))
            .collect::<Vec<_>>();
        let low_weights = (0..low)
            .map(|index| field(29 + index as u64))
            .collect::<Vec<_>>();
        let carrier = opening_carrier(&rows, &high_weights, low);
        assert_eq!(
            reduce_openings(&carrier, &low_weights),
            direct_openings(&rows, &high_weights, &low_weights)
        );
        for (slot, &column) in SHIFT_COLUMNS.iter().enumerate() {
            for x_low in 0..low {
                let expected = (1..high).fold(AkitaField::zero(), |sum, x_high| {
                    sum + high_weights[x_high - 1] * rows[x_high * low + x_low][column]
                });
                assert_eq!(carrier.successor[slot][x_low], expected);
            }
        }
    }

    #[test]
    fn deferred_signed_dot_matches_extrema_and_cancellation() {
        let modulus = u128::MAX - u128::from(AKITA_OFFSET) + 1;
        let weights = [
            AkitaField::from_u128(modulus - 1),
            AkitaField::from_u128(modulus - 2),
            AkitaField::from_u128((1_u128 << 127) + 17),
            field(0),
            field(1),
            field(41),
            AkitaField::from_u128(modulus - 33),
            field(7),
            field(11),
            field(13),
        ];
        let maximum = [u32::MAX, u32::MAX, u32::MAX, u32::MAX, 3];
        let terms = [
            SignedMagnitude160::new(maximum, false).unwrap(),
            SignedMagnitude160::new(maximum, true).unwrap(),
            SignedMagnitude160::new([0, 0, 0, 0x8000_0000, 0], false).unwrap(),
            SignedMagnitude160::new(maximum, false).unwrap(),
            SignedMagnitude160::new([1, 0, 0, 0, 0], true).unwrap(),
            SignedMagnitude160::new([u32::MAX, 17, 0, 0, 0], false).unwrap(),
            SignedMagnitude160::new([u32::MAX, 17, 0, 0, 0], true).unwrap(),
            SignedMagnitude160::new([5, 4, 3, 2, 1], false).unwrap(),
            SignedMagnitude160::new([5, 4, 3, 2, 1], true).unwrap(),
            SignedMagnitude160::new([0; 5], true).unwrap(),
        ];
        assert_eq!(
            deferred_signed_dot(&weights, &terms).unwrap(),
            direct_signed_dot(&weights, &terms)
        );
        let same_sign_terms = [SignedMagnitude160::new(maximum, false).unwrap(); 10];
        assert_eq!(
            deferred_signed_dot(&weights, &same_sign_terms).unwrap(),
            direct_signed_dot(&weights, &same_sign_terms)
        );
    }

    #[test]
    fn deferred_signed_dot_matches_deterministic_mixed_inputs() {
        let modulus = u128::MAX - u128::from(AKITA_OFFSET) + 1;
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        for case in 0..256 {
            let mut next = || {
                state ^= state << 7;
                state ^= state >> 9;
                state ^= state << 8;
                state
            };
            let weights: [AkitaField; 10] = std::array::from_fn(|_| {
                let value = (u128::from(next()) << 64) | u128::from(next());
                AkitaField::from_u128(if value >= modulus {
                    value - modulus
                } else {
                    value
                })
            });
            let terms: [SignedMagnitude160; 10] = std::array::from_fn(|index| {
                let limbs = [
                    next() as u32,
                    next() as u32,
                    next() as u32,
                    next() as u32,
                    (next() & 3) as u32,
                ];
                SignedMagnitude160::new(limbs, (case + index) % 3 == 0).unwrap()
            });
            assert_eq!(
                deferred_signed_dot(&weights, &terms).unwrap(),
                direct_signed_dot(&weights, &terms),
                "case {case}"
            );
        }
    }

    #[test]
    fn deferred_signed_dot_rejects_a_row_outside_the_proven_bound() {
        assert_eq!(
            SignedMagnitude160::new([0, 0, 0, 0, 4], false),
            Err(DeferredDotError::MagnitudeOutOfRange)
        );
    }
}
