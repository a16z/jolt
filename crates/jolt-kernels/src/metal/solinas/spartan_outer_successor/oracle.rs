use jolt_field::Field;

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
    use jolt_field::AkitaField;

    use super::*;

    fn field(value: u64) -> AkitaField {
        AkitaField::from_u64(value)
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
}
