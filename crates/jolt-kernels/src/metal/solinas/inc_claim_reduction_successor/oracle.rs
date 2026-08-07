//! Scalar oracle for the dense and two-scan increment reductions.

use core::ops::{Add, Mul, Sub};

const MODULUS: u64 = (1u64 << 61) - 1;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Elem(u64);

impl Elem {
    pub fn new(value: u64) -> Self {
        Self(value % MODULUS)
    }

    pub const fn zero() -> Self {
        Self(0)
    }

    pub const fn one() -> Self {
        Self(1)
    }

    pub const fn value(self) -> u64 {
        self.0
    }
}

impl Add for Elem {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let sum = self.0 + rhs.0;
        Self(if sum >= MODULUS { sum - MODULUS } else { sum })
    }
}

impl Sub for Elem {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(if self.0 >= rhs.0 {
            self.0 - rhs.0
        } else {
            MODULUS - (rhs.0 - self.0)
        })
    }
}

impl Mul for Elem {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(((self.0 as u128 * rhs.0 as u128) % MODULUS as u128) as u64)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RelationInput {
    pub ram: Vec<Elem>,
    pub rd: Vec<Elem>,
    pub points: [Vec<Elem>; 4],
    pub gamma: Elem,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TranscriptResult {
    pub messages_at_zero_and_two: Vec<[Elem; 2]>,
    pub ram_output: Elem,
    pub rd_output: Elem,
}

pub fn dense_oracle(input: &RelationInput, challenges: &[Elem]) -> TranscriptResult {
    validate(input, challenges);
    let mut ram = input.ram.clone();
    let mut rd = input.rd.clone();
    let mut equalities = input.points.each_ref().map(|point| eq_table(point));
    let powers = gamma_powers(input.gamma);
    let mut messages = Vec::with_capacity(challenges.len());
    for &challenge in challenges {
        messages.push(dense_message(&ram, &rd, &equalities, powers));
        bind_low(&mut ram, challenge);
        bind_low(&mut rd, challenge);
        for equality in &mut equalities {
            bind_low(equality, challenge);
        }
    }
    TranscriptResult {
        messages_at_zero_and_two: messages,
        ram_output: ram[0],
        rd_output: rd[0],
    }
}

pub fn split_oracle(input: &RelationInput, challenges: &[Elem]) -> TranscriptResult {
    validate(input, challenges);
    let rounds = challenges.len();
    let prefix_bits = rounds / 2;
    let suffix_bits = rounds - prefix_bits;
    let prefix_len = 1usize << prefix_bits;
    let suffix_len = 1usize << suffix_bits;
    let powers = gamma_powers(input.gamma);

    let mut low_equalities = input
        .points
        .each_ref()
        .map(|point| eq_table(&point[point.len() - prefix_bits..]));
    let high_equalities = input
        .points
        .each_ref()
        .map(|point| eq_table(&point[..suffix_bits]));
    let mut q = core::array::from_fn(|_| vec![Elem::zero(); prefix_len]);
    for lo in 0..prefix_len {
        for hi in 0..suffix_len {
            let row = hi * prefix_len + lo;
            q[0][lo] = q[0][lo] + high_equalities[0][hi] * input.ram[row];
            q[1][lo] = q[1][lo] + high_equalities[1][hi] * input.ram[row];
            q[2][lo] = q[2][lo] + high_equalities[2][hi] * input.rd[row];
            q[3][lo] = q[3][lo] + high_equalities[3][hi] * input.rd[row];
        }
    }

    let mut messages = Vec::with_capacity(rounds);
    for &challenge in &challenges[..prefix_bits] {
        messages.push(projected_message(&q, &low_equalities, powers));
        for table in &mut q {
            bind_low(table, challenge);
        }
        for table in &mut low_equalities {
            bind_low(table, challenge);
        }
    }

    let suffix_powers = core::array::from_fn(|term| powers[term] * low_equalities[term][0]);

    let low_point = challenges[..prefix_bits]
        .iter()
        .rev()
        .copied()
        .collect::<Vec<_>>();
    let low_weights = eq_table(&low_point);
    let mut ram = vec![Elem::zero(); suffix_len];
    let mut rd = vec![Elem::zero(); suffix_len];
    for hi in 0..suffix_len {
        for lo in 0..prefix_len {
            let row = hi * prefix_len + lo;
            ram[hi] = ram[hi] + low_weights[lo] * input.ram[row];
            rd[hi] = rd[hi] + low_weights[lo] * input.rd[row];
        }
    }

    let mut high_equalities = high_equalities;
    for &challenge in &challenges[prefix_bits..] {
        messages.push(dense_message(&ram, &rd, &high_equalities, suffix_powers));
        bind_low(&mut ram, challenge);
        bind_low(&mut rd, challenge);
        for equality in &mut high_equalities {
            bind_low(equality, challenge);
        }
    }
    TranscriptResult {
        messages_at_zero_and_two: messages,
        ram_output: ram[0],
        rd_output: rd[0],
    }
}

pub fn multilinear_evaluation(table: &[Elem], point: &[Elem]) -> Elem {
    assert_eq!(table.len(), 1usize << point.len());
    let weights = eq_table(point);
    table
        .iter()
        .zip(weights)
        .fold(Elem::zero(), |sum, (&value, weight)| sum + value * weight)
}

fn validate(input: &RelationInput, challenges: &[Elem]) {
    let rows = input.ram.len();
    assert!(rows >= 4 && rows.is_power_of_two());
    assert_eq!(input.rd.len(), rows);
    assert_eq!(challenges.len(), rows.ilog2() as usize);
    for point in &input.points {
        assert_eq!(point.len(), challenges.len());
    }
}

fn gamma_powers(gamma: Elem) -> [Elem; 4] {
    let gamma_squared = gamma * gamma;
    [Elem::one(), gamma, gamma_squared, gamma_squared * gamma]
}

fn eq_table(point: &[Elem]) -> Vec<Elem> {
    let mut table = vec![Elem::one()];
    for &coordinate in point {
        let mut next = Vec::with_capacity(2 * table.len());
        for &value in &table {
            next.push(value * (Elem::one() - coordinate));
            next.push(value * coordinate);
        }
        table = next;
    }
    table
}

fn bind_low(table: &mut Vec<Elem>, challenge: Elem) {
    let half = table.len() / 2;
    for index in 0..half {
        let low = table[2 * index];
        let high = table[2 * index + 1];
        table[index] = low + challenge * (high - low);
    }
    table.truncate(half);
}

fn dense_message(
    ram: &[Elem],
    rd: &[Elem],
    equalities: &[Vec<Elem>; 4],
    powers: [Elem; 4],
) -> [Elem; 2] {
    let mut message = [Elem::zero(); 2];
    for index in 0..ram.len() / 2 {
        let row = 2 * index;
        let ram_zero = ram[row];
        let rd_zero = rd[row];
        let ram_two = at_two(ram[row], ram[row + 1]);
        let rd_two = at_two(rd[row], rd[row + 1]);
        let eq_zero = equalities.each_ref().map(|table| table[row]);
        let eq_two = equalities
            .each_ref()
            .map(|table| at_two(table[row], table[row + 1]));
        message[0] = message[0]
            + ram_zero * (powers[0] * eq_zero[0] + powers[1] * eq_zero[1])
            + rd_zero * (powers[2] * eq_zero[2] + powers[3] * eq_zero[3]);
        message[1] = message[1]
            + ram_two * (powers[0] * eq_two[0] + powers[1] * eq_two[1])
            + rd_two * (powers[2] * eq_two[2] + powers[3] * eq_two[3]);
    }
    message
}

fn projected_message(
    q: &[Vec<Elem>; 4],
    equalities: &[Vec<Elem>; 4],
    powers: [Elem; 4],
) -> [Elem; 2] {
    let mut message = [Elem::zero(); 2];
    for index in 0..q[0].len() / 2 {
        let row = 2 * index;
        for term in 0..4 {
            let zero = equalities[term][row] * q[term][row];
            let two = at_two(equalities[term][row], equalities[term][row + 1])
                * at_two(q[term][row], q[term][row + 1]);
            message[0] = message[0] + powers[term] * zero;
            message[1] = message[1] + powers[term] * two;
        }
    }
    message
}

fn at_two(low: Elem, high: Elem) -> Elem {
    high + high - low
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(rounds: usize, gamma: Elem) -> (RelationInput, Vec<Elem>) {
        let rows = 1usize << rounds;
        let mut ram = vec![Elem::zero(); rows];
        let mut rd = vec![Elem::zero(); rows];
        for index in 0..rows {
            let value = Elem::new((17 * index + 9) as u64);
            if index % 3 == 0 || index % 7 == 0 {
                ram[index] = value;
            } else {
                rd[index] = value;
            }
        }
        let point = |seed: u64| {
            (0..rounds)
                .map(|index| Elem::new(seed + 13 * index as u64))
                .collect::<Vec<_>>()
        };
        let challenges = (0..rounds)
            .map(|index| Elem::new(101 + 19 * index as u64))
            .collect();
        (
            RelationInput {
                ram,
                rd,
                points: [point(3), point(5), point(7), point(11)],
                gamma,
            },
            challenges,
        )
    }

    #[test]
    fn split_matches_independent_dense_messages_at_even_and_odd_logs() {
        for rounds in [4, 5, 6, 7] {
            let (input, challenges) = fixture(rounds, Elem::new(29));
            assert_eq!(
                split_oracle(&input, &challenges),
                dense_oracle(&input, &challenges)
            );
        }
    }

    #[test]
    fn split_matches_at_gamma_zero_and_one() {
        for gamma in [Elem::zero(), Elem::one()] {
            let (input, challenges) = fixture(6, gamma);
            assert_eq!(
                split_oracle(&input, &challenges),
                dense_oracle(&input, &challenges)
            );
        }
    }

    #[test]
    fn terminal_outputs_use_the_reversed_sumcheck_point() {
        let (input, challenges) = fixture(5, Elem::new(23));
        let result = split_oracle(&input, &challenges);
        let opening_point = challenges.iter().rev().copied().collect::<Vec<_>>();
        assert_eq!(
            result.ram_output,
            multilinear_evaluation(&input.ram, &opening_point)
        );
        assert_eq!(
            result.rd_output,
            multilinear_evaluation(&input.rd, &opening_point)
        );
    }

    #[test]
    fn modular_element_arithmetic_handles_wraparound() {
        let high = Elem::new(MODULUS - 1);
        assert_eq!((high + Elem::new(2)).value(), 1);
        assert_eq!((Elem::one() - Elem::new(2)).value(), MODULUS - 1);
        assert_eq!((high * high).value(), 1);
    }
}
