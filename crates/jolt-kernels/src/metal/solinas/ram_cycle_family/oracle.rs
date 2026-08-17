use jolt_field::Field;

use super::frontier::{RamCycleError, RamCycleMember};
use super::owner::RamCycleFamilyOwner;
use super::ram_val_check::{RamValMessage, RamValTerminalFactors};

const MEMBER: RamCycleMember = RamCycleMember::ValCheck;

/// Independent dense oracle for the sparse RAM value-check sequence.
///
/// This oracle materializes all three `T`-sized columns. It does not read the
/// union topology or use the split-LT representation.
pub struct DenseRamValCheckOracle<F> {
    ram_increment: Vec<F>,
    ram_ra: Vec<F>,
    lt_cycle_plus_gamma: Vec<F>,
    round: usize,
    rounds: usize,
}

impl<F: Field> DenseRamValCheckOracle<F> {
    pub fn new(
        owner: &RamCycleFamilyOwner,
        r_address: &[F],
        r_cycle: &[F],
        gamma: F,
    ) -> Result<Self, RamCycleError> {
        let receipt = owner.receipt();
        if r_address.len() != receipt.log_k() {
            return Err(RamCycleError::AddressPointLength {
                member: MEMBER,
                expected: receipt.log_k(),
                got: r_address.len(),
            });
        }
        if r_cycle.len() != receipt.log_t() {
            return Err(RamCycleError::CyclePointLength {
                member: MEMBER,
                expected: receipt.log_t(),
                got: r_cycle.len(),
            });
        }

        let eq_address = dense_eq_evaluations(r_address)?;
        if eq_address.len() != receipt.address_domain() {
            return Err(RamCycleError::AddressTableLength {
                member: MEMBER,
                expected: receipt.address_domain(),
                got: eq_address.len(),
            });
        }
        let mut ram_increment = vec![F::zero(); receipt.cycles()];
        let mut ram_ra = vec![F::zero(); receipt.cycles()];
        for record in owner.access_records() {
            let address_weight = eq_address.get(record.address() as usize).copied().ok_or(
                RamCycleError::AccessAddressOutOfRange {
                    member: MEMBER,
                    address: record.address(),
                },
            )?;
            let cycle = record.cycle() as usize;
            let destination = ram_ra
                .get_mut(cycle)
                .ok_or(RamCycleError::DenseCycleOutOfRange {
                    member: MEMBER,
                    cycle: u64::from(record.cycle()),
                })?;
            *destination = address_weight;
        }
        for increment in owner.increment_records() {
            let destination = ram_increment
                .get_mut(
                    usize::try_from(increment.cycle())
                        .map_err(|_| RamCycleError::Overflow { member: MEMBER })?,
                )
                .ok_or(RamCycleError::DenseCycleOutOfRange {
                    member: MEMBER,
                    cycle: increment.cycle(),
                })?;
            *destination = F::from_i128(increment.increment());
        }
        let lt_cycle_plus_gamma = dense_lt_evaluations(r_cycle)?
            .into_iter()
            .map(|value| value + gamma)
            .collect::<Vec<_>>();
        if lt_cycle_plus_gamma.len() != receipt.cycles() {
            return Err(RamCycleError::DenseTableLength {
                member: MEMBER,
                expected: receipt.cycles(),
                got: lt_cycle_plus_gamma.len(),
            });
        }

        Ok(Self {
            ram_increment,
            ram_ra,
            lt_cycle_plus_gamma,
            round: 0,
            rounds: receipt.log_t(),
        })
    }

    pub const fn num_rounds(&self) -> usize {
        self.rounds
    }

    pub const fn round(&self) -> usize {
        self.round
    }

    pub fn current_len(&self) -> usize {
        self.ram_increment.len()
    }

    pub fn message(&self) -> Result<RamValMessage<F>, RamCycleError> {
        if self.round >= self.rounds {
            return Err(RamCycleError::AlreadyFullyBound { member: MEMBER });
        }
        self.validate_lengths()?;
        if self.ram_increment.len() < 2 || !self.ram_increment.len().is_multiple_of(2) {
            return Err(RamCycleError::InvalidDenseState { member: MEMBER });
        }

        let mut at_zero = F::zero();
        let mut at_two = F::zero();
        let mut at_three = F::zero();
        let increment_pairs = self.ram_increment.chunks_exact(2);
        let ra_pairs = self.ram_ra.chunks_exact(2);
        let lt_pairs = self.lt_cycle_plus_gamma.chunks_exact(2);
        for ((increment, ra), lt) in increment_pairs.zip(ra_pairs).zip(lt_pairs) {
            let inc_zero = pair_entry(increment, 0)?;
            let inc_one = pair_entry(increment, 1)?;
            let ra_zero = pair_entry(ra, 0)?;
            let ra_one = pair_entry(ra, 1)?;
            let lt_zero = pair_entry(lt, 0)?;
            let lt_one = pair_entry(lt, 1)?;
            let inc_slope = inc_one - inc_zero;
            let ra_slope = ra_one - ra_zero;
            let lt_slope = lt_one - lt_zero;
            let inc_two = inc_one + inc_slope;
            let ra_two = ra_one + ra_slope;
            let lt_two = lt_one + lt_slope;

            at_zero += inc_zero * ra_zero * lt_zero;
            at_two += inc_two * ra_two * lt_two;
            at_three += (inc_two + inc_slope) * (ra_two + ra_slope) * (lt_two + lt_slope);
        }
        Ok(RamValMessage::new(at_zero, at_two, at_three))
    }

    pub fn bind(&mut self, challenge: F) -> Result<(), RamCycleError> {
        if self.round >= self.rounds {
            return Err(RamCycleError::AlreadyFullyBound { member: MEMBER });
        }
        self.validate_lengths()?;
        self.ram_increment = bind_dense(&self.ram_increment, challenge)?;
        self.ram_ra = bind_dense(&self.ram_ra, challenge)?;
        self.lt_cycle_plus_gamma = bind_dense(&self.lt_cycle_plus_gamma, challenge)?;
        self.round = self
            .round
            .checked_add(1)
            .ok_or(RamCycleError::Overflow { member: MEMBER })?;
        self.validate_lengths()?;
        Ok(())
    }

    pub fn terminal_factors(&self) -> Result<RamValTerminalFactors<F>, RamCycleError> {
        if self.round != self.rounds {
            return Err(RamCycleError::NotFullyBound {
                member: MEMBER,
                remaining: self.rounds - self.round,
            });
        }
        self.validate_lengths()?;
        Ok(RamValTerminalFactors::new(
            only_value(&self.ram_increment)?,
            only_value(&self.ram_ra)?,
            only_value(&self.lt_cycle_plus_gamma)?,
        ))
    }

    fn validate_lengths(&self) -> Result<(), RamCycleError> {
        let expected = 1usize
            .checked_shl(
                u32::try_from(self.rounds - self.round)
                    .map_err(|_| RamCycleError::Overflow { member: MEMBER })?,
            )
            .ok_or(RamCycleError::Overflow { member: MEMBER })?;
        if self.ram_increment.len() != expected
            || self.ram_ra.len() != expected
            || self.lt_cycle_plus_gamma.len() != expected
        {
            return Err(RamCycleError::DenseTableLength {
                member: MEMBER,
                expected,
                got: self.ram_increment.len(),
            });
        }
        Ok(())
    }
}

fn bind_dense<F: Field>(values: &[F], challenge: F) -> Result<Vec<F>, RamCycleError> {
    if values.len() < 2 || !values.len().is_multiple_of(2) {
        return Err(RamCycleError::InvalidDenseState { member: MEMBER });
    }
    let mut output = Vec::with_capacity(values.len() / 2);
    for pair in values.chunks_exact(2) {
        let low = pair_entry(pair, 0)?;
        let high = pair_entry(pair, 1)?;
        output.push(low + challenge * (high - low));
    }
    Ok(output)
}

fn pair_entry<F: Copy>(pair: &[F], index: usize) -> Result<F, RamCycleError> {
    pair.get(index)
        .copied()
        .ok_or(RamCycleError::InvalidDenseState { member: MEMBER })
}

fn only_value<F: Copy>(values: &[F]) -> Result<F, RamCycleError> {
    match values {
        [value] => Ok(*value),
        _ => Err(RamCycleError::InvalidDenseState { member: MEMBER }),
    }
}

fn dense_eq_evaluations<F: Field>(point: &[F]) -> Result<Vec<F>, RamCycleError> {
    let expected = dense_domain_size(point.len())?;
    let mut table = vec![F::one()];
    for &challenge in point {
        let mut next = Vec::with_capacity(
            table
                .len()
                .checked_mul(2)
                .ok_or(RamCycleError::Overflow { member: MEMBER })?,
        );
        for &base in &table {
            next.push(base * (F::one() - challenge));
            next.push(base * challenge);
        }
        table = next;
    }
    if table.len() != expected {
        return Err(RamCycleError::InvalidEqualityTable { member: MEMBER });
    }
    Ok(table)
}

fn dense_lt_evaluations<F: Field>(point: &[F]) -> Result<Vec<F>, RamCycleError> {
    let expected = dense_domain_size(point.len())?;
    let mut output = Vec::with_capacity(expected);
    for index in 0..expected {
        let mut less_than = F::zero();
        let mut equal_prefix = F::one();
        for (position, &challenge) in point.iter().enumerate() {
            let shift = point
                .len()
                .checked_sub(position)
                .and_then(|remaining| remaining.checked_sub(1))
                .ok_or(RamCycleError::Overflow { member: MEMBER })?;
            let bit = index
                .checked_shr(
                    u32::try_from(shift).map_err(|_| RamCycleError::Overflow { member: MEMBER })?,
                )
                .ok_or(RamCycleError::Overflow { member: MEMBER })?
                & 1;
            let bit_field = F::from_u64(bit as u64);
            less_than += (F::one() - bit_field) * challenge * equal_prefix;
            equal_prefix *= bit_field * challenge + (F::one() - bit_field) * (F::one() - challenge);
        }
        output.push(less_than);
    }
    Ok(output)
}

fn dense_domain_size(log_size: usize) -> Result<usize, RamCycleError> {
    let shift = u32::try_from(log_size).map_err(|_| RamCycleError::Overflow { member: MEMBER })?;
    1usize
        .checked_shl(shift)
        .ok_or(RamCycleError::Overflow { member: MEMBER })
}
