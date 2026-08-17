//! Sparse RAM value-check sequence over the access/increment union topology.
//!
//! Unlike the weighted members this one carries two lanes (increment and RA),
//! pairs them with a split-LT companion table bound alongside the frontier,
//! and uses the driver's uncached walk mechanic.

use std::sync::Arc;

use jolt_field::Field;

use super::frontier::{FrontierDriver, RamCycleError, RamCycleMember};
use super::owner::RamCycleFamilyOwner;

const MEMBER: RamCycleMember = RamCycleMember::ValCheck;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[expect(
    clippy::struct_field_names,
    reason = "the field names are the protocol evaluation points"
)]
pub struct RamValMessage<F> {
    at_zero: F,
    at_two: F,
    at_three: F,
}

impl<F: Field> RamValMessage<F> {
    copy_field_getters! { pub, {
        at_zero: F,
        at_two: F,
        at_three: F,
    }}

    pub const fn sampled_evaluations(self) -> [F; 3] {
        [self.at_zero, self.at_two, self.at_three]
    }

    pub fn evaluations_with_hint(self, previous_claim: F) -> [F; 4] {
        [
            self.at_zero,
            previous_claim - self.at_zero,
            self.at_two,
            self.at_three,
        ]
    }

    pub(crate) const fn new(at_zero: F, at_two: F, at_three: F) -> Self {
        Self {
            at_zero,
            at_two,
            at_three,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamValTerminalFactors<F> {
    ram_increment: F,
    ram_ra: F,
    lt_cycle_plus_gamma: F,
}

impl<F: Field> RamValTerminalFactors<F> {
    copy_field_getters! { pub, {
        ram_increment: F,
        ram_ra: F,
        lt_cycle_plus_gamma: F,
    }}

    pub(crate) const fn new(ram_increment: F, ram_ra: F, lt_cycle_plus_gamma: F) -> Self {
        Self {
            ram_increment,
            ram_ra,
            lt_cycle_plus_gamma,
        }
    }
}

/// Authoritative sparse sequence over the access/increment union topology.
///
/// The sequence contains no transcript logic. The caller reads a message,
/// absorbs it on the host, and supplies the resulting challenge to [`Self::bind`].
pub struct HostSparseRamValCheck<F> {
    owner: Arc<RamCycleFamilyOwner>,
    core: FrontierDriver<F>,
    lt: SplitLt<F>,
}

impl<F: Field> HostSparseRamValCheck<F> {
    pub fn new(
        owner: Arc<RamCycleFamilyOwner>,
        r_address: &[F],
        r_cycle: &[F],
        gamma: F,
    ) -> Result<Self, RamCycleError> {
        let receipt = owner.receipt();
        let rounds = receipt.log_t();
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

        let eq_address = eq_evaluations(r_address)?;
        if eq_address.len() != receipt.address_domain() {
            return Err(RamCycleError::AddressTableLength {
                member: MEMBER,
                expected: receipt.address_domain(),
                got: eq_address.len(),
            });
        }
        let (blocks, increments, ra) = seed_frontier(&owner, &eq_address)?;
        if !blocks.iter().copied().eq(owner
            .block_topology()
            .leaf_cycles()
            .iter()
            .map(|&cycle| u64::from(cycle)))
        {
            return Err(RamCycleError::UnionLeafMismatch { member: MEMBER });
        }
        let core = FrontierDriver::new(MEMBER, rounds, blocks, vec![increments, ra]);
        core.validate_frontier(owner.block_topology().census())?;

        Ok(Self {
            owner,
            core,
            lt: SplitLt::new_plus_constant(r_cycle, gamma)?,
        })
    }

    pub const fn num_rounds(&self) -> usize {
        self.core.num_rounds()
    }

    pub const fn round(&self) -> usize {
        self.core.round()
    }

    pub fn frontier_len(&self) -> usize {
        self.core.frontier_len()
    }

    pub fn message(&self) -> Result<RamValMessage<F>, RamCycleError> {
        let mut at_zero = F::zero();
        let mut at_two = F::zero();
        let mut at_three = F::zero();
        self.core
            .walk_round(self.owner.block_topology(), |parent, lows, highs| {
                let (lt_zero, lt_one) = self.lt.pair(u64_to_usize(parent)?)?;
                let (inc_zero, ra_zero) = (lows[0], lows[1]);
                let (inc_one, ra_one) = (highs[0], highs[1]);
                let inc_slope = inc_one - inc_zero;
                let ra_slope = ra_one - ra_zero;
                let lt_slope = lt_one - lt_zero;
                let inc_two = inc_one + inc_slope;
                let ra_two = ra_one + ra_slope;
                let lt_two = lt_one + lt_slope;

                at_zero += inc_zero * ra_zero * lt_zero;
                at_two += inc_two * ra_two * lt_two;
                at_three += (inc_two + inc_slope) * (ra_two + ra_slope) * (lt_two + lt_slope);
                Ok(())
            })?;
        Ok(RamValMessage::new(at_zero, at_two, at_three))
    }

    pub fn bind(&mut self, challenge: F) -> Result<(), RamCycleError> {
        let lt = &mut self.lt;
        self.core
            .bind_walk(self.owner.block_topology(), challenge, || {
                lt.bind(challenge)
            })
    }

    pub fn terminal_factors(&self) -> Result<RamValTerminalFactors<F>, RamCycleError> {
        let lanes = self
            .core
            .terminal_values(self.owner.block_topology().census())?;
        let (ram_increment, ram_ra) = match lanes {
            None => (F::zero(), F::zero()),
            Some(lanes) => (lanes[0][0], lanes[1][0]),
        };
        Ok(RamValTerminalFactors::new(
            ram_increment,
            ram_ra,
            self.lt.final_value()?,
        ))
    }
}

type SeededLanes<F> = (Vec<u64>, Vec<F>, Vec<F>);

fn seed_frontier<F: Field>(
    owner: &RamCycleFamilyOwner,
    eq_address: &[F],
) -> Result<SeededLanes<F>, RamCycleError> {
    let records = owner.access_records();
    let (increment_cycles, increments) = owner.increment_slices();
    if increment_cycles.len() != increments.len() {
        return Err(RamCycleError::IncrementPayloadLength { member: MEMBER });
    }

    let capacity = records
        .len()
        .checked_add(increments.len())
        .ok_or(RamCycleError::Overflow { member: MEMBER })?;
    let mut blocks = Vec::with_capacity(capacity);
    let mut increment_lane = Vec::with_capacity(capacity);
    let mut ra_lane = Vec::with_capacity(capacity);
    let mut access_index = 0usize;
    let mut increment_index = 0usize;
    while access_index < records.len() || increment_index < increments.len() {
        let access = records.get(access_index);
        let increment_cycle = increment_cycles.get(increment_index).copied();
        let increment_value = match increment_cycle {
            Some(_) => Some(
                increments
                    .get(increment_index)
                    .copied()
                    .ok_or(RamCycleError::IncrementPayloadLength { member: MEMBER })?,
            ),
            None => None,
        };
        let (cycle, access_at_cycle, increment_at_cycle) = match (access, increment_cycle) {
            (Some(access), Some(increment_cycle)) => {
                let access_cycle = u64::from(access.cycle());
                match access_cycle.cmp(&increment_cycle) {
                    std::cmp::Ordering::Less => {
                        access_index += 1;
                        (access_cycle, Some(access), None)
                    }
                    std::cmp::Ordering::Equal => {
                        access_index += 1;
                        increment_index += 1;
                        (access_cycle, Some(access), increment_value)
                    }
                    std::cmp::Ordering::Greater => {
                        increment_index += 1;
                        (increment_cycle, None, increment_value)
                    }
                }
            }
            (Some(access), None) => {
                access_index += 1;
                (u64::from(access.cycle()), Some(access), None)
            }
            (None, Some(increment_cycle)) => {
                increment_index += 1;
                (increment_cycle, None, increment_value)
            }
            (None, None) => break,
        };
        let ram_ra = match access_at_cycle {
            Some(access) => eq_address.get(access.address() as usize).copied().ok_or(
                RamCycleError::AccessAddressOutOfRange {
                    member: MEMBER,
                    address: access.address(),
                },
            )?,
            None => F::zero(),
        };
        blocks.push(cycle);
        increment_lane.push(increment_at_cycle.map_or(F::zero(), F::from_i128));
        ra_lane.push(ram_ra);
    }
    Ok((blocks, increment_lane, ra_lane))
}

enum SplitLt<F> {
    Split {
        lt_lo: Vec<F>,
        lt_hi: Vec<F>,
        eq_hi: Vec<F>,
    },
    Dense(Vec<F>),
}

impl<F: Field> SplitLt<F> {
    fn new_plus_constant(r_cycle: &[F], constant: F) -> Result<Self, RamCycleError> {
        let low_variables = r_cycle.len() / 2;
        let high_variables = r_cycle
            .len()
            .checked_sub(low_variables)
            .ok_or(RamCycleError::Overflow { member: MEMBER })?;
        let r_hi = r_cycle
            .get(..high_variables)
            .ok_or(RamCycleError::InvalidLtState { member: MEMBER })?;
        let r_lo = r_cycle
            .get(high_variables..)
            .ok_or(RamCycleError::InvalidLtState { member: MEMBER })?;
        if r_lo.is_empty() {
            let dense = lt_evaluations(r_hi)?
                .into_iter()
                .map(|value| value + constant)
                .collect();
            return Ok(Self::Dense(dense));
        }
        Ok(Self::Split {
            lt_lo: lt_evaluations(r_lo)?,
            lt_hi: lt_evaluations(r_hi)?
                .into_iter()
                .map(|value| value + constant)
                .collect(),
            eq_hi: eq_evaluations(r_hi)?,
        })
    }

    fn pair(&self, parent_block: usize) -> Result<(F, F), RamCycleError> {
        let first = parent_block
            .checked_mul(2)
            .ok_or(RamCycleError::Overflow { member: MEMBER })?;
        let second = first
            .checked_add(1)
            .ok_or(RamCycleError::Overflow { member: MEMBER })?;
        match self {
            Self::Split {
                lt_lo,
                lt_hi,
                eq_hi,
            } => {
                if lt_lo.len() < 2 || lt_hi.len() != eq_hi.len() {
                    return Err(RamCycleError::InvalidLtState { member: MEMBER });
                }
                let high_index = first / lt_lo.len();
                let low_index = first % lt_lo.len();
                let base =
                    lt_hi
                        .get(high_index)
                        .copied()
                        .ok_or(RamCycleError::LtIndexOutOfRange {
                            member: MEMBER,
                            index: first,
                        })?;
                let scale =
                    eq_hi
                        .get(high_index)
                        .copied()
                        .ok_or(RamCycleError::LtIndexOutOfRange {
                            member: MEMBER,
                            index: first,
                        })?;
                let low =
                    lt_lo
                        .get(low_index)
                        .copied()
                        .ok_or(RamCycleError::LtIndexOutOfRange {
                            member: MEMBER,
                            index: first,
                        })?;
                let high = lt_lo.get(second % lt_lo.len()).copied().ok_or(
                    RamCycleError::LtIndexOutOfRange {
                        member: MEMBER,
                        index: second,
                    },
                )?;
                Ok((base + scale * low, base + scale * high))
            }
            Self::Dense(table) => Ok((
                table
                    .get(first)
                    .copied()
                    .ok_or(RamCycleError::LtIndexOutOfRange {
                        member: MEMBER,
                        index: first,
                    })?,
                table
                    .get(second)
                    .copied()
                    .ok_or(RamCycleError::LtIndexOutOfRange {
                        member: MEMBER,
                        index: second,
                    })?,
            )),
        }
    }

    fn bind(&mut self, challenge: F) -> Result<(), RamCycleError> {
        match self {
            Self::Split {
                lt_lo,
                lt_hi,
                eq_hi,
            } => {
                let bound = bind_adjacent(lt_lo, challenge)?;
                if bound.len() == 1 {
                    if lt_hi.len() != eq_hi.len() {
                        return Err(RamCycleError::InvalidLtState { member: MEMBER });
                    }
                    let low_scalar = bound
                        .first()
                        .copied()
                        .ok_or(RamCycleError::InvalidLtState { member: MEMBER })?;
                    let dense = lt_hi
                        .iter()
                        .copied()
                        .zip(eq_hi.iter().copied())
                        .map(|(lt, eq)| lt + eq * low_scalar)
                        .collect();
                    *self = Self::Dense(dense);
                } else {
                    *lt_lo = bound;
                }
            }
            Self::Dense(table) => {
                *table = bind_adjacent(table, challenge)?;
            }
        }
        Ok(())
    }

    fn final_value(&self) -> Result<F, RamCycleError> {
        match self {
            Self::Dense(table) if table.len() == 1 => table
                .first()
                .copied()
                .ok_or(RamCycleError::InvalidLtState { member: MEMBER }),
            Self::Dense(_) | Self::Split { .. } => {
                Err(RamCycleError::InvalidLtState { member: MEMBER })
            }
        }
    }
}

fn bind_adjacent<F: Field>(values: &[F], challenge: F) -> Result<Vec<F>, RamCycleError> {
    if values.len() < 2 || !values.len().is_multiple_of(2) {
        return Err(RamCycleError::InvalidLtState { member: MEMBER });
    }
    let mut bound = Vec::with_capacity(values.len() / 2);
    for pair in values.chunks_exact(2) {
        let low = pair
            .first()
            .copied()
            .ok_or(RamCycleError::InvalidLtState { member: MEMBER })?;
        let high = pair
            .get(1)
            .copied()
            .ok_or(RamCycleError::InvalidLtState { member: MEMBER })?;
        bound.push(low + challenge * (high - low));
    }
    Ok(bound)
}

fn eq_evaluations<F: Field>(point: &[F]) -> Result<Vec<F>, RamCycleError> {
    let expected = checked_domain_size(point.len())?;
    let mut table = Vec::with_capacity(expected);
    table.push(F::one());
    for &challenge in point {
        let next_capacity = table
            .len()
            .checked_mul(2)
            .ok_or(RamCycleError::Overflow { member: MEMBER })?;
        let mut next = Vec::with_capacity(next_capacity);
        let complement = F::one() - challenge;
        for &base in &table {
            next.push(base * complement);
            next.push(base * challenge);
        }
        table = next;
    }
    if table.len() != expected {
        return Err(RamCycleError::InvalidEqualityTable { member: MEMBER });
    }
    Ok(table)
}

fn lt_evaluations<F: Field>(point: &[F]) -> Result<Vec<F>, RamCycleError> {
    let expected = checked_domain_size(point.len())?;
    let mut table = vec![F::zero()];
    for &challenge in point.iter().rev() {
        let next_capacity = table
            .len()
            .checked_mul(2)
            .ok_or(RamCycleError::Overflow { member: MEMBER })?;
        let mut low = Vec::with_capacity(table.len());
        let mut high = Vec::with_capacity(table.len());
        for &value in &table {
            let propagated = value * challenge;
            low.push(value + challenge - propagated);
            high.push(propagated);
        }
        let mut next = Vec::with_capacity(next_capacity);
        next.extend(low);
        next.extend(high);
        table = next;
    }
    if table.len() != expected {
        return Err(RamCycleError::InvalidLtState { member: MEMBER });
    }
    Ok(table)
}

fn checked_domain_size(log_size: usize) -> Result<usize, RamCycleError> {
    let shift = u32::try_from(log_size).map_err(|_| RamCycleError::Overflow { member: MEMBER })?;
    1usize
        .checked_shl(shift)
        .ok_or(RamCycleError::Overflow { member: MEMBER })
}

fn u64_to_usize(value: u64) -> Result<usize, RamCycleError> {
    usize::try_from(value).map_err(|_| RamCycleError::Overflow { member: MEMBER })
}
