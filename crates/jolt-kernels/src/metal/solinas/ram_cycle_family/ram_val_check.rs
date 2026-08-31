use std::sync::Arc;

use jolt_field::Field;
use thiserror::Error;

use super::owner::RamCycleFamilyOwner;
use super::topology::{BlockMerge, TopologyError};

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

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamValFrontierEntry<F> {
    block: u64,
    ram_increment: F,
    ram_ra: F,
}

impl<F: Field> RamValFrontierEntry<F> {
    copy_field_getters! { pub, {
        block: u64,
        ram_increment: F,
        ram_ra: F,
    }}
}

/// Authoritative sparse sequence over the access/increment union topology.
///
/// The sequence contains no transcript logic. The caller reads a message,
/// absorbs it on the host, and supplies the resulting challenge to [`Self::bind`].
pub struct HostSparseRamValCheck<F> {
    owner: Arc<RamCycleFamilyOwner>,
    frontier: Vec<RamValFrontierEntry<F>>,
    lt: SplitLt<F>,
    round: usize,
    rounds: usize,
}

impl<F: Field> HostSparseRamValCheck<F> {
    pub fn new(
        owner: Arc<RamCycleFamilyOwner>,
        r_address: &[F],
        r_cycle: &[F],
        gamma: F,
    ) -> Result<Self, RamValError> {
        let receipt = owner.receipt();
        let rounds = receipt.log_t();
        if r_address.len() != receipt.log_k() {
            return Err(RamValError::AddressPointLength {
                expected: receipt.log_k(),
                got: r_address.len(),
            });
        }
        if r_cycle.len() != receipt.log_t() {
            return Err(RamValError::CyclePointLength {
                expected: receipt.log_t(),
                got: r_cycle.len(),
            });
        }

        let eq_address = eq_evaluations(r_address)?;
        if eq_address.len() != receipt.address_domain() {
            return Err(RamValError::AddressTableLength {
                expected: receipt.address_domain(),
                got: eq_address.len(),
            });
        }
        let frontier = seed_frontier(&owner, &eq_address)?;
        if !frontier.iter().map(|value| value.block).eq(owner
            .block_topology()
            .leaf_cycles()
            .iter()
            .map(|&cycle| u64::from(cycle)))
        {
            return Err(RamValError::UnionLeafMismatch);
        }
        let expected = owner
            .block_topology()
            .census()
            .first()
            .ok_or(RamValError::MissingTopologyLevel)?
            .entries();
        if usize_to_u64(frontier.len())? != expected {
            return Err(RamValError::FrontierLength {
                round: 0,
                expected,
                got: usize_to_u64(frontier.len())?,
            });
        }

        Ok(Self {
            owner,
            frontier,
            lt: SplitLt::new_plus_constant(r_cycle, gamma)?,
            round: 0,
            rounds,
        })
    }

    copy_field_getters! { pub, {
        num_rounds => rounds: usize,
        round: usize,
    }}

    pub fn frontier_len(&self) -> usize {
        self.frontier.len()
    }

    pub fn frontier(&self) -> &[RamValFrontierEntry<F>] {
        &self.frontier
    }

    pub fn message(&self) -> Result<RamValMessage<F>, RamValError> {
        if self.round >= self.rounds {
            return Err(RamValError::AlreadyFullyBound);
        }
        self.validate_frontier_length()?;
        let merges = self.owner.block_topology().merges_for_round(self.round)?;
        let mut at_zero = F::zero();
        let mut at_two = F::zero();
        let mut at_three = F::zero();

        for merge in merges {
            let (low, high, parent_block) = merge_children(&self.frontier, *merge, self.round)?;
            let (lt_zero, lt_one) = self.lt.pair(parent_block)?;
            let (inc_zero, ra_zero) = low.map_or((F::zero(), F::zero()), |value| {
                (value.ram_increment, value.ram_ra)
            });
            let (inc_one, ra_one) = high.map_or((F::zero(), F::zero()), |value| {
                (value.ram_increment, value.ram_ra)
            });
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

    pub fn bind(&mut self, challenge: F) -> Result<(), RamValError> {
        if self.round >= self.rounds {
            return Err(RamValError::AlreadyFullyBound);
        }
        self.validate_frontier_length()?;
        let merges = self.owner.block_topology().merges_for_round(self.round)?;
        let mut next = Vec::with_capacity(merges.len());
        let mut previous_parent = None;

        for merge in merges {
            let (low, high, parent_block) = merge_children(&self.frontier, *merge, self.round)?;
            if previous_parent.is_some_and(|previous| previous >= parent_block) {
                return Err(RamValError::UnorderedParentBlocks { round: self.round });
            }
            previous_parent = Some(parent_block);
            let (inc_zero, ra_zero) = low.map_or((F::zero(), F::zero()), |value| {
                (value.ram_increment, value.ram_ra)
            });
            let (inc_one, ra_one) = high.map_or((F::zero(), F::zero()), |value| {
                (value.ram_increment, value.ram_ra)
            });
            next.push(RamValFrontierEntry {
                block: usize_to_u64(parent_block)?,
                ram_increment: inc_zero + challenge * (inc_one - inc_zero),
                ram_ra: ra_zero + challenge * (ra_one - ra_zero),
            });
        }

        self.lt.bind(challenge)?;
        self.frontier = next;
        self.round = self.round.checked_add(1).ok_or(RamValError::Overflow)?;
        self.validate_frontier_length()?;
        Ok(())
    }

    pub fn terminal_factors(&self) -> Result<RamValTerminalFactors<F>, RamValError> {
        if self.round != self.rounds {
            return Err(RamValError::NotFullyBound {
                remaining: self.rounds - self.round,
            });
        }
        self.validate_frontier_length()?;
        let (ram_increment, ram_ra) = match self.frontier.as_slice() {
            [] => (F::zero(), F::zero()),
            [value] if value.block == 0 => (value.ram_increment, value.ram_ra),
            _ => return Err(RamValError::InvalidTerminalFrontier),
        };
        Ok(RamValTerminalFactors::new(
            ram_increment,
            ram_ra,
            self.lt.final_value()?,
        ))
    }

    fn validate_frontier_length(&self) -> Result<(), RamValError> {
        let expected = self
            .owner
            .block_topology()
            .census()
            .get(self.round)
            .ok_or(RamValError::MissingTopologyLevel)?
            .entries();
        let got = usize_to_u64(self.frontier.len())?;
        if got != expected {
            return Err(RamValError::FrontierLength {
                round: self.round,
                expected,
                got,
            });
        }
        Ok(())
    }
}

fn seed_frontier<F: Field>(
    owner: &RamCycleFamilyOwner,
    eq_address: &[F],
) -> Result<Vec<RamValFrontierEntry<F>>, RamValError> {
    let records = owner.access_records();
    let (increment_cycles, increments) = owner.increment_slices();
    if increment_cycles.len() != increments.len() {
        return Err(RamValError::IncrementPayloadLength);
    }

    let mut frontier = Vec::with_capacity(
        records
            .len()
            .checked_add(increments.len())
            .ok_or(RamValError::Overflow)?,
    );
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
                    .ok_or(RamValError::IncrementPayloadLength)?,
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
                RamValError::AccessAddressOutOfRange {
                    address: access.address(),
                },
            )?,
            None => F::zero(),
        };
        let ram_increment = increment_at_cycle.map_or(F::zero(), F::from_i128);
        frontier.push(RamValFrontierEntry {
            block: cycle,
            ram_increment,
            ram_ra,
        });
    }
    Ok(frontier)
}

type RamValChildren<'a, F> = (
    Option<&'a RamValFrontierEntry<F>>,
    Option<&'a RamValFrontierEntry<F>>,
    usize,
);

fn merge_children<F>(
    frontier: &[RamValFrontierEntry<F>],
    merge: BlockMerge,
    round: usize,
) -> Result<RamValChildren<'_, F>, RamValError> {
    let low = match merge.low_state() {
        Some(index) => Some(
            frontier
                .get(index)
                .ok_or(RamValError::InvalidFrontierIndex { round, index })?,
        ),
        None => None,
    };
    let high = match merge.high_state() {
        Some(index) => Some(
            frontier
                .get(index)
                .ok_or(RamValError::InvalidFrontierIndex { round, index })?,
        ),
        None => None,
    };
    let parent = match (low, high) {
        (Some(low), Some(high)) => {
            if low.block & 1 != 0 || high.block & 1 != 1 || low.block >> 1 != high.block >> 1 {
                return Err(RamValError::InvalidMergeChildren { round });
            }
            low.block >> 1
        }
        (Some(low), None) => {
            if low.block & 1 != 0 {
                return Err(RamValError::InvalidMergeChildren { round });
            }
            low.block >> 1
        }
        (None, Some(high)) => {
            if high.block & 1 != 1 {
                return Err(RamValError::InvalidMergeChildren { round });
            }
            high.block >> 1
        }
        (None, None) => return Err(RamValError::EmptyMerge { round }),
    };
    Ok((low, high, u64_to_usize(parent)?))
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
    fn new_plus_constant(r_cycle: &[F], constant: F) -> Result<Self, RamValError> {
        let low_variables = r_cycle.len() / 2;
        let high_variables = r_cycle
            .len()
            .checked_sub(low_variables)
            .ok_or(RamValError::Overflow)?;
        let r_hi = r_cycle
            .get(..high_variables)
            .ok_or(RamValError::InvalidLtState)?;
        let r_lo = r_cycle
            .get(high_variables..)
            .ok_or(RamValError::InvalidLtState)?;
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

    fn pair(&self, parent_block: usize) -> Result<(F, F), RamValError> {
        let first = parent_block.checked_mul(2).ok_or(RamValError::Overflow)?;
        let second = first.checked_add(1).ok_or(RamValError::Overflow)?;
        match self {
            Self::Split {
                lt_lo,
                lt_hi,
                eq_hi,
            } => {
                if lt_lo.len() < 2 || lt_hi.len() != eq_hi.len() {
                    return Err(RamValError::InvalidLtState);
                }
                let high_index = first / lt_lo.len();
                let low_index = first % lt_lo.len();
                let base = lt_hi
                    .get(high_index)
                    .copied()
                    .ok_or(RamValError::LtIndexOutOfRange { index: first })?;
                let scale = eq_hi
                    .get(high_index)
                    .copied()
                    .ok_or(RamValError::LtIndexOutOfRange { index: first })?;
                let low = lt_lo
                    .get(low_index)
                    .copied()
                    .ok_or(RamValError::LtIndexOutOfRange { index: first })?;
                let high = lt_lo
                    .get(second % lt_lo.len())
                    .copied()
                    .ok_or(RamValError::LtIndexOutOfRange { index: second })?;
                Ok((base + scale * low, base + scale * high))
            }
            Self::Dense(table) => Ok((
                table
                    .get(first)
                    .copied()
                    .ok_or(RamValError::LtIndexOutOfRange { index: first })?,
                table
                    .get(second)
                    .copied()
                    .ok_or(RamValError::LtIndexOutOfRange { index: second })?,
            )),
        }
    }

    fn bind(&mut self, challenge: F) -> Result<(), RamValError> {
        match self {
            Self::Split {
                lt_lo,
                lt_hi,
                eq_hi,
            } => {
                let bound = bind_adjacent(lt_lo, challenge)?;
                if bound.len() == 1 {
                    if lt_hi.len() != eq_hi.len() {
                        return Err(RamValError::InvalidLtState);
                    }
                    let low_scalar = bound.first().copied().ok_or(RamValError::InvalidLtState)?;
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

    fn final_value(&self) -> Result<F, RamValError> {
        match self {
            Self::Dense(table) if table.len() == 1 => {
                table.first().copied().ok_or(RamValError::InvalidLtState)
            }
            Self::Dense(_) | Self::Split { .. } => Err(RamValError::InvalidLtState),
        }
    }
}

fn bind_adjacent<F: Field>(values: &[F], challenge: F) -> Result<Vec<F>, RamValError> {
    if values.len() < 2 || !values.len().is_multiple_of(2) {
        return Err(RamValError::InvalidLtState);
    }
    let mut bound = Vec::with_capacity(values.len() / 2);
    for pair in values.chunks_exact(2) {
        let low = pair.first().copied().ok_or(RamValError::InvalidLtState)?;
        let high = pair.get(1).copied().ok_or(RamValError::InvalidLtState)?;
        bound.push(low + challenge * (high - low));
    }
    Ok(bound)
}

fn eq_evaluations<F: Field>(point: &[F]) -> Result<Vec<F>, RamValError> {
    let expected = checked_domain_size(point.len())?;
    let mut table = Vec::with_capacity(expected);
    table.push(F::one());
    for &challenge in point {
        let next_capacity = table.len().checked_mul(2).ok_or(RamValError::Overflow)?;
        let mut next = Vec::with_capacity(next_capacity);
        let complement = F::one() - challenge;
        for &base in &table {
            next.push(base * complement);
            next.push(base * challenge);
        }
        table = next;
    }
    if table.len() != expected {
        return Err(RamValError::InvalidEqualityTable);
    }
    Ok(table)
}

fn lt_evaluations<F: Field>(point: &[F]) -> Result<Vec<F>, RamValError> {
    let expected = checked_domain_size(point.len())?;
    let mut table = vec![F::zero()];
    for &challenge in point.iter().rev() {
        let next_capacity = table.len().checked_mul(2).ok_or(RamValError::Overflow)?;
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
        return Err(RamValError::InvalidLtState);
    }
    Ok(table)
}

fn checked_domain_size(log_size: usize) -> Result<usize, RamValError> {
    let shift = u32::try_from(log_size).map_err(|_| RamValError::Overflow)?;
    1usize.checked_shl(shift).ok_or(RamValError::Overflow)
}

fn usize_to_u64(value: usize) -> Result<u64, RamValError> {
    u64::try_from(value).map_err(|_| RamValError::Overflow)
}

fn u64_to_usize(value: u64) -> Result<usize, RamValError> {
    usize::try_from(value).map_err(|_| RamValError::Overflow)
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum RamValError {
    #[error(transparent)]
    Topology(#[from] TopologyError),
    #[error("RAM value-check address point has length {got}, expected {expected}")]
    AddressPointLength { expected: usize, got: usize },
    #[error("RAM value-check cycle point has length {got}, expected {expected}")]
    CyclePointLength { expected: usize, got: usize },
    #[error("RAM value-check address table has length {got}, expected {expected}")]
    AddressTableLength { expected: usize, got: usize },
    #[error("RAM value-check access address {address} is out of range")]
    AccessAddressOutOfRange { address: u32 },
    #[error("RAM value-check increment cycle and value payloads differ in length")]
    IncrementPayloadLength,
    #[error("RAM value-check topology has no census for the requested level")]
    MissingTopologyLevel,
    #[error("RAM value-check union leaves do not match the owner topology")]
    UnionLeafMismatch,
    #[error("RAM value-check frontier at round {round} has {got} entries, expected {expected}")]
    FrontierLength {
        round: usize,
        expected: u64,
        got: u64,
    },
    #[error("RAM value-check frontier index {index} is invalid at round {round}")]
    InvalidFrontierIndex { round: usize, index: usize },
    #[error("RAM value-check merge has no child at round {round}")]
    EmptyMerge { round: usize },
    #[error("RAM value-check merge children disagree at round {round}")]
    InvalidMergeChildren { round: usize },
    #[error("RAM value-check parent blocks are not strictly ordered at round {round}")]
    UnorderedParentBlocks { round: usize },
    #[error("RAM value-check LT index {index} is out of range")]
    LtIndexOutOfRange { index: usize },
    #[error("RAM value-check LT state is invalid")]
    InvalidLtState,
    #[error("RAM value-check equality table is invalid")]
    InvalidEqualityTable,
    #[error("RAM value-check is already fully bound")]
    AlreadyFullyBound,
    #[error("RAM value-check is not fully bound; {remaining} rounds remain")]
    NotFullyBound { remaining: usize },
    #[error("RAM value-check terminal frontier is invalid")]
    InvalidTerminalFrontier,
    #[error("RAM value-check dense cycle {cycle} is out of range")]
    DenseCycleOutOfRange { cycle: u64 },
    #[error("RAM value-check dense table has length {got}, expected {expected}")]
    DenseTableLength { expected: usize, got: usize },
    #[error("RAM value-check dense oracle state is invalid")]
    InvalidDenseState,
    #[error("RAM value-check arithmetic overflowed")]
    Overflow,
}
