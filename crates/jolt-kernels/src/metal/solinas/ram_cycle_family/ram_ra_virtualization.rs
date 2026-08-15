//! Sparse cycle-phase sequence for `RamRaVirtualization`.
//!
//! For an accessed cycle `j`, factor `i` is
//! `f_i(j) = eq(r_chunk_i, chunk_i(address[j]))`. All factors are zero on
//! cycles without a remapped RAM access. This includes increment-only leaves
//! retained by the shared RAM block topology. The round relation is
//! `eq(r_cycle, j) * product_i f_i(j)` and binds cycle variables low-to-high.

use std::sync::Arc;

use jolt_claims::protocols::jolt::geometry::dimensions::committed_address_chunks;
use jolt_field::Field;
use thiserror::Error;

use super::owner::RamCycleFamilyOwner;
use super::topology::{BlockMerge, TopologyError};

pub const MAX_RAM_RA_VIRTUALIZATION_FACTORS: usize = u32::BITS as usize;
pub const MAX_RAM_RA_VIRTUALIZATION_EVALUATIONS: usize = MAX_RAM_RA_VIRTUALIZATION_FACTORS + 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamRaVirtualizationMessage<F> {
    evaluations: [F; MAX_RAM_RA_VIRTUALIZATION_EVALUATIONS],
    len: usize,
}

impl<F> RamRaVirtualizationMessage<F> {
    pub fn evaluations(&self) -> &[F] {
        &self.evaluations[..self.len]
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamRaVirtualizationTerminal<F> {
    ram_ra: [F; MAX_RAM_RA_VIRTUALIZATION_FACTORS],
    factors: usize,
    eq_cycle: F,
}

impl<F> RamRaVirtualizationTerminal<F> {
    pub fn ram_ra(&self) -> &[F] {
        &self.ram_ra[..self.factors]
    }

    pub const fn eq_cycle(&self) -> F
    where
        F: Copy,
    {
        self.eq_cycle
    }
}

pub struct HostSparseRamRaVirtualization<F> {
    owner: Arc<RamCycleFamilyOwner>,
    frontier_blocks: Vec<u64>,
    frontier_values: Vec<Vec<F>>,
    scratch_blocks: Vec<u64>,
    scratch_values: Vec<Vec<F>>,
    cached_parent_blocks: Vec<u64>,
    cached_lows: Vec<Vec<F>>,
    cached_slopes: Vec<Vec<F>>,
    cached_round: Option<usize>,
    parent_weights: Vec<Vec<F>>,
    cycle_point: Vec<F>,
    eq_scale: F,
    factors: usize,
    round: usize,
    rounds: usize,
}

impl<F: Field> HostSparseRamRaVirtualization<F> {
    pub fn new(
        owner: Arc<RamCycleFamilyOwner>,
        r_address: &[F],
        committed_chunk_bits: usize,
        r_cycle: &[F],
    ) -> Result<Self, RamRaVirtualizationError> {
        validate_chunk_bits(committed_chunk_bits)?;
        let receipt = owner.receipt();
        if r_address.len() != receipt.log_k() {
            return Err(RamRaVirtualizationError::AddressPointLength {
                expected: receipt.log_k(),
                got: r_address.len(),
            });
        }
        if r_cycle.len() != receipt.log_t() {
            return Err(RamRaVirtualizationError::CyclePointLength {
                expected: receipt.log_t(),
                got: r_cycle.len(),
            });
        }

        let chunks = committed_address_chunks(r_address, committed_chunk_bits);
        let factors = chunks.len();
        if factors == 0 || factors > MAX_RAM_RA_VIRTUALIZATION_FACTORS {
            return Err(RamRaVirtualizationError::FactorCount { factors });
        }
        let (frontier_blocks, frontier_values) =
            seed_frontier(&owner, &chunks, committed_chunk_bits)?;
        let capacity = frontier_blocks.len();
        let rounds = receipt.log_t();
        let parent_weights = build_parent_weights(&owner, r_cycle)?;
        let sequence = Self {
            owner,
            frontier_blocks,
            frontier_values,
            scratch_blocks: Vec::with_capacity(capacity),
            scratch_values: factor_buffers(factors, capacity),
            cached_parent_blocks: Vec::with_capacity(capacity),
            cached_lows: factor_buffers(factors, capacity),
            cached_slopes: factor_buffers(factors, capacity),
            cached_round: None,
            parent_weights,
            cycle_point: r_cycle.to_vec(),
            eq_scale: F::one(),
            factors,
            round: 0,
            rounds,
        };
        sequence.validate_frontier()?;
        Ok(sequence)
    }

    pub fn owned_heap_bytes(&self) -> usize {
        let nested = |values: &[Vec<F>], outer_capacity: usize| {
            outer_capacity * std::mem::size_of::<Vec<F>>()
                + values
                    .iter()
                    .map(|value| value.capacity() * std::mem::size_of::<F>())
                    .sum::<usize>()
        };
        self.frontier_blocks.capacity() * std::mem::size_of::<u64>()
            + nested(&self.frontier_values, self.frontier_values.capacity())
            + self.scratch_blocks.capacity() * std::mem::size_of::<u64>()
            + nested(&self.scratch_values, self.scratch_values.capacity())
            + self.cached_parent_blocks.capacity() * std::mem::size_of::<u64>()
            + nested(&self.cached_lows, self.cached_lows.capacity())
            + nested(&self.cached_slopes, self.cached_slopes.capacity())
            + self.parent_weights.capacity() * std::mem::size_of::<Vec<F>>()
            + self
                .parent_weights
                .iter()
                .map(|level| level.capacity() * std::mem::size_of::<F>())
                .sum::<usize>()
            + self.cycle_point.capacity() * std::mem::size_of::<F>()
    }

    copy_field_getters! { pub, {
        num_rounds => rounds: usize,
        round: usize,
    }}

    pub fn message(&mut self) -> Result<RamRaVirtualizationMessage<F>, RamRaVirtualizationError> {
        if self.round >= self.rounds {
            return Err(RamRaVirtualizationError::AlreadyFullyBound);
        }
        self.validate_frontier()?;
        let merges = self.owner.block_topology().merges_for_round(self.round)?;
        let weights = self
            .parent_weights
            .get(self.round + 1)
            .ok_or(RamRaVirtualizationError::MissingWeightLevel { round: self.round })?;
        if weights.len() != merges.len() {
            return Err(RamRaVirtualizationError::WeightLength {
                round: self.round,
                expected: merges.len(),
                got: weights.len(),
            });
        }

        self.cached_parent_blocks.clear();
        for values in &mut self.cached_lows {
            values.clear();
        }
        for values in &mut self.cached_slopes {
            values.clear();
        }

        let coordinate = self.cycle_point[self.rounds - 1 - self.round];
        let eq_at_zero = F::one() - coordinate;
        let eq_step = coordinate + coordinate - F::one();
        let points = self.factors + 2;
        let mut evaluations = [F::zero(); MAX_RAM_RA_VIRTUALIZATION_EVALUATIONS];
        let mut factor_values = [F::zero(); MAX_RAM_RA_VIRTUALIZATION_FACTORS];
        let mut factor_slopes = [F::zero(); MAX_RAM_RA_VIRTUALIZATION_FACTORS];

        for (merge, &weight) in merges.iter().zip(weights) {
            let parent_block = merge_parent_block(&self.frontier_blocks, *merge, self.round)?;
            self.cached_parent_blocks.push(parent_block);
            for factor in 0..self.factors {
                let low =
                    frontier_value(&self.frontier_values[factor], merge.low_state(), self.round)?;
                let high = frontier_value(
                    &self.frontier_values[factor],
                    merge.high_state(),
                    self.round,
                )?;
                let slope = high - low;
                self.cached_lows[factor].push(low);
                self.cached_slopes[factor].push(slope);
                factor_values[factor] = low;
                factor_slopes[factor] = slope;
            }

            let weighted_scale = weight * self.eq_scale;
            let mut eq_value = weighted_scale * eq_at_zero;
            let eq_delta = weighted_scale * eq_step;
            for evaluation in &mut evaluations[..points] {
                let mut product = factor_values[0];
                for &value in &factor_values[1..self.factors] {
                    product *= value;
                }
                *evaluation += eq_value * product;
                for factor in 0..self.factors {
                    factor_values[factor] += factor_slopes[factor];
                }
                eq_value += eq_delta;
            }
        }
        self.cached_round = Some(self.round);
        Ok(RamRaVirtualizationMessage {
            evaluations,
            len: points,
        })
    }

    pub fn bind(&mut self, challenge: F) -> Result<(), RamRaVirtualizationError> {
        if self.round >= self.rounds {
            return Err(RamRaVirtualizationError::AlreadyFullyBound);
        }
        if self.cached_round != Some(self.round) {
            return Err(RamRaVirtualizationError::MessageNotPrepared { round: self.round });
        }
        let expected = self.cached_parent_blocks.len();
        for factor in 0..self.factors {
            if self.cached_lows[factor].len() != expected
                || self.cached_slopes[factor].len() != expected
            {
                return Err(RamRaVirtualizationError::CacheLength {
                    round: self.round,
                    factor,
                    expected,
                    lows: self.cached_lows[factor].len(),
                    slopes: self.cached_slopes[factor].len(),
                });
            }
        }

        self.scratch_blocks.clear();
        self.scratch_blocks
            .extend_from_slice(&self.cached_parent_blocks);
        for factor in 0..self.factors {
            self.scratch_values[factor].clear();
            for (&low, &slope) in self.cached_lows[factor]
                .iter()
                .zip(&self.cached_slopes[factor])
            {
                self.scratch_values[factor].push(low + challenge * slope);
            }
        }
        let coordinate = self.cycle_point[self.rounds - 1 - self.round];
        self.eq_scale *= (F::one() - coordinate) + challenge * (coordinate + coordinate - F::one());

        std::mem::swap(&mut self.frontier_blocks, &mut self.scratch_blocks);
        std::mem::swap(&mut self.frontier_values, &mut self.scratch_values);
        self.cached_parent_blocks.clear();
        for values in &mut self.cached_lows {
            values.clear();
        }
        for values in &mut self.cached_slopes {
            values.clear();
        }
        self.cached_round = None;
        self.round += 1;
        self.validate_frontier()
    }

    pub fn terminal(&self) -> Result<RamRaVirtualizationTerminal<F>, RamRaVirtualizationError> {
        if self.round != self.rounds {
            return Err(RamRaVirtualizationError::NotFullyBound {
                remaining: self.rounds - self.round,
            });
        }
        self.validate_frontier()?;
        let mut ram_ra = [F::zero(); MAX_RAM_RA_VIRTUALIZATION_FACTORS];
        match self.frontier_blocks.as_slice() {
            [] => {}
            [0] => {
                for (factor, values) in self.frontier_values.iter().enumerate() {
                    ram_ra[factor] = values[0];
                }
            }
            _ => return Err(RamRaVirtualizationError::InvalidTerminalFrontier),
        }
        Ok(RamRaVirtualizationTerminal {
            ram_ra,
            factors: self.factors,
            eq_cycle: self.eq_scale,
        })
    }

    fn validate_frontier(&self) -> Result<(), RamRaVirtualizationError> {
        let expected = usize::try_from(
            self.owner
                .block_topology()
                .census()
                .get(self.round)
                .ok_or(RamRaVirtualizationError::MissingTopologyLevel { round: self.round })?
                .entries(),
        )
        .map_err(|_| RamRaVirtualizationError::Overflow)?;
        if self.frontier_blocks.len() != expected {
            return Err(RamRaVirtualizationError::FrontierLength {
                round: self.round,
                factor: None,
                expected,
                got: self.frontier_blocks.len(),
            });
        }
        if self.frontier_values.len() != self.factors {
            return Err(RamRaVirtualizationError::FactorCount {
                factors: self.frontier_values.len(),
            });
        }
        for (factor, values) in self.frontier_values.iter().enumerate() {
            if values.len() != expected {
                return Err(RamRaVirtualizationError::FrontierLength {
                    round: self.round,
                    factor: Some(factor),
                    expected,
                    got: values.len(),
                });
            }
        }
        Ok(())
    }
}

pub fn estimated_ram_ra_virtualization_products(
    owner: &RamCycleFamilyOwner,
    committed_chunk_bits: usize,
) -> Result<u128, RamRaVirtualizationError> {
    validate_chunk_bits(committed_chunk_bits)?;
    let receipt = owner.receipt();
    let factors = receipt.log_k().div_ceil(committed_chunk_bits);
    if factors == 0 || factors > MAX_RAM_RA_VIRTUALIZATION_FACTORS {
        return Err(RamRaVirtualizationError::FactorCount { factors });
    }
    let census = owner.block_topology().census();
    let parent_nodes = census.iter().skip(1).try_fold(0u128, |sum, level| {
        sum.checked_add(u128::from(level.entries()))
    });
    let parent_nodes = parent_nodes.ok_or(RamRaVirtualizationError::Overflow)?;
    let middle_nodes = census
        .iter()
        .skip(1)
        .take(receipt.log_t().saturating_sub(1))
        .try_fold(0u128, |sum, level| {
            sum.checked_add(u128::from(level.entries()))
        })
        .ok_or(RamRaVirtualizationError::Overflow)?;
    let factors = u128::try_from(factors).map_err(|_| RamRaVirtualizationError::Overflow)?;
    let chunk_bits =
        u128::try_from(committed_chunk_bits).map_err(|_| RamRaVirtualizationError::Overflow)?;
    let rounds = u128::try_from(receipt.log_t()).map_err(|_| RamRaVirtualizationError::Overflow)?;
    let address_products = u128::try_from(receipt.access_count())
        .map_err(|_| RamRaVirtualizationError::Overflow)?
        .checked_mul(factors)
        .and_then(|value| value.checked_mul(chunk_bits))
        .ok_or(RamRaVirtualizationError::Overflow)?;
    let message_products_per_parent = factors
        .checked_mul(
            factors
                .checked_add(2)
                .ok_or(RamRaVirtualizationError::Overflow)?,
        )
        .and_then(|value| value.checked_add(3))
        .ok_or(RamRaVirtualizationError::Overflow)?;
    let message_products = parent_nodes
        .checked_mul(message_products_per_parent)
        .ok_or(RamRaVirtualizationError::Overflow)?;
    let bind_products = parent_nodes
        .checked_mul(factors)
        .ok_or(RamRaVirtualizationError::Overflow)?;
    address_products
        .checked_add(middle_nodes)
        .and_then(|value| value.checked_add(message_products))
        .and_then(|value| value.checked_add(bind_products))
        .and_then(|value| value.checked_add(rounds.checked_mul(2)?))
        .ok_or(RamRaVirtualizationError::Overflow)
}

fn validate_chunk_bits(committed_chunk_bits: usize) -> Result<(), RamRaVirtualizationError> {
    if committed_chunk_bits == 0 || committed_chunk_bits > u32::BITS as usize {
        Err(RamRaVirtualizationError::ChunkBits {
            got: committed_chunk_bits,
        })
    } else {
        Ok(())
    }
}

fn factor_buffers<F>(factors: usize, capacity: usize) -> Vec<Vec<F>> {
    (0..factors).map(|_| Vec::with_capacity(capacity)).collect()
}

fn seed_frontier<F: Field>(
    owner: &RamCycleFamilyOwner,
    chunks: &[Vec<F>],
    committed_chunk_bits: usize,
) -> Result<(Vec<u64>, Vec<Vec<F>>), RamRaVirtualizationError> {
    let leaves = owner.block_topology().leaf_cycles();
    let records = owner.access_records();
    let mut record_index = 0;
    let mut blocks = Vec::with_capacity(leaves.len());
    let mut values = factor_buffers(chunks.len(), leaves.len());
    for &cycle in leaves {
        blocks.push(u64::from(cycle));
        let address = match records.get(record_index) {
            Some(record) if record.cycle() == cycle => {
                record_index += 1;
                Some(record.address())
            }
            Some(record) if record.cycle() < cycle => {
                return Err(RamRaVirtualizationError::AccessOutsideTopology {
                    cycle: record.cycle(),
                });
            }
            _ => None,
        };
        for (factor, chunk) in chunks.iter().enumerate() {
            let value = match address {
                Some(address) => {
                    let index = address_chunk(address, factor, chunks.len(), committed_chunk_bits)?;
                    eq_at_boolean_index(chunk, u64::from(index))?
                }
                None => F::zero(),
            };
            values[factor].push(value);
        }
    }
    if let Some(record) = records.get(record_index) {
        return Err(RamRaVirtualizationError::AccessOutsideTopology {
            cycle: record.cycle(),
        });
    }
    Ok((blocks, values))
}

fn address_chunk(
    address: u32,
    factor: usize,
    factors: usize,
    committed_chunk_bits: usize,
) -> Result<u32, RamRaVirtualizationError> {
    let remaining = factors
        .checked_sub(factor + 1)
        .ok_or(RamRaVirtualizationError::InvalidFactorIndex { factor })?;
    let shift = remaining
        .checked_mul(committed_chunk_bits)
        .ok_or(RamRaVirtualizationError::Overflow)?;
    if shift >= u32::BITS as usize {
        return Err(RamRaVirtualizationError::ChunkShift { shift });
    }
    let mask = if committed_chunk_bits == u32::BITS as usize {
        u32::MAX
    } else {
        (1u32 << committed_chunk_bits) - 1
    };
    Ok((address >> shift) & mask)
}

fn build_parent_weights<F: Field>(
    owner: &RamCycleFamilyOwner,
    cycle_point: &[F],
) -> Result<Vec<Vec<F>>, RamRaVirtualizationError> {
    let rounds = owner.receipt().log_t();
    let census = owner.block_topology().census();
    let mut levels = vec![Vec::new(); rounds + 1];
    let root_entries = usize::try_from(
        census
            .get(rounds)
            .ok_or(RamRaVirtualizationError::MissingTopologyLevel { round: rounds })?
            .entries(),
    )
    .map_err(|_| RamRaVirtualizationError::Overflow)?;
    if root_entries > 1 {
        return Err(RamRaVirtualizationError::InvalidRootCensus { got: root_entries });
    }
    if root_entries == 1 {
        levels[rounds].push(F::one());
    }

    for round in (1..rounds).rev() {
        let current_len = usize::try_from(
            census
                .get(round)
                .ok_or(RamRaVirtualizationError::MissingTopologyLevel { round })?
                .entries(),
        )
        .map_err(|_| RamRaVirtualizationError::Overflow)?;
        let merges = owner.block_topology().merges_for_round(round)?;
        let parents = levels
            .get(round + 1)
            .ok_or(RamRaVirtualizationError::MissingWeightLevel { round })?;
        if merges.len() != parents.len() {
            return Err(RamRaVirtualizationError::WeightLength {
                round,
                expected: merges.len(),
                got: parents.len(),
            });
        }
        let mut current = vec![F::zero(); current_len];
        let mut filled = vec![false; current_len];
        let coordinate = cycle_point[rounds - 1 - round];
        for (merge, &parent) in merges.iter().zip(parents) {
            for (child, high) in [(merge.low_state(), false), (merge.high_state(), true)] {
                let Some(child) = child else {
                    continue;
                };
                if child >= current.len() || filled[child] {
                    return Err(RamRaVirtualizationError::InvalidWeightChild { round, child });
                }
                current[child] = parent
                    * if high {
                        coordinate
                    } else {
                        F::one() - coordinate
                    };
                filled[child] = true;
            }
        }
        if filled.iter().any(|filled| !filled) {
            return Err(RamRaVirtualizationError::IncompleteWeightLevel { round });
        }
        levels[round] = current;
    }
    Ok(levels)
}

fn eq_at_boolean_index<F: Field>(point: &[F], index: u64) -> Result<F, RamRaVirtualizationError> {
    if point.len() >= u64::BITS as usize || index >= (1u64 << point.len()) {
        return Err(RamRaVirtualizationError::BooleanIndex { index });
    }
    let mut value = F::one();
    for (bit, &coordinate) in point.iter().rev().enumerate() {
        value *= if index & (1u64 << bit) == 0 {
            F::one() - coordinate
        } else {
            coordinate
        };
    }
    Ok(value)
}

fn frontier_value<F: Field>(
    values: &[F],
    index: Option<usize>,
    round: usize,
) -> Result<F, RamRaVirtualizationError> {
    index.map_or(Ok(F::zero()), |index| {
        values
            .get(index)
            .copied()
            .ok_or(RamRaVirtualizationError::InvalidFrontierIndex { round, index })
    })
}

fn merge_parent_block(
    frontier_blocks: &[u64],
    merge: BlockMerge,
    round: usize,
) -> Result<u64, RamRaVirtualizationError> {
    let low = merge
        .low_state()
        .map(|index| {
            frontier_blocks
                .get(index)
                .copied()
                .ok_or(RamRaVirtualizationError::InvalidFrontierIndex { round, index })
        })
        .transpose()?;
    let high = merge
        .high_state()
        .map(|index| {
            frontier_blocks
                .get(index)
                .copied()
                .ok_or(RamRaVirtualizationError::InvalidFrontierIndex { round, index })
        })
        .transpose()?;
    match (low, high) {
        (Some(low), Some(high))
            if low.is_multiple_of(2) && !high.is_multiple_of(2) && low >> 1 == high >> 1 =>
        {
            Ok(low >> 1)
        }
        (Some(low), None) if low.is_multiple_of(2) => Ok(low >> 1),
        (None, Some(high)) if !high.is_multiple_of(2) => Ok(high >> 1),
        (None, None) => Err(RamRaVirtualizationError::EmptyMerge { round }),
        _ => Err(RamRaVirtualizationError::InvalidMergeChildren { round }),
    }
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum RamRaVirtualizationError {
    #[error(transparent)]
    Topology(#[from] TopologyError),
    #[error("RAM RA virtualization address point has length {got}, expected {expected}")]
    AddressPointLength { expected: usize, got: usize },
    #[error("RAM RA virtualization cycle point has length {got}, expected {expected}")]
    CyclePointLength { expected: usize, got: usize },
    #[error("RAM RA virtualization chunk width {got} is unsupported")]
    ChunkBits { got: usize },
    #[error("RAM RA virtualization has unsupported factor count {factors}")]
    FactorCount { factors: usize },
    #[error("RAM RA virtualization factor index {factor} is invalid")]
    InvalidFactorIndex { factor: usize },
    #[error("RAM RA virtualization chunk shift {shift} is outside a u32 address")]
    ChunkShift { shift: usize },
    #[error("RAM RA virtualization topology is missing level {round}")]
    MissingTopologyLevel { round: usize },
    #[error("RAM RA virtualization weights are missing level {round}")]
    MissingWeightLevel { round: usize },
    #[error("RAM RA virtualization weight level {round} has {got} entries, expected {expected}")]
    WeightLength {
        round: usize,
        expected: usize,
        got: usize,
    },
    #[error(
        "RAM RA virtualization frontier at round {round}, factor {factor:?}, has {got} entries, expected {expected}"
    )]
    FrontierLength {
        round: usize,
        factor: Option<usize>,
        expected: usize,
        got: usize,
    },
    #[error("RAM RA virtualization frontier index {index} is invalid at round {round}")]
    InvalidFrontierIndex { round: usize, index: usize },
    #[error("RAM RA virtualization cache for factor {factor} is malformed at round {round}")]
    CacheLength {
        round: usize,
        factor: usize,
        expected: usize,
        lows: usize,
        slopes: usize,
    },
    #[error("RAM RA virtualization topology has an invalid root census of {got}")]
    InvalidRootCensus { got: usize },
    #[error("RAM RA virtualization weight child {child} is invalid at round {round}")]
    InvalidWeightChild { round: usize, child: usize },
    #[error("RAM RA virtualization weight level {round} is incomplete")]
    IncompleteWeightLevel { round: usize },
    #[error("RAM RA virtualization access cycle {cycle} is absent from the topology")]
    AccessOutsideTopology { cycle: u32 },
    #[error("RAM RA virtualization Boolean index {index} is outside its chunk")]
    BooleanIndex { index: u64 },
    #[error("RAM RA virtualization merge has no child at round {round}")]
    EmptyMerge { round: usize },
    #[error("RAM RA virtualization merge children disagree at round {round}")]
    InvalidMergeChildren { round: usize },
    #[error("RAM RA virtualization message was not prepared at round {round}")]
    MessageNotPrepared { round: usize },
    #[error("RAM RA virtualization is already fully bound")]
    AlreadyFullyBound,
    #[error("RAM RA virtualization is not fully bound; {remaining} rounds remain")]
    NotFullyBound { remaining: usize },
    #[error("RAM RA virtualization terminal frontier is invalid")]
    InvalidTerminalFrontier,
    #[error("RAM RA virtualization arithmetic overflowed")]
    Overflow,
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_field::AkitaField;

    use super::super::owner::{OwnerConfig, RamAccessRecord, RamIncrementRecord};
    use super::*;

    fn field(value: u64) -> AkitaField {
        AkitaField::from_u64(value)
    }

    fn fixture_owner() -> RamCycleFamilyOwner {
        let config = OwnerConfig::new(3, 3, 41, 16).unwrap();
        let records = vec![
            RamAccessRecord::new(0, 1, 0, 2),
            RamAccessRecord::new(3, 5, 0, 3),
            RamAccessRecord::new(4, 1, 2, 2),
            RamAccessRecord::new(7, 5, 3, 1),
        ];
        let increments = vec![
            RamIncrementRecord::new(0, 2),
            RamIncrementRecord::new(2, -3),
            RamIncrementRecord::new(3, 3),
            RamIncrementRecord::new(6, 4),
            RamIncrementRecord::new(7, -2),
        ];
        RamCycleFamilyOwner::from_sparse_records(
            config,
            records,
            increments,
            vec![0, 2, 0, 0, 0, 1, 0, 0],
        )
        .unwrap()
    }

    struct DenseOracle<F> {
        factors: Vec<Vec<F>>,
        eq_cycle: Vec<F>,
    }

    impl<F: Field> DenseOracle<F> {
        fn new(
            owner: &RamCycleFamilyOwner,
            r_address: &[F],
            committed_chunk_bits: usize,
            r_cycle: &[F],
        ) -> Self {
            let chunks = committed_address_chunks(r_address, committed_chunk_bits);
            let mut addresses = vec![None; owner.receipt().cycles()];
            for record in owner.access_records() {
                addresses[record.cycle() as usize] = Some(record.address());
            }
            let factors = chunks
                .iter()
                .enumerate()
                .map(|(factor, chunk)| {
                    addresses
                        .iter()
                        .map(|address| {
                            address.map_or(F::zero(), |address| {
                                let index = address_chunk(
                                    address,
                                    factor,
                                    chunks.len(),
                                    committed_chunk_bits,
                                )
                                .unwrap();
                                eq_at_boolean_index(chunk, u64::from(index)).unwrap()
                            })
                        })
                        .collect()
                })
                .collect();
            let eq_cycle = (0..owner.receipt().cycles())
                .map(|index| eq_at_boolean_index(r_cycle, index as u64).unwrap())
                .collect();
            Self { factors, eq_cycle }
        }

        fn message(&self) -> Vec<F> {
            let points = self.factors.len() + 2;
            let mut evaluations = vec![F::zero(); points];
            for pair in 0..self.eq_cycle.len() / 2 {
                let low = 2 * pair;
                let high = low + 1;
                for (sample, evaluation) in evaluations.iter_mut().enumerate() {
                    let sample = F::from_u64(sample as u64);
                    let eq =
                        self.eq_cycle[low] + sample * (self.eq_cycle[high] - self.eq_cycle[low]);
                    let mut product = eq;
                    for factor in &self.factors {
                        product *= factor[low] + sample * (factor[high] - factor[low]);
                    }
                    *evaluation += product;
                }
            }
            evaluations
        }

        fn bind(&mut self, challenge: F) {
            bind_dense(&mut self.eq_cycle, challenge);
            for factor in &mut self.factors {
                bind_dense(factor, challenge);
            }
        }
    }

    fn bind_dense<F: Field>(values: &mut Vec<F>, challenge: F) {
        let bound = values.len() / 2;
        for index in 0..bound {
            let low = values[2 * index];
            let high = values[2 * index + 1];
            values[index] = low + challenge * (high - low);
        }
        values.truncate(bound);
    }

    #[test]
    fn sparse_sequence_matches_independent_dense_relation() {
        let owner = Arc::new(fixture_owner());
        let r_address = [field(2), field(3), field(5)];
        let r_cycle = [field(7), field(11), field(13)];
        let mut sparse =
            HostSparseRamRaVirtualization::new(Arc::clone(&owner), &r_address, 2, &r_cycle)
                .unwrap();
        let mut dense = DenseOracle::new(&owner, &r_address, 2, &r_cycle);

        assert_eq!(
            estimated_ram_ra_virtualization_products(&owner, 2).unwrap(),
            119
        );
        for challenge in [field(17), field(19), field(23)] {
            assert_eq!(sparse.message().unwrap().evaluations(), dense.message());
            sparse.bind(challenge).unwrap();
            dense.bind(challenge);
        }
        let terminal = sparse.terminal().unwrap();
        let expected = dense
            .factors
            .iter()
            .map(|factor| factor[0])
            .collect::<Vec<_>>();
        assert_eq!(terminal.ram_ra(), expected);
        assert_eq!(terminal.eq_cycle(), dense.eq_cycle[0]);
    }

    #[test]
    fn increment_only_leaf_seeds_zero_ra_factors() {
        let config = OwnerConfig::new(2, 1, 43, 4).unwrap();
        let owner = Arc::new(
            RamCycleFamilyOwner::from_sparse_records(
                config,
                Vec::new(),
                vec![RamIncrementRecord::new(1, 9)],
                vec![0, 0],
            )
            .unwrap(),
        );
        let mut sparse =
            HostSparseRamRaVirtualization::new(owner, &[field(3)], 1, &[field(5), field(7)])
                .unwrap();
        for challenge in [field(11), field(13)] {
            assert!(sparse
                .message()
                .unwrap()
                .evaluations()
                .iter()
                .all(|value| *value == AkitaField::zero()));
            sparse.bind(challenge).unwrap();
        }
        assert_eq!(sparse.terminal().unwrap().ram_ra(), &[AkitaField::zero()]);
    }
}
