use std::sync::Arc;

use jolt_field::Field;
use thiserror::Error;

use super::owner::RamCycleFamilyOwner;
use super::topology::{BlockMerge, TopologyError};

const TERMS: usize = 3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamRaClaimMessage<F> {
    at_zero: F,
    at_two: F,
}

impl<F: Field> RamRaClaimMessage<F> {
    pub const fn sampled_evaluations(self) -> [F; 2] {
        [self.at_zero, self.at_two]
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamRaClaimTerminal<F> {
    ram_ra: F,
    eq_cycles: [F; TERMS],
}

impl<F: Field> RamRaClaimTerminal<F> {
    copy_field_getters! { pub, {
        ram_ra: F,
        eq_cycles: [F; TERMS],
    }}
}

#[derive(Clone, Copy)]
struct FrontierEntry<F> {
    block: u64,
    value: F,
}

#[derive(Clone, Copy)]
struct CachedPair<F> {
    parent_block: u64,
    at_zero: F,
    slope: F,
}

pub struct HostSparseRamRaClaimReduction<F> {
    owner: Arc<RamCycleFamilyOwner>,
    frontier: Vec<FrontierEntry<F>>,
    scratch: Vec<FrontierEntry<F>>,
    cached_pairs: Vec<CachedPair<F>>,
    cached_round: Option<usize>,
    parent_weights: Vec<Vec<[F; TERMS]>>,
    cycle_points: [Vec<F>; TERMS],
    scales: [F; TERMS],
    round: usize,
    rounds: usize,
}

impl<F: Field> HostSparseRamRaClaimReduction<F> {
    pub fn new(
        owner: Arc<RamCycleFamilyOwner>,
        r_address: &[F],
        cycle_points: [&[F]; TERMS],
        gamma: F,
    ) -> Result<Self, RamRaClaimError> {
        let receipt = owner.receipt();
        if r_address.len() != receipt.log_k() {
            return Err(RamRaClaimError::AddressPointLength {
                expected: receipt.log_k(),
                got: r_address.len(),
            });
        }
        for point in cycle_points {
            if point.len() != receipt.log_t() {
                return Err(RamRaClaimError::CyclePointLength {
                    expected: receipt.log_t(),
                    got: point.len(),
                });
            }
        }

        let frontier = seed_frontier(&owner, r_address)?;
        let capacity = frontier.len();
        let rounds = receipt.log_t();
        let cycle_points = cycle_points.map(<[F]>::to_vec);
        let gamma_powers = [F::one(), gamma, gamma * gamma];
        let parent_weights = build_parent_weights(&owner, &cycle_points, gamma_powers)?;
        let sequence = Self {
            owner,
            frontier,
            scratch: Vec::with_capacity(capacity),
            cached_pairs: Vec::with_capacity(capacity),
            cached_round: None,
            parent_weights,
            cycle_points,
            scales: [F::one(); TERMS],
            round: 0,
            rounds,
        };
        sequence.validate_frontier_length()?;
        Ok(sequence)
    }

    pub fn owned_heap_bytes(&self) -> usize {
        self.frontier.capacity() * std::mem::size_of::<FrontierEntry<F>>()
            + self.scratch.capacity() * std::mem::size_of::<FrontierEntry<F>>()
            + self.cached_pairs.capacity() * std::mem::size_of::<CachedPair<F>>()
            + self
                .parent_weights
                .iter()
                .map(|level| level.capacity() * std::mem::size_of::<[F; TERMS]>())
                .sum::<usize>()
            + self
                .cycle_points
                .iter()
                .map(|point| point.capacity() * std::mem::size_of::<F>())
                .sum::<usize>()
    }

    copy_field_getters! { pub, {
        num_rounds => rounds: usize,
        round: usize,
    }}

    pub fn message(&mut self) -> Result<RamRaClaimMessage<F>, RamRaClaimError> {
        if self.round >= self.rounds {
            return Err(RamRaClaimError::AlreadyFullyBound);
        }
        self.validate_frontier_length()?;
        let merges = self.owner.block_topology().merges_for_round(self.round)?;
        let weights = self
            .parent_weights
            .get(self.round + 1)
            .ok_or(RamRaClaimError::MissingWeightLevel { round: self.round })?;
        if weights.len() != merges.len() {
            return Err(RamRaClaimError::WeightLength {
                round: self.round,
                expected: merges.len(),
                got: weights.len(),
            });
        }

        let mut c_zero = [F::zero(); TERMS];
        let mut c_two = [F::zero(); TERMS];
        for term in 0..TERMS {
            let coordinate = self.cycle_points[term][self.rounds - 1 - self.round];
            c_zero[term] = self.scales[term] * (F::one() - coordinate);
            c_two[term] = self.scales[term] * (coordinate + coordinate + coordinate - F::one());
        }

        self.cached_pairs.clear();
        let mut at_zero = F::zero();
        let mut at_two = F::zero();
        for (merge, weight) in merges.iter().zip(weights) {
            let (low, high, parent_block) = merge_children(&self.frontier, *merge, self.round)?;
            let h_zero = low.map_or(F::zero(), |entry| entry.value);
            let h_one = high.map_or(F::zero(), |entry| entry.value);
            let slope = h_one - h_zero;
            let mut g_zero = F::zero();
            let mut g_two = F::zero();
            for term in 0..TERMS {
                g_zero += weight[term] * c_zero[term];
                g_two += weight[term] * c_two[term];
            }
            at_zero += h_zero * g_zero;
            at_two += (h_one + slope) * g_two;
            self.cached_pairs.push(CachedPair {
                parent_block,
                at_zero: h_zero,
                slope,
            });
        }
        self.cached_round = Some(self.round);
        Ok(RamRaClaimMessage { at_zero, at_two })
    }

    pub fn bind(&mut self, challenge: F) -> Result<(), RamRaClaimError> {
        if self.round >= self.rounds {
            return Err(RamRaClaimError::AlreadyFullyBound);
        }
        if self.cached_round != Some(self.round) {
            return Err(RamRaClaimError::MessageNotPrepared { round: self.round });
        }
        self.scratch.clear();
        for pair in &self.cached_pairs {
            self.scratch.push(FrontierEntry {
                block: pair.parent_block,
                value: pair.at_zero + challenge * pair.slope,
            });
        }
        for term in 0..TERMS {
            let coordinate = self.cycle_points[term][self.rounds - 1 - self.round];
            self.scales[term] *=
                (F::one() - coordinate) + challenge * (coordinate + coordinate - F::one());
        }
        std::mem::swap(&mut self.frontier, &mut self.scratch);
        self.cached_pairs.clear();
        self.cached_round = None;
        self.round += 1;
        self.validate_frontier_length()
    }

    pub fn terminal(&self) -> Result<RamRaClaimTerminal<F>, RamRaClaimError> {
        if self.round != self.rounds {
            return Err(RamRaClaimError::NotFullyBound {
                remaining: self.rounds - self.round,
            });
        }
        self.validate_frontier_length()?;
        let ram_ra = match self.frontier.as_slice() {
            [] => F::zero(),
            [entry] if entry.block == 0 => entry.value,
            _ => return Err(RamRaClaimError::InvalidTerminalFrontier),
        };
        Ok(RamRaClaimTerminal {
            ram_ra,
            eq_cycles: self.scales,
        })
    }

    fn validate_frontier_length(&self) -> Result<(), RamRaClaimError> {
        let expected = self
            .owner
            .block_topology()
            .census()
            .get(self.round)
            .ok_or(RamRaClaimError::MissingTopologyLevel { round: self.round })?
            .entries();
        let got = u64::try_from(self.frontier.len()).map_err(|_| RamRaClaimError::Overflow)?;
        if got != expected {
            return Err(RamRaClaimError::FrontierLength {
                round: self.round,
                expected,
                got,
            });
        }
        Ok(())
    }
}

pub fn estimated_ram_ra_claim_products(
    owner: &RamCycleFamilyOwner,
) -> Result<u128, RamRaClaimError> {
    let receipt = owner.receipt();
    let census = owner.block_topology().census();
    let parent_nodes = census
        .iter()
        .skip(1)
        .try_fold(0u128, |sum, level| {
            sum.checked_add(u128::from(level.entries()))
        })
        .ok_or(RamRaClaimError::Overflow)?;
    let middle_nodes = census
        .iter()
        .skip(1)
        .take(receipt.log_t().saturating_sub(1))
        .try_fold(0u128, |sum, level| {
            sum.checked_add(u128::from(level.entries()))
        })
        .ok_or(RamRaClaimError::Overflow)?;
    let address = u128::try_from(receipt.access_count())
        .map_err(|_| RamRaClaimError::Overflow)?
        .checked_mul(u128::try_from(receipt.log_k()).map_err(|_| RamRaClaimError::Overflow)?)
        .ok_or(RamRaClaimError::Overflow)?;
    address
        .checked_add(1)
        .and_then(|value| value.checked_add(3 * middle_nodes))
        .and_then(|value| value.checked_add(9 * parent_nodes))
        .and_then(|value| value.checked_add(12 * u128::try_from(receipt.log_t()).ok()?))
        .ok_or(RamRaClaimError::Overflow)
}

fn seed_frontier<F: Field>(
    owner: &RamCycleFamilyOwner,
    r_address: &[F],
) -> Result<Vec<FrontierEntry<F>>, RamRaClaimError> {
    let leaves = owner.block_topology().leaf_cycles();
    let records = owner.access_records();
    let mut record_index = 0;
    let mut frontier = Vec::with_capacity(leaves.len());
    for &cycle in leaves {
        let value = match records.get(record_index) {
            Some(record) if record.cycle() == cycle => {
                record_index += 1;
                eq_at_boolean_index(r_address, u64::from(record.address()))?
            }
            Some(record) if record.cycle() < cycle => {
                return Err(RamRaClaimError::AccessOutsideTopology {
                    cycle: record.cycle(),
                });
            }
            _ => F::zero(),
        };
        frontier.push(FrontierEntry {
            block: u64::from(cycle),
            value,
        });
    }
    if let Some(record) = records.get(record_index) {
        return Err(RamRaClaimError::AccessOutsideTopology {
            cycle: record.cycle(),
        });
    }
    Ok(frontier)
}

fn build_parent_weights<F: Field>(
    owner: &RamCycleFamilyOwner,
    cycle_points: &[Vec<F>; TERMS],
    gamma_powers: [F; TERMS],
) -> Result<Vec<Vec<[F; TERMS]>>, RamRaClaimError> {
    let rounds = owner.receipt().log_t();
    let census = owner.block_topology().census();
    let mut levels = vec![Vec::new(); rounds + 1];
    let root_entries = usize::try_from(
        census
            .get(rounds)
            .ok_or(RamRaClaimError::MissingTopologyLevel { round: rounds })?
            .entries(),
    )
    .map_err(|_| RamRaClaimError::Overflow)?;
    if root_entries > 1 {
        return Err(RamRaClaimError::InvalidRootCensus { got: root_entries });
    }
    if root_entries == 1 {
        levels[rounds].push(gamma_powers);
    }

    for round in (1..rounds).rev() {
        let current_len = usize::try_from(
            census
                .get(round)
                .ok_or(RamRaClaimError::MissingTopologyLevel { round })?
                .entries(),
        )
        .map_err(|_| RamRaClaimError::Overflow)?;
        let merges = owner.block_topology().merges_for_round(round)?;
        let parents = levels
            .get(round + 1)
            .ok_or(RamRaClaimError::MissingWeightLevel { round })?;
        if merges.len() != parents.len() {
            return Err(RamRaClaimError::WeightLength {
                round,
                expected: merges.len(),
                got: parents.len(),
            });
        }
        let mut current = vec![[F::zero(); TERMS]; current_len];
        let mut filled = vec![false; current_len];
        for (merge, parent) in merges.iter().zip(parents) {
            for (child, high) in [(merge.low_state(), false), (merge.high_state(), true)] {
                let Some(child) = child else {
                    continue;
                };
                if child >= current.len() || filled[child] {
                    return Err(RamRaClaimError::InvalidWeightChild { round, child });
                }
                for term in 0..TERMS {
                    let coordinate = cycle_points[term][rounds - 1 - round];
                    current[child][term] = parent[term]
                        * if high {
                            coordinate
                        } else {
                            F::one() - coordinate
                        };
                }
                filled[child] = true;
            }
        }
        if filled.iter().any(|filled| !filled) {
            return Err(RamRaClaimError::IncompleteWeightLevel { round });
        }
        levels[round] = current;
    }
    Ok(levels)
}

fn eq_at_boolean_index<F: Field>(point: &[F], index: u64) -> Result<F, RamRaClaimError> {
    if point.len() >= u64::BITS as usize || index >= (1u64 << point.len()) {
        return Err(RamRaClaimError::AddressOutOfRange { index });
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

type Children<'a, F> = (
    Option<&'a FrontierEntry<F>>,
    Option<&'a FrontierEntry<F>>,
    u64,
);

fn merge_children<F>(
    frontier: &[FrontierEntry<F>],
    merge: BlockMerge,
    round: usize,
) -> Result<Children<'_, F>, RamRaClaimError> {
    let low = merge
        .low_state()
        .map(|index| {
            frontier
                .get(index)
                .ok_or(RamRaClaimError::InvalidFrontierIndex { round, index })
        })
        .transpose()?;
    let high = merge
        .high_state()
        .map(|index| {
            frontier
                .get(index)
                .ok_or(RamRaClaimError::InvalidFrontierIndex { round, index })
        })
        .transpose()?;
    let parent = match (low, high) {
        (Some(low), Some(high))
            if low.block.is_multiple_of(2)
                && !high.block.is_multiple_of(2)
                && low.block >> 1 == high.block >> 1 =>
        {
            low.block >> 1
        }
        (Some(low), None) if low.block.is_multiple_of(2) => low.block >> 1,
        (None, Some(high)) if !high.block.is_multiple_of(2) => high.block >> 1,
        (None, None) => return Err(RamRaClaimError::EmptyMerge { round }),
        _ => return Err(RamRaClaimError::InvalidMergeChildren { round }),
    };
    Ok((low, high, parent))
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum RamRaClaimError {
    #[error(transparent)]
    Topology(#[from] TopologyError),
    #[error("RAM RA claim address point has length {got}, expected {expected}")]
    AddressPointLength { expected: usize, got: usize },
    #[error("RAM RA claim cycle point has length {got}, expected {expected}")]
    CyclePointLength { expected: usize, got: usize },
    #[error("RAM RA claim topology is missing level {round}")]
    MissingTopologyLevel { round: usize },
    #[error("RAM RA claim weights are missing level {round}")]
    MissingWeightLevel { round: usize },
    #[error("RAM RA claim weight level {round} has {got} entries, expected {expected}")]
    WeightLength {
        round: usize,
        expected: usize,
        got: usize,
    },
    #[error("RAM RA claim frontier at round {round} has {got} entries, expected {expected}")]
    FrontierLength {
        round: usize,
        expected: u64,
        got: u64,
    },
    #[error("RAM RA claim frontier index {index} is invalid at round {round}")]
    InvalidFrontierIndex { round: usize, index: usize },
    #[error("RAM RA claim topology has an invalid root census of {got}")]
    InvalidRootCensus { got: usize },
    #[error("RAM RA claim weight child {child} is invalid at round {round}")]
    InvalidWeightChild { round: usize, child: usize },
    #[error("RAM RA claim weight level {round} is incomplete")]
    IncompleteWeightLevel { round: usize },
    #[error("RAM RA claim access cycle {cycle} is absent from the topology")]
    AccessOutsideTopology { cycle: u32 },
    #[error("RAM RA claim Boolean address {index} is out of range")]
    AddressOutOfRange { index: u64 },
    #[error("RAM RA claim merge has no child at round {round}")]
    EmptyMerge { round: usize },
    #[error("RAM RA claim merge children disagree at round {round}")]
    InvalidMergeChildren { round: usize },
    #[error("RAM RA claim message was not prepared at round {round}")]
    MessageNotPrepared { round: usize },
    #[error("RAM RA claim is already fully bound")]
    AlreadyFullyBound,
    #[error("RAM RA claim is not fully bound; {remaining} rounds remain")]
    NotFullyBound { remaining: usize },
    #[error("RAM RA claim terminal frontier is invalid")]
    InvalidTerminalFrontier,
    #[error("RAM RA claim arithmetic overflowed")]
    Overflow,
}
