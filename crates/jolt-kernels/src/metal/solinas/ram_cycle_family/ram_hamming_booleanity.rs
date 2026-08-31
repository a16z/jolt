//! Sparse cycle sequence for RAM Hamming-weight booleanity.

use std::sync::Arc;

use jolt_field::Field;
#[cfg(test)]
use jolt_field::{One as _, Zero as _};
use thiserror::Error;

use super::owner::RamCycleFamilyOwner;
use super::topology::{BlockMerge, RamBlockTopology, TopologyError};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamHammingMessage<F> {
    coefficients: [F; 4],
}

impl<F> RamHammingMessage<F> {
    pub const fn coefficients(&self) -> &[F; 4] {
        &self.coefficients
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamHammingTerminal<F> {
    ram_hamming_weight: F,
    eq_cycle: F,
}

pub struct RamHammingSparsePlan {
    topology: RamBlockTopology,
    source_generation: u64,
    source_fingerprint: u64,
    log_t: usize,
    access_leaves: usize,
    parent_nodes: usize,
    middle_nodes: usize,
    estimated_products: u128,
    topology_bytes: usize,
}

impl RamHammingSparsePlan {
    pub fn new(owner: &RamCycleFamilyOwner) -> Result<Self, RamHammingError> {
        let receipt = owner.receipt();
        let topology = RamBlockTopology::build(receipt.log_t(), owner.access_records(), &[])?;
        let access_leaves = topology.leaf_cycles().len();
        let parent_nodes = topology
            .census()
            .iter()
            .skip(1)
            .try_fold(0usize, |sum, level| {
                sum.checked_add(
                    usize::try_from(level.entries()).map_err(|_| RamHammingError::Overflow)?,
                )
                .ok_or(RamHammingError::Overflow)
            })?;
        let middle_nodes = topology
            .census()
            .iter()
            .skip(1)
            .take(receipt.log_t().saturating_sub(1))
            .try_fold(0usize, |sum, level| {
                sum.checked_add(
                    usize::try_from(level.entries()).map_err(|_| RamHammingError::Overflow)?,
                )
                .ok_or(RamHammingError::Overflow)
            })?;
        let rounds = receipt.log_t() as u128;
        let estimated_products = (parent_nodes as u128)
            .checked_mul(7)
            .and_then(|value| value.checked_add(middle_nodes as u128))
            .and_then(|value| value.checked_add(rounds.checked_mul(10)?))
            .ok_or(RamHammingError::Overflow)?;
        let topology_bytes = topology.owned_heap_bytes();
        Ok(Self {
            topology,
            source_generation: receipt.source_generation(),
            source_fingerprint: receipt.fingerprint(),
            log_t: receipt.log_t(),
            access_leaves,
            parent_nodes,
            middle_nodes,
            estimated_products,
            topology_bytes,
        })
    }

    copy_field_getters! { pub, {
        source_generation: u64,
        source_fingerprint: u64,
        log_t: usize,
        access_leaves: usize,
        parent_nodes: usize,
        middle_nodes: usize,
        estimated_products: u128,
        topology_bytes: usize,
    }}
}

impl<F: Copy> RamHammingTerminal<F> {
    copy_field_getters! { pub, {
        ram_hamming_weight: F,
        eq_cycle: F,
    }}
}

pub struct HostSparseRamHammingBooleanity<F> {
    _owner: Arc<RamCycleFamilyOwner>,
    topology: RamBlockTopology,
    frontier_blocks: Vec<u64>,
    frontier_values: Vec<F>,
    scratch_blocks: Vec<u64>,
    scratch_values: Vec<F>,
    cached_parent_blocks: Vec<u64>,
    cached_lows: Vec<F>,
    cached_slopes: Vec<F>,
    cached_round: Option<usize>,
    parent_weights: Vec<Vec<F>>,
    cycle_binding: Vec<F>,
    eq_scale: F,
    round: usize,
    rounds: usize,
}

impl<F: Field> HostSparseRamHammingBooleanity<F> {
    pub fn new(
        owner: Arc<RamCycleFamilyOwner>,
        stage1_cycle_binding: &[F],
    ) -> Result<Self, RamHammingError> {
        let plan = RamHammingSparsePlan::new(&owner)?;
        Self::new_from_plan(owner, stage1_cycle_binding, plan)
    }

    pub(crate) fn new_from_plan(
        owner: Arc<RamCycleFamilyOwner>,
        stage1_cycle_binding: &[F],
        plan: RamHammingSparsePlan,
    ) -> Result<Self, RamHammingError> {
        let receipt = owner.receipt();
        if stage1_cycle_binding.len() != receipt.log_t() {
            return Err(RamHammingError::CyclePointLength {
                expected: receipt.log_t(),
                got: stage1_cycle_binding.len(),
            });
        }
        if plan.source_generation != receipt.source_generation()
            || plan.source_fingerprint != receipt.fingerprint()
            || plan.log_t != receipt.log_t()
            || plan.access_leaves != owner.access_records().len()
        {
            return Err(RamHammingError::PlanReceiptMismatch);
        }
        let topology = plan.topology;
        let frontier_blocks = topology
            .leaf_cycles()
            .iter()
            .map(|&cycle| u64::from(cycle))
            .collect::<Vec<_>>();
        let capacity = frontier_blocks.len();
        let rounds = receipt.log_t();
        let parent_weights = build_parent_weights(&topology, stage1_cycle_binding)?;
        let sequence = Self {
            _owner: owner,
            topology,
            frontier_blocks,
            frontier_values: vec![F::one(); capacity],
            scratch_blocks: Vec::with_capacity(capacity),
            scratch_values: Vec::with_capacity(capacity),
            cached_parent_blocks: Vec::with_capacity(capacity),
            cached_lows: Vec::with_capacity(capacity),
            cached_slopes: Vec::with_capacity(capacity),
            cached_round: None,
            parent_weights,
            cycle_binding: stage1_cycle_binding.to_vec(),
            eq_scale: F::one(),
            round: 0,
            rounds,
        };
        sequence.validate_frontier()?;
        Ok(sequence)
    }

    pub fn owned_heap_bytes(&self) -> usize {
        self.topology.owned_heap_bytes()
            + self.frontier_blocks.capacity() * std::mem::size_of::<u64>()
            + self.frontier_values.capacity() * std::mem::size_of::<F>()
            + self.scratch_blocks.capacity() * std::mem::size_of::<u64>()
            + self.scratch_values.capacity() * std::mem::size_of::<F>()
            + self.cached_parent_blocks.capacity() * std::mem::size_of::<u64>()
            + self.cached_lows.capacity() * std::mem::size_of::<F>()
            + self.cached_slopes.capacity() * std::mem::size_of::<F>()
            + self.parent_weights.capacity() * std::mem::size_of::<Vec<F>>()
            + self
                .parent_weights
                .iter()
                .map(|level| level.capacity() * std::mem::size_of::<F>())
                .sum::<usize>()
            + self.cycle_binding.capacity() * std::mem::size_of::<F>()
    }

    copy_field_getters! { pub, {
        num_rounds => rounds: usize,
        round: usize,
    }}

    pub fn message(&mut self) -> Result<RamHammingMessage<F>, RamHammingError> {
        if self.round >= self.rounds {
            return Err(RamHammingError::AlreadyFullyBound);
        }
        self.validate_frontier()?;
        let merges = self.topology.merges_for_round(self.round)?;
        let weights = self
            .parent_weights
            .get(self.round + 1)
            .ok_or(RamHammingError::MissingWeightLevel { round: self.round })?;
        if weights.len() != merges.len() {
            return Err(RamHammingError::WeightLength {
                round: self.round,
                expected: merges.len(),
                got: weights.len(),
            });
        }

        self.cached_parent_blocks.clear();
        self.cached_lows.clear();
        self.cached_slopes.clear();
        let mut q = [F::zero(); 3];
        for (merge, &weight) in merges.iter().zip(weights) {
            let parent = merge_parent_block(&self.frontier_blocks, *merge, self.round)?;
            let low = frontier_value(&self.frontier_values, merge.low_state(), self.round)?;
            let high = frontier_value(&self.frontier_values, merge.high_state(), self.round)?;
            let slope = high - low;
            self.cached_parent_blocks.push(parent);
            self.cached_lows.push(low);
            self.cached_slopes.push(slope);

            let q_0 = low * low - low;
            let q_1 = slope * (low + low - F::one());
            let q_2 = slope * slope;
            q[0] += weight * q_0;
            q[1] += weight * q_1;
            q[2] += weight * q_2;
        }

        let coordinate = self.cycle_binding[self.round];
        let l_0 = self.eq_scale * (F::one() - coordinate);
        let l_1 = self.eq_scale * (coordinate + coordinate - F::one());
        let coefficients = [
            l_0 * q[0],
            l_0 * q[1] + l_1 * q[0],
            l_0 * q[2] + l_1 * q[1],
            l_1 * q[2],
        ];
        self.cached_round = Some(self.round);
        Ok(RamHammingMessage { coefficients })
    }

    pub fn bind(&mut self, challenge: F) -> Result<(), RamHammingError> {
        if self.round >= self.rounds {
            return Err(RamHammingError::AlreadyFullyBound);
        }
        if self.cached_round != Some(self.round) {
            return Err(RamHammingError::MessageNotPrepared { round: self.round });
        }
        let expected = self.cached_parent_blocks.len();
        if self.cached_lows.len() != expected || self.cached_slopes.len() != expected {
            return Err(RamHammingError::CacheLength {
                round: self.round,
                expected,
                lows: self.cached_lows.len(),
                slopes: self.cached_slopes.len(),
            });
        }

        self.scratch_blocks.clear();
        self.scratch_blocks
            .extend_from_slice(&self.cached_parent_blocks);
        self.scratch_values.clear();
        self.scratch_values.reserve(expected);
        for (&low, &slope) in self.cached_lows.iter().zip(&self.cached_slopes) {
            self.scratch_values.push(low + challenge * slope);
        }
        let coordinate = self.cycle_binding[self.round];
        self.eq_scale *= (F::one() - coordinate) + challenge * (coordinate + coordinate - F::one());

        std::mem::swap(&mut self.frontier_blocks, &mut self.scratch_blocks);
        std::mem::swap(&mut self.frontier_values, &mut self.scratch_values);
        self.cached_parent_blocks.clear();
        self.cached_lows.clear();
        self.cached_slopes.clear();
        self.cached_round = None;
        self.round += 1;
        self.validate_frontier()
    }

    pub fn terminal(&self) -> Result<RamHammingTerminal<F>, RamHammingError> {
        if self.round != self.rounds {
            return Err(RamHammingError::NotFullyBound {
                remaining: self.rounds - self.round,
            });
        }
        self.validate_frontier()?;
        let ram_hamming_weight = match (
            self.frontier_blocks.as_slice(),
            self.frontier_values.as_slice(),
        ) {
            ([], []) => F::zero(),
            ([0], [value]) => *value,
            _ => return Err(RamHammingError::InvalidTerminalFrontier),
        };
        Ok(RamHammingTerminal {
            ram_hamming_weight,
            eq_cycle: self.eq_scale,
        })
    }

    fn validate_frontier(&self) -> Result<(), RamHammingError> {
        let expected = usize::try_from(
            self.topology
                .census()
                .get(self.round)
                .ok_or(RamHammingError::MissingTopologyLevel { round: self.round })?
                .entries(),
        )
        .map_err(|_| RamHammingError::Overflow)?;
        if self.frontier_blocks.len() != expected || self.frontier_values.len() != expected {
            return Err(RamHammingError::FrontierLength {
                round: self.round,
                expected,
                blocks: self.frontier_blocks.len(),
                values: self.frontier_values.len(),
            });
        }
        Ok(())
    }
}

pub fn estimated_ram_hamming_products(
    owner: &RamCycleFamilyOwner,
) -> Result<u128, RamHammingError> {
    Ok(RamHammingSparsePlan::new(owner)?.estimated_products())
}

fn build_parent_weights<F: Field>(
    topology: &RamBlockTopology,
    cycle_binding: &[F],
) -> Result<Vec<Vec<F>>, RamHammingError> {
    let rounds = topology.log_t();
    let census = topology.census();
    let mut levels = vec![Vec::new(); rounds + 1];
    let root_entries = usize::try_from(
        census
            .get(rounds)
            .ok_or(RamHammingError::MissingTopologyLevel { round: rounds })?
            .entries(),
    )
    .map_err(|_| RamHammingError::Overflow)?;
    if root_entries > 1 {
        return Err(RamHammingError::InvalidRootCensus { got: root_entries });
    }
    if root_entries == 1 {
        levels[rounds].push(F::one());
    }

    for round in (1..rounds).rev() {
        let current_len = usize::try_from(
            census
                .get(round)
                .ok_or(RamHammingError::MissingTopologyLevel { round })?
                .entries(),
        )
        .map_err(|_| RamHammingError::Overflow)?;
        let merges = topology.merges_for_round(round)?;
        let parents = levels
            .get(round + 1)
            .ok_or(RamHammingError::MissingWeightLevel { round })?;
        if merges.len() != parents.len() {
            return Err(RamHammingError::WeightLength {
                round,
                expected: merges.len(),
                got: parents.len(),
            });
        }
        let mut current = vec![F::zero(); current_len];
        let mut filled = vec![false; current_len];
        let coordinate = cycle_binding[round];
        for (merge, &parent) in merges.iter().zip(parents) {
            for (child, high) in [(merge.low_state(), false), (merge.high_state(), true)] {
                let Some(child) = child else {
                    continue;
                };
                if child >= current.len() || filled[child] {
                    return Err(RamHammingError::InvalidWeightChild { round, child });
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
            return Err(RamHammingError::IncompleteWeightLevel { round });
        }
        levels[round] = current;
    }
    Ok(levels)
}

fn frontier_value<F: Field>(
    values: &[F],
    index: Option<usize>,
    round: usize,
) -> Result<F, RamHammingError> {
    index.map_or(Ok(F::zero()), |index| {
        values
            .get(index)
            .copied()
            .ok_or(RamHammingError::InvalidFrontierIndex { round, index })
    })
}

fn merge_parent_block(
    frontier_blocks: &[u64],
    merge: BlockMerge,
    round: usize,
) -> Result<u64, RamHammingError> {
    let low = merge
        .low_state()
        .map(|index| {
            frontier_blocks
                .get(index)
                .copied()
                .ok_or(RamHammingError::InvalidFrontierIndex { round, index })
        })
        .transpose()?;
    let high = merge
        .high_state()
        .map(|index| {
            frontier_blocks
                .get(index)
                .copied()
                .ok_or(RamHammingError::InvalidFrontierIndex { round, index })
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
        (None, None) => Err(RamHammingError::EmptyMerge { round }),
        _ => Err(RamHammingError::InvalidMergeChildren { round }),
    }
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum RamHammingError {
    #[error(transparent)]
    Topology(#[from] TopologyError),
    #[error("RAM Hamming cycle point has length {got}, expected {expected}")]
    CyclePointLength { expected: usize, got: usize },
    #[error("RAM Hamming topology is missing level {round}")]
    MissingTopologyLevel { round: usize },
    #[error("RAM Hamming weights are missing level {round}")]
    MissingWeightLevel { round: usize },
    #[error("RAM Hamming weight level {round} has {got} entries, expected {expected}")]
    WeightLength {
        round: usize,
        expected: usize,
        got: usize,
    },
    #[error(
        "RAM Hamming frontier at round {round} has {blocks} blocks and {values} values, expected {expected}"
    )]
    FrontierLength {
        round: usize,
        expected: usize,
        blocks: usize,
        values: usize,
    },
    #[error("RAM Hamming frontier index {index} is invalid at round {round}")]
    InvalidFrontierIndex { round: usize, index: usize },
    #[error("RAM Hamming cache is malformed at round {round}")]
    CacheLength {
        round: usize,
        expected: usize,
        lows: usize,
        slopes: usize,
    },
    #[error("RAM Hamming topology has an invalid root census of {got}")]
    InvalidRootCensus { got: usize },
    #[error("RAM Hamming weight child {child} is invalid at round {round}")]
    InvalidWeightChild { round: usize, child: usize },
    #[error("RAM Hamming weight level {round} is incomplete")]
    IncompleteWeightLevel { round: usize },
    #[error("RAM Hamming merge has no child at round {round}")]
    EmptyMerge { round: usize },
    #[error("RAM Hamming merge children disagree at round {round}")]
    InvalidMergeChildren { round: usize },
    #[error("RAM Hamming message was not prepared at round {round}")]
    MessageNotPrepared { round: usize },
    #[error("RAM Hamming sequence is already fully bound")]
    AlreadyFullyBound,
    #[error("RAM Hamming sequence is not fully bound; {remaining} rounds remain")]
    NotFullyBound { remaining: usize },
    #[error("RAM Hamming terminal frontier is invalid")]
    InvalidTerminalFrontier,
    #[error("RAM Hamming sparse plan does not match its owner receipt")]
    PlanReceiptMismatch,
    #[error("RAM Hamming arithmetic overflowed")]
    Overflow,
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_field::Prime128OffsetA7F7 as AkitaField;
    use jolt_field::Ring as _;

    use super::super::owner::{OwnerConfig, RamAccessRecord, RamIncrementRecord};
    use super::*;

    fn field(value: u64) -> AkitaField {
        AkitaField::from_u64(value)
    }

    fn fixture_owner() -> RamCycleFamilyOwner {
        let config = OwnerConfig::new(3, 3, 47, 16).unwrap();
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

    fn dense_eq(binding: &[AkitaField]) -> Vec<AkitaField> {
        (0..1usize << binding.len())
            .map(|index| {
                binding
                    .iter()
                    .enumerate()
                    .fold(AkitaField::one(), |value, (bit, coordinate)| {
                        value
                            * if index & (1 << bit) == 0 {
                                AkitaField::one() - *coordinate
                            } else {
                                *coordinate
                            }
                    })
            })
            .collect()
    }

    fn dense_message(hamming: &[AkitaField], eq: &[AkitaField]) -> [AkitaField; 4] {
        let mut coefficients = [AkitaField::zero(); 4];
        for pair in 0..hamming.len() / 2 {
            let low = 2 * pair;
            let high = low + 1;
            let h_0 = hamming[low];
            let h_slope = hamming[high] - h_0;
            let q = [
                h_0 * h_0 - h_0,
                h_slope * (h_0 + h_0 - AkitaField::one()),
                h_slope * h_slope,
            ];
            let l = [eq[low], eq[high] - eq[low]];
            coefficients[0] += l[0] * q[0];
            coefficients[1] += l[0] * q[1] + l[1] * q[0];
            coefficients[2] += l[0] * q[2] + l[1] * q[1];
            coefficients[3] += l[1] * q[2];
        }
        coefficients
    }

    fn bind_dense(values: &mut Vec<AkitaField>, challenge: AkitaField) {
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
        let binding = [field(7), field(11), field(13)];
        let mut sparse = HostSparseRamHammingBooleanity::new(Arc::clone(&owner), &binding).unwrap();
        let mut hamming = vec![AkitaField::zero(); owner.receipt().cycles()];
        for record in owner.access_records() {
            hamming[record.cycle() as usize] = AkitaField::one();
        }
        let mut eq = dense_eq(&binding);

        assert_eq!(sparse.frontier_values.len(), owner.access_records().len());
        assert!(estimated_ram_hamming_products(&owner).unwrap() < 1_000);
        for challenge in [field(17), field(19), field(23)] {
            assert_eq!(
                sparse.message().unwrap().coefficients(),
                &dense_message(&hamming, &eq)
            );
            sparse.bind(challenge).unwrap();
            bind_dense(&mut hamming, challenge);
            bind_dense(&mut eq, challenge);
        }
        let terminal = sparse.terminal().unwrap();
        assert_eq!(terminal.ram_hamming_weight(), hamming[0]);
        assert_eq!(terminal.eq_cycle(), eq[0]);
    }

    #[test]
    fn sparse_plan_freezes_one_topology_and_owner_receipt() {
        let owner = Arc::new(fixture_owner());
        let plan = RamHammingSparsePlan::new(&owner).unwrap();
        assert_eq!(plan.source_generation(), 47);
        assert_eq!(plan.source_fingerprint(), owner.receipt().fingerprint());
        assert_eq!(plan.log_t(), 3);
        assert_eq!(plan.access_leaves(), 4);
        assert_eq!(plan.parent_nodes(), 7);
        assert_eq!(plan.middle_nodes(), 6);
        assert_eq!(plan.estimated_products(), 85);
        assert_eq!(plan.topology_bytes(), 128);
        let topology_bytes = plan.topology_bytes();

        let sequence = HostSparseRamHammingBooleanity::new_from_plan(
            owner,
            &[field(7), field(11), field(13)],
            plan,
        )
        .unwrap();
        assert_eq!(sequence.num_rounds(), 3);
        assert!(sequence.owned_heap_bytes() > topology_bytes);
    }

    #[test]
    fn empty_support_remains_zero_without_dense_storage() {
        let config = OwnerConfig::new(3, 2, 53, 8).unwrap();
        let owner = Arc::new(
            RamCycleFamilyOwner::from_sparse_records(config, Vec::new(), Vec::new(), vec![0; 4])
                .unwrap(),
        );
        let mut sparse =
            HostSparseRamHammingBooleanity::new(owner, &[field(3), field(5), field(7)]).unwrap();
        for challenge in [field(11), field(13), field(17)] {
            assert_eq!(
                sparse.message().unwrap().coefficients(),
                &[AkitaField::zero(); 4]
            );
            sparse.bind(challenge).unwrap();
        }
        assert_eq!(
            sparse.terminal().unwrap().ram_hamming_weight(),
            AkitaField::zero()
        );
    }

    #[test]
    fn bind_requires_a_prepared_message() {
        let owner = Arc::new(fixture_owner());
        let mut sparse =
            HostSparseRamHammingBooleanity::new(owner, &[field(3), field(5), field(7)]).unwrap();
        assert_eq!(
            sparse.bind(field(11)),
            Err(RamHammingError::MessageNotPrepared { round: 0 })
        );
    }
}
