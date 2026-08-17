//! Shared sparse-frontier driver for the RAM cycle-family members.
//!
//! Every member collapses a sparse frontier of `(block, lane values)` pairs
//! along the shared block topology, one merge level per sumcheck round. The
//! driver owns the skeleton — census validation, merge legality, the cached
//! message/bind round mechanic and the uncached walk mechanic, terminal
//! collapse — while each member supplies its leaf seeding, per-merge value
//! math, weight construction, and output claim mapping.

use jolt_field::Field;
use thiserror::Error;

use super::owner::RamCycleFamilyOwner;
use super::topology::{BlockMerge, LevelCensus, RamBlockTopology, TopologyError};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RamCycleMember {
    ValCheck,
    RaClaimReduction,
    RaVirtualization,
    HammingBooleanity,
}

impl std::fmt::Display for RamCycleMember {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::ValCheck => "RAM value-check",
            Self::RaClaimReduction => "RAM RA claim",
            Self::RaVirtualization => "RAM RA virtualization",
            Self::HammingBooleanity => "RAM Hamming",
        })
    }
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum RamCycleError {
    #[error(transparent)]
    Topology(#[from] TopologyError),
    #[error("{member} address point has length {got}, expected {expected}")]
    AddressPointLength {
        member: RamCycleMember,
        expected: usize,
        got: usize,
    },
    #[error("{member} cycle point has length {got}, expected {expected}")]
    CyclePointLength {
        member: RamCycleMember,
        expected: usize,
        got: usize,
    },
    #[error("{member} address table has length {got}, expected {expected}")]
    AddressTableLength {
        member: RamCycleMember,
        expected: usize,
        got: usize,
    },
    #[error("{member} access address {address} is out of range")]
    AccessAddressOutOfRange {
        member: RamCycleMember,
        address: u32,
    },
    #[error("{member} increment cycle and value payloads differ in length")]
    IncrementPayloadLength { member: RamCycleMember },
    #[error("{member} union leaves do not match the owner topology")]
    UnionLeafMismatch { member: RamCycleMember },
    #[error("{member} chunk width {got} is unsupported")]
    ChunkBits { member: RamCycleMember, got: usize },
    #[error("{member} has unsupported factor count {factors}")]
    FactorCount {
        member: RamCycleMember,
        factors: usize,
    },
    #[error("{member} factor index {factor} is invalid")]
    InvalidFactorIndex {
        member: RamCycleMember,
        factor: usize,
    },
    #[error("{member} chunk shift {shift} is outside a u32 address")]
    ChunkShift {
        member: RamCycleMember,
        shift: usize,
    },
    #[error("{member} topology is missing level {round}")]
    MissingTopologyLevel {
        member: RamCycleMember,
        round: usize,
    },
    #[error("{member} weights are missing level {round}")]
    MissingWeightLevel {
        member: RamCycleMember,
        round: usize,
    },
    #[error("{member} weight level {round} has {got} entries, expected {expected}")]
    WeightLength {
        member: RamCycleMember,
        round: usize,
        expected: usize,
        got: usize,
    },
    #[error(
        "{member} frontier at round {round}, lane {lane:?}, has {got} entries, expected {expected}"
    )]
    FrontierLength {
        member: RamCycleMember,
        round: usize,
        lane: Option<usize>,
        expected: u64,
        got: u64,
    },
    #[error("{member} frontier index {index} is invalid at round {round}")]
    InvalidFrontierIndex {
        member: RamCycleMember,
        round: usize,
        index: usize,
    },
    #[error("{member} cache for lane {lane} is malformed at round {round}")]
    CacheLength {
        member: RamCycleMember,
        round: usize,
        lane: usize,
        expected: usize,
        lows: usize,
        slopes: usize,
    },
    #[error("{member} topology has an invalid root census of {got}")]
    InvalidRootCensus { member: RamCycleMember, got: usize },
    #[error("{member} weight child {child} is invalid at round {round}")]
    InvalidWeightChild {
        member: RamCycleMember,
        round: usize,
        child: usize,
    },
    #[error("{member} weight level {round} is incomplete")]
    IncompleteWeightLevel {
        member: RamCycleMember,
        round: usize,
    },
    #[error("{member} access cycle {cycle} is absent from the topology")]
    AccessOutsideTopology { member: RamCycleMember, cycle: u32 },
    #[error("{member} boolean index {index} is out of range")]
    BooleanIndex { member: RamCycleMember, index: u64 },
    #[error("{member} merge has no child at round {round}")]
    EmptyMerge {
        member: RamCycleMember,
        round: usize,
    },
    #[error("{member} merge children disagree at round {round}")]
    InvalidMergeChildren {
        member: RamCycleMember,
        round: usize,
    },
    #[error("{member} parent blocks are not strictly ordered at round {round}")]
    UnorderedParentBlocks {
        member: RamCycleMember,
        round: usize,
    },
    #[error("{member} message was not prepared at round {round}")]
    MessageNotPrepared {
        member: RamCycleMember,
        round: usize,
    },
    #[error("{member} is already fully bound")]
    AlreadyFullyBound { member: RamCycleMember },
    #[error("{member} is not fully bound; {remaining} rounds remain")]
    NotFullyBound {
        member: RamCycleMember,
        remaining: usize,
    },
    #[error("{member} terminal frontier is invalid")]
    InvalidTerminalFrontier { member: RamCycleMember },
    #[error("{member} LT index {index} is out of range")]
    LtIndexOutOfRange {
        member: RamCycleMember,
        index: usize,
    },
    #[error("{member} LT state is invalid")]
    InvalidLtState { member: RamCycleMember },
    #[error("{member} equality table is invalid")]
    InvalidEqualityTable { member: RamCycleMember },
    #[error("{member} dense cycle {cycle} is out of range")]
    DenseCycleOutOfRange { member: RamCycleMember, cycle: u64 },
    #[error("{member} dense table has length {got}, expected {expected}")]
    DenseTableLength {
        member: RamCycleMember,
        expected: usize,
        got: usize,
    },
    #[error("{member} dense oracle state is invalid")]
    InvalidDenseState { member: RamCycleMember },
    #[error("{member} sparse plan does not match its owner receipt")]
    PlanReceiptMismatch { member: RamCycleMember },
    #[error("{member} arithmetic overflowed")]
    Overflow { member: RamCycleMember },
}

/// Frontier state shared by every member: blocks plus one value lane per
/// committed factor, with scratch and cache buffers for the round mechanics.
///
/// Two round mechanics exist. The cached mechanic (`prepare_round` then
/// `bind_cached`) walks the merges once per round and reuses the cached
/// `(parent, low, slope)` triples to bind; the RA claim, RA virtualization,
/// and Hamming members use it. The walk mechanic (`walk_round` then
/// `bind_walk`) recomputes the merge pass on bind and additionally enforces
/// strictly increasing parent blocks; only the value-check member uses it.
pub(super) struct FrontierDriver<F> {
    member: RamCycleMember,
    blocks: Vec<u64>,
    lanes: Box<[Vec<F>]>,
    scratch_blocks: Vec<u64>,
    scratch_lanes: Box<[Vec<F>]>,
    cached_blocks: Vec<u64>,
    cached_lows: Box<[Vec<F>]>,
    cached_slopes: Box<[Vec<F>]>,
    cached_round: Option<usize>,
    merge_lows: Vec<F>,
    merge_highs: Vec<F>,
    merge_slopes: Vec<F>,
    round: usize,
    rounds: usize,
}

impl<F: Field> FrontierDriver<F> {
    pub(super) fn new(
        member: RamCycleMember,
        rounds: usize,
        blocks: Vec<u64>,
        lanes: Vec<Vec<F>>,
    ) -> Self {
        let capacity = blocks.len();
        let lane_count = lanes.len();
        let lane_buffers = || {
            (0..lane_count)
                .map(|_| Vec::with_capacity(capacity))
                .collect::<Box<[_]>>()
        };
        Self {
            member,
            blocks,
            lanes: lanes.into_boxed_slice(),
            scratch_blocks: Vec::with_capacity(capacity),
            scratch_lanes: lane_buffers(),
            cached_blocks: Vec::with_capacity(capacity),
            cached_lows: lane_buffers(),
            cached_slopes: lane_buffers(),
            cached_round: None,
            merge_lows: Vec::with_capacity(lane_count),
            merge_highs: Vec::with_capacity(lane_count),
            merge_slopes: Vec::with_capacity(lane_count),
            round: 0,
            rounds,
        }
    }

    copy_field_getters! { pub(super), {
        num_rounds => rounds: usize,
        round: usize,
    }}

    pub(super) fn frontier_len(&self) -> usize {
        self.blocks.len()
    }

    pub(super) fn owned_heap_bytes(&self) -> usize {
        let lane_bytes = |lanes: &[Vec<F>]| {
            std::mem::size_of_val(lanes)
                + lanes
                    .iter()
                    .map(|lane| lane.capacity() * std::mem::size_of::<F>())
                    .sum::<usize>()
        };
        self.blocks.capacity() * std::mem::size_of::<u64>()
            + self.scratch_blocks.capacity() * std::mem::size_of::<u64>()
            + self.cached_blocks.capacity() * std::mem::size_of::<u64>()
            + lane_bytes(&self.lanes)
            + lane_bytes(&self.scratch_lanes)
            + lane_bytes(&self.cached_lows)
            + lane_bytes(&self.cached_slopes)
    }

    pub(super) fn ensure_active(&self) -> Result<(), RamCycleError> {
        if self.round >= self.rounds {
            return Err(RamCycleError::AlreadyFullyBound {
                member: self.member,
            });
        }
        Ok(())
    }

    pub(super) fn validate_frontier(&self, census: &[LevelCensus]) -> Result<(), RamCycleError> {
        let expected = census
            .get(self.round)
            .ok_or(RamCycleError::MissingTopologyLevel {
                member: self.member,
                round: self.round,
            })?
            .entries();
        let got = self.frontier_width(self.blocks.len())?;
        if got != expected {
            return Err(RamCycleError::FrontierLength {
                member: self.member,
                round: self.round,
                lane: None,
                expected,
                got,
            });
        }
        for (lane, values) in self.lanes.iter().enumerate() {
            let got = self.frontier_width(values.len())?;
            if got != expected {
                return Err(RamCycleError::FrontierLength {
                    member: self.member,
                    round: self.round,
                    lane: Some(lane),
                    expected,
                    got,
                });
            }
        }
        Ok(())
    }

    fn frontier_width(&self, len: usize) -> Result<u64, RamCycleError> {
        u64::try_from(len).map_err(|_| RamCycleError::Overflow {
            member: self.member,
        })
    }

    /// Cached message walk: validates the frontier, checks the parent weight
    /// level, caches `(parent block, per-lane low/slope)`, and folds each
    /// merge as `fold(weight, lows, highs, slopes)`.
    pub(super) fn prepare_round<W: Copy>(
        &mut self,
        topology: &RamBlockTopology,
        weights: &[Vec<W>],
        mut fold: impl FnMut(W, &[F], &[F], &[F]),
    ) -> Result<(), RamCycleError> {
        self.ensure_active()?;
        self.validate_frontier(topology.census())?;
        let merges = topology.merges_for_round(self.round)?;
        let level = weights
            .get(self.round + 1)
            .ok_or(RamCycleError::MissingWeightLevel {
                member: self.member,
                round: self.round,
            })?;
        if level.len() != merges.len() {
            return Err(RamCycleError::WeightLength {
                member: self.member,
                round: self.round,
                expected: merges.len(),
                got: level.len(),
            });
        }

        self.cached_blocks.clear();
        for lane in self
            .cached_lows
            .iter_mut()
            .chain(self.cached_slopes.iter_mut())
        {
            lane.clear();
        }
        for (merge, &weight) in merges.iter().zip(level) {
            let parent = merge_parent_block(self.member, &self.blocks, *merge, self.round)?;
            self.cached_blocks.push(parent);
            self.merge_lows.clear();
            self.merge_highs.clear();
            self.merge_slopes.clear();
            for (lane, values) in self.lanes.iter().enumerate() {
                let low = lane_value(self.member, values, merge.low_state(), self.round)?;
                let high = lane_value(self.member, values, merge.high_state(), self.round)?;
                let slope = high - low;
                self.cached_lows[lane].push(low);
                self.cached_slopes[lane].push(slope);
                self.merge_lows.push(low);
                self.merge_highs.push(high);
                self.merge_slopes.push(slope);
            }
            fold(
                weight,
                &self.merge_lows,
                &self.merge_highs,
                &self.merge_slopes,
            );
        }
        self.cached_round = Some(self.round);
        Ok(())
    }

    /// Cached bind: rebuilds each lane as `low + challenge * slope` from the
    /// cache, runs `update_scale` (the member's eq-scale fold) before the
    /// frontier swap, then advances and revalidates.
    pub(super) fn bind_cached(
        &mut self,
        topology: &RamBlockTopology,
        challenge: F,
        update_scale: impl FnOnce(),
    ) -> Result<(), RamCycleError> {
        self.ensure_active()?;
        if self.cached_round != Some(self.round) {
            return Err(RamCycleError::MessageNotPrepared {
                member: self.member,
                round: self.round,
            });
        }
        let expected = self.cached_blocks.len();
        for (lane, (lows, slopes)) in self
            .cached_lows
            .iter()
            .zip(self.cached_slopes.iter())
            .enumerate()
        {
            if lows.len() != expected || slopes.len() != expected {
                return Err(RamCycleError::CacheLength {
                    member: self.member,
                    round: self.round,
                    lane,
                    expected,
                    lows: lows.len(),
                    slopes: slopes.len(),
                });
            }
        }

        self.scratch_blocks.clear();
        self.scratch_blocks.extend_from_slice(&self.cached_blocks);
        for (lane, scratch) in self.scratch_lanes.iter_mut().enumerate() {
            scratch.clear();
            for (&low, &slope) in self.cached_lows[lane].iter().zip(&self.cached_slopes[lane]) {
                scratch.push(low + challenge * slope);
            }
        }
        update_scale();

        std::mem::swap(&mut self.blocks, &mut self.scratch_blocks);
        std::mem::swap(&mut self.lanes, &mut self.scratch_lanes);
        self.cached_blocks.clear();
        for lane in self
            .cached_lows
            .iter_mut()
            .chain(self.cached_slopes.iter_mut())
        {
            lane.clear();
        }
        self.cached_round = None;
        self.round += 1;
        self.validate_frontier(topology.census())
    }

    /// Stateless message walk: folds each merge as
    /// `fold(parent block, lows, highs)` without touching the cache.
    pub(super) fn walk_round(
        &self,
        topology: &RamBlockTopology,
        mut fold: impl FnMut(u64, &[F], &[F]) -> Result<(), RamCycleError>,
    ) -> Result<(), RamCycleError> {
        self.ensure_active()?;
        self.validate_frontier(topology.census())?;
        let merges = topology.merges_for_round(self.round)?;
        let mut lows = Vec::with_capacity(self.lanes.len());
        let mut highs = Vec::with_capacity(self.lanes.len());
        for merge in merges {
            let parent = merge_parent_block(self.member, &self.blocks, *merge, self.round)?;
            lows.clear();
            highs.clear();
            for values in &self.lanes {
                lows.push(lane_value(
                    self.member,
                    values,
                    merge.low_state(),
                    self.round,
                )?);
                highs.push(lane_value(
                    self.member,
                    values,
                    merge.high_state(),
                    self.round,
                )?);
            }
            fold(parent, &lows, &highs)?;
        }
        Ok(())
    }

    /// Uncached bind: recomputes the merge pass, enforces strictly increasing
    /// parent blocks, rebuilds each lane as `low + challenge * (high - low)`,
    /// runs `after_walk` (the member's companion bind) before the frontier
    /// swap, then advances and revalidates.
    pub(super) fn bind_walk(
        &mut self,
        topology: &RamBlockTopology,
        challenge: F,
        after_walk: impl FnOnce() -> Result<(), RamCycleError>,
    ) -> Result<(), RamCycleError> {
        self.ensure_active()?;
        self.validate_frontier(topology.census())?;
        let merges = topology.merges_for_round(self.round)?;
        self.scratch_blocks.clear();
        for lane in &mut self.scratch_lanes {
            lane.clear();
        }
        let mut previous_parent = None;
        for merge in merges {
            let parent = merge_parent_block(self.member, &self.blocks, *merge, self.round)?;
            if previous_parent.is_some_and(|previous| previous >= parent) {
                return Err(RamCycleError::UnorderedParentBlocks {
                    member: self.member,
                    round: self.round,
                });
            }
            previous_parent = Some(parent);
            self.scratch_blocks.push(parent);
            for (lane, values) in self.lanes.iter().enumerate() {
                let low = lane_value(self.member, values, merge.low_state(), self.round)?;
                let high = lane_value(self.member, values, merge.high_state(), self.round)?;
                self.scratch_lanes[lane].push(low + challenge * (high - low));
            }
        }
        after_walk()?;
        std::mem::swap(&mut self.blocks, &mut self.scratch_blocks);
        std::mem::swap(&mut self.lanes, &mut self.scratch_lanes);
        self.round += 1;
        self.validate_frontier(topology.census())
    }

    /// Terminal collapse shared by every member: after the last round the
    /// frontier is either empty (`None`, an all-zero claim) or the single
    /// block 0 whose lane values are the outputs.
    pub(super) fn terminal_values(
        &self,
        census: &[LevelCensus],
    ) -> Result<Option<&[Vec<F>]>, RamCycleError> {
        if self.round != self.rounds {
            return Err(RamCycleError::NotFullyBound {
                member: self.member,
                remaining: self.rounds - self.round,
            });
        }
        self.validate_frontier(census)?;
        match self.blocks.as_slice() {
            [] => Ok(None),
            [0] => Ok(Some(&self.lanes)),
            _ => Err(RamCycleError::InvalidTerminalFrontier {
                member: self.member,
            }),
        }
    }
}

fn lane_value<F: Field>(
    member: RamCycleMember,
    values: &[F],
    index: Option<usize>,
    round: usize,
) -> Result<F, RamCycleError> {
    index.map_or(Ok(F::zero()), |index| {
        values
            .get(index)
            .copied()
            .ok_or(RamCycleError::InvalidFrontierIndex {
                member,
                round,
                index,
            })
    })
}

fn merge_parent_block(
    member: RamCycleMember,
    frontier_blocks: &[u64],
    merge: BlockMerge,
    round: usize,
) -> Result<u64, RamCycleError> {
    let low = merge
        .low_state()
        .map(|index| {
            frontier_blocks
                .get(index)
                .copied()
                .ok_or(RamCycleError::InvalidFrontierIndex {
                    member,
                    round,
                    index,
                })
        })
        .transpose()?;
    let high = merge
        .high_state()
        .map(|index| {
            frontier_blocks
                .get(index)
                .copied()
                .ok_or(RamCycleError::InvalidFrontierIndex {
                    member,
                    round,
                    index,
                })
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
        (None, None) => Err(RamCycleError::EmptyMerge { member, round }),
        _ => Err(RamCycleError::InvalidMergeChildren { member, round }),
    }
}

/// Builds the per-level parent weights for the weighted members: the root
/// carries `root_seed`, and each child inherits
/// `child_weight(parent, is_high_child, round)`.
pub(super) fn build_parent_weights<W: Copy>(
    member: RamCycleMember,
    topology: &RamBlockTopology,
    zero: W,
    root_seed: W,
    mut child_weight: impl FnMut(&W, bool, usize) -> W,
) -> Result<Vec<Vec<W>>, RamCycleError> {
    let rounds = topology.log_t();
    let census = topology.census();
    let mut levels = vec![Vec::new(); rounds + 1];
    let root_entries = usize::try_from(
        census
            .get(rounds)
            .ok_or(RamCycleError::MissingTopologyLevel {
                member,
                round: rounds,
            })?
            .entries(),
    )
    .map_err(|_| RamCycleError::Overflow { member })?;
    if root_entries > 1 {
        return Err(RamCycleError::InvalidRootCensus {
            member,
            got: root_entries,
        });
    }
    if root_entries == 1 {
        levels[rounds].push(root_seed);
    }

    for round in (1..rounds).rev() {
        let current_len = usize::try_from(
            census
                .get(round)
                .ok_or(RamCycleError::MissingTopologyLevel { member, round })?
                .entries(),
        )
        .map_err(|_| RamCycleError::Overflow { member })?;
        let merges = topology.merges_for_round(round)?;
        let parents = levels
            .get(round + 1)
            .ok_or(RamCycleError::MissingWeightLevel { member, round })?;
        if merges.len() != parents.len() {
            return Err(RamCycleError::WeightLength {
                member,
                round,
                expected: merges.len(),
                got: parents.len(),
            });
        }
        let mut current = vec![zero; current_len];
        let mut filled = vec![false; current_len];
        for (merge, parent) in merges.iter().zip(parents) {
            for (child, high) in [(merge.low_state(), false), (merge.high_state(), true)] {
                let Some(child) = child else {
                    continue;
                };
                if child >= current.len() || filled[child] {
                    return Err(RamCycleError::InvalidWeightChild {
                        member,
                        round,
                        child,
                    });
                }
                current[child] = child_weight(parent, high, round);
                filled[child] = true;
            }
        }
        if filled.iter().any(|filled| !filled) {
            return Err(RamCycleError::IncompleteWeightLevel { member, round });
        }
        levels[round] = current;
    }
    Ok(levels)
}

pub(super) fn weight_level_bytes<W>(levels: &[Vec<W>]) -> usize {
    std::mem::size_of_val(levels)
        + levels
            .iter()
            .map(|level| level.capacity() * std::mem::size_of::<W>())
            .sum::<usize>()
}

pub(super) fn eq_at_boolean_index<F: Field>(
    member: RamCycleMember,
    point: &[F],
    index: u64,
) -> Result<F, RamCycleError> {
    if point.len() >= u64::BITS as usize || index >= (1u64 << point.len()) {
        return Err(RamCycleError::BooleanIndex { member, index });
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

/// Walks the owner topology's leaf cycles in order, pairing each with the
/// access record at that cycle (if any). Every access record must land on a
/// leaf cycle.
pub(super) fn for_each_leaf_access(
    member: RamCycleMember,
    owner: &RamCycleFamilyOwner,
    mut leaf: impl FnMut(u32, Option<u32>) -> Result<(), RamCycleError>,
) -> Result<(), RamCycleError> {
    let records = owner.access_records();
    let mut record_index = 0;
    for &cycle in owner.block_topology().leaf_cycles() {
        let address = match records.get(record_index) {
            Some(record) if record.cycle() == cycle => {
                record_index += 1;
                Some(record.address())
            }
            Some(record) if record.cycle() < cycle => {
                return Err(RamCycleError::AccessOutsideTopology {
                    member,
                    cycle: record.cycle(),
                });
            }
            _ => None,
        };
        leaf(cycle, address)?;
    }
    if let Some(record) = records.get(record_index) {
        return Err(RamCycleError::AccessOutsideTopology {
            member,
            cycle: record.cycle(),
        });
    }
    Ok(())
}
