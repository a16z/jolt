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

use super::frontier::{
    build_parent_weights, eq_at_boolean_index, for_each_leaf_access, weight_level_bytes,
    FrontierDriver, RamCycleError, RamCycleMember,
};
use super::owner::RamCycleFamilyOwner;

const MEMBER: RamCycleMember = RamCycleMember::RaVirtualization;

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
    core: FrontierDriver<F>,
    parent_weights: Vec<Vec<F>>,
    cycle_point: Vec<F>,
    eq_scale: F,
    factors: usize,
}

impl<F: Field> HostSparseRamRaVirtualization<F> {
    pub fn new(
        owner: Arc<RamCycleFamilyOwner>,
        r_address: &[F],
        committed_chunk_bits: usize,
        r_cycle: &[F],
    ) -> Result<Self, RamCycleError> {
        validate_chunk_bits(committed_chunk_bits)?;
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

        let chunks = committed_address_chunks(r_address, committed_chunk_bits);
        let factors = chunks.len();
        if factors == 0 || factors > MAX_RAM_RA_VIRTUALIZATION_FACTORS {
            return Err(RamCycleError::FactorCount {
                member: MEMBER,
                factors,
            });
        }
        let (blocks, lanes) = seed_frontier(&owner, &chunks, committed_chunk_bits)?;
        let rounds = receipt.log_t();
        let parent_weights = build_parent_weights(
            MEMBER,
            owner.block_topology(),
            F::zero(),
            F::one(),
            |&parent, high, round| {
                let coordinate = r_cycle[rounds - 1 - round];
                parent
                    * if high {
                        coordinate
                    } else {
                        F::one() - coordinate
                    }
            },
        )?;
        let core = FrontierDriver::new(MEMBER, rounds, blocks, lanes);
        core.validate_frontier(owner.block_topology().census())?;
        Ok(Self {
            owner,
            core,
            parent_weights,
            cycle_point: r_cycle.to_vec(),
            eq_scale: F::one(),
            factors,
        })
    }

    pub fn owned_heap_bytes(&self) -> usize {
        self.core.owned_heap_bytes()
            + weight_level_bytes(&self.parent_weights)
            + self.cycle_point.capacity() * std::mem::size_of::<F>()
    }

    pub const fn num_rounds(&self) -> usize {
        self.core.num_rounds()
    }

    pub const fn round(&self) -> usize {
        self.core.round()
    }

    pub fn message(&mut self) -> Result<RamRaVirtualizationMessage<F>, RamCycleError> {
        self.core.ensure_active()?;
        let round = self.core.round();
        let rounds = self.core.num_rounds();
        let coordinate = self.cycle_point[rounds - 1 - round];
        let eq_at_zero = F::one() - coordinate;
        let eq_step = coordinate + coordinate - F::one();
        let factors = self.factors;
        let points = factors + 2;
        let eq_scale = self.eq_scale;
        let mut evaluations = [F::zero(); MAX_RAM_RA_VIRTUALIZATION_EVALUATIONS];
        let mut factor_values = [F::zero(); MAX_RAM_RA_VIRTUALIZATION_FACTORS];
        let mut factor_slopes = [F::zero(); MAX_RAM_RA_VIRTUALIZATION_FACTORS];

        self.core.prepare_round(
            self.owner.block_topology(),
            &self.parent_weights,
            |weight, lows, _highs, slopes| {
                factor_values[..factors].copy_from_slice(&lows[..factors]);
                factor_slopes[..factors].copy_from_slice(&slopes[..factors]);
                let weighted_scale = weight * eq_scale;
                let mut eq_value = weighted_scale * eq_at_zero;
                let eq_delta = weighted_scale * eq_step;
                for evaluation in &mut evaluations[..points] {
                    let mut product = factor_values[0];
                    for &value in &factor_values[1..factors] {
                        product *= value;
                    }
                    *evaluation += eq_value * product;
                    for factor in 0..factors {
                        factor_values[factor] += factor_slopes[factor];
                    }
                    eq_value += eq_delta;
                }
            },
        )?;
        Ok(RamRaVirtualizationMessage {
            evaluations,
            len: points,
        })
    }

    pub fn bind(&mut self, challenge: F) -> Result<(), RamCycleError> {
        let round = self.core.round();
        let rounds = self.core.num_rounds();
        let cycle_point = &self.cycle_point;
        let eq_scale = &mut self.eq_scale;
        self.core
            .bind_cached(self.owner.block_topology(), challenge, || {
                let coordinate = cycle_point[rounds - 1 - round];
                *eq_scale *=
                    (F::one() - coordinate) + challenge * (coordinate + coordinate - F::one());
            })
    }

    pub fn terminal(&self) -> Result<RamRaVirtualizationTerminal<F>, RamCycleError> {
        let lanes = self
            .core
            .terminal_values(self.owner.block_topology().census())?;
        let mut ram_ra = [F::zero(); MAX_RAM_RA_VIRTUALIZATION_FACTORS];
        if let Some(lanes) = lanes {
            for (factor, values) in lanes.iter().enumerate() {
                ram_ra[factor] = values[0];
            }
        }
        Ok(RamRaVirtualizationTerminal {
            ram_ra,
            factors: self.factors,
            eq_cycle: self.eq_scale,
        })
    }
}

pub fn estimated_ram_ra_virtualization_products(
    owner: &RamCycleFamilyOwner,
    committed_chunk_bits: usize,
) -> Result<u128, RamCycleError> {
    let overflow = || RamCycleError::Overflow { member: MEMBER };
    validate_chunk_bits(committed_chunk_bits)?;
    let receipt = owner.receipt();
    let factors = receipt.log_k().div_ceil(committed_chunk_bits);
    if factors == 0 || factors > MAX_RAM_RA_VIRTUALIZATION_FACTORS {
        return Err(RamCycleError::FactorCount {
            member: MEMBER,
            factors,
        });
    }
    let census = owner.block_topology().census();
    let parent_nodes = census.iter().skip(1).try_fold(0u128, |sum, level| {
        sum.checked_add(u128::from(level.entries()))
    });
    let parent_nodes = parent_nodes.ok_or_else(overflow)?;
    let middle_nodes = census
        .iter()
        .skip(1)
        .take(receipt.log_t().saturating_sub(1))
        .try_fold(0u128, |sum, level| {
            sum.checked_add(u128::from(level.entries()))
        })
        .ok_or_else(overflow)?;
    let factors = u128::try_from(factors).map_err(|_| overflow())?;
    let chunk_bits = u128::try_from(committed_chunk_bits).map_err(|_| overflow())?;
    let rounds = u128::try_from(receipt.log_t()).map_err(|_| overflow())?;
    let address_products = u128::try_from(receipt.access_count())
        .map_err(|_| overflow())?
        .checked_mul(factors)
        .and_then(|value| value.checked_mul(chunk_bits))
        .ok_or_else(overflow)?;
    let message_products_per_parent = factors
        .checked_mul(factors.checked_add(2).ok_or_else(overflow)?)
        .and_then(|value| value.checked_add(3))
        .ok_or_else(overflow)?;
    let message_products = parent_nodes
        .checked_mul(message_products_per_parent)
        .ok_or_else(overflow)?;
    let bind_products = parent_nodes.checked_mul(factors).ok_or_else(overflow)?;
    address_products
        .checked_add(middle_nodes)
        .and_then(|value| value.checked_add(message_products))
        .and_then(|value| value.checked_add(bind_products))
        .and_then(|value| value.checked_add(rounds.checked_mul(2)?))
        .ok_or_else(overflow)
}

fn validate_chunk_bits(committed_chunk_bits: usize) -> Result<(), RamCycleError> {
    if committed_chunk_bits == 0 || committed_chunk_bits > u32::BITS as usize {
        Err(RamCycleError::ChunkBits {
            member: MEMBER,
            got: committed_chunk_bits,
        })
    } else {
        Ok(())
    }
}

fn seed_frontier<F: Field>(
    owner: &RamCycleFamilyOwner,
    chunks: &[Vec<F>],
    committed_chunk_bits: usize,
) -> Result<(Vec<u64>, Vec<Vec<F>>), RamCycleError> {
    let leaves = owner.block_topology().leaf_cycles().len();
    let mut blocks = Vec::with_capacity(leaves);
    let mut lanes = chunks
        .iter()
        .map(|_| Vec::with_capacity(leaves))
        .collect::<Vec<_>>();
    for_each_leaf_access(MEMBER, owner, |cycle, address| {
        blocks.push(u64::from(cycle));
        for (factor, chunk) in chunks.iter().enumerate() {
            let value = match address {
                Some(address) => {
                    let index = address_chunk(address, factor, chunks.len(), committed_chunk_bits)?;
                    eq_at_boolean_index(MEMBER, chunk, u64::from(index))?
                }
                None => F::zero(),
            };
            lanes[factor].push(value);
        }
        Ok(())
    })?;
    Ok((blocks, lanes))
}

fn address_chunk(
    address: u32,
    factor: usize,
    factors: usize,
    committed_chunk_bits: usize,
) -> Result<u32, RamCycleError> {
    let remaining = factors
        .checked_sub(factor + 1)
        .ok_or(RamCycleError::InvalidFactorIndex {
            member: MEMBER,
            factor,
        })?;
    let shift = remaining
        .checked_mul(committed_chunk_bits)
        .ok_or(RamCycleError::Overflow { member: MEMBER })?;
    if shift >= u32::BITS as usize {
        return Err(RamCycleError::ChunkShift {
            member: MEMBER,
            shift,
        });
    }
    let mask = if committed_chunk_bits == u32::BITS as usize {
        u32::MAX
    } else {
        (1u32 << committed_chunk_bits) - 1
    };
    Ok((address >> shift) & mask)
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
                                eq_at_boolean_index(MEMBER, chunk, u64::from(index)).unwrap()
                            })
                        })
                        .collect()
                })
                .collect();
            let eq_cycle = (0..owner.receipt().cycles())
                .map(|index| eq_at_boolean_index(MEMBER, r_cycle, index as u64).unwrap())
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
